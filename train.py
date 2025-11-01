# In (root)/train.py
import jax
# === BEGIN JAX INIT FIX ===
import jax.numpy as jnp
from tqdm import trange
import numpy as np
import optax
import networkx as nx
import pickle

from numpy.random import default_rng
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io

import json
import pandas as pd
from dag_gflownet.moe import MoEDAGGFlowNet # <-- IMPORT NEW CLASS
# Note: NpEncoder is often in utils, but defining it here is fine.
# from dag_gflownet.utils import NpEncoder

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and JAX types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # === FIX: Handle JAX array types (e.g., ArrayImpl, DeviceArray) ===
        # JAX arrays expose a .tolist() method for conversion.
        if hasattr(obj, 'tolist'):
             return obj.tolist()
        # ==================================================================

        return super(NpEncoder, self).default(obj)

def main(args):
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Create the environment
    scorer, data, graph = get_scorer(args, rng=rng)
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context,
        cache_max_size=args.cache_max_size  # <-- ADD THIS
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables
    )

    # --- Define dummy inputs for model initialization ---
    dummy_s_0_adj = replay.dummy['adjacency']
    dummy_s_0_mask = replay.dummy['mask']
    # ----------------------------------------------------------

    # Create the GFlowNet & initialize parameters
    optimizer = optax.adam(args.lr)

    if args.model_type == 'baseline':
        gflownet = DAGGFlowNet(
            delta=args.delta,
            update_target_every=args.update_target_every
        )
    else: # 'moe'
        gflownet = MoEDAGGFlowNet(
            config=args, # Pass all args
            delta=args.delta,
            update_target_every=args.update_target_every
        )

    params, state = gflownet.init(
        subkey,
        optimizer,
        dummy_s_0_adj,
        dummy_s_0_mask
    )

    # In jax-dag-gflownet/train.py, line 208
    exploration_schedule = jax.jit(optax.linear_schedule(
        init_value=jnp.array(1.0),  # <-- START at 100% exploration
        end_value=jnp.array(args.min_exploration), # <-- END at the minimum rate
        transition_steps=args.num_iterations // 2,
        transition_begin=args.prefill,
    ))

    # Training loop
    indices = None
    observations_tuple = env.reset()
    current_obs_dict = observations_tuple[0]
    results_history = []
    with trange(args.prefill + args.num_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            actions, key, logs = gflownet.act(params.online, key, current_obs_dict, epsilon)
            next_observations, delta_scores, dones, truncated, _ = env.step(np.asarray(actions))
            # Must include logs['is_exploration']
            # === FIX: Pass only the obs_dict (index 0) from the obs tuples ===
            indices = replay.add(
                current_obs_dict,         # Pass the old obs_dict
                actions,
                logs['is_exploration'],
                next_observations,     # CORRECTED: from env.step()
                delta_scores,
                dones,
                prev_indices=indices
            )
            #-------------------------------------------

            current_obs_dict = next_observations

            if iteration >= args.prefill:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=args.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples)

                pbar.set_postfix(
                    loss=f"{logs['loss']:.2f}",
                    L_gfn=f"{logs.get('l_gfn', logs['loss']):.2f}",
                    L_bal=f"{logs.get('l_balance', 0):.2f}",
                    L_div=f"{logs.get('l_diversity', 0):.2f}",
                    eps=f"{epsilon:.2f}"
                )
                logs['step'] = iteration
                results_history.append(logs)

            if (iteration + 1) % args.eval_every == 0 and iteration >= args.prefill:
                eval_posterior, _ = posterior_estimate(
                    gflownet, params.online, env, key,
                    num_samples=args.eval_samples, verbose=False
                )

                ground_truth = nx.to_numpy_array(graph, weight=None)
                eval_shd = expected_shd(eval_posterior, ground_truth)

                if results_history: # Ensure list is not empty
                    results_history[-1]['eval_shd'] = eval_shd

                # --- FIX ---
                # Reconstruct the full postfix dictionary instead of unpacking pbar.postfix
                # This avoids the TypeError.
                pbar.set_postfix(
                    loss=f"{logs['loss']:.2f}",
                    L_gfn=f"{logs.get('l_gfn', logs['loss']):.2f}",
                    L_bal=f"{logs.get('l_balance', 0):.2f}",
                    L_div=f"{logs.get('l_diversity', 0):.2f}",
                    eps=f"{epsilon:.2f}",
                    eval_shd=f"{eval_shd:.2f}"  # Add the new key
                )

    # Evaluate the posterior estimate
    posterior, _ = posterior_estimate(
        gflownet, params.online, env, key,
        # Use 'num_samples_posterior' from args
        num_samples=args.num_samples_posterior, verbose=True
    )

    # Compute the metrics
    ground_truth = nx.to_numpy_array(graph, weight=None)
    results = {
        'expected_shd': expected_shd(posterior, ground_truth),
        'expected_edges': expected_edges(posterior),
        **threshold_metrics(posterior, ground_truth)
    }

    # Save model, data & results
    args.output_folder.mkdir(parents=True, exist_ok=True)

    data.to_csv(args.output_folder / 'data.csv')
    with open(args.output_folder / 'graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    io.save(args.output_folder / 'model.npz', params=params.online)
    replay.save(args.output_folder / 'replay_buffer.npz')
    np.save(args.output_folder / 'posterior.npy', posterior)

    # Save training history and final results
    results['training_history'] = results_history
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=4)

    # Save arguments
    with open(args.output_folder / 'arguments.json', 'w') as f:
        # Convert Path object to string for JSON serialization
        args_dict = vars(args).copy()
        args_dict['output_folder'] = str(args_dict['output_folder'])
        # Handle data_path if it exists (it might not for Sachs)
        if 'data_path' in args_dict and args_dict['data_path'] is not None:
             args_dict['data_path'] = str(args_dict['data_path'])

        json.dump(args_dict, f, cls=NpEncoder, indent=4)

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import json

    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
        help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=150_000,
        help='Number of iterations (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=150_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')

    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--update_epsilon_every', type=int, default=10,
        help='Frequency of update for epsilon (default: %(default)s)')

    # Mixture of Experts (MoE)
    moe = parser.add_argument_group('Mixture of Experts (MoE)')
    moe.add_argument('--model_type', type=str, default='baseline',
        choices=['baseline', 'moe'],
        help='Which model to train (default: %(default)s)')
    moe.add_argument('--num_experts', type=int, default=5,
        help='Number of experts (default: %(default)s)')
    moe.add_argument('--context_dim', type=int, default=64,
        help='Dimension of the global context vector for AHR (default: %(default)s)')
    moe.add_argument('--top_k', type=int, default=2,
        help='Top-K routing (default: %(default)s)')
    moe.add_argument('--load_balance_alpha', type=float, default=0.1,
        help='Coefficient for load balancing loss (default: %(default)s)')
    moe.add_argument('--diversity_beta', type=float, default=2.0,
        help='Coefficient for diversity loss (default: %(default)s)')

    # Ablation flags
    moe.add_argument('--use_ahr', action='store_true',
        help='Enable AHR gating (global context)')
    moe.add_argument('--use_diversity', action='store_true',
        help='Enable diversity loss')
    moe.add_argument('--use_load_balancing', action='store_true',
        help='Enable load balancing loss')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--cache_max_size', type=int, default=500_000,
    help='Maximum size of the scorer cache (default: %(default)s)')
    misc.add_argument('--eval_every', type=int, default=1000,
        help='Run evaluation every N steps (default: %(default)s)')
    misc.add_argument('--eval_samples', type=int, default=100,
        help='Number of samples for intermediate eval (default: %(default)s)')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--update_target_every', type=int, default=1000,
        help='Frequency of update for the target network (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--output_folder', type=Path, default='output',
        help='Output folder (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
        help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
        help='Number of samples')

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser('sachs_continuous')

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser('sachs_interventional')

    args = parser.parse_args()

    main(args)
