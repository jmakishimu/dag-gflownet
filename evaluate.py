import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import pickle
import json
import pandas as pd
import argparse
from argparse import Namespace
from pathlib import Path
import math

# --- Imports from your project ---
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.moe import MoEDAGGFlowNet
from dag_gflownet.utils.factories import get_prior
from dag_gflownet.utils.gflownet import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io
from dag_gflownet.scores import BDeScore, BGeScore

# --- FIX: Remove get_data, as we must load, not regenerate ---
# from dag_gflownet.utils.data import get_data

def get_score_name(graph_type):
    """Helper to determine score type from graph type in args."""
    if graph_type == 'erdos_renyi_lingauss':
        return 'bge'
    elif graph_type == 'sachs_continuous':
        return 'bge'
    elif graph_type == 'sachs_interventional':
        return 'bde'
    raise ValueError(f"Unknown graph type: {graph_type}")

# --- Post-Processing Logic (Adapted from plot_results.py) ---

def calculate_shd(G_true, G_pred):
    """Calculates the Structural Hamming Distance (SHD) between two graphs."""
    if not isinstance(G_true, nx.DiGraph) or not isinstance(G_pred, nx.DiGraph):
        if isinstance(G_true, nx.DiGraph) and G_pred.number_of_nodes() == 0:
            return G_true.number_of_edges()
        raise TypeError(f"Inputs must be nx.DiGraph. Got {type(G_true)} and {type(G_pred)}")

    true_edges = set(G_true.edges())
    pred_edges = set(G_pred.edges())

    missing_edges = len(true_edges - pred_edges)

    false_positive_or_wrong_direction = 0
    for u, v in pred_edges:
        if (u, v) not in true_edges:
            if (v, u) not in true_edges:
                false_positive_or_wrong_direction += 1 # False Positive
            else:
                false_positive_or_wrong_direction += 1 # Wrong Direction (Penalty: 1)

    shd = missing_edges + false_positive_or_wrong_direction
    return shd

def find_elbow_point(eml_values):
    """Finds the elbow point using the largest distance from the line method."""
    points = np.array(eml_values)
    if len(points) < 3:
        return len(points)

    n_points = len(points)
    x_data = np.arange(n_points)

    # Rescale coordinates to be between 0 and 1
    x_norm = x_data / x_data[-1] if x_data[-1] > 0 else x_data
    if np.max(points) - np.min(points) < 1e-6:
         y_norm = np.zeros_like(points)
    else:
        y_norm = (points - np.min(points)) / (np.max(points) - np.min(points))

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    max_dist = -1
    elbow_index = n_points - 1

    for i in range(1, n_points - 1):
        p3 = np.array([x_norm[i], y_norm[i]])

        # Distance from point to line (p1, p2)
        numerator = np.abs(
            (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]
        )
        denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

        distance = numerator / denominator if denominator > 1e-6 else 0

        if distance > max_dist:
            max_dist = distance
            elbow_index = i

    return elbow_index + 1
# In evaluate.py

def post_processing_refinement(G_true, posterior, scorer):
    """
    Performs mode refinement using EML and adaptive sparsity based on the Elbow method.
    """
    if scorer is None:
        print("Skipping refinement: Scorer is None.")
        return np.nan

    num_nodes = G_true.number_of_nodes()
    # Posterior marginal edge probabilities (EML)
    posterior_probs = np.mean(posterior, axis=0)

    # --- FIX 1: Get the mapping from integer index (0..N-1) to the
    # node names used in G_true (e.g., 'A', 'B', 'C' or 'Raf', 'Mek') ---
    # The scorer's column_names list maintains this exact order.
    try:
        node_names = scorer.column_names
        idx_to_name = dict(enumerate(node_names))
    except AttributeError:
        print("Error: Scorer object does not have 'column_names' attribute.")
        # Fallback to assuming G_true uses integer nodes 0..N-1.
        if isinstance(list(G_true.nodes())[0], int):
             idx_to_name = {i: i for i in range(num_nodes)}
        else:
             print("Falling back and relabeling G_true to use integer nodes.")
             # This is a failsafe, but the scorer should always have column_names
             g_true_mapping = {name: i for i, name in enumerate(G_true.nodes())}
             G_true = nx.relabel_nodes(G_true, g_true_mapping, copy=True)
             idx_to_name = {i: i for i in range(num_nodes)}


    all_possible_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Edges are stored using their integer indices
                all_possible_edges.append(((i, j), posterior_probs[i, j]))

    # Sort edges by EML
    sorted_eml = sorted(all_possible_edges, key=lambda item: item[1], reverse=True)
    eml_values_only = [eml for edge, eml in sorted_eml]

    # Create the empty graph for fallback, ensuring it uses G_true's node names
    G_empty = nx.DiGraph()
    G_empty.add_nodes_from(G_true.nodes())

    if not eml_values_only:
        return calculate_shd(G_true, G_empty)

    # --- Step 2: Adaptive Sparsity Determination ---
    K_adaptive = find_elbow_point(eml_values_only)
    max_edges = num_nodes * (num_nodes - 1)
    K_adaptive = max(1, min(K_adaptive, max_edges))
    K_SPARSITY_OPTIONS = sorted(list(set([
        max(1, K_adaptive - 2),
        K_adaptive,
        min(max_edges, K_adaptive + 2)
    ])))

    # --- Step 3: Refine Graph Structure ---
    refined_results = {}

    for K in K_SPARSITY_OPTIONS:
        top_k_edges = [edge for edge, eml in sorted_eml[:K]]

        G_refine_int = nx.DiGraph()
        # Create the graph using integer nodes (0..N-1) for scoring
        G_refine_int.add_nodes_from(range(num_nodes))
        G_refine_int.add_edges_from(top_k_edges)

        is_dag = nx.is_directed_acyclic_graph(G_refine_int)

        if is_dag:
            log_score_refine = 0.0
            try:
                # Score the graph using integer indices
                for node_idx in G_refine_int.nodes():
                    parents_indices = tuple(sorted(list(G_refine_int.predecessors(node_idx))))
                    local_score_namedtuple = scorer.local_score(node_idx, parents_indices)

                    # --- BEGIN ROBUST FIX ---
                    # Combine score and prior first
                    current_score = local_score_namedtuple.score + local_score_namedtuple.prior

                    # Check for nan OR inf (positive or negative)
                    if np.isnan(current_score) or np.isinf(current_score):
                        log_score_refine = -np.inf
                        break  # This graph is invalid

                    log_score_refine += current_score
                    # --- END ROBUST FIX ---
            except Exception as e:
                print(f"Warning: Scorer failed for K={K}. {e}")
                log_score_refine = -np.inf

            # --- FIX 2: Relabel the graph to match G_true's node names
            # (e.g., 'A', 'B') *before* calculating SHD ---
            G_refine_named = nx.relabel_nodes(G_refine_int, mapping=idx_to_name, copy=True)
            shd_refine = calculate_shd(G_true, G_refine_named)
        else:
            log_score_refine = -np.inf
            shd_refine = np.nan

        refined_results[K] = {
            'log_score': log_score_refine,
            'shd': shd_refine
        }

    # Select the SHD of the structure with the highest log-score (MAP estimate proxy)
    best_log_score = -np.inf
    best_shd_by_score = calculate_shd(G_true, G_empty) # Fallback SHD

    for K, res in refined_results.items():
        if res.get('log_score', -np.inf) > best_log_score and not np.isnan(res.get('shd', np.inf)):
             best_log_score = res['log_score']
             best_shd_by_score = res['shd']

    if best_log_score == -np.inf:
       print("Warning: All candidate structures failed scoring checks. Falling back to 0-edge SHD.")

    return best_shd_by_score

# --- Evaluation Setup Utilities ---
# In evaluate.py

def get_evaluation_scorer_env(args_dict, rng):
    """Recreates the scorer, data, graph, and environment needed for evaluation."""
    args = Namespace(**args_dict)
    output_folder = Path(args.output_folder) # Get path from loaded args

    # --- ADDED DEBUG PRINT ---
    print(f"\n[DEBUG] Inside get_evaluation_scorer_env for folder: {output_folder}")
    print(f"[DEBUG] Loaded model_type from args_dict: {args_dict.get('model_type', 'NOT FOUND')}")
    # --- END DEBUG PRINT ---


    # --- FIX 1: Load graph and data from files, do not regenerate ---
    try:
        with open(output_folder / 'graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        # The train.py script saves data with an index
        data = pd.read_csv(output_folder / 'data.csv', index_col=0)
        score_name = get_score_name(args.graph)
    except FileNotFoundError as e:
        print(f"Error: Could not load data.csv or graph.pkl from {output_folder}")
        print(f"Did training complete successfully? Missing file: {e}")
        raise
        # --- Handle categorical data for BDeScore (which is lost in CSV) ---
    if score_name == 'bde':
        for col in data.columns:
            if not pd.api.types.is_categorical_dtype(data[col]):
                data[col] = pd.Categorical(data[col])
    # --- END FIX 1 ---

    # 2. Get the Prior
    prior_kwargs = args_dict.get('prior_kwargs', {})
    prior_name = args_dict.get('prior', 'uniform')
    prior = get_prior(prior_name, **prior_kwargs)

    # 3. Get the Scorer
    # This ensures BDeScore gets its 'equivalent_sample_size' etc.
    scorer_kwargs = args_dict.get('scorer_kwargs', {})
    if score_name == 'bde' and 'equivalent_sample_size' in args_dict:
         # Handle legacy args where ess was outside scorer_kwargs
         scorer_kwargs.setdefault('equivalent_sample_size', args_dict['equivalent_sample_size'])

    scores = {'bde': BDeScore, 'bge': BGeScore}
    scorer = scores[score_name](data, prior, **scorer_kwargs)
    # --- END FIX ---

    # 4. Create the Environment
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs, # Use args.num_envs from loaded config
        scorer=scorer,
        num_workers=0, # Disable multiprocessing for standalone script
        # context=args.mp_context # context may not be needed if num_workers=0
    )

    # 5. Determine Model Class and get instance
    model_type = args.model_type if hasattr(args, 'model_type') else 'baseline'

    # Arguments common to both classes
    gflownet_kwargs = {
        'delta': args.delta,
        'update_target_every': args.update_target_every
    }

    if model_type == 'baseline':
        gflownet_class = DAGGFlowNet
    elif model_type == 'moe': # Be explicit
        gflownet_class = MoEDAGGFlowNet
        # MoEDAGGFlowNet explicitly takes 'config'
        gflownet_kwargs['config'] = args # args is Namespace(args_dict)
    else:
        raise ValueError(f"Unknown model_type loaded from arguments.json: {model_type}")

    gflownet_instance = gflownet_class( **gflownet_kwargs )

    # --- ADDED DEBUG PRINT ---
    print(f"[DEBUG] Determined model_type: {model_type}")
    print(f"[DEBUG] Instantiated gflownet_class: {gflownet_class.__name__}")
    print(f"[DEBUG] Instantiated gflownet_instance type: {type(gflownet_instance)}")
    # --- END DEBUG PRINT ---


    return gflownet_instance, scorer, graph, env

# --- Main Evaluation Function ---

def main_evaluate(output_folder, num_samples_posterior, seed):
    output_folder = Path(output_folder)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    # 1. Load arguments, model, and graph
    try:
        with open(output_folder / 'arguments.json', 'r') as f:
            args_dict = json.load(f)
        loaded_data = io.load(output_folder / 'model.npz')
        params = loaded_data['params']
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {output_folder}. Please check the path and ensure training completed.")
        print(f"Missing file: {e}")
        return

    # 2. Recreate the infrastructure
    gflownet, scorer, graph, env = get_evaluation_scorer_env(args_dict, rng)

    # --- ADDED DEBUG PRINT ---
    print(f"\n[DEBUG] Before posterior_estimate in main_evaluate:")
    print(f"[DEBUG]   Folder: {output_folder}")
    print(f"[DEBUG]   gflownet object type: {type(gflownet)}")
    print(f"[DEBUG]   Number of top-level keys in loaded params: {len(params)}")
    # --- END DEBUG PRINT ---

    # 3. Sample the Posterior
    print(f"Sampling {num_samples_posterior} graphs from posterior...")
    # NOTE: Ensure posterior_estimate uses the CORRECT gflownet.act method
    # based on the type of the 'gflownet' object passed to it.
    posterior, logs = posterior_estimate(
        gflownet, # Pass the instantiated object
        params,   # Pass the loaded parameters
        env,
        subkey,
        num_samples=num_samples_posterior,
        verbose=True,
        desc='Final Sampling'
    )

    # 4. Compute Metrics
    # We use the original 'graph' (nx.DiGraph) for SHD, not the numpy array
    ground_truth_adj = nx.to_numpy_array(graph, weight=None)

    standard_metrics = {
        'expected_shd': expected_shd(posterior, ground_truth_adj),
        'expected_edges': expected_edges(posterior),
        **threshold_metrics(posterior, ground_truth_adj)
    }

    # 5. Compute Post-Processing Refinement SHD (EML + Elbow)
    print("Performing post-processing refinement...")
    # Pass the networkx graph (G_true) and the scorer
    refined_shd = post_processing_refinement(graph, posterior, scorer)
    standard_metrics['refined_shd'] = refined_shd

    # 6. Save and Print Results
    results = {
        'final_metrics': standard_metrics,
        'args_used': args_dict,
        'posterior_logs': logs # Include logs (e.g., expert usage)
    }

    class NpEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types."""
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            # --- ADDED: Handle JAX arrays if they somehow appear ---
            if hasattr(obj, 'tolist'):
                 return obj.tolist()
            # --- END ADDITION ---
            return super(NpEncoder, self).default(obj)

    # Save to evaluation_results_full.json (includes logs)
    results_file_path = output_folder / 'evaluation_results_full.json'
    with open(results_file_path, 'w') as f:
        json.dump(results, f, cls=NpEncoder, indent=4)

    print("\n--- Final Evaluation Metrics ---")
    print(f"Expected SHD: {standard_metrics['expected_shd']:.4f}")
    print(f"Refined SHD (EML+Elbow): {standard_metrics['refined_shd']:.4f}")
    print(f"AUROC: {standard_metrics['roc_auc']:.4f}")
    print(f"Results saved to {results_file_path}")

    # Close the environment's workers (even if num_workers=0)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation script for DAG-GFlowNet.')
    parser.add_argument('output_folder', type=str,
                        help='Path to the experiment output folder (containing arguments.json and model.npz).')
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of samples for the final posterior estimate (default: 5000).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42).')

    # --- Optional: Clear pycache before running ---
    # parser.add_argument('--clear-cache', action='store_true', help='Clear __pycache__ folders before running.')
    args = parser.parse_args()

    # if args.clear_cache:
    #     import shutil, glob
    #     print("Clearing __pycache__ directories...")
    #     for path in glob.glob('./**/__pycache__', recursive=True):
    #         print(f"Removing {path}")
    #         shutil.rmtree(path, ignore_errors=True)

    # Use the final logic function
    main_evaluate(args.output_folder, args.num_samples, args.seed)
