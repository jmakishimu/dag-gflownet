# In jax-dag-gflownet/optimize_hyperparams.py
import optuna
import train  # Main training script
import evaluate # Main evaluation script
import argparse
import json
import functools
from pathlib import Path
from argparse import Namespace

# --- DEFAULT HPO CONFIGURATION ---
# Use shorter training runs for hyperparameter optimization
DEFAULT_HPO_ARGS = {
    # Env args
    'num_envs': 8,
    'cache_max_size': 500000,

    # Replay Buffer
    'replay_capacity': 100000, # Capacity
    'prefill': 1000,           # Prefill steps

    # Misc
    'seed': 42,
    'num_workers': 4,
    'mp_context': 'spawn',

    # --- Training duration for HPO ---
    # We use 50k iterations for HPO.
    # Full training (Phase 3) will use more (e.g., 150k).
    'num_iterations': 50000,
    'eval_every': 50000,     # Only evaluate at the end of the HPO run
    'eval_samples': 500,
    'num_samples_posterior': 2000 # Samples for final evaluation
}
# -----------------------------

# --- DATASET-SPECIFIC ARGS ---
DATASET_DEFAULTS = {
    "lingauss": {
        'graph': 'erdos_renyi_lingauss',
        'num_variables': 20,
        'num_edges': 40,  # ER2 (as in paper)
        'num_samples': 100,
        'scorer_kwargs': {},
        'prior': 'uniform',
        'prior_kwargs': {},
    },
    "sachs": {
        'graph': 'sachs_interventional',
        'scorer_kwargs': {'equivalent_sample_size': 1.0}, # BDe score
        'prior': 'uniform',
        'prior_kwargs': {},
        # num_variables, num_edges, num_samples are not needed for sachs
    }
}
# -----------------------------


def define_search_space(trial: optuna.Trial, ablation_config: str) -> dict:
    """
    Defines the hyperparameter search space for a given ablation.
    """
    hparams = {}

    # === 1. Common Hyperparameters (All models) ===
    hparams['lr'] = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    hparams['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
    hparams['delta'] = trial.suggest_float("delta", 0.5, 5.0)
    hparams['min_exploration'] = trial.suggest_float("min_exploration", 0.05, 0.2)
    hparams['update_target_every'] = trial.suggest_int("update_target_every", 500, 2000)

    # === 2. Ablation-Specific Hyperparameters ===

    # --- Baseline ---
    if ablation_config == 'baseline':
        hparams['model_type'] = 'baseline'

    # --- MoE Models ---
    if ablation_config.startswith('moe'):
        hparams['model_type'] = 'moe'
        hparams['num_experts'] = trial.suggest_int("num_experts", 2, 8)
        hparams['context_dim'] = trial.suggest_categorical("context_dim", [32, 64, 128])
        # Top-K is suggested relative to num_experts
        hparams['top_k'] = trial.suggest_int("top_k", 1, hparams['num_experts'])
        hparams['use_ahr'] = trial.suggest_categorical("use_ahr", [True, False])

    # --- MoE Loss Configurations ---
    if ablation_config == 'moe_base':
        hparams['use_load_balancing'] = False
        hparams['use_diversity'] = False

    elif ablation_config == 'moe_ld': # Load Balancing + Diversity
        hparams['use_load_balancing'] = True
        hparams['use_diversity'] = True
        hparams['load_balance_alpha'] = trial.suggest_float("load_balance_alpha", 0.001, 0.1, log=True)
        hparams['diversity_beta'] = trial.suggest_float("diversity_beta", 0.1, 5.0)

    elif ablation_config == 'moe_l': # Load Balancing Only
        hparams['use_load_balancing'] = True
        hparams['use_diversity'] = False
        hparams['load_balance_alpha'] = trial.suggest_float("load_balance_alpha", 0.001, 0.1, log=True)

    elif ablation_config == 'moe_d': # Diversity Only
        hparams['use_load_balancing'] = False
        hparams['use_diversity'] = True
        hparams['diversity_beta'] = trial.suggest_float("diversity_beta", 0.1, 5.0)

    return hparams


def objective(trial: optuna.Trial, ablation_config: str, dataset_args: dict) -> float:
    """
    Objective function for Optuna to minimize.
    Trains and evaluates a model, returning the final refined SHD.
    """
    # --- 1. Create argument namespace ---
    args_dict = DEFAULT_HPO_ARGS.copy()

    # Add dataset-specific defaults
    args_dict.update(dataset_args)

    # Add hyperparameters from search space
    hparams = define_search_space(trial, ablation_config)
    args_dict.update(hparams)

    # --- 2. Set unique output folder ---
    output_folder = Path(f"optuna_runs/{trial.study.study_name}/trial_{trial.number}")
    args_dict['output_folder'] = output_folder

    # Convert dict to Namespace for the train/evaluate functions
    args = Namespace(**args_dict)

    try:
        # --- 3. Run Training ---
        print(f"\n--- Starting Optuna Trial {trial.number} ({trial.study.study_name}) ---")
        print(f"Params: {json.dumps(trial.params, indent=2)}")
        print(f"Output: {output_folder}")

        train.main(args)

        # --- 4. Run Evaluation ---
        print(f"\n--- Evaluating Trial {trial.number} ---")
        # Use main_evaluate from evaluate.py
        evaluate.main_evaluate(
            output_folder=str(output_folder),
            num_samples_posterior=args.num_samples_posterior,
            seed=args.seed
        )

        # --- 5. Load results and return objective ---
        results_file = output_folder / 'evaluation_results_full.json'
        if not results_file.exists():
            print(f"ERROR: Results file not found for trial {trial.number}")
            # Prune if training/eval failed (e.g., OOM)
            raise optuna.exceptions.TrialPruned()

        with open(results_file, 'r') as f:
            results = json.load(f)

        # The metric to minimize is the 'refined_shd'
        refined_shd = results['final_metrics']['refined_shd']

        # Handle potential NaN/Inf values from failed scoring
        if not np.isfinite(refined_shd):
            print(f"Warning: Trial {trial.number} resulted in non-finite SHD. Pruning.")
            raise optuna.exceptions.TrialPruned()

        print(f"--- Trial {trial.number} Complete ---")
        print(f"Refined SHD: {refined_shd}")

        return refined_shd

    except optuna.exceptions.TrialPruned as e:
        # Re-raise prune exceptions
        raise e
    except Exception as e:
        print(f"--- Trial {trial.number} FAILED ---")
        print(f"Error: {e}")
        # Prune trial if it fails (e.g., OOM, JAX error)
        raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for DAG-GFlowNet.')

    parser.add_argument('--ablation_config', type=str, required=True,
                        choices=['baseline', 'moe_base', 'moe_ld', 'moe_l', 'moe_d'],
                        help='The specific model ablation to run.')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['lingauss', 'sachs'],
                        help='The dataset to use for the study.')

    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials to run.')

    parser.add_argument('--study_name_prefix', type=str, default="dag_gflownet_hpo",
                        help='Prefix for the Optuna study name.')

    parser.add_argument('--db_url', type=str, default="sqlite:///optuna_study.db",
                        help='Optuna storage database URL.')

    args = parser.parse_args()

    # --- Study Configuration ---
    storage_name = args.db_url
    study_name = f"{args.study_name_prefix}_{args.dataset}_{args.ablation_config}"
    dataset_args = DATASET_DEFAULTS[args.dataset]

    print(f"Starting Optuna study: {study_name}")
    print(f"Storage: {storage_name}")
    print(f"Dataset Config: {dataset_args}")
    print(f"Ablation Config: {args.ablation_config}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",  # We want to minimize Refined SHD
        load_if_exists=True    # Resume study if it already exists
    )

    # Pass fixed arguments to the objective function
    obj_fn = functools.partial(
        objective,
        ablation_config=args.ablation_config,
        dataset_args=dataset_args
    )

    # --- Run Optimization ---
    study.optimize(obj_fn, n_trials=args.n_trials)

    print("\n--- Optimization Complete ---")
    print(f"Study: {study_name}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best Refined SHD: {study.best_value}")
    print("Best hyperparameters:")
    print(json.dumps(study.best_params, indent=2))
