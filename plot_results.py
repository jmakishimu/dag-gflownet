# In (root)/plot_results.py
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from argparse import Namespace

# --- FIX 7 & 8: Import correct scorers and prior factory ---
from dag_gflownet.scores.bde_score import BDeScore
from dag_gflownet.scores.bge_score import BGeScore
from dag_gflownet.utils.factories import get_prior
# --------------------------------------------------------

# --- 1. Post-processing and SHD Calculation ---
# (Logic adapted from your PyTorch script)

def calculate_shd(G_true, G_pred):
    """Calculates the Structural Hamming Distance (SHD)."""
    if not isinstance(G_true, nx.DiGraph) or not isinstance(G_pred, nx.DiGraph):
        raise TypeError("Inputs must be nx.DiGraph")

    true_edges = set(G_true.edges())
    pred_edges = set(G_pred.edges())

    missing_edges = len(true_edges - pred_edges)

    false_positive_or_wrong_direction = 0
    for u, v in pred_edges:
        if (u, v) not in true_edges:
            # Check for reversed edge in true_edges
            if (v, u) not in true_edges:
                false_positive_or_wrong_direction += 1 # False Positive
            else:
                false_positive_or_wrong_direction += 1 # Wrong Direction (Penalty: 1)

    shd = missing_edges + false_positive_or_wrong_direction
    return shd

def find_elbow_point(eml_values):
    """
    Finds the elbow point using the largest distance from the line method.
    Input: eml_values (list of sorted EMLs from highest to lowest)
    Output: K_adaptive (integer, number of edges)
    """
    points = np.array(eml_values)
    if len(points) < 3:
        return len(points)

    n_points = len(points)
    x_data = np.arange(n_points)

    # Rescale coordinates to be between 0 and 1
    x_norm = x_data / x_data[-1] if x_data[-1] > 0 else x_data

    # Check for flat line
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

        numerator = np.abs(
            (p2[1] - p1[1]) * p3[0] - (p2[0] - p1[0]) * p3[1] + p2[0] * p1[1] - p2[1] * p1[0]
        )
        denominator = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)

        distance = numerator / denominator if denominator > 1e-6 else 0

        if distance > max_dist:
            max_dist = distance
            elbow_index = i

    return elbow_index + 1

def post_processing_refinement(G_true, posterior_probs, scorer, num_nodes):
    """
    Performs mode refinement using EML and adaptive sparsity.
    Adapted for the JAX repo: posterior_probs is the (N, N) matrix of
    marginal edge probabilities (the EML).
    """
    if posterior_probs is None or scorer is None:
        print("Skipping refinement: Scorer is None.")
        return np.nan

    all_possible_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                all_possible_edges.append(((i, j), posterior_probs[i, j]))

    # Sort edges by EML (posterior probability)
    sorted_eml = sorted(all_possible_edges, key=lambda item: item[1], reverse=True)
    eml_values_only = [eml for edge, eml in sorted_eml]

    if not eml_values_only:
        return calculate_shd(G_true, nx.DiGraph(G_true.nodes()))

    # --- Step 2: Adaptive Sparsity Determination ---
    K_adaptive = find_elbow_point(eml_values_only)

    max_edges = num_nodes * (num_nodes - 1)
    K_adaptive = max(1, min(K_adaptive, max_edges))

    K_SPARSITY_OPTIONS = [
        max(1, K_adaptive - 2),
        K_adaptive,
        min(max_edges, K_adaptive + 2)
    ]
    K_SPARSITY_OPTIONS = sorted(list(set(K_SPARSITY_OPTIONS)))

    # --- Step 3: Refine Graph Structure ---
    refined_results = {}

    for K in K_SPARSITY_OPTIONS:
        top_k_edges = [edge for edge, eml in sorted_eml[:K]]

        G_refine = nx.DiGraph()
        G_refine.add_nodes_from(range(num_nodes))
        G_refine.add_edges_from(top_k_edges)

        is_dag = nx.is_directed_acyclic_graph(G_refine)

        if is_dag:
            # Use the JAX-repo's scorer object
            log_bic_refine = 0.0
            try:
                for node in G_refine.nodes():
                    parents = tuple(sorted(list(G_refine.predecessors(node))))
                    # Use local_score from BDe/BGe, which requires (target, parents)
                    local_score_namedtuple = scorer.local_score(node, parents)
                    log_bic_refine += local_score_namedtuple.score
            except Exception as e:
                print(f"Warning: Scorer failed for K={K}. {e}")
                log_bic_refine = -np.inf

            shd_refine = calculate_shd(G_true, G_refine)
        else:
            log_bic_refine = -np.inf
            shd_refine = np.nan

        refined_results[K] = {
            'log_bic': log_bic_refine,
            'shd': shd_refine
        }

    # Find best SHD among valid (DAG) refined graphs
    best_shd = min(
        [res.get('shd', np.inf) for res in refined_results.values() if not np.isnan(res.get('shd', np.inf))],
        default=np.inf
    )

    if best_shd == np.inf: # Handle case where no DAGs were found
       best_shd = calculate_shd(G_true, nx.DiGraph(G_true.nodes()))

    return best_shd


# --- 2. Data Loading Helpers (with Corrections) ---

# --- FIX 7 & 8: Rewritten get_plot_scorer ---
def get_plot_scorer(args_dict, graph):
    """Minimal version of get_scorer for plotting."""
    args = Namespace(**args_dict) # Convert dict to Namespace

    try:
        # data_path might not be in args for sachs
        data_path_str = args.data_path if hasattr(args, 'data_path') else None

        if data_path_str:
            data_path = Path(data_path_str)
        else:
            # Try to find data relative to experiment folder
            data_path = Path(args.output_folder) / 'data.csv'

        if not data_path.exists():
            # Try again with output folder if first attempt failed
            data_path = Path(args.output_folder) / 'data.csv'
            if not data_path.exists():
                 raise FileNotFoundError(f"data.csv not found in {args.output_folder} or {data_path_str}")

        data = pd.read_csv(data_path, index_col=0)
    except Exception as e:
        print(f"Warning: Could not load data from {data_path_str} or 'data.csv'.")
        print(f"Error: {e}")
        print("Post-processing refinement will be skipped.")
        return None

    variables = list(graph.nodes)
    data = data[variables]

    # --- ADDED PRIOR LOGIC ---
    # The scorer object requires a prior object, which was missing
    try:
        prior_kwargs = args.prior_kwargs if hasattr(args, 'prior_kwargs') else {}
        prior_name = args.prior if hasattr(args, 'prior') else 'uniform'
        prior = get_prior(prior_name, **prior_kwargs)
    except Exception as e:
        print(f"Warning: Could not create prior: {e}. Defaulting to Uniform.")
        prior = get_prior('uniform')
    # -------------------------

    # Map scorer names from your plotting script to the jax repo's classes
    scorer_name = args.scorer.lower()
    if scorer_name == 'bdeu' or scorer_name == 'bde':
        print("Info: Using BDeScore for refinement.")
        # Ensure data is categorical for BDeScore
        for col in data.columns:
            if not pd.api.types.is_categorical_dtype(data[col]):
                data[col] = pd.Categorical(data[col])
        ess = args.bdeu_ess if hasattr(args, 'bdeu_ess') else 1.0
        return BDeScore(data, prior, equivalent_sample_size=ess)

    elif scorer_name == 'bic' or scorer_name == 'bge':
        print("Info: Using BGeScore for refinement.")
        return BGeScore(data, prior)
    else:
        print(f"Warning: Unknown scorer {args.scorer}. Skipping refinement.")
        return None
# --- END FIX ---

def load_experiment_data(exp_folder):
    """Loads all data for one experiment run."""
    exp_folder = Path(exp_folder)
    print(f"Loading experiment: {exp_folder.name}")

    results_file = exp_folder / 'results.json'
    graph_file = exp_folder / 'graph.pkl'
    posterior_file = exp_folder / 'posterior.npy'
    args_file = exp_folder / 'arguments.json'

    missing_files = []
    if not results_file.exists(): missing_files.append('results.json')
    if not graph_file.exists(): missing_files.append('graph.pkl')
    if not posterior_file.exists(): missing_files.append('posterior.npy')
    if not args_file.exists(): missing_files.append('arguments.json')

    if missing_files:
        raise FileNotFoundError(f"Missing required files in {exp_folder}: {', '.join(missing_files)}")

    with open(results_file, 'r') as f:
        results = json.load(f)
    with open(graph_file, 'rb') as f:
        G_true = pickle.load(f)

    posterior = np.load(posterior_file)

    with open(args_file, 'r') as f:
        args_dict = json.load(f)
        # Add output_folder to args_dict if not present, as get_plot_scorer needs it
        if 'output_folder' not in args_dict:
            args_dict['output_folder'] = str(exp_folder)

    scorer = get_plot_scorer(args_dict, G_true)
    num_nodes = G_true.number_of_nodes()

    refined_shd = post_processing_refinement(
        G_true, posterior, scorer, num_nodes
    )

    return {
        'name': exp_folder.name,
        'history': pd.DataFrame(results['training_history']),
        'final_shd': results['expected_shd'],
        'final_auroc': results['roc_auc'],
        'refined_shd': refined_shd,
        'args': args_dict
    }

# --- 3. Main Plotting Function ---
def main(args):
    sns.set_theme(style="whitegrid")

    try:
        all_data = [load_experiment_data(folder) for folder in args.experiment_folders]
    except FileNotFoundError as e:
        print(f"Error: Could not find required file. {e}")
        print("Please ensure all experiment folders are correct and contain all required files.")
        return

    plot_folder = Path(args.plot_folder)
    plot_folder.mkdir(exist_ok=True)

    # Get eval_samples from the first experiment's args for plot label
    try:
        eval_samples = all_data[0]['args'].get('eval_samples', 100)
    except Exception:
        eval_samples = 100

    # === Plot 1: SHD over Training ===
    plt.figure(figsize=(10, 6))
    for data in all_data:
        df = data['history'].dropna(subset=['eval_shd'])
        if not df.empty:
            sns.lineplot(data=df, x='step', y='eval_shd', label=data['name'])
    plt.title('Mean SHD During Training')
    plt.xlabel('Training Step')
    plt.ylabel(f'Expected SHD (over {eval_samples} samples)')
    plt.legend()
    plt.savefig(plot_folder / 'shd_over_training.png')
    plt.close()

    # === Plot 2: Loss Curves ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    loss_cols = ['loss', 'l_gfn', 'l_balance', 'l_diversity']
    titles = ['Total Loss (Smoothed)', 'L_GFN (Smoothed)', 'L_Balance (Smoothed)', 'L_Diversity (Smoothed)']

    for data in all_data:
        # Check if training history is empty or missing columns
        if data['history'].empty:
            print(f"Skipping loss plot for {data['name']}: No training history found.")
            continue

        # Ensure columns exist, fill with 0 if not (e.g., for baseline)
        history_df = data['history'].copy()
        for col in loss_cols:
            if col not in history_df.columns:
                history_df[col] = 0.0

        df = history_df.rolling(window=100).mean() # Smooth curves

        sns.lineplot(data=df, x='step', y=loss_cols[0], label=data['name'], ax=axes[0, 0])
        sns.lineplot(data=df, x='step', y=loss_cols[1], label=data['name'], ax=axes[0, 1])
        sns.lineplot(data=df, x='step', y=loss_cols[2], label=data['name'], ax=axes[1, 0])
        sns.lineplot(data=df, x='step', y=loss_cols[3], label=data['name'], ax=axes[1, 1])

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.legend()

    axes[1,0].set_xlabel('Training Step')
    axes[1,1].set_xlabel('Training Step')
    plt.tight_layout()
    plt.savefig(plot_folder / 'loss_curves.png')
    plt.close()

    # === Plot 3: Final Metrics Bar Chart ===
    final_df = pd.DataFrame([
        {
            'Model': d['name'],
            'Expected SHD': d['final_shd'],
            'Refined SHD': d['refined_shd'],
            'AUROC': d['final_auroc']
        }
        for d in all_data
    ])
    final_df_melted = final_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 7))
    sns.barplot(data=final_df_melted, x='Model', y='Score', hue='Metric')
    plt.title('Final Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(plot_folder / 'final_metrics_bar.png')
    plt.close()

    print(f"\nAll plots saved to {plot_folder}")

    # Print final table to console
    print("\n" + "="*50)
    print("      FINAL METRICS COMPARISON")
    print("="*50)
    print(final_df.to_string(index=False, float_format="%.2f"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting script for MoE-GFN results.')
    parser.add_argument('experiment_folders', type=str, nargs='+',
                        help='List of experiment output folders to compare.')
    parser.add_argument('--plot_folder', type=str, default='plots',
                        help='Folder to save plots (default: %(default)s)')
    args = parser.parse_args()

    main(args)
