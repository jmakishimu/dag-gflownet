# DAG-GFlowNet with Mixture of Experts

[Paper](https://arxiv.org/abs/2202.13903) - [Installation](#installation) - [Basic Training](#basic-training) - [Full Experimentation Workflow](#full-experimentation-workflow) - [Citation](#citation)

This repository contains an extended JAX implementation of DAG-GFlowNet ([Deleu et al., 2022](https://arxiv.org/abs/2202.13903)), a Bayesian structure learning algorithm based on Generative Flow Networks (GFlowNets; [Bengio et al., 2021](http://folinoid.com/w/gflownet/)).

This fork extends the original implementation with a **Mixture of Experts (MoE)** architecture, including **Astrocyte-Hierarchical Routing (AHR)**, and provides a comprehensive framework for hyperparameter optimization and multi-seed experimentation.

## Core New Features

  * **Mixture of Experts (MoE) Model:** A new `MoEDAGGFlowNet` class (`dag_gflownet/moe.py`) that uses multiple expert networks instead of a single monolithic model.
  * **Hybrid Loss Function:** The MoE model is trained with a three-part loss:
    1.  **`L_GFN`:** The standard Detailed Balance GFlowNet loss.
    2.  **`L_bal` (Load Balancing):** An auxiliary loss to encourage the gating network to distribute states evenly among experts.
    3.  **`L_div` (Diversity):** An auxiliary loss that penalizes parameter similarity between experts, encouraging specialization.
  * **Astrocyte-Hierarchical Routing (AHR):** A specialized, context-aware gating network (`dag_gflownet/nets/ahr.py`) that can be enabled as an ablation.
  * **Full Experimentation Suite:** A complete, multi-phase workflow for:
    1.  **Hyperparameter Optimization** (`optimize_hyperparams.py` using Optuna).
    2.  **Full-Scale Training** (`run_experiments.sh` for multi-seed runs).
    3.  **Robust Evaluation** (`evaluate.py` with a new "Refined SHD" metric).
    4.  **Plotting & Analysis** (`plot_results.py` and `plot_optuna_results.py`).
  * **Environment & Stability Fixes:** The environment (`dag_gflownet/env.py`) has been refactored to use a `gymnasium`-like API and includes robust action validation to prevent crashes from invalid (e.g., cycle-creating) actions. Numerous numerical stability fixes have been applied to utility and scoring functions.

-----

## Installation

We suggest working in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Follow the [official instructions](https://github.com/google/jax#installation) to install JAX with the correct CUDA and CuDNN versions for your setup.

Then, install the repository and its dependencies:

```bash
git clone https://github.com/jmakishimu/dag-gflownet.git
cd dag-gflownet
pip install -r requirements.txt
```

Note: The environment has been updated to a `gymnasium`-like API. The `requirements.txt` file includes `flax`, `optuna`, and `plotly` for the new experimentation workflow.

-----

## Basic Training

You can train a single model using `train.py`.

### Example 1: Baseline Model

This runs the original single-network model, equivalent to the original paper.

```bash
python train.py \
    --model_type baseline \
    --output_folder output/baseline_run \
    --batch_size 64 \
    --lr 1e-5 \
    --num_iterations 150000 \
    erdos_renyi_lingauss \
    --num_variables 10 \
    --num_edges 10 \
    --num_samples 100
```

### Example 2: Mixture of Experts (MoE) Model

This runs the new MoE model with 4 experts, load balancing, and diversity loss.

```bash
python train.py \
    --model_type moe \
    --output_folder output/moe_run \
    --batch_size 64 \
    --lr 1e-5 \
    --num_iterations 150000 \
    --num_experts 4 \
    --top_k 2 \
    --use_load_balancing \
    --use_diversity \
    --load_balance_alpha 0.01 \
    --diversity_beta 1.0 \
    sachs_interventional
```

-----

## Full Experimentation Workflow

This repository is designed for a multi-phase experimentation workflow, from hyperparameter optimization to final plotting.

### Phase 1: Hyperparameter Optimization (HPO)

Use `optimize_hyperparams.py` to find the best hyperparameters for a specific model configuration and dataset using Optuna.

The script runs shorter training jobs (50k iterations by default) and evaluates the final **Refined SHD** as the metric to minimize.

**Command:**

```bash
python optimize_hyperparams.py \
    --ablation_config moe_ld \
    --dataset lingauss \
    --n_trials 50 \
    --study_name_prefix "hpo_lingauss_d20" \
    --db_url "sqlite:///optuna_study.db"
```

  * `--ablation_config`: The model to test. Choices:
      * `baseline`: The original DAG-GFlowNet.
      * `moe_base`: MoE without auxiliary losses.
      * `moe_l`: MoE with Load Balancing only.
      * `moe_d`: MoE with Diversity only.
      * `moe_ld`: MoE with both Load Balancing and Diversity.
  * `--dataset`: The dataset to use. Choices: `lingauss` (synthetic) or `sachs`.

This will run 50 trials and store results in `optuna_study.db`. The `study_name` will be automatically generated (e.g., `hpo_lingauss_d20_lingauss_moe_ld`).

### Phase 2: HPO Analysis (Optional)

You can visualize the HPO results using `plot_optuna_results.py`.

**Note:** You must **manually edit** the `STUDY_NAME` and `STORAGE_DB` variables inside `plot_optuna_results.py` to match the database and study name from Phase 1.

```bash
# First, edit plot_optuna_results.py to set:
# STUDY_NAME = "hpo_lingauss_d20_lingauss_moe_ld"
# STORAGE_DB = "sqlite:///optuna_study.db"

python plot_optuna_results.py
```

This will create an `optuna_plots/` directory with interactive HTML plots for optimization history, parameter importances, and parameter slices. It will also save the best parameters to `best_hyperparams.json`.

### Phase 3: Full-Scale, Multi-Seed Training

After identifying the best hyperparameters (from Phase 1 or 2), use the `run_experiments.sh` script to launch a full, multi-seed experiment.

**Setup:**

1.  **Get Best Parameters:** Run Phase 1 (and optionally Phase 2) to generate `best_hyperparams.json` files for each ablation you want to test.

2.  **Edit the Script:** You **must edit `dag_gflownet/run_experiments.sh`** to point to the correct `.json` files.

    ```bash
    # Inside dag_gflownet/run_experiments.sh
    declare -A config_files
    config_files=(
        # Example for Sachs
        ["sachs_baseline"]="path/to/your/sachs_baseline_params.json"
        ["sachs_moe_ld"]="path/to/your/sachs_moe_ld_params.json"

        # Example for Lingauss
        ["lingauss_d20_e40_n100_baseline"]="path/to/your/lingauss_baseline_params.json"
        ["lingauss_d20_e40_n100_moe_ld"]="path/to/your/lingauss_moe_ld_params.json"
        # ... and so on for all 10 configurations
    )
    ```

**Command:**

The script will run `train.py` for the full number of iterations (150k by default) and then automatically run `evaluate.py` for each seed.

```bash
# This script is located inside the dag_gflownet sub-directory
bash dag_gflownet/run_experiments.sh
```

This will create a `results/` directory containing one folder for each model and seed (e.g., `results/sachs_moe_ld_seed1/`, `results/sachs_moe_ld_seed2/`, etc.).

### Phase 4: Evaluation and Plotting

While `run_experiments.sh` automatically calls `evaluate.py`, you can also run it manually on any output folder.

#### Manual Evaluation (Single Run)

The `evaluate.py` script loads a trained model, samples from the posterior, and calculates all metrics, including the **Refined SHD**.

```bash
python evaluate.py results/sachs_moe_ld_seed1/ --num_samples 5000
```

This saves the final metrics and posterior logs to `evaluation_results_full.json` inside that folder.

#### Final Plotting (Comparing Runs)

Use `plot_results.py` to compare multiple experiment folders and generate final plots.

```bash
python plot_results.py \
    results/sachs_baseline_seed1 \
    results/sachs_moe_ld_seed1 \
    --plot_folder final_plots/sachs_comparison
```

This command will generate three plots in the `final_plots/sachs_comparison` directory:

1.  `shd_over_training.png`: A line plot of Expected SHD vs. training steps.
2.  `loss_curves.png`: A 2x2 grid of the (smoothed) `Total Loss`, `L_GFN`, `L_Balance`, and `L_Diversity` losses.
3.  `final_metrics_bar.png`: A bar chart comparing `Expected SHD`, `Refined SHD`, and `AUROC` for the specified models.

It will also print a summary table of the final metrics to the console.

-----

## Citation

If you use the original DAG-GFlowNet work, please cite:

```
TBD
```
