#!/bin/bash

# ======================================================================
# PHASE 3: FULL-SCALE MULTI-SEED & SCALING EXPERIMENT SCRIPT
# ======================================================================

# --- Configuration ---
SEEDS=(1 2 3 4 5) # Run N=5 seeds
FULL_TRAIN_ITERATIONS=150000
FULL_EVAL_SAMPLES=5000 # Number of samples for final posterior
BASE_RESULTS_DIR="results"

# 1. DEFINE YOUR 10 EXPERIMENT CONFIGS
# Manually add the paths to the 'best_hyperparams.json' files
# you generated in Phase 2.
# ----------------------------------------------------------------------
declare -A config_files
config_files=(
    ["sachs_baseline"]="optuna_plots/pub_exp_sachs_baseline/best_hyperparams.json"
    ["sachs_moe_base"]="optuna_plots/pub_exp_sachs_moe_base/best_hyperparams.json"
    ["sachs_moe_ld"]="optuna_plots/pub_exp_sachs_moe_ld/best_hyperparams.json"
    ["sachs_moe_l"]="optuna_plots/pub_exp_sachs_moe_l/best_hyperparams.json"
    ["sachs_moe_d"]="optuna_plots/pub_exp_sachs_moe_d/best_hyperparams.json"

    ["lingauss_d20_e40_n100_baseline"]="optuna_plots/pub_exp_lingauss_baseline/best_hyperparams.json"
    ["lingauss_d20_e40_n100_moe_base"]="optuna_plots/pub_exp_lingauss_moe_base/best_hyperparams.json"
    ["lingauss_d20_e40_n100_moe_ld"]="optuna_plots/pub_exp_lingauss_moe_ld/best_hyperparams.json"
    ["lingauss_d20_e40_n100_moe_l"]="optuna_plots/pub_exp_lingauss_moe_l/best_hyperparams.json"
    ["lingauss_d20_e40_n100_moe_d"]="optuna_plots/pub_exp_lingauss_moe_d/best_hyperparams.json"

    # Add configs for d=50 after you run HPO for them
    # ["lingauss_d50_e100_n100_baseline"]="optuna_plots/pub_exp_lingauss_d50_baseline/best_hyperparams.json"
    # ["lingauss_d50_e100_n100_moe_ld"]="optuna_plots/pub_exp_lingauss_d50_moe_ld/best_hyperparams.json"
)
# ----------------------------------------------------------------------


# --- Helper function to run one experiment ---
run_one() {
    local exp_name=$1
    local seed=$2
    local dataset_args=$3
    local hparam_file=$4

    local output_folder="$BASE_RESULTS_DIR/${exp_name}_seed${seed}"

    # Check if this run already exists
    if [ -f "$output_folder/evaluation_results_full.json" ]; then
        echo "Skipping ${exp_name}_seed${seed} (results already exist)"
        return
    fi

    echo "======================================================================"
    echo "Starting: ${exp_name}_seed${seed}"
    echo "Output: $output_folder"
    echo "HParams: $hparam_file"
    echo "======================================================================"

    # Build the training command
    # We read the JSON file and convert it to --key value arguments
    local train_args=$(jq -r 'to_entries | .[] | "--\(.key) \(.value)"' < $hparam_file | tr '\n' ' ')

    # Run Training
    python train.py \
        --output_folder "$output_folder" \
        --seed "$seed" \
        --num_iterations "$FULL_TRAIN_ITERATIONS" \
        --num_samples_posterior "$FULL_EVAL_SAMPLES" \
        $dataset_args \
        $train_args \
        --eval_every 5000 # Evaluate more often for plotting

    # Check if training succeeded (model.npz exists)
    if [ ! -f "$output_folder/model.npz" ]; then
        echo "ERROR: Training failed for ${exp_name}_seed${seed}"
        return
    fi

    # Run Evaluation
    python evaluate.py \
        "$output_folder" \
        --num_samples "$FULL_EVAL_SAMPLES" \
        --seed "$seed"

    echo "Finished: ${exp_name}_seed${seed}"
}


# ======================================================================
# --- EXPERIMENT SUITE 1: SACHS (d=11) ---
# ======================================================================

echo "--- RUNNING SACHS EXPERIMENTS ---"
sachs_dataset_args="sachs_interventional --scorer_kwargs '{\"equivalent_sample_size\": 1.0}'"

for seed in "${SEEDS[@]}"; do
    run_one "sachs_baseline" $seed "$sachs_dataset_args" "${config_files[sachs_baseline]}"
    run_one "sachs_moe_base" $seed "$sachs_dataset_args" "${config_files[sachs_moe_base]}"
    run_one "sachs_moe_ld" $seed "$sachs_dataset_args" "${config_files[sachs_moe_ld]}"
    run_one "sachs_moe_l" $seed "$sachs_dataset_args" "${config_files[sachs_moe_l]}"
    run_one "sachs_moe_d" $seed "$sachs_dataset_args" "${config_files[sachs_moe_d]}"
done


# ======================================================================
# --- EXPERIMENT SUITE 2: SYNTHETIC (d=20, ER2, N=100) ---
# ======================================================================

echo "--- RUNNING SYNTHETIC (d=20) EXPERIMENTS ---"
d=20
edges=40
n_samples=100
exp_prefix="lingauss_d${d}_e${edges}_n${n_samples}"
lingauss_dataset_args="erdos_renyi_lingauss --num_variables $d --num_edges $edges --num_samples $n_samples"

for seed in "${SEEDS[@]}"; do
    run_one "${exp_prefix}_baseline" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_baseline]}"
    run_one "${exp_prefix}_moe_base" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_moe_base]}"
    run_one "${exp_prefix}_moe_ld" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_moe_ld]}"
    run_one "${exp_prefix}_moe_l" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_moe_l]}"
    run_one "${exp_prefix}_moe_d" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_moe_d]}"
done


# ======================================================================
# --- EXPERIMENT SUITE 3: SYNTHETIC SCALING (d=50, ER2, N=100) ---
# ======================================================================
# NOTE: You must first run HPO for d=50 and add the paths to config_files above
#
# echo "--- RUNNING SYNTHETIC (d=50) EXPERIMENTS ---"
# d=50
# edges=100
# n_samples=100
# exp_prefix="lingauss_d${d}_e${edges}_n${n_samples}"
# lingauss_dataset_args="erdos_renyi_lingauss --num_variables $d --num_edges $edges --num_samples $n_samples"
#
# for seed in "${SEEDS[@]}"; do
#     run_one "${exp_prefix}_baseline" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_baseline]}"
#     run_one "${exp_prefix}_moe_ld" $seed "$lingauss_dataset_args" "${config_files[${exp_prefix}_moe_ld]}"
# done

echo "--- ALL EXPERIMENTS COMPLETE ---"
