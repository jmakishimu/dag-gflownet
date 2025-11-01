# In jax-dag-gflownet/plot_optuna_results.py
import optuna
import json
from pathlib import Path
import plotly.io as pio

# --- Configuration ---
STUDY_NAME = "dag-gflownet-optimization"
STORAGE_DB = "sqlite:///optuna_study.db"
PLOT_DIR = Path("optuna_plots")
BEST_PARAMS_FILE = Path("best_hyperparams.json")

def main():
    # Create plot directory
    PLOT_DIR.mkdir(exist_ok=True)

    print(f"Loading study '{STUDY_NAME}' from '{STORAGE_DB}'...")
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
    except KeyError:
        print(f"Error: Study '{STUDY_NAME}' not found in '{STORAGE_DB}'.")
        print("Please run 'optimize_hyperparams.py' first.")
        return

    print(f"Study loaded with {len(study.trials)} trials.")

    # --- 1. Save Best Hyperparameters ---
    print(f"\nSaving best hyperparameters to {BEST_PARAMS_FILE}...")
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Best trial ({study.best_trial.number}) score (SHD): {study.best_value}")
    print(json.dumps(study.best_params, indent=2))

    # --- 2. Generate and Save Plots ---
    print(f"\nGenerating and saving plots to {PLOT_DIR}/")

    # Optimization History
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(PLOT_DIR / "1_optimization_history.html")
        print("Saved 1_optimization_history.html")
    except Exception as e:
        print(f"Could not generate optimization history plot: {e}")

    # Parameter Importances
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(PLOT_DIR / "2_param_importances.html")
        print("Saved 2_param_importances.html")
    except Exception as e:
        print(f"Could not generate parameter importance plot: {e}")

    # Slice Plot (shows 1D slice for each parameter)
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(PLOT_DIR / "3_slice_plot.html")
        print("Saved 3_slice_plot.html")
    except Exception as e:
        print(f"Could not generate slice plot: {e}")

    # Contour Plot (example for two parameters)
    # You can customize this by finding important params from plot 2
    try:
        important_params = [
            p[0] for p in optuna.importance.get_param_importances(study).items()
            if p[1] is not None
        ]

        if len(important_params) >= 2:
            fig = optuna.visualization.plot_contour(
                study, params=important_params[:2]
            )
            fig.write_html(PLOT_DIR / "4_contour_plot.html")
            print(f"Saved 4_contour_plot.html (for {important_params[0]} vs {important_params[1]})")
        else:
            print("Skipping contour plot: not enough completed trials or parameters.")

    except Exception as e:
        print(f"Could not generate contour plot: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
