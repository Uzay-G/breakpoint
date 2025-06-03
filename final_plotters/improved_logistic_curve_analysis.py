"""
Improved logistic curve analysis with proper prediction intervals.

This script creates plots of logistic curves for complexity and harmonic centrality
with bootstrap-based prediction intervals that correctly represent prediction uncertainty.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import scienceplots
from pathlib import Path
import os
import logging
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from numpy.linalg import LinAlgError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and settings
BASE_DIR = Path("/home/ubuntu/breakpoint")
CONSOLIDATED_RESULTS = (
    BASE_DIR / "consolidated_results/consolidated_results_remove_all_normalized.jsonl"
)
OUTPUT_DIR = BASE_DIR / "final_plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Try to use scienceplots style
try:
    plt.style.use(["science"])
    # Apply solarize color scheme with white background
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Use solarize color cycle
            "axes.prop_cycle": plt.cycler(
                "color",
                [
                    "#268bd2",
                    "#dc322f",
                    "#859900",
                    "#d33682",
                    "#2aa198",
                    "#b58900",
                    "#6c71c4",
                    "#cb4b16",
                ],
            ),
            'font.size': 26,
            'axes.titlesize': 36,
            'axes.labelsize': 34,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'legend.fontsize': 30,
        }
    )
except:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            'font.size': 26,
            'axes.titlesize': 36,
            'axes.labelsize': 34,
            'xtick.labelsize': 30,
            'ytick.labelsize': 30,
            'legend.fontsize': 30,
        }
    )


def load_data(filepath):
    """Load and parse the consolidated results file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_data_for_regression(data):
    """
    Prepare data for regression analysis, including all models and all REMOVE mode test configs.

    Returns:
        pandas DataFrame ready for regression analysis
    """
    regression_data = []

    for item in data:
        function_id = item.get("function_id", "unknown")

        # Skip entries without required data
        if "complexity_metrics" not in item or "model_performance" not in item:
            continue

        # Extract complexity metrics
        if "normalized_metrics" in item and "complexity" in item["normalized_metrics"]:
            code_lines = item["normalized_metrics"]["complexity"].get(
                "code_line_count_z",
                item["normalized_metrics"]["complexity"].get("line_count_z", None),
            )
        else:
            code_lines = item["complexity_metrics"].get(
                "code_line_count", item["complexity_metrics"].get("line_count", None)
            )

        # Extract harmonic centrality
        if "normalized_metrics" in item and "centrality" in item["normalized_metrics"]:
            harmonic = item["normalized_metrics"]["centrality"].get(
                "harmonic_z",
                item["normalized_metrics"]["centrality"].get("harmonic_log_z", None),
            )
        else:
            if "centrality" in item["complexity_metrics"]:
                harmonic = item["complexity_metrics"]["centrality"].get(
                    "harmonic", None
                )
            else:
                harmonic = None

        # Skip entries with missing complexity data
        if code_lines is None or harmonic is None:
            continue

        # Process all models and their REMOVE mode test configs
        for model_name, model_data in item["model_performance"].items():
            for test_config, success in model_data.items():
                # Only include REMOVE mode test configs
                if not test_config.startswith("remove_"):
                    continue

                # Extract iteration count from test config (e.g., remove_16iter_4tests -> 16)
                parts = test_config.split("_")
                iterations = None
                for part in parts:
                    if part.endswith("iter"):
                        try:
                            iterations = int(part.replace("iter", ""))
                            break
                        except ValueError:
                            pass

                if iterations is None:
                    continue

                regression_data.append(
                    {
                        "function_id": function_id,
                        "model": model_name,
                        "test_config": test_config,
                        "iterations": iterations,
                        "code_lines": code_lines,
                        "harmonic": harmonic,
                        "success": (
                            1.0 if success == 1.0 else 0.0
                        ),  # Ensure binary outcome
                    }
                )

    return pd.DataFrame(regression_data)


def bootstrap_prediction_intervals(X, y, n_bootstrap=1000, alpha=0.05):
    """
    Generate bootstrap-based prediction intervals for logistic regression.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (e.g., 0.05 for 95% intervals)

    Returns:
        Dictionary with mean predictions and lower/upper bounds
    """
    n_samples = X.shape[0]

    # Create grid for predictions
    x_grid = np.linspace(-3, 3, 100).reshape(-1, 1)
    X_grid = sm.add_constant(x_grid)

    # Store successful predictions only
    successful_predictions = []
    failed_iterations = 0

    # Fit on bootstrap samples
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]

        try:
            # Suppress convergence warnings but catch them
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings('error', category=ConvergenceWarning)
                
                # Fit logistic regression model
                model = sm.Logit(y_boot, X_boot)
                result = model.fit(disp=0, maxiter=100)

                # Check if converged
                if not result.converged:
                    logger.debug(f"Bootstrap iteration {i}: Model did not converge")
                    failed_iterations += 1
                    continue

                # Predict on grid
                y_pred = result.predict(X_grid)
                successful_predictions.append(y_pred)
                
        except ConvergenceWarning as e:
            logger.debug(f"Bootstrap iteration {i}: Convergence warning - {str(e)}")
            failed_iterations += 1
            continue
            
        except LinAlgError as e:
            logger.warning(f"Bootstrap iteration {i}: Linear algebra error - {str(e)}")
            failed_iterations += 1
            continue
            
        except Exception as e:
            logger.warning(f"Bootstrap iteration {i}: Unexpected error - {type(e).__name__}: {str(e)}")
            failed_iterations += 1
            continue

    # Check if we have enough successful fits
    n_successful = len(successful_predictions)
    if n_successful < n_bootstrap * 0.5:  # Require at least 50% success rate
        logger.warning(f"Only {n_successful}/{n_bootstrap} bootstrap iterations succeeded. "
                      f"Results may be unreliable.")
    
    if n_successful == 0:
        raise ValueError("All bootstrap iterations failed. Cannot compute prediction intervals.")
    
    # Convert to array for easier manipulation
    all_predictions = np.array(successful_predictions)
    
    # Log success rate
    logger.info(f"Bootstrap completed: {n_successful}/{n_bootstrap} iterations successful "
                f"({n_successful/n_bootstrap*100:.1f}% success rate)")

    # Calculate mean and intervals
    mean_pred = np.mean(all_predictions, axis=0)
    lower_bound = np.percentile(all_predictions, alpha / 2 * 100, axis=0)
    upper_bound = np.percentile(all_predictions, (1 - alpha / 2) * 100, axis=0)

    return {
        "x_values": x_grid.flatten(),
        "mean_predictions": mean_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_successful": n_successful,
        "n_failed": failed_iterations,
    }


def fit_logistic_regression_by_model(df, filter_iterations=16):
    """
    Fit separate logistic regression models for each model at a specific iteration count,
    with bootstrap-based prediction intervals.

    Args:
        df: DataFrame with prepared data
        filter_iterations: Only include test configs with this many iterations

    Returns:
        Dictionary of regression results by model
    """
    results = {}

    # Filter data for the specified iteration count
    filtered_df = df[df["iterations"] == filter_iterations]

    # Get unique models
    models = filtered_df["model"].unique()

    for model in models:
        # Filter data for this model
        model_df = filtered_df[filtered_df["model"] == model]

        # Ensure we have enough data points
        if len(model_df) < 10:
            print(f"Not enough data points for {model}: {len(model_df)}")
            continue

        # Get feature and target variables
        X_complexity = sm.add_constant(model_df[["code_lines"]].values)
        X_harmonic = sm.add_constant(model_df[["harmonic"]].values)
        y = model_df["success"].values

        # First get standard logistic regression models (for parameter estimates)
        try:
            # Fit standard logistic regression for complexity
            complexity_model = sm.Logit(y, X_complexity)
            complexity_result = complexity_model.fit(disp=0)

            # Fit standard logistic regression for harmonic centrality
            harmonic_model = sm.Logit(y, X_harmonic)
            harmonic_result = harmonic_model.fit(disp=0)

            # Generate bootstrap prediction intervals
            complexity_intervals = bootstrap_prediction_intervals(
                X_complexity, y, n_bootstrap=1000
            )
            harmonic_intervals = bootstrap_prediction_intervals(
                X_harmonic, y, n_bootstrap=1000
            )

            results[model] = {
                "complexity_result": complexity_result,
                "harmonic_result": harmonic_result,
                "complexity_intervals": complexity_intervals,
                "harmonic_intervals": harmonic_intervals,
                "n_samples": len(model_df),
                "avg_performance": model_df["success"].mean(),
            }
        except Exception as e:
            print(f"Error fitting regression model for {model}: {e}")

    return results


def fit_logistic_regression_by_iteration(df, model_name="o4-mini"):
    """
    Fit separate logistic regression models for a specific model at different iteration counts,
    with bootstrap-based prediction intervals.

    Args:
        df: DataFrame with prepared data
        model_name: Only include this specific model

    Returns:
        Dictionary of regression results by iteration count
    """
    results = {}

    # Filter data for the specified model
    filtered_df = df[df["model"] == model_name]

    # Get unique iteration counts
    iterations = sorted(filtered_df["iterations"].unique())

    for iter_count in iterations:
        # Filter data for this iteration count
        iter_df = filtered_df[filtered_df["iterations"] == iter_count]

        # Ensure we have enough data points
        if len(iter_df) < 10:
            print(
                f"Not enough data points for {model_name} at {iter_count} iterations: {len(iter_df)}"
            )
            continue

        # Get feature and target variables
        X_complexity = sm.add_constant(iter_df[["code_lines"]].values)
        X_harmonic = sm.add_constant(iter_df[["harmonic"]].values)
        y = iter_df["success"].values

        try:
            # Fit standard logistic regression for complexity
            complexity_model = sm.Logit(y, X_complexity)
            complexity_result = complexity_model.fit(disp=0)

            # Fit standard logistic regression for harmonic centrality
            harmonic_model = sm.Logit(y, X_harmonic)
            harmonic_result = harmonic_model.fit(disp=0)

            # Generate bootstrap prediction intervals
            complexity_intervals = bootstrap_prediction_intervals(
                X_complexity, y, n_bootstrap=1000
            )
            harmonic_intervals = bootstrap_prediction_intervals(
                X_harmonic, y, n_bootstrap=1000
            )

            results[iter_count] = {
                "complexity_result": complexity_result,
                "harmonic_result": harmonic_result,
                "complexity_intervals": complexity_intervals,
                "harmonic_intervals": harmonic_intervals,
                "n_samples": len(iter_df),
                "avg_performance": iter_df["success"].mean(),
            }
        except Exception as e:
            print(
                f"Error fitting regression model for {model_name} at {iter_count} iterations: {e}"
            )

    return results


def plot_logistic_curves_by_model(regression_results, output_path=None):
    """
    Plot logistic curves with bootstrap prediction intervals for different models
    at a fixed iteration count in REMOVE mode.

    Args:
        regression_results: Dictionary of regression results by model
        output_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot complexity curves
    for model, results in regression_results.items():
        # Get prediction intervals
        complexity_intervals = results["complexity_intervals"]
        x_vals = complexity_intervals["x_values"]
        mean_pred = complexity_intervals["mean_predictions"]
        lower_bound = complexity_intervals["lower_bound"]
        upper_bound = complexity_intervals["upper_bound"]

        # Get model performance for the legend
        avg_perf = results["avg_performance"]

        # Simplify model names for legend
        model_display = model.replace("claude-3-7-sonnet-latest", "Claude-3.7")
        
        # Plot the curve
        (line,) = ax1.plot(
            x_vals,
            mean_pred,
            linewidth=2,
            label=model_display,
        )
        color = line.get_color()

        # Add shaded prediction interval
        ax1.fill_between(x_vals, lower_bound, upper_bound, alpha=0.2, color=color)

    # Add axis labels and title for complexity
    ax1.set_xlabel("Code Complexity (z-score)", fontsize=14)
    ax1.set_ylabel("Success Rate", fontsize=14)
    ax1.set_title("Code Complexity", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', labelsize=12)

    # Plot harmonic centrality curves
    for model, results in regression_results.items():
        # Get prediction intervals
        harmonic_intervals = results["harmonic_intervals"]
        x_vals = harmonic_intervals["x_values"]
        mean_pred = harmonic_intervals["mean_predictions"]
        lower_bound = harmonic_intervals["lower_bound"]
        upper_bound = harmonic_intervals["upper_bound"]

        # Get model performance for the legend
        avg_perf = results["avg_performance"]

        # Simplify model names for legend
        model_display = model.replace("claude-3-7-sonnet-latest", "Claude-3.7")
        
        # Plot the curve
        (line,) = ax2.plot(
            x_vals,
            mean_pred,
            linewidth=2,
            label=model_display,
        )
        color = line.get_color()

        # Add shaded prediction interval
        ax2.fill_between(x_vals, lower_bound, upper_bound, alpha=0.2, color=color)

    # Add axis labels and title for harmonic centrality
    ax2.set_xlabel("Harmonic Centrality (z-score)", fontsize=14)
    ax2.set_ylabel("Success Rate", fontsize=14)
    ax2.set_title("Harmonic Centrality", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='both', labelsize=12)

    # Add legends with larger font
    ax1.legend(loc="best", fontsize=12)
    ax2.legend(loc="best", fontsize=12)

    plt.tight_layout()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


def plot_logistic_curves_by_iteration(
    regression_results, model_name="o4-mini", output_path=None
):
    """
    Plot logistic curves with bootstrap prediction intervals for different iteration counts
    for a specific model in REMOVE mode.

    Args:
        regression_results: Dictionary of regression results by iteration count
        model_name: Name of the model being analyzed
        output_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort iteration counts
    iter_counts = sorted(regression_results.keys())

    # Plot complexity curves
    for iter_count in iter_counts:
        results = regression_results[iter_count]

        # Get prediction intervals
        complexity_intervals = results["complexity_intervals"]
        x_vals = complexity_intervals["x_values"]
        mean_pred = complexity_intervals["mean_predictions"]
        lower_bound = complexity_intervals["lower_bound"]
        upper_bound = complexity_intervals["upper_bound"]

        # Get performance metrics for the legend
        avg_perf = results["avg_performance"]
        n_samples = results["n_samples"]

        # Plot the curve
        (line,) = ax1.plot(
            x_vals,
            mean_pred,
            linewidth=2,
            label=f"{iter_count} iterations",
        )
        color = line.get_color()

        # Add shaded prediction interval
        ax1.fill_between(x_vals, lower_bound, upper_bound, alpha=0.2, color=color)

    # Add axis labels and title for complexity
    ax1.set_xlabel("Code Complexity (z-score)", fontsize=14)
    ax1.set_ylabel("Success Rate", fontsize=14)
    ax1.set_title(f"{model_name}: Code Complexity", fontsize=16)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', labelsize=12)

    # Plot harmonic centrality curves
    for iter_count in iter_counts:
        results = regression_results[iter_count]

        # Get prediction intervals
        harmonic_intervals = results["harmonic_intervals"]
        x_vals = harmonic_intervals["x_values"]
        mean_pred = harmonic_intervals["mean_predictions"]
        lower_bound = harmonic_intervals["lower_bound"]
        upper_bound = harmonic_intervals["upper_bound"]

        # Get performance metrics for the legend
        avg_perf = results["avg_performance"]
        n_samples = results["n_samples"]

        # Plot the curve
        (line,) = ax2.plot(
            x_vals,
            mean_pred,
            linewidth=2,
            label=f"{iter_count} iterations",
        )
        color = line.get_color()

        # Add shaded prediction interval
        ax2.fill_between(x_vals, lower_bound, upper_bound, alpha=0.2, color=color)

    # Add axis labels and title for harmonic centrality
    ax2.set_xlabel("Harmonic Centrality (z-score)", fontsize=14)
    ax2.set_ylabel("Success Rate", fontsize=14)
    ax2.set_title(f"{model_name}: Harmonic Centrality", fontsize=16)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='both', labelsize=12)

    # Add legends with larger font
    ax1.legend(loc="best", fontsize=12)
    ax2.legend(loc="best", fontsize=12)

    plt.tight_layout()

    # Save the figure if output_path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.show()


def add_scatter_points(ax, df, x_column, model_name=None, iterations=None, alpha=0.2):
    """
    Add scatter points to the plot to show the raw data.
    """
    if model_name is not None:
        df = df[df["model"] == model_name]

    if iterations is not None:
        df = df[df["iterations"] == iterations]

    # Jitter the y values slightly for better visualization
    jitter = 0.03
    success_y = np.ones(len(df[df["success"] == 1])) - np.random.uniform(
        0, jitter, len(df[df["success"] == 1])
    )
    failure_y = np.zeros(len(df[df["success"] == 0])) + np.random.uniform(
        0, jitter, len(df[df["success"] == 0])
    )

    # Plot successes
    ax.scatter(
        df[df["success"] == 1][x_column],
        success_y,
        marker="o",
        color="green",
        alpha=alpha,
        s=30,
        edgecolor=None,
    )

    # Plot failures
    ax.scatter(
        df[df["success"] == 0][x_column],
        failure_y,
        marker="x",
        color="red",
        alpha=alpha,
        s=30,
        edgecolor=None,
    )


def main():
    # Load data
    print(f"Loading data from {CONSOLIDATED_RESULTS}...")
    data = load_data(CONSOLIDATED_RESULTS)
    print(f"Loaded {len(data)} entries")

    # Prepare data for regression
    df = prepare_data_for_regression(data)
    print(f"Prepared {len(df)} data points for regression analysis")

    # Summary of data
    print("\nData Summary:")
    print(f"Number of unique functions: {df['function_id'].nunique()}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Models included: {', '.join(df['model'].unique())}")
    print(f"Iteration counts: {', '.join(map(str, sorted(df['iterations'].unique())))}")

    # 1. Compare models at 16 iterations
    print(
        "\nFitting logistic regression models for different models at 16 iterations..."
    )
    model_results = fit_logistic_regression_by_model(df, filter_iterations=16)

    if model_results:
        output_path = OUTPUT_DIR / "improved_logistic_curves_by_model_16iter_remove.png"
        plot_logistic_curves_by_model(model_results, output_path)

    # 2. Compare o4-mini at different iterations
    print("\nFitting logistic regression models for o4-mini at different iterations...")
    iteration_results = fit_logistic_regression_by_iteration(df, model_name="o4-mini")

    if iteration_results:
        output_path = (
            OUTPUT_DIR / "improved_logistic_curves_by_iteration_o4mini_remove.png"
        )
        plot_logistic_curves_by_iteration(
            iteration_results, model_name="o4-mini", output_path=output_path
        )


if __name__ == "__main__":
    main()
