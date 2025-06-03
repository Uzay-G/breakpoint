"""
Regression analysis to predict model success based on code complexity metrics.

This script fits a logistic regression model to predict whether a model will 
successfully solve a problem in REMOVE mode based on code line count and harmonic centrality metrics,
while accounting for the model's average performance.
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

# Try to use scienceplots style if available

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
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
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

    Args:
        data: List of dictionaries containing problem data

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

                regression_data.append(
                    {
                        "function_id": function_id,
                        "model": model_name,
                        "test_config": test_config,
                        "code_lines": code_lines,
                        "harmonic": harmonic,
                        "success": (
                            1.0 if success == 1.0 else 0.0
                        ),  # Ensure binary outcome
                    }
                )

    return pd.DataFrame(regression_data)


def calculate_model_performance(df):
    """Calculate average performance for each model/test_config combination."""
    model_perf = df.groupby(["model", "test_config"])["success"].mean().reset_index()
    model_perf.rename(columns={"success": "avg_performance"}, inplace=True)
    return model_perf


def fit_logistic_regression(df):
    """Fit a logistic regression model to predict success."""
    # Ensure we have enough data points
    if len(df) < 10:
        print(f"Not enough data points: {len(df)}")
        return None, None

    # Add model's average performance
    model_perf = calculate_model_performance(df)
    df = pd.merge(df, model_perf, on=["model", "test_config"])

    # Create design matrix and response variable
    X = df[["code_lines", "harmonic", "avg_performance"]]
    y = df["success"]

    # Add constant for intercept
    X_sm = sm.add_constant(X)

    # Fit logistic regression model
    model = sm.Logit(y, X_sm)

    try:
        result = model.fit()
        return result, df
    except Exception as e:
        print(f"Error fitting regression model: {e}")
        return None, None


def plot_success_probability_by_complexity(result, df, output_path=None):
    """Create visualization of regression results showing success probability vs. complexity metrics."""
    if result is None:
        return

    # Print regression summary
    print("\nRegression Results for all models in REMOVE mode:")
    print(result.summary())

    # Extract coefficients
    intercept = result.params["const"]
    coef_code_lines = result.params["code_lines"]
    coef_harmonic = result.params["harmonic"]
    coef_avg_performance = result.params["avg_performance"]

    # Calculate the actual average performance across all models
    avg_perf = df["avg_performance"].mean()

    # Create a single figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid for predicted probabilities
    x_min, x_max = df["code_lines"].min() - 0.5, df["code_lines"].max() + 0.5
    y_min, y_max = df["harmonic"].min() - 0.5, df["harmonic"].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calculate predicted probabilities using the average performance
    Z = (
        intercept
        + coef_code_lines * xx
        + coef_harmonic * yy
        + coef_avg_performance * avg_perf
    )
    Z = 1 / (1 + np.exp(-Z))  # Convert from logit to probability

    # Plot contour lines for different probability levels
    contour_lines = ax.contour(
        xx,
        yy,
        Z,
        levels=[0.05, 0.25, 0.5, 0.75, 0.95],
        colors="black",
        linestyles=[":", "--", "-", "--", ":"],
        linewidths=[1, 1, 2, 1, 1],
    )
    ax.clabel(contour_lines, inline=True, fontsize=14, fmt="%.2f")

    # Separate the data by success/failure
    success_data = df[df["success"] == 1]
    failure_data = df[df["success"] == 0]

    # Create hexbin plot for success rates
    from matplotlib.colors import LinearSegmentedColormap

    # Create a hexbin plot showing success rates in each bin
    # First, combine all data points and add a success column
    hb_data = pd.DataFrame(
        {
            "code_lines": np.concatenate(
                [success_data["code_lines"], failure_data["code_lines"]]
            ),
            "harmonic": np.concatenate(
                [success_data["harmonic"], failure_data["harmonic"]]
            ),
            "success": np.concatenate(
                [np.ones(len(success_data)), np.zeros(len(failure_data))]
            ),
        }
    )

    # Function to compute success rate in each bin
    def success_rate(x):
        return np.sum(x) / len(x) if len(x) > 0 else np.nan

    # Create hexbin for density visualization
    hexbin = ax.hexbin(
        hb_data["code_lines"],
        hb_data["harmonic"],
        C=hb_data["success"],  # Color by success
        reduce_C_function=success_rate,  # Calculate success rate in each bin
        gridsize=50,  # Adjust for fewer, larger hexagons
        cmap="coolwarm",  # Blue for high success, red for low success
        alpha=0.7,
        mincnt=1,  # Only show bins with at least 1 point
        vmin=0,
        vmax=1,  # Success rate ranges from 0 to 1
    )

    # Add a colorbar for success rate
    cbar_hex = plt.colorbar(hexbin, ax=ax, label="Success Rate", orientation="vertical")

    # Add handles for legend
    legend_handles = [
        plt.Line2D(
            [0], [0], color="black", linestyle=":", label="5%"
        ),
        plt.Line2D(
            [0], [0], color="black", linestyle="--", label="25%"
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=2,
            label="50%",
        ),
        plt.Line2D(
            [0], [0], color="black", linestyle="--", label="75%"
        ),
        plt.Line2D(
            [0], [0], color="black", linestyle=":", label="95%"
        ),
    ]

    # Add axis labels and title
    ax.set_xlabel("Code Line Count (z-score)")
    ax.set_ylabel("Harmonic Centrality (z-score)")
    ax.set_title("Success Probability by Code Complexity (Remove)")



    # Add model information
    model_info = f"Average model performance: {avg_perf:.2f}"
    ax.text(
        0.02,
        0.98,
        model_info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # We already have a colorbar for the hexbin, so we don't need another one for the contour

    # Add legend with the contour line explanations
    ax.legend(handles=legend_handles, loc="upper right", title="Predicted Success Rate", title_fontsize=14)

    # Save the figure if output_path is provided
    if output_path:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    plt.tight_layout()
    plt.show()


def main():
    # Configuration
    file_path = "/home/ubuntu/breakpoint/consolidated_results/consolidated_results_remove_all_normalized.jsonl"
    output_dir = "/home/ubuntu/breakpoint/final_plots/"

    # Load data
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    print(f"Loaded {len(data)} entries")

    # Prepare data for regression
    df = prepare_data_for_regression(data)
    print(f"Prepared {len(df)} data points for regression analysis")

    # Summary of data
    print("\nData Summary:")
    print(f"Number of unique functions: {df['function_id'].nunique()}")
    print(f"Number of models: {df['model'].nunique()}")
    print(f"Models included: {', '.join(df['model'].unique())}")
    print(f"Test configurations: {', '.join(df['test_config'].unique())}")

    # Fit logistic regression
    result, df_with_avg = fit_logistic_regression(df)

    if result is not None:
        # Plot regression results
        output_path = f"{output_dir}success_heatmap_all_models_remove_mode.png"
        plot_success_probability_by_complexity(result, df_with_avg, output_path)


if __name__ == "__main__":
    main()
