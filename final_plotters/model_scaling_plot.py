"""
Plot model scaling results across iterations for o4-mini and gpt-4o.

This script creates a publication-quality plot showing performance scaling 
with iteration count (4, 8, 16) for both remove and discovery modes.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
from scipy import stats as scipy_stats

# Constants and settings
BASE_DIR = Path("/home/ubuntu/breakpoint")
CONSOLIDATED_DISCOVERY = (
    BASE_DIR
    / "consolidated_results/consolidated_results_discovery_all_with_tools_filters.jsonl"
)
CONSOLIDATED_REMOVE = (
    BASE_DIR
    / "consolidated_results/consolidated_results_remove_all_with_tools_filters.jsonl"
)
OUTPUT_DIR = BASE_DIR / "final_plots"
OUTPUT_FILE = OUTPUT_DIR / "model_scaling_plot.png"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up the plotting style
try:
    import scienceplots

    plt.style.use(["science", "grid"])
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
except ImportError:
    print("For better plots, install scienceplots: pip install scienceplots")
    plt.style.use("seaborn-v0_8-whitegrid")
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


def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_performance_data(data, mode_prefix, models=None):
    """
    Extract performance data for each model and iteration count.

    Args:
        data: List of dictionaries containing problem data
        mode_prefix: Prefix for test configurations (either 'discovery_' or 'remove_')
        models: List of model names to include (None for all models)

    Returns:
        DataFrame with performance metrics by model and iteration count
    """
    results = []

    for item in data:
        function_id = item.get("function_id", "unknown")

        # Extract model performance data
        model_performance = item.get("model_performance", {})
        if not model_performance:
            continue

        # Process each model's performance
        for model_name, metrics in model_performance.items():
            # Skip models not in the specified list (if provided)
            if models and model_name not in models:
                continue

            # Process each test configuration
            for test_config, score in metrics.items():
                # Filter by mode prefix
                if not test_config.startswith(mode_prefix):
                    continue

                # Extract iteration count from test config (e.g., remove_16iter_4tests -> 16)
                iter_match = re.search(r"(\d+)iter", test_config)
                if not iter_match:
                    continue

                iter_count = int(iter_match.group(1))

                # Ensure binary score (perfect solve mode)
                binary_score = 1.0 if score == 1.0 else 0.0

                # Store the result
                results.append(
                    {
                        "function_id": function_id,
                        "model": model_name,
                        "iterations": iter_count,
                        "test_config": test_config,
                        "mode": (
                            "discovery" if mode_prefix == "discovery_" else "remove"
                        ),
                        "score": binary_score,
                    }
                )

    return pd.DataFrame(results)


def calculate_performance_stats(df):
    """
    Calculate performance statistics by model, mode, and iteration count.

    Args:
        df: DataFrame with performance data

    Returns:
        DataFrame with aggregated statistics
    """
    # Group by model, mode, and iteration count
    grouped = df.groupby(["model", "mode", "iterations"])

    # Calculate statistics
    stats = (
        grouped["score"]
        .agg(
            [
                ("mean", "mean"),
                ("std", "std"),
                ("count", "count"),
                ("sem", lambda x: scipy_stats.sem(x)),  # Standard error of the mean
            ]
        )
        .reset_index()
    )

    # Calculate 95% confidence intervals
    stats["ci_lower"] = stats["mean"] - 1.96 * stats["sem"]
    stats["ci_upper"] = stats["mean"] + 1.96 * stats["sem"]

    # Ensure confidence intervals are within valid range [0, 1]
    stats["ci_lower"] = stats["ci_lower"].clip(0, 1)
    stats["ci_upper"] = stats["ci_upper"].clip(0, 1)

    return stats


def plot_scaling_results(stats_df, output_path):
    """
    Create a publication-quality plot showing performance scaling with iteration count.

    Args:
        stats_df: DataFrame with performance statistics
        output_path: Path to save the output figure
    """
    # Set up the figure with two subplots (one for each mode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Define markers and colors for models
    model_styles = {
        "o4-mini": {
            "color": "#268bd2",
            "marker": "o",
            "linestyle": "-",
            "label": "o4-mini",
        },
        "gpt-4o": {
            "color": "#dc322f",
            "marker": "s",
            "linestyle": "--",
            "label": "GPT-4o",
        },
    }

    # Plot discovery mode (left subplot)
    for model, style in model_styles.items():
        # Filter data for this model and discovery mode
        model_data = stats_df[
            (stats_df["model"] == model) & (stats_df["mode"] == "discovery")
        ]

        if len(model_data) > 0:
            # Sort by iteration count
            model_data = model_data.sort_values("iterations")

            # Plot the line
            ax1.errorbar(
                model_data["iterations"],
                model_data["mean"],
                yerr=[
                    model_data["mean"] - model_data["ci_lower"],
                    model_data["ci_upper"] - model_data["mean"],
                ],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
                markersize=8,
                capsize=4,
                elinewidth=2,
                linewidth=2,
            )

    # Plot remove mode (right subplot)
    for model, style in model_styles.items():
        # Filter data for this model and remove mode
        model_data = stats_df[
            (stats_df["model"] == model) & (stats_df["mode"] == "remove")
        ]

        if len(model_data) > 0:
            # Sort by iteration count
            model_data = model_data.sort_values("iterations")

            # Plot the line
            ax2.errorbar(
                model_data["iterations"],
                model_data["mean"],
                yerr=[
                    model_data["mean"] - model_data["ci_lower"],
                    model_data["ci_upper"] - model_data["mean"],
                ],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
                markersize=8,
                capsize=4,
                elinewidth=2,
                linewidth=2,
            )

    # Set axis labels and titles
    ax1.set_xlabel("Number of Iterations")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Discovery Mode")

    ax2.set_xlabel("Number of Iterations")
    ax2.set_title("Remove Mode")

    # Set x-axis ticks to match the iteration counts we have
    iteration_counts = sorted(stats_df["iterations"].unique())

    ax1.set_xticks(iteration_counts)
    ax2.set_xticks(iteration_counts)

    # Increase tick label size
    ax1.tick_params(axis="both", which="major", labelsize=30)
    ax2.tick_params(axis="both", which="major", labelsize=30)

    # Set y-axis limits
    ax1.set_ylim(0, 1.05)

    # Add a grid
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add a legend
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    # Add a suptitle
    plt.suptitle("Model Performance Scaling with Iteration Count", y=0.98)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Scaling plot saved to {output_path}")

    # Show the plot
    plt.show()


def main():
    # Load data
    print(f"Loading discovery data from {CONSOLIDATED_DISCOVERY}")
    discovery_data = load_jsonl(CONSOLIDATED_DISCOVERY)
    print(f"Loading remove data from {CONSOLIDATED_REMOVE}")
    remove_data = load_jsonl(CONSOLIDATED_REMOVE)

    # Models to analyze
    target_models = ["o4-mini", "gpt-4o"]

    # Extract performance data for each mode
    print("Extracting performance data...")
    discovery_df = extract_performance_data(discovery_data, "discovery_", target_models)
    remove_df = extract_performance_data(remove_data, "remove_", target_models)

    # Combine both modes' data
    combined_df = pd.concat([discovery_df, remove_df])

    # Calculate statistics
    print("Calculating performance statistics...")
    stats_df = calculate_performance_stats(combined_df)

    # Print summary statistics
    print("\nPerformance Statistics by Model, Mode, and Iteration Count:")
    print(stats_df[["model", "mode", "iterations", "mean", "sem", "count"]])

    # Create the scaling plot
    print("Creating scaling plot...")
    plot_scaling_results(stats_df, OUTPUT_FILE)


if __name__ == "__main__":
    main()
