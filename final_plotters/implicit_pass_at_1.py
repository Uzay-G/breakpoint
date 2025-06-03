"""
Calculate implicit pass@1 for models with multiple iterations.

This script analyzes consolidated result files to calculate what would be the pass@1 
score for models that had multiple iterations available - essentially analyzing 
only their first attempt at each problem.
"""

import json
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import re
from scipy import stats as scipy_stats

# Constants and paths
BASE_DIR = Path("/home/ubuntu/breakpoint")
CONSOLIDATED_DISCOVERY = (
    BASE_DIR
    / "consolidated_results/consolidated_results_discovery_all_with_tools_filters.jsonl"
)
CONSOLIDATED_REMOVE = (
    BASE_DIR
    / "consolidated_results/consolidated_results_remove_all_with_tools_filters_full.jsonl"
)
OUTPUT_DIR = BASE_DIR / "final_plots"

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
            'legend.fontsize': 22,
        }
    )
except ImportError:
    print("For better plots, install scienceplots: pip install scienceplots")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
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
            'legend.fontsize': 22,
        }
    )

# Define consistent Solarized colors for models
SOLARIZED_COLORS = {
    "o4-mini": "#268bd2",  # Blue
    "gpt-4o": "#dc322f",  # Red
    "o3-mini": "#6C71C4",  # Violet
    "claude-3-7-sonnet-latest": "#d33682",  # Magenta
}


def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def calculate_implicit_pass_at_1(data, mode_prefix, models=None):
    """
    Calculate the implicit pass@1 rate for models with multiple iterations.
    This tracks whether the first submission for each problem was successful.

    Args:
        data: List of dictionaries containing problem data
        mode_prefix: Prefix for test configurations (either 'discovery_' or 'remove_')
        models: List of model names to include (None for all models)

    Returns:
        DataFrame with model, iterations, and first-attempt success data
    """
    results = []

    for item in data:
        function_id = item.get("function_id", "unknown")

        # Extract model performance data
        model_performance = item.get("model_performance", {})
        if not model_performance:
            continue

        # Extract tool call lists
        tool_call_lists = item.get("tool_call_lists", {})
        if not tool_call_lists:
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

                # Get tool calls for this model and config
                if test_config in tool_call_lists.get(model_name, {}):
                    tool_calls = tool_call_lists[model_name][test_config]

                    # Filter to only submit attempts
                    submit_indices = [
                        i
                        for i, tool in enumerate(tool_calls)
                        if tool == "submit_attempt" or tool == "replace_function"
                    ]

                    # If there was at least one submission
                    if submit_indices:
                        # Check if the first attempt was successful
                        first_attempt_successful = False

                        # If we have intermediate scores, check if the first attempt was successful
                        if (
                            "intermediate_scores" in item
                            and model_name in item["intermediate_scores"]
                            and test_config in item["intermediate_scores"][model_name]
                        ):
                            scores = item["intermediate_scores"][model_name][
                                test_config
                            ]
                            if len(scores) >= 1 and scores[0] == 1.0:
                                first_attempt_successful = True
                        # Otherwise, if there's only one submission and the final score is 1.0,
                        # then the first attempt must have been successful
                        elif len(submit_indices) == 1 and score == 1.0:
                            first_attempt_successful = True

                        # Store the result
                        results.append(
                            {
                                "function_id": function_id,
                                "model": model_name,
                                "iterations": iter_count,
                                "test_config": test_config,
                                "mode": (
                                    "discovery"
                                    if mode_prefix == "discovery_"
                                    else "remove"
                                ),
                                "first_attempt_success": (
                                    1.0 if first_attempt_successful else 0.0
                                ),
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
        grouped["first_attempt_success"]
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


def plot_16iter_bar_chart(stats_df, output_path):
    """
    Create a bar chart showing pass@1 rates for different models with 16 iterations.

    Args:
        stats_df: DataFrame with performance statistics
        output_path: Path to save the output figure
    """
    # Filter for 16 iterations
    data_16iter = stats_df[stats_df["iterations"] == 16]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set width of bars
    bar_width = 0.35

    # Extract models
    models = sorted(data_16iter["model"].unique())
    x_positions = np.arange(len(models))

    # Plot discovery mode results
    discovery_data = data_16iter[data_16iter["mode"] == "discovery"]
    discovery_means = [
        (
            discovery_data[discovery_data["model"] == model]["mean"].values[0]
            if not discovery_data[discovery_data["model"] == model].empty
            else 0
        )
        for model in models
    ]
    discovery_ci_lower = [
        (
            discovery_data[discovery_data["model"] == model]["ci_lower"].values[0]
            if not discovery_data[discovery_data["model"] == model].empty
            else 0
        )
        for model in models
    ]
    discovery_ci_upper = [
        (
            discovery_data[discovery_data["model"] == model]["ci_upper"].values[0]
            if not discovery_data[discovery_data["model"] == model].empty
            else 0
        )
        for model in models
    ]

    discovery_bars = ax.bar(
        x_positions - bar_width / 2,
        [m * 100 for m in discovery_means],  # Convert to percentage
        bar_width,
        label="Discovery Mode",
        color="#268bd2",  # Blue
        yerr=[(m - l) * 100 for m, l in zip(discovery_means, discovery_ci_lower)],
        capsize=5,
    )

    # Plot remove mode results
    remove_data = data_16iter[data_16iter["mode"] == "remove"]
    remove_means = [
        (
            remove_data[remove_data["model"] == model]["mean"].values[0]
            if not remove_data[remove_data["model"] == model].empty
            else 0
        )
        for model in models
    ]
    remove_ci_lower = [
        (
            remove_data[remove_data["model"] == model]["ci_lower"].values[0]
            if not remove_data[remove_data["model"] == model].empty
            else 0
        )
        for model in models
    ]
    remove_ci_upper = [
        (
            remove_data[remove_data["model"] == model]["ci_upper"].values[0]
            if not remove_data[remove_data["model"] == model].empty
            else 0
        )
        for model in models
    ]

    remove_bars = ax.bar(
        x_positions + bar_width / 2,
        [m * 100 for m in remove_means],  # Convert to percentage
        bar_width,
        label="Remove Mode",
        color="#dc322f",  # Red
        yerr=[(m - l) * 100 for m, l in zip(remove_means, remove_ci_lower)],
        capsize=5,
    )

    # Add labels, title, etc.
    ax.set_xlabel("Model")
    ax.set_ylabel("First Attempt Success Rate (%)")
    ax.set_title("Implicit Pass@1 Performance (16 Iterations)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models)

    # Format the y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Add value labels above bars
    for bar in discovery_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=22,
        )

    for bar in remove_bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=22,
        )

    # Add grid for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Add legend
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Pass@1 bar chart saved to {output_path}")

    # Show the plot
    plt.show()


def plot_iteration_scaling(stats_df, output_path):
    """
    Create a plot showing how pass@1 performance scales with iteration count.

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
        # 'o3-mini': {'color': '#6C71C4', 'marker': '^', 'linestyle': '-.', 'label': 'o3-mini'},
        # 'claude-3-7-sonnet-latest': {'color': '#d33682', 'marker': 'D', 'linestyle': ':', 'label': 'Claude 3.7 Sonnet'}
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
                model_data["mean"] * 100,  # Convert to percentage
                yerr=[
                    (model_data["mean"] - model_data["ci_lower"]) * 100,
                    (model_data["ci_upper"] - model_data["mean"]) * 100,
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
                model_data["mean"] * 100,  # Convert to percentage
                yerr=[
                    (model_data["mean"] - model_data["ci_lower"]) * 100,
                    (model_data["ci_upper"] - model_data["mean"]) * 100,
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
    ax1.set_ylabel("First Attempt Success Rate (%)")
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
    ax1.set_ylim(0, 100)

    # Format the y-axis as percentages
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # Add a grid
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add a legend
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")

    # Add a suptitle
    plt.suptitle(
        "First Attempt Success Rate by Model and Iteration Count", y=0.98
    )

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Iteration scaling plot saved to {output_path}")

    # Show the plot
    plt.show()


def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Target models
    target_models = ["o4-mini", "gpt-4o", "o3-mini", "claude-3-7-sonnet-latest"]

    # Load data
    print(f"Loading discovery data from {CONSOLIDATED_DISCOVERY}")
    discovery_data = load_jsonl(CONSOLIDATED_DISCOVERY)
    print(f"Loading remove data from {CONSOLIDATED_REMOVE}")
    remove_data = load_jsonl(CONSOLIDATED_REMOVE)

    # Extract performance data for each mode
    print("Extracting performance data...")
    discovery_df = calculate_implicit_pass_at_1(
        discovery_data, "discovery_", target_models
    )
    remove_df = calculate_implicit_pass_at_1(remove_data, "remove_", target_models)

    # Combine both modes' data
    combined_df = pd.concat([discovery_df, remove_df])

    # Calculate statistics
    print("Calculating performance statistics...")
    stats_df = calculate_performance_stats(combined_df)

    # Print summary statistics
    print("\nFirst Attempt Success Rates by Model, Mode, and Iteration Count:")
    print(stats_df[["model", "mode", "iterations", "mean", "sem", "count"]])

    # Create the bar chart for 16 iterations
    print("Creating 16 iteration bar chart...")
    plot_16iter_bar_chart(stats_df, OUTPUT_DIR / "implicit_pass_at_1_16iter.png")

    # Create the iteration scaling plot
    print("Creating iteration scaling plot...")
    plot_iteration_scaling(stats_df, OUTPUT_DIR / "implicit_pass_at_1_scaling.png")


if __name__ == "__main__":
    main()
