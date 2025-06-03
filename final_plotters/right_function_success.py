#!/usr/bin/env python3
"""
This script analyzes the relationship between finding the right function
and successfully solving problems in discovery mode.

It shows that when models identify the correct function, they almost always 
solve the problem correctly, demonstrating that finding the right function
is a necessary but not sufficient condition for solving the problem.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from collections import defaultdict
import os

# Use science plots style with solarize color scheme and white backgrounds
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
            'font.size': 18,
            'axes.titlesize': 22,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
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
            'font.size': 18,
            'axes.titlesize': 22,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16,
        }
    )


def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data


def analyze_right_function_correlation(data, test_config="discovery_16iter_4tests"):
    """
    Analyzes the correlation between finding the right function and solving the problem.

    Args:
        data: List of problem data dictionaries
        test_config: String indicating the test configuration to analyze

    Returns:
        Dictionary with models as keys and dictionaries of metrics as values
    """
    results = {}
    model_results = defaultdict(
        lambda: {
            "correct_solution": 0,
            "total_problems": 0,
            "right_function_found": 0,
            "solved_when_right_function": 0,
        }
    )

    for problem in data:
        # Extract model performances for the specified test configuration
        model_performance = problem.get("model_performance", {})
        right_function_found = problem.get("right_function_found", {})
        tool_call_lists = problem.get("tool_call_lists", {})

        for model, configs in model_performance.items():
            if test_config in configs:
                # Get performance score for this model and config
                score = configs[test_config]

                # Count total problems attempted by this model
                model_results[model]["total_problems"] += 1

                # Check if the problem was solved correctly (score of 1.0)
                if score == 1.0:
                    model_results[model]["correct_solution"] += 1

                # Get the right_function_found value directly from the field
                found_right_function = False
                if (
                    model in right_function_found
                    and test_config in right_function_found.get(model, {})
                ):
                    found_right_function = right_function_found[model][test_config]

                # If the model found the right function by either method
                if found_right_function:
                    model_results[model]["right_function_found"] += 1

                    # If the model found the right function AND solved the problem
                    if score == 1.0:
                        model_results[model]["solved_when_right_function"] += 1

    # Calculate percentages and other metrics
    for model, metrics in model_results.items():
        # Calculate overall success rate
        success_rate = (
            metrics["correct_solution"] / metrics["total_problems"]
            if metrics["total_problems"] > 0
            else 0
        )

        # Calculate percentage of problems where right function was found
        right_function_rate = (
            metrics["right_function_found"] / metrics["total_problems"]
            if metrics["total_problems"] > 0
            else 0
        )

        # Calculate success rate when right function was found (conditional probability)
        success_when_right = (
            metrics["solved_when_right_function"] / metrics["right_function_found"]
            if metrics["right_function_found"] > 0
            else 0
        )

        results[model] = {
            "total_problems": metrics["total_problems"],
            "success_rate": success_rate,
            "right_function_rate": right_function_rate,
            "success_when_right": success_when_right,
            "raw": metrics,
        }

    return results


def plot_nested_bar_chart(results, output_dir="final_plots"):
    """
    Creates a nested bar chart showing the relationship between finding the right function
    and successfully solving the problem.

    The outer (unfilled) bar represents the rate at which the right function was found.
    The inner (filled) bar represents the overall success rate.
    The conditional probability (success rate when right function found) is displayed on each bar.

    Args:
        results: Dictionary with models as keys and dictionaries of metrics as values
        output_dir: Directory to save the output plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Model name mappings to make them more readable - use the same as in main_figure.py
    model_name_map = {
        "o4-mini": "o4-mini",
        "gpt-4o": "GPT-4o",
        "o3-mini": "o3-mini",
        "claude-3-7-sonnet-latest": "Claude 3.7\nSonnet",
    }

    # Prepare data for plotting
    models = list(results.keys())
    # Apply name mapping if available
    display_names = [model_name_map.get(model, model) for model in models]

    right_function_rates = [
        results[model]["right_function_rate"] * 100 for model in models
    ]
    success_rates = [results[model]["success_rate"] * 100 for model in models]
    conditional_probs = [results[model]["success_when_right"] * 100 for model in models]

    # Sort all data by right function identification rate (descending)
    sorted_indices = np.argsort(right_function_rates)[::-1]
    display_names = [display_names[i] for i in sorted_indices]
    success_rates = [success_rates[i] for i in sorted_indices]
    right_function_rates = [right_function_rates[i] for i in sorted_indices]
    conditional_probs = [conditional_probs[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set the positions of the bars on the x-axis
    x = np.arange(len(display_names))
    bar_width = 0.5

    # Colors - use solarize colors that match the main figure
    edge_color = "#268bd2"  # Blue
    fill_color = "#859900"  # Green

    # Create the nested bars
    # First, create the outer bars (right function identification rate)
    outer_bars = ax.bar(
        x,
        right_function_rates,
        width=bar_width,
        color="none",
        edgecolor=edge_color,
        linewidth=2,
        label="Right Function Identification Rate",
    )

    # Then, create the inner bars (overall success rate)
    inner_bars = ax.bar(
        x,
        success_rates,
        width=bar_width,
        color=fill_color,
        alpha=0.7,
        label="Overall Success Rate",
    )

    # Add labels, title, and legend
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel(r"Success Rate (\%)", fontweight="bold", labelpad=10)
    ax.set_title(
        "Finding the Right Function vs. Problem Success Rate",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=0, ha="center")
    
    # Create custom legend with explanation
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Create custom legend handles that look like vertical bars (patches)
    legend_elements = [
        Patch(facecolor='none', edgecolor=edge_color, linewidth=2, label='Right Function Identification Rate'),
        Patch(facecolor=fill_color, alpha=0.7, label='Overall Success Rate'),
        Line2D([0], [0], color='none', label=r"Values above bars: P(success $|$ right function found)")
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Add the conditional probability as text vertically above each bar
    for i, (rf_rate, cond_prob) in enumerate(
        zip(right_function_rates, conditional_probs)
    ):
        if not np.isnan(cond_prob) and cond_prob > 0:
            # Place the text above the outer bar
            y_pos = rf_rate + 2
            ax.text(
                i,
                y_pos,
                rf"{cond_prob:.1f}\%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
                color="black",
            )
    
    # Customize y-axis to show percentages - set after adding text to ensure proper limits
    ax.set_ylim(0, 70)  # Set fixed limit to ensure text is visible
    ax.yaxis.set_ticks(range(0, 71, 10))

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/right_function_success.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(f"{output_dir}/right_function_success.pdf", bbox_inches="tight")
    plt.savefig(f"{output_dir}/right_function_success.svg", bbox_inches="tight")
    plt.close()

    # Create a summary table file for reference
    rows = []
    for i, model in enumerate(models):
        idx = sorted_indices[i]
        rows.append(
            [
                display_names[i],
                f"{results[models[idx]]['total_problems']}",
                f"{results[models[idx]]['raw']['right_function_found']} ({right_function_rates[i]:.1f}%)",
                f"{results[models[idx]]['raw']['correct_solution']} ({success_rates[i]:.1f}%)",
                f"{results[models[idx]]['raw']['solved_when_right_function']} ({conditional_probs[i]:.1f}%)",
            ]
        )

    # Create a pandas DataFrame for the table
    table_data = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Total Problems",
            "Right Function Found",
            "Problems Solved",
            "Solved when Right Function",
        ],
    )

    # Save the summary table for reference
    table_data.to_csv(f"{output_dir}/right_function_analysis.csv", index=False)

    return table_data


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and plot relationship between finding the right function and solving problems"
    )
    parser.add_argument(
        "--input",
        default="/home/ubuntu/breakpoint/consolidated_results/consolidated_results_discovery_all_with_tools_filters.jsonl",
        help="Path to the consolidated results file",
    )
    parser.add_argument(
        "--output", default="final_plots", help="Directory to save output plots"
    )
    parser.add_argument(
        "--config",
        default="discovery_16iter_4tests",
        help="Test configuration to analyze (e.g., discovery_16iter_4tests)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    print(f"Loaded {len(data)} problems")

    # Analyze data
    print(
        f"Analyzing relationship between right function identification and success rate for {args.config}..."
    )
    results = analyze_right_function_correlation(data, test_config=args.config)

    # Plot results
    print(f"Generating nested bar chart in {args.output}...")
    summary_df = plot_nested_bar_chart(results, output_dir=args.output)

    print("\nSummary of findings:")
    print(summary_df.to_string(index=False))
    print(f"\nAnalysis complete. Plots saved to {args.output}/")


if __name__ == "__main__":
    main()
