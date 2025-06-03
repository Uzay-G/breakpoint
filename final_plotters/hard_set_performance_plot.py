#!/usr/bin/env python3
"""
Hard Set Performance Plot
========================

This script generates a bar chart showing model performance on the hardest problems,
defined as those above the 85th percentile for both harmonic centrality and
log-transformed code line count.

Usage:
    python hard_set_performance_plot.py

The script analyzes consolidated data from:
/home/ubuntu/breakpoint/consolidated_results/consolidated_results_remove_all_normalized.jsonl
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import math

# Configure paths
BASE_DIR = Path("/home/ubuntu/breakpoint")
CONSOLIDATED_DATA = (
    BASE_DIR / "consolidated_results/consolidated_results_remove_all_normalized.jsonl"
)
OUTPUT_DIR = BASE_DIR / "final_plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure analysis settings
PERCENTILE = 90  # Using 85th percentile for both metrics
MODELS = ["o4-mini", "gpt-4o", "o3-mini", "claude-3-7-sonnet-latest"]
MODEL_KEY = "remove_16iter_4tests"

# Model display names for better readability
MODEL_DISPLAY = {
    "o4-mini": "o4-mini",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
    "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet",
}

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


def load_data(file_path):
    """Load and prepare data from JSONL file."""
    # Enable perfect solve mode (binary scores)
    PERFECT_SOLVE_MODE = True

    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Extract relevant metrics into a DataFrame
    rows = []
    for item in data:
        complexity = item.get("complexity_metrics", {})
        centrality = complexity.get("centrality", {}) if complexity else {}
        model_perf = item.get("model_performance", {})

        # Basic function info
        row = {
            "function_id": item.get("function_id", ""),
        }

        # Extract complexity metric (log-transformed code line count)
        if "code_line_count" in complexity:
            code_line_count = complexity.get("code_line_count", 0)
            row["log_code_line_count"] = (
                math.log(code_line_count + 1) if code_line_count > 0 else 0
            )
        else:
            row["log_code_line_count"] = np.nan

        # Extract harmonic centrality
        row["harmonic"] = centrality.get("harmonic", np.nan) if centrality else np.nan

        # Extract model performance with perfect solve mode (binary scores)
        for model in MODELS:
            if model in model_perf:
                model_metrics = model_perf[model]
                score = model_metrics.get(MODEL_KEY, np.nan)
                # Apply perfect solve mode - convert to binary (1.0 or 0.0)
                if PERFECT_SOLVE_MODE and not np.isnan(score):
                    score = 1.0 if score == 1.0 else 0.0
                row[f"{model}_{MODEL_KEY}"] = score

        # Add average model performance
        model_scores = [row.get(f"{model}_{MODEL_KEY}", np.nan) for model in MODELS]
        valid_scores = [score for score in model_scores if not np.isnan(score)]
        row["avg_model_score"] = np.mean(valid_scores) if valid_scores else np.nan
        row["models_solved_count"] = sum(1 for score in valid_scores if score == 1.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def identify_hard_set(df, percentile=85):
    """
    Identify the hard set as functions above the specified percentile
    for both log_code_line_count and harmonic centrality.
    """
    # Calculate threshold values
    log_line_count_threshold = np.percentile(
        df["log_code_line_count"].dropna(), percentile
    )
    harmonic_threshold = np.percentile(df["harmonic"].dropna(), percentile)

    # Identify hard set (both metrics above threshold)
    hard_set = df[
        (df["log_code_line_count"] >= log_line_count_threshold)
        & (df["harmonic"] >= harmonic_threshold)
    ].copy()

    print(
        f"Log Code Line Count threshold ({percentile}th percentile): {log_line_count_threshold:.2f}"
    )
    print(
        f"Harmonic Centrality threshold ({percentile}th percentile): {harmonic_threshold:.2f}"
    )
    print(
        f"Hard set size: {len(hard_set)} functions ({len(hard_set)/len(df)*100:.2f}% of total)"
    )

    return hard_set, log_line_count_threshold, harmonic_threshold


def plot_model_performance(hard_set, output_dir, percentile):
    """
    Create a bar chart showing model performance on the hard set.
    """
    # Calculate performance for each model
    model_performance = []

    for model in MODELS:
        model_key = f"{model}_{MODEL_KEY}"
        if model_key in hard_set.columns:
            # Calculate mean score (perfect solve rate) and count valid evaluations
            valid_scores = hard_set[model_key].dropna()
            mean_score = valid_scores.mean() if len(valid_scores) > 0 else 0
            count = len(valid_scores)

            # Calculate standard error
            std_err = np.std(valid_scores, ddof=1) / np.sqrt(count) if count > 1 else 0

            # Calculate 95% confidence interval
            ci = 1.96 * std_err if count > 1 else 0

            model_performance.append(
                {
                    "model": model,
                    "display_name": MODEL_DISPLAY.get(model, model),
                    "mean_score": mean_score,
                    "count": count,
                    "std_err": std_err,
                    "ci": ci,
                }
            )

    # Convert to DataFrame and sort by performance
    perf_df = pd.DataFrame(model_performance)
    perf_df = perf_df.sort_values("mean_score", ascending=False)

    # Create the bar chart
    plt.figure(figsize=(10, 6))

    # Define bar colors using solarize palette
    colors = [
        "#268bd2",
        "#d33682",
        "#859900",
        "#b58900",
    ]  # blue, magenta, green, yellow

    # Create bars (without error bars)
    bars = plt.bar(
        perf_df["display_name"],
        perf_df["mean_score"],
        color=colors[: len(perf_df)],
        edgecolor="black",
        linewidth=1.5,
    )

    # Add bar values
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{perf_df['mean_score'].iloc[i]:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=24,
        )

    # Sample size annotations removed

    # Add labels and title
    plt.xlabel("Model")
    plt.ylabel("Perfect Solve Rate")
    plt.title(
        f"Model Performance on Hard Functions \n ({percentile}th percentile for both log line count and harmonic centrality)"
    )

    # Set y-axis limits
    plt.ylim(0, min(1.15, perf_df["mean_score"].max() * 1.2))

    # Add grid
    plt.grid(axis="y", alpha=0.3)

    # Tight layout and save
    plt.tight_layout()

    # Save figure in multiple formats
    output_path_base = output_dir / f"hard_set_performance_{percentile}percentile"
    plt.savefig(f"{output_path_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_path_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{output_path_base}.svg", bbox_inches="tight")

    print(f"Created performance plot: {output_path_base}.png")

    # Save results to CSV
    results_df = pd.DataFrame(
        {
            "Model": perf_df["display_name"],
            "Perfect Solve Rate": perf_df["mean_score"],
            "Standard Error": perf_df["std_err"],
            "Confidence Interval": perf_df["ci"],
            "Sample Size": perf_df["count"],
        }
    )

    results_df.to_csv(f"{output_path_base}_results.csv", index=False)
    print(f"Saved results to: {output_path_base}_results.csv")

    return results_df


def create_scatter_visualization(
    df, hard_set, log_line_count_threshold, harmonic_threshold, output_dir, percentile
):
    """
    Create a scatter plot showing all functions with the hard set highlighted.
    """
    plt.figure(figsize=(12, 10))

    # Plot all functions in gray
    plt.scatter(
        df["log_code_line_count"],
        df["harmonic"],
        c="lightgray",
        label="All Functions",
        alpha=0.3,
        edgecolors="none",
        s=50,
    )

    # Plot hard set in red
    plt.scatter(
        hard_set["log_code_line_count"],
        hard_set["harmonic"],
        c="#dc322f",  # solarize red
        label=f"Hard Set (n={len(hard_set)})",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        s=70,
    )

    # Add threshold lines
    plt.axvline(
        x=log_line_count_threshold, color="black", linestyle="--", linewidth=1.5
    )
    plt.axhline(y=harmonic_threshold, color="black", linestyle="--", linewidth=1.5)

    # Add threshold labels
    plt.text(
        log_line_count_threshold,
        df["harmonic"].min(),
        f"log(line count) = {log_line_count_threshold:.2f}\n({percentile}th percentile)",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=22,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )

    plt.text(
        df["log_code_line_count"].min(),
        harmonic_threshold,
        f"harmonic = {harmonic_threshold:.2f}\n({percentile}th percentile)",
        ha="left",
        va="center",
        fontsize=22,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
    )

    # Set axis labels and title
    plt.xlabel("Log Code Line Count")
    plt.ylabel("Harmonic Centrality")
    plt.title(
        f"Hard Set: Functions Above {percentile}th Percentile\nfor both Log Code Line Count and Harmonic Centrality"
    )

    # Add legend
    plt.legend(loc="upper right")

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"hard_set_distribution_{percentile}percentile.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Created distribution plot: {output_path}")


def main():
    """Main function to execute the analysis."""
    print(f"Loading data from {CONSOLIDATED_DATA}")
    df = load_data(CONSOLIDATED_DATA)
    print(f"Loaded {len(df)} functions")

    # Identify the hard set
    print(f"\nIdentifying hard set at {PERCENTILE}th percentile...")
    hard_set, log_line_count_threshold, harmonic_threshold = identify_hard_set(
        df, PERCENTILE
    )

    # Create visualization of the hard set
    print("\nCreating distribution visualization...")
    create_scatter_visualization(
        df,
        hard_set,
        log_line_count_threshold,
        harmonic_threshold,
        OUTPUT_DIR,
        PERCENTILE,
    )

    # Plot model performance on the hard set
    print("\nPlotting model performance...")
    results = plot_model_performance(hard_set, OUTPUT_DIR, PERCENTILE)

    # Print results
    print("\nModel performance on hard set:")
    print(results.to_string(index=False))

    print(f"\nAnalysis complete. All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
