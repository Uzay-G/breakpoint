#!/usr/bin/env python3
"""
Main Figure Generator
====================

This script generates the main comparison figure showing model performance
for both discovery and remove modes at 16 iterations with 4 test budget.

Features:
- Perfect Solve Mode (binary scores where 1.0 = solved, 0.0 = not solved)
- Standard error bars assuming IID problems 
- Side-by-side comparison of discovery and remove modes
- SciencePlots style for publication-quality visualization

Usage:
    python main_figure.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from collections import defaultdict

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
        }
    )

# Configure paths
BASE_DIR = Path("/home/ubuntu/breakpoint")
CONSOLIDATED_RESULTS_DIR = BASE_DIR / "consolidated_results"
REMOVE_RESULTS = (
    CONSOLIDATED_RESULTS_DIR
    / "consolidated_results_remove_all_with_tools_filters.jsonl"
)
DISCOVERY_RESULTS = (
    CONSOLIDATED_RESULTS_DIR
    / "consolidated_results_discovery_all_with_tools_filters_recompute.jsonl"
)
# REMOVE_RESULTS = CONSOLIDATED_RESULTS_DIR / "consolidated_results_remove_all.jsonl"
# DISCOVERY_RESULTS = CONSOLIDATED_RESULTS_DIR / "consolidated_results_discovery_all.jsonl"
OUTPUT_DIR = BASE_DIR / "final_plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure analysis settings
PERFECT_SOLVE_MODE = True  # Use binary scores (1.0 = solved, 0.0 = not solved)
MODELS = ["o4-mini", "gpt-4o", "o3-mini", "claude-3-7-sonnet-latest"]
ITER_COUNT = 16
TEST_BUDGET = 4

# Model display names for better readability
MODEL_DISPLAY = {
    "o4-mini": "o4-mini",
    "gpt-4o": "GPT-4o",
    "o3-mini": "o3-mini",
    "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet",
}


def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_performance_metrics(data):
    """
    Extract performance metrics from the consolidated data.
    Returns a DataFrame with columns: function_id, model, config, score
    """
    rows = []

    for item in data:
        function_id = item.get("function_id", "")
        model_performance = item.get("model_performance", {})

        if not function_id or not model_performance:
            continue

        for model, configs in model_performance.items():
            for config, score in configs.items():
                # Parse the config string to extract iterations and test budget
                # Format: "mode_Niter_Mtests"
                config_parts = config.split("_")
                if len(config_parts) >= 3:
                    mode = config_parts[0]
                    iterations = int(config_parts[1].replace("iter", ""))
                    tests = int(config_parts[2].replace("tests", ""))

                    # Add complexity metrics if available
                    complexity_metrics = item.get("complexity_metrics", {})

                    row = {
                        "function_id": function_id,
                        "model": model,
                        "config": config,
                        "mode": mode,
                        "iterations": iterations,
                        "tests": tests,
                        "score": score,
                    }

                    # Add complexity metrics
                    for metric, value in complexity_metrics.items():
                        if isinstance(value, dict):
                            # Handle nested metrics (like centrality)
                            for sub_metric, sub_value in value.items():
                                row[f"complexity_{metric}_{sub_metric}"] = sub_value
                        else:
                            row[f"complexity_{metric}"] = value

                    rows.append(row)

    return pd.DataFrame(rows)


def estimate_problem_difficulty(df):
    """
    Estimate problem difficulty based on how many models solve each problem.
    This helps in grouping similar difficulty problems for better error estimation.

    Returns the DataFrame with added difficulty columns.
    """
    # Group by function_id and calculate how many models solved each problem
    function_stats = (
        df.groupby("function_id")["score"]
        .agg(
            mean_score=np.mean,
            solved_count=lambda x: sum(x == 1.0),
            total_evals=lambda x: len(x),
        )
        .reset_index()
    )

    # Calculate solve rate for each function
    function_stats["solve_rate"] = (
        function_stats["solved_count"] / function_stats["total_evals"]
    )

    # Create difficulty clusters using K-means
    # Use solve_rate as the feature for clustering
    X = function_stats[["solve_rate"]].values

    # Choose number of clusters (5 is a reasonable default)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    function_stats["difficulty_cluster"] = kmeans.fit_predict(X)

    # Sort clusters by mean solve rate to get difficulty levels
    cluster_difficulty = (
        function_stats.groupby("difficulty_cluster")["solve_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    difficulty_mapping = {
        cluster: i for i, cluster in enumerate(cluster_difficulty.index)
    }
    function_stats["difficulty_level"] = function_stats["difficulty_cluster"].map(
        difficulty_mapping
    )

    # Create labels for difficulty levels
    difficulty_labels = ["Very Easy", "Easy", "Medium", "Hard", "Very Hard"]
    function_stats["difficulty"] = function_stats["difficulty_level"].apply(
        lambda x: difficulty_labels[x] if x < len(difficulty_labels) else f"Level {x}"
    )

    # Merge difficulty information back to the original DataFrame
    difficulty_cols = ["function_id", "difficulty_level", "difficulty", "solve_rate"]
    return df.merge(function_stats[difficulty_cols], on="function_id", how="left")


def calculate_iid_errors(df, groupby_cols, value_col="score"):
    """
    Calculate standard error bars assuming problems are IID.

    This is a simpler approach that assumes all problems are independent and
    identically distributed (IID).

    Returns a DataFrame with mean, std, se, and confidence intervals.
    """
    result = []

    # Group by specified columns
    for group_name, group_df in df.groupby(groupby_cols):
        # Extract as tuple if needed
        if not isinstance(group_name, tuple):
            group_name = (group_name,)

        if len(group_df) < 2:
            # Skip if too few samples
            continue

        # Standard approach: calculate mean and standard error
        mean = group_df[value_col].mean()
        std = group_df[value_col].std()
        se = std / np.sqrt(len(group_df))
        ci = 1.96 * se  # 95% confidence interval

        result.append(
            {
                **dict(zip(groupby_cols, group_name)),
                "mean": mean,
                "std": std,
                "se": se,
                "ci": ci,
                "count": len(group_df),
            }
        )

    return pd.DataFrame(result)


def create_model_comparison_plots(remove_df, discovery_df):
    """
    Create side-by-side comparison plots showing model performance for both
    remove and discovery modes.
    """
    # We're using simple IID error bars, so no need for difficulty estimation
    # Just proceed with the original data

    # Filter for the specified iterations and test budget
    remove_fair = remove_df[
        (remove_df["iterations"] == ITER_COUNT) & (remove_df["tests"] == TEST_BUDGET)
    ]
    discovery_fair = discovery_df[
        (discovery_df["iterations"] == ITER_COUNT)
        & (discovery_df["tests"] == TEST_BUDGET)
    ]

    # Calculate performance statistics with IID error bars (simpler approach)
    remove_stats = calculate_iid_errors(remove_fair, ["model"])
    discovery_stats = calculate_iid_errors(discovery_fair, ["model"])

    # Sort by mean score
    remove_stats = remove_stats.sort_values("mean", ascending=False)
    discovery_stats = discovery_stats.sort_values("mean", ascending=False)

    # Create the main comparison plot with larger font sizes
    plt.figure(figsize=(16, 10))
    
    # Set default font sizes
    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 36,
        'axes.labelsize': 32,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'legend.fontsize': 22,
    })

    # Set up the data for side-by-side bars
    models_remove = [MODEL_DISPLAY.get(m, m) for m in remove_stats["model"]]
    models_discovery = [MODEL_DISPLAY.get(m, m) for m in discovery_stats["model"]]

    # Get all unique models in display format
    all_models = sorted(set(models_remove + models_discovery))
    model_indices = np.arange(len(all_models))

    # Bar width and positioning
    bar_width = 0.35
    remove_offset = -bar_width / 2
    discovery_offset = bar_width / 2

    # Create bars for remove mode
    remove_bars = plt.bar(
        [model_indices[all_models.index(m)] + remove_offset for m in models_remove],
        remove_stats["mean"],
        width=bar_width,
        label="Remove Mode",
        color="#268bd2",  # Solarize blue
        edgecolor="black",
        linewidth=2.5,
        yerr=remove_stats["ci"],
        capsize=10,
        error_kw={"elinewidth": 3, "capthick": 3},
    )

    # Create bars for discovery mode
    discovery_bars = plt.bar(
        [
            model_indices[all_models.index(m)] + discovery_offset
            for m in models_discovery
        ],
        discovery_stats["mean"],
        width=bar_width,
        label="Discovery Mode",
        color="#d33682",  # Solarize magenta
        edgecolor="black",
        linewidth=2.5,
        yerr=discovery_stats["ci"],
        capsize=10,
        error_kw={"elinewidth": 3, "capthick": 3},
    )

    # Add bar values
    for i, bar in enumerate(remove_bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{remove_stats['mean'].iloc[i]:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=24,
        )

    for i, bar in enumerate(discovery_bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{discovery_stats['mean'].iloc[i]:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=24,
        )

    # Add sample size annotations
    for i, model in enumerate(models_remove):
        idx = all_models.index(model)
        plt.text(
            idx + remove_offset,
            0.05,
            f"n={remove_stats['count'].iloc[i]}",
            ha="center",
            va="bottom",
            fontsize=20,
            color="#268bd2",  # Solarize blue
        )

    for i, model in enumerate(models_discovery):
        idx = all_models.index(model)
        plt.text(
            idx + discovery_offset,
            0.02,
            f"n={discovery_stats['count'].iloc[i]}",
            ha="center",
            va="bottom",
            fontsize=20,
            color="#d33682",  # Solarize magenta
        )

    # Add labels, title, and legend
    plt.xlabel("Model", fontsize=32)
    plt.ylabel("Success Rate (Perfect Solve Mode)", fontsize=32)
    plt.title(
        f"Model Performance Comparison ({ITER_COUNT} iterations, {TEST_BUDGET} test budget)",
        fontsize=36,
    )
    plt.xticks(model_indices, all_models, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0, 1.15)  # Leave room for bar values
    plt.legend(fontsize=22)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_clean.png", dpi=300, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "model_comparison_clean.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "model_comparison_clean.svg", bbox_inches="tight")

    # Save results to CSV for reference
    results = []

    for i, row in remove_stats.iterrows():
        model_name = MODEL_DISPLAY.get(row["model"], row["model"])
        results.append(
            {
                "Model": model_name,
                "Mode": "Remove",
                "Success Rate": row["mean"],
                "SE": row["se"],
                "CI": row["ci"],
                "Sample Size": row["count"],
            }
        )

    for i, row in discovery_stats.iterrows():
        model_name = MODEL_DISPLAY.get(row["model"], row["model"])
        results.append(
            {
                "Model": model_name,
                "Mode": "Discovery",
                "Success Rate": row["mean"],
                "SE": row["se"],
                "CI": row["ci"],
                "Sample Size": row["count"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison_clean_results.csv", index=False)

    print(
        f"Created performance comparison plot in: {OUTPUT_DIR / 'model_comparison_clean.png'}"
    )
    print(f"Results saved to: {OUTPUT_DIR / 'model_comparison_clean_results.csv'}")

    return results_df


def main():
    print("Loading data...")
    remove_data = load_jsonl(REMOVE_RESULTS)
    discovery_data = load_jsonl(DISCOVERY_RESULTS)

    print(f"Loaded {len(remove_data)} rows from remove results")
    print(f"Loaded {len(discovery_data)} rows from discovery results")

    # Extract performance metrics
    print("Extracting performance metrics...")
    remove_df = extract_performance_metrics(remove_data)
    discovery_df = extract_performance_metrics(discovery_data)

    # Apply perfect solve mode
    if PERFECT_SOLVE_MODE:
        print("Converting scores to binary (Perfect Solve Mode)")
        remove_df["score"] = (remove_df["score"] == 1.0).astype(float)
        discovery_df["score"] = (discovery_df["score"] == 1.0).astype(float)

    print(f"Extracted {len(remove_df)} performance data points from remove results")
    print(
        f"Extracted {len(discovery_df)} performance data points from discovery results"
    )

    # Create model comparison plots
    print("Creating model comparison plots...")
    results = create_model_comparison_plots(remove_df, discovery_df)

    print("Analysis complete.")
    print(results)


if __name__ == "__main__":
    main()
