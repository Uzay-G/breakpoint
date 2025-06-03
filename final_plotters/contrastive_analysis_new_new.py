#!/usr/bin/env python3
"""
compare_runs.py — Unified contrastive analysis for two model runs
----------------------------------------------------------------
This refactored script supersedes the original *contrastive_analysis.py* and the
ad‑hoc resource‑comparison utilities.  Pass **exactly two** run specifications
(model name plus, optionally, a resource/setting tag) and the tool will:
  • pull the requested scores from your consolidated JSONL file;
  • build a tidy DataFrame with performance + complexity metrics;
  • generate all plots, statistics and summary CSVs in a single output folder.

Run specification format
========================
    model_name[:setting]

Examples
========
    # Compare two different models (their default settings)
    python compare_runs.py data/consolidated.jsonl gpt‑3.5‑turbo gpt‑4o

    # Compare two resource settings of the same model
    python compare_runs.py data/consolidated.jsonl gpt‑4o:1k gpt‑4o:32k \
        --run1-label "GPT‑4o 1k ctx" --run2-label "GPT‑4o 32k ctx"

    # A full invocation with custom output dir and filters
    python compare_runs.py data/consolidated.jsonl claude‑3‑sonnet claude‑3‑opus:high‑precision \
"""

import argparse
import json
import os
from typing import Dict, Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# SciencePlots style setup
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

# Optional/soft dependencies --------------------------------------------------
try:
    import seaborn as sns  # noqa: F401 — only for boxplots
except ImportError:
    sns = None
try:
    from matplotlib_venn import venn2  # noqa: F401 — for Venn diagram
except ImportError:
    venn2 = None

from scipy.stats import mannwhitneyu  # noqa: F401 — for Mann-Whitney U test

# ----------------------------------------------------------------------------
# ---------------------------  Helper functions  -----------------------------
# ----------------------------------------------------------------------------


def parse_run_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Split a *model[:setting]* spec into its parts."""
    if ":" in spec:
        model, setting = spec.split(":", 1)
        return model.strip(), setting.strip()
    return spec.strip(), None


def load_consolidated_data(file_path: str) -> List[dict]:
    data: List[dict] = []
    with open(file_path, "r", encoding="utf‑8") as fh:
        for line in fh:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip bad lines but keep going
                continue
    print(f"Loaded {len(data):,} function records from {file_path}")
    return data


def extract_results(
    data: List[dict], model_name: str, setting: Optional[str]
) -> Dict[str, float]:
    """Return *best attainable* score for each function under the requested run."""
    results: Dict[str, float] = {}
    for item in data:
        function_id = item.get("function_id")
        perf = item.get("model_performance", {}).get(model_name, {})
        if not perf:
            continue  # model absent for this function

        if setting is None:
            # No setting specified → take *best* score across tests
            score = max(perf.values()) if perf else None
        else:
            score = perf.get(setting)
        if score is not None:
            results[function_id] = score
    print(
        f"{model_name}{':' + setting if setting else ''}: "
        f"scores for {len(results):,} functions"
    )
    return results


def categorise_functions(
    run1: Dict[str, float], run2: Dict[str, float]
) -> Dict[str, List[str]]:
    """Partition problems into *both / run1_only / run2_only / neither*."""
    all_fns = set(run1) | set(run2)
    cat = {"both": [], "run1_only": [], "run2_only": [], "neither": []}
    for fn in all_fns:
        r1 = run1.get(fn, 0.0) == 1.0
        r2 = run2.get(fn, 0.0) == 1.0
        if r1 and r2:
            cat["both"].append(fn)
        elif r1:
            cat["run1_only"].append(fn)
        elif r2:
            cat["run2_only"].append(fn)
        else:
            cat["neither"].append(fn)
    return cat


# ---------------------------------------------------------------------------
# -------------------------  DataFrame assembly  ----------------------------
# ---------------------------------------------------------------------------


def build_dataframe(
    consolidated: List[dict],
    categories: Dict[str, List[str]],
    run1_scores: Dict[str, float],
    run2_scores: Dict[str, float],
) -> pd.DataFrame:
    """Gather complexity metrics & scores for all relevant functions."""
    lookup = {d["function_id"]: d for d in consolidated}
    rows: List[dict] = []
    for fn in set(run1_scores) | set(run2_scores):
        meta = lookup.get(fn)
        if not meta:
            continue
        comp = meta.get("complexity_metrics", {})
        central = comp.get("centrality", {})
        if fn in categories["both"]:
            cat = "Both Runs"
        elif fn in categories["run1_only"]:
            cat = "Run 1 Only"
        elif fn in categories["run2_only"]:
            cat = "Run 2 Only"
        else:
            cat = "Neither Run"
        rows.append(
            {
                "function": fn,
                "category": cat,
                "run1_score": run1_scores.get(fn, 0.0),
                "run2_score": run2_scores.get(fn, 0.0),
                # complexity metrics ------------------------------------
                "line_count": comp.get("line_count", 0),
                "code_line_count": comp.get("code_line_count", 0),
                "cyclomatic": comp.get("cyclomatic", 0),
                "halstead_volume": comp.get("halstead_volume", 0),
                "halstead_difficulty": comp.get("halstead_difficulty", 0),
                "in_degree": central.get("in_degree", 0),
                "out_degree": central.get("out_degree", 0),
                "harmonic": central.get("harmonic", 0),
                "betweenness": central.get("betweenness", 0),
                "pagerank": central.get("pagerank", 0),
                "degree": central.get("degree", 0),
                "repo_fn_cnt": meta.get("stats", {}).get("functions_count", 0),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# ---------------------------  Plotting helpers  -----------------------------
# ----------------------------------------------------------------------------
# (Most of the original visualisation routines are kept verbatim but we moved
#  them into a dedicated module‑like section so they can be imported / tested.)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ❶ Venn diagram -------------------------------------------------------------


def plot_venn(categories: Dict[str, List[str]], labels: Tuple[str, str], out: str):
    if venn2 is None:
        print("matplotlib‑venn not installed — skipping Venn diagram")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    venn2(
        [
            set(categories["run1_only"]) | set(categories["both"]),
            set(categories["run2_only"]) | set(categories["both"]),
        ],
        labels,
    )
    ax.set_title("Function Solution Overlap")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "venn.png"), dpi=300)
    plt.close(fig)


# ❷ Category bar -------------------------------------------------------------


def plot_category_distribution(df: pd.DataFrame, labels: Tuple[str, str], out: str):
    cats = (
        df.assign(
            display=df["category"].replace(
                {"Run 1 Only": f"{labels[0]} Only", "Run 2 Only": f"{labels[1]} Only"}
            )
        )["display"]
        .value_counts()
        .reindex([f"{labels[0]} Only", f"{labels[1]} Only", "Both Runs", "Neither Run"])
        .fillna(0)
    )
    colors = ["#D33682", "#6C71C4", "#2ecc71", "#95a5a6"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cats.index, cats.values, color=colors, edgecolor="white")
    for b in bars:
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.5,
            int(b.get_height()),
            ha="center",
            va="bottom",
            fontsize=30,
        )
    ax.set_ylabel("Number of Functions")
    ax.set_title("Distribution by Solution Category")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "category_distribution.png"), dpi=300)
    plt.close(fig)


# ❸ Scatter: complexity ------------------------------------------------------


def plot_complexity_scatter(
    df: pd.DataFrame, x: str, y: str, labels: Tuple[str, str], out: str
):
    palette = {
        "Both Runs": "#2ecc71",
        f"{labels[0]} Only": "#D33682",
        f"{labels[1]} Only": "#6C71C4",
        # "Neither Run": "#95a5a6",
    }
    df = df.assign(
        display=df["category"].replace(
            {"Run 1 Only": f"{labels[0]} Only", "Run 2 Only": f"{labels[1]} Only"}
        )
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat, col in palette.items():
        sub = df[df["display"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub[x],
            sub[y],
            s=60,
            alpha=0.7,
            c=col,
            edgecolors="w",
            linewidths=0.4,
            label=cat,
        )
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(f"{x.title()} vs {y.title()}")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"scatter_{x}_vs_{y}.png"), dpi=300)
    plt.close(fig)


# ❹ PCA clustering -----------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------  All-Solved complexity comparison & Mann-Whitney tests -----------
# ---------------------------------------------------------------------------


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled‑SD Cohen's *d* for two independent samples"""
    a, b = a.dropna(), b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)


def _add_sig_bracket(
    ax, x1: float, x2: float, y: float, text: str, linewidth: float = 1.0
):
    """Draw a significance bracket between two x‑positions."""
    ax.plot([x1, x1, x2, x2], [y, y + 0.5, y + 0.5, y], lw=linewidth, c="black")
    ax.text((x1 + x2) / 2, y + 0.7, text, ha="center", va="bottom", fontsize=26)


# --------------------------
# Main function
# --------------------------


def compare_all_solved_problems(
    df: pd.DataFrame,
    labels: Tuple[str, str],
    out_dir: str,
) -> pd.DataFrame:
    """Visualise and test differences in *solved* functions across two runs.

    Parameters
    ----------
    df : pd.DataFrame
        Input table with at least columns ``run1_score``/``run2_score`` and the
        metrics ``code_line_count`` & ``out_degree``.
    labels : Tuple[str, str]
        Human‑readable names for the two runs (``run1``, ``run2``).
    out_dir : str
        Where to save the PNG and summary CSV.

    Returns
    -------
    pd.DataFrame
        A tidy table with mean, median, p‑value and Cohen's *d* for each metric.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ----------  Preparation  ----------
    run1_name, run2_name = labels
    solved1 = df[df["run1_score"] == 1.0]
    solved2 = df[df["run2_score"] == 1.0]

    metrics = {
        "code_line_count": "Code Line Count",
        "harmonic": "Harmonic Centrality",
    }

    # color palette - use solarize colors from scienceplots
    colors = ["#D33682", "#6C71C4"]  # Solarize magenta and violet

    # Set up SciencePlots style
    try:
        import scienceplots

        plt.style.use(["science", "grid"])
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "font.family": "serif",
                "font.size": 26,
                "axes.labelsize": 34,
                "axes.titlesize": 36,
                "xtick.labelsize": 30,
                "ytick.labelsize": 30,
                "legend.fontsize": 30,
                "axes.grid": True,
                "grid.linestyle": "--",
                "grid.alpha": 0.7,
            }
        )
    except ImportError:
        print("SciencePlots not available - using default style")

    # ----------  Figure  ----------
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5.5 * len(metrics), 4.5), squeeze=False
    )

    summary_rows = []

    # Generate a caption based on what differs between run1 and run2
    model1, setting1 = (
        parse_run_spec(run1_name) if ":" in run1_name else (run1_name, None)
    )
    model2, setting2 = (
        parse_run_spec(run2_name) if ":" in run2_name else (run2_name, None)
    )

    # Extract iteration counts if present in settings
    iter1 = (
        int(setting1.split("_")[1].replace("iter", ""))
        if setting1 and "iter" in setting1
        else None
    )
    iter2 = (
        int(setting2.split("_")[1].replace("iter", ""))
        if setting2 and "iter" in setting2
        else None
    )

    # Create figure caption based on what differs
    if model1 != model2:
        figure_caption = f"Comparison between {model1} and {model2}"
    elif iter1 is not None and iter2 is not None:
        figure_caption = f"Comparison between {iter1} iterations and {iter2} iterations"
    else:
        figure_caption = f"Comparison between {run1_name} and {run2_name}"

    # Add figure caption
    fig.suptitle(figure_caption, fontsize=36, y=0.98)

    for i, (metric_key, metric_label) in enumerate(metrics.items()):
        ax = axes[0, i]

        # Build tidy DF
        plot_df = pd.concat(
            [
                solved1[[metric_key]].assign(Model=run1_name),
                solved2[[metric_key]].assign(Model=run2_name),
            ]
        ).rename(columns={metric_key: "Value"})

        # Use hue for proper coloring in seaborn
        # Violin plots
        sns.violinplot(
            x="Model",
            y="Value",
            hue="Model",  # Set hue parameter to fix deprecation warning
            data=plot_df,
            ax=ax,
            inner=None,
            linewidth=0,
            palette={run1_name: colors[0], run2_name: colors[1]},
            alpha=0.25,
            legend=False,  # Hide legend since we have x-labels
        )

        # Box plots (median / IQR)
        sns.boxplot(
            x="Model",
            y="Value",
            hue="Model",  # Set hue parameter to fix deprecation warning
            data=plot_df,
            ax=ax,
            width=0.25,
            showcaps=True,
            boxprops=dict(alpha=1, linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            medianprops=dict(linewidth=1.5, color="black"),
            showfliers=True,
            flierprops=dict(
                markeredgecolor="grey", markerfacecolor="none", markersize=3, alpha=0.4
            ),
            palette={run1_name: colors[0], run2_name: colors[1]},
            saturation=1,
            legend=False,  # Hide legend since we have x-labels
        )

        # Fix x-tick labels properly
        ax.set_xticks(range(2))
        ax.set_xticklabels([run1_name, run2_name], fontsize=30)

        # Titles & axes with science style
        ax.set_title(metric_label, fontsize=32, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel(metric_label, fontsize=34)

        # Improve grid appearance
        ax.grid(True, linestyle="--", alpha=0.7)

        # ----------  Statistics  ----------
        a = solved1[metric_key].dropna()
        b = solved2[metric_key].dropna()
        stat, p_val = mannwhitneyu(a, b, alternative="two-sided")
        d = _cohen_d(a, b)

        # Significance annotation
        y_max = plot_df["Value"].max()
        star = "*" if p_val < 0.05 else "ns"
        # _add_sig_bracket(ax, 0, 1, y_max * 1.05, f"{star}  p={p_val:.3f}\n d={d:.2f}")
        ax.set_ylim(top=y_max * 1.14)

        summary_rows.append(
            {
                "Metric": metric_label,
                f"{run1_name} Mean": round(a.mean(), 2),
                f"{run1_name} Median": round(a.median(), 2),
                f"{run2_name} Mean": round(b.mean(), 2),
                f"{run2_name} Median": round(b.median(), 2),
                "p-value": round(p_val, 4),
                "Significant (<0.05)": p_val < 0.05,
                "Cohen d": round(d, 2),
            }
        )

    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "all_solved_complexity_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ----------  Summary table  ----------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(out_dir, "all_solved_complexity_summary.csv"), index=False
    )
    return summary_df


def plot_pca(df: pd.DataFrame, labels: Tuple[str, str], out: str):
    numeric = [
        "code_line_count",
        "cyclomatic",
        "halstead_volume",
        "halstead_difficulty",
        "in_degree",
        "out_degree",
        "betweenness",
        "pagerank",
        "degree",
    ]
    df_clean = df.dropna(subset=numeric)
    if len(df_clean) < 10:
        print("Not enough data for PCA plot – skipping")
        return
    scaled = StandardScaler().fit_transform(df_clean[numeric])
    p = PCA(n_components=2).fit_transform(scaled)
    df_clean = df_clean.assign(
        pca1=p[:, 0],
        pca2=p[:, 1],
        display=df_clean["category"].replace(
            {"Run 1 Only": f"{labels[0]} Only", "Run 2 Only": f"{labels[1]} Only"}
        ),
    )
    palette = {
        "Both Runs": "#2ecc71",
        f"{labels[0]} Only": "#D33682",
        f"{labels[1]} Only": "#6C71C4",
        "Neither Run": "#95a5a6",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for cat, col in palette.items():
        sub = df_clean[df_clean["display"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["pca1"],
            sub["pca2"],
            c=col,
            s=70,
            alpha=0.7,
            edgecolors="w",
            linewidths=0.3,
            label=cat,
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Complexity Metrics")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pca.png"), dpi=300)
    plt.close(fig)


# ❺ Score comparison ---------------------------------------------------------


def plot_score_comparison(df: pd.DataFrame, labels: Tuple[str, str], out: str):
    cmap = LinearSegmentedColormap.from_list(
        "difficulty", ["#2ecc71", "#f39c12", "#e74c3c"]
    )
    metrics = ["code_line_count", "cyclomatic", "halstead_volume", "out_degree"]
    norm = df.copy()
    for m in metrics:
        rng = norm[m].max() - norm[m].min()
        norm[m] = (norm[m] - norm[m].min()) / rng if rng else 0
    norm["difficulty"] = norm[metrics].mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        norm["run1_score"],
        norm["run2_score"],
        c=norm["difficulty"],
        cmap=cmap,
        s=70,
        alpha=0.75,
        edgecolors="w",
        linewidths=0.4,
    )
    fig.colorbar(sc, label="Function Complexity (norm)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.fill_between([0, 1], [0, 0], [1, 1], color="#D33682", alpha=0.1)
    ax.fill_between([0, 1], [1, 1], [0, 0], color="#6C71C4", alpha=0.1)
    ax.text(0.75, 0.25, f"{labels[0]} better", color="#D33682", ha="center")
    ax.text(0.25, 0.75, f"{labels[1]} better", color="#6C71C4", ha="center")
    ax.set_xlabel(f"{labels[0]} Score")
    ax.set_ylabel(f"{labels[1]} Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "score_comparison.png"), dpi=300)
    plt.close(fig)


# (Other visualisations — boxplots, histograms, statistical tests — omitted for
# brevity but can be ported verbatim from the original script if still needed.)


# ----------------------------------------------------------------------------
# -------------------------------  main()  -----------------------------------
# ----------------------------------------------------------------------------


def run_analysis(
    consolidated_file,
    run1_spec,
    run2_spec,
    output_dir=None,
    run1_label=None,
    run2_label=None,
    exclude_neither=False,
):
    """Run a complete analysis for a pair of model/resource runs.

    Parameters
    ----------
    consolidated_file : str
        Path to the consolidated JSONL file with performance data
    run1_spec : str
        Run #1 specification: model[:setting]
    run2_spec : str
        Run #2 specification: model[:setting]
    output_dir : str, optional
        Directory to save outputs (will be auto-generated if None)
    run1_label : str, optional
        Custom display label for run 1
    run2_label : str, optional
        Custom display label for run 2
    exclude_neither : bool
        Whether to exclude problems solved by neither run

    Returns
    -------
    pd.DataFrame
        Summary of the analysis results
    """
    run1_model, run1_setting = parse_run_spec(run1_spec)
    run2_model, run2_setting = parse_run_spec(run2_spec)

    label1 = (
        run1_label
        or f"{run1_model}{'' if run1_setting is None else f' ({run1_setting})'}"
    )
    label2 = (
        run2_label
        or f"{run2_model}{'' if run2_setting is None else f' ({run2_setting})'}"
    )

    output_dir = f"compare_runs_{run1_model}_{run2_model}"

    out_dir = ensure_dir(output_dir)

    data = load_consolidated_data(consolidated_file)
    run1_scores = extract_results(data, run1_model, run1_setting)
    run2_scores = extract_results(data, run2_model, run2_setting)

    categories = categorise_functions(run1_scores, run2_scores)
    print("\nCategory counts:")
    for k, v in categories.items():
        print(f"  {k.replace('_', ' ').title()}: {len(v):,}")

    df = build_dataframe(data, categories, run1_scores, run2_scores)
    if exclude_neither:
        before = len(df)
        df = df[df["category"] != "Neither Run"]
        print(f"Excluded {before - len(df):,} 'neither' rows → {len(df):,} remain")

    # Visualisations ---------------------------------------------------------
    print("Generating plots …")
    plot_venn(categories, (label1, label2), out_dir)
    plot_category_distribution(df, (label1, label2), out_dir)
    plot_complexity_scatter(
        df, "out_degree", "code_line_count", (label1, label2), out_dir
    )
    plot_complexity_scatter(
        df, "cyclomatic", "halstead_volume", (label1, label2), out_dir
    )
    plot_pca(df, (label1, label2), out_dir)
    plot_score_comparison(df, (label1, label2), out_dir)
    summary_df = compare_all_solved_problems(df, (label1, label2), out_dir)
    print("\nAll-solved complexity summary:")
    print(summary_df.to_string(index=False))

    print(f"Done! All artefacts saved to '{out_dir}'")
    return summary_df


def main():
    p = argparse.ArgumentParser(
        description="Contrastive analysis for any two model/resource runs"
    )
    p.add_argument(
        "consolidated_file", nargs="?", help="Path to consolidated JSONL file"
    )
    p.add_argument("run1", nargs="?", help="Run #1 specification: model[:setting]")
    p.add_argument("run2", nargs="?", help="Run #2 specification: model[:setting]")
    p.add_argument("--run1-label", help="Custom display label for run 1")
    p.add_argument("--run2-label", help="Custom display label for run 2")
    p.add_argument(
        "--exclude-neither",
        action="store_true",
        help="Drop problems solved by neither run",
    )
    p.add_argument(
        "--run-all", action="store_true", help="Run both default comparison scenarios"
    )
    args = p.parse_args()

    # If no specific runs provided or --run-all flag is set, run the default scenarios
    if args.consolidated_file and args.run1 and args.run2:
        run_analysis(
            consolidated_file=args.consolidated_file,
            run1_spec=args.run1,
            run2_spec=args.run2,
            run1_label=args.run1_label,
            run2_label=args.run2_label,
            exclude_neither=args.exclude_neither,
        )
    else:
        print(
            "Error: Please provide all three required arguments (consolidated_file, run1, run2) or use --run-all flag."
        )
        p.print_help()
        return 1

    return 0


if __name__ == "__main__":
    main()
