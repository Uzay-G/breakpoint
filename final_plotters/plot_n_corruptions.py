#!/usr/bin/env python3
# python3 final_plotters/plot_n_corruptions.py results/o3-mini-32-n-corruptions-1.jsonl results/o3-mini-32-n-corruptions-2.jsonl results/o3-mini-32-n-corruption-4.jsonl
"""
Plot recovery-performance curves for three experiment logs.

For each file we read the *first* JSON object and expect at least
    success_rate          (float, 0–1 or 0–100)
    perfect_solve_rate    (float, 0–1 or 0–100)

We also need an integer 'n' = number of simultaneous corruptions.
The script tries, in order:
  1. record['n']
  2. record['num_corruptions']
  3. len(record['corruption_functions'])      (if present)
  4. last integer in the filename

Runs are plotted in ascending order of n.

Output:  final_plots/recovery_performance.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401 – needed for plt.style.use

# ——————————————————————  styling  ——————————————————————
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

TEST_RECOVERY_COLOR = "#D33682"  # Solarized magenta
SOLVE_RATE_COLOR = "#6C71C4"  # Solarized blue

LAST_INT_RE = re.compile(r"(\d+)(?!.*\d)")


# ——————————————————  helpers  ——————————————————
def percentage(val):
    """Convert 0-1 floats to 0-100, leave ≥1 untouched."""
    return val * 100 if isinstance(val, (int, float)) and 0 <= val <= 1 else val


def infer_n(path: Path, rec: dict) -> int:
    """Best-effort guess of the # of simultaneous corruptions."""
    for key in ("n", "num_corruptions"):
        if key in rec and isinstance(rec[key], int):
            return rec[key]

    if "corruption_functions" in rec and isinstance(rec["corruption_functions"], list):
        return len(rec["corruption_functions"])

    m = LAST_INT_RE.search(path.stem)
    if m:
        return int(m.group(1))

    raise ValueError(f"Could not infer 'n' for {path}")


def extract_metrics(path: Path) -> dict:
    """
    Return a dict with keys:
        n                   – inferred number of simultaneous corruptions
        success_rate        – percentage of rows whose score == 1
        perfect_solve_rate  – percentage of rows that are both
                              score == 1  AND  metadata.right_function == True
    """
    with path.open(encoding="utf-8") as f:
        # ---------- read the header line ----------
        header_line = f.readline().strip()
        if not header_line:
            raise ValueError(f"{path} is empty")

        header = json.loads(header_line)

        # try to discover the “n” value as before
        n_val = infer_n(path, header)

        # total problems (if absent we'll count them ourselves)
        total_problems = header.get("total_problems")

        # ---------- scan the per-problem rows ----------
        success_count = 0
        perfect_solves = 0
        problem_rows = 0
        found = 0

        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            problem_rows += 1

            if obj.get("metadata", {}).get("right_function") is True:
                found += 1
            if obj.get("score") == 1:
                success_count += 1
                if obj.get("metadata", {}).get("right_function") is True:
                    perfect_solves += 1

        # in case the header lacked total_problems
        if total_problems is None:
            total_problems = problem_rows or 1  # avoid divide-by-zero

    # convert to percentages
    success_rate_pct = (success_count / total_problems) * 100
    perfect_rate_pct = (perfect_solves / total_problems) * 100

    return {
        "n": n_val,
        "test_recovery": success_rate_pct,
        "perfect_solve_rate": perfect_rate_pct,
        "discovery_rate": found / total_problems * 100,
    }


# ————————————————————  main  ————————————————————
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot recovery & solve rates vs. simultaneous corruptions"
    )
    ap.add_argument("files", nargs=3, type=Path, help="three jsonl result files")
    args = ap.parse_args()

    records = [extract_metrics(p) for p in args.files]
    df = pd.DataFrame(records).sort_values("n")

    # ——— plot ———
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        df["n"],
        df["perfect_solve_rate"],
        marker="o",
        markersize=5,
        label="Solve Rate",
        color=SOLVE_RATE_COLOR,
    )

    ax.plot(
        df["n"],
        df["discovery_rate"],
        marker="o",
        markersize=5,
        label="Discovery Rate",
        color=TEST_RECOVERY_COLOR,
    )
    # labels & cosmetics
    ax.set_xlabel("Number of Simultaneous Corruptions")
    ax.set_ylabel("Percentage")
    ax.set_title("Solve Rate vs. Number of Simultaneous Corruptions")
    ax.legend()

    ax.set_xticks(df["n"])
    ymax = max(df["test_recovery"].max(), df["perfect_solve_rate"].max())
    ax.set_ylim(0, min(100, ymax * 1.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    # save + show
    Path("final_plots").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("final_plots/recovery_performance.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
