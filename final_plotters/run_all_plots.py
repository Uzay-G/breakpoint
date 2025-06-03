#!/usr/bin/env python3
"""
Run All Plots Script
===================

This script runs all the plotting scripts in the final_plotters directory
with the appropriate data files based on the CLAUDE.md instructions and
analysis of the existing scripts.

Usage:
    python run_all_plots.py
"""

import subprocess
import sys
from pathlib import Path
import json

# Configure base paths
BASE_DIR = Path("/home/ubuntu/breakpoint")
FINAL_PLOTTERS_DIR = BASE_DIR / "final_plotters"
VENV_PYTHON = BASE_DIR / "venv" / "bin" / "python"

# Primary data files based on CLAUDE.md
DATA_FILES = {
    "remove": BASE_DIR / "data" / "new_all_functions_cleaner.json",
    "discovery": BASE_DIR / "data" / "o4_corruptions.json",
}

# Consolidated result files
CONSOLIDATED_FILES = {
    "remove_all": BASE_DIR / "consolidated_results" / "consolidated_results_remove_all.jsonl",
    "remove_all_normalized": BASE_DIR / "consolidated_results" / "consolidated_results_remove_all_normalized.jsonl",
    "remove_all_with_tools": BASE_DIR / "consolidated_results" / "consolidated_results_remove_all_with_tools.jsonl",
    "remove_all_with_tools_filters": BASE_DIR / "consolidated_results" / "consolidated_results_remove_all_with_tools_filters.jsonl",
    "remove_16iter_4tests": BASE_DIR / "consolidated_results" / "consolidated_results_remove_16iter_4tests_openai.jsonl",
    "discovery_all": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_all.jsonl",
    "discovery_all_normalized": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_all_normalized.jsonl",
    "discovery_all_with_tools": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_all_with_tools.jsonl",
    "discovery_all_with_tools_filters": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_all_with_tools_filters.jsonl",
    "discovery_all_with_tools_filters_recompute": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_all_with_tools_filters_recompute.jsonl",
    "discovery_16iter_4tests": BASE_DIR / "consolidated_results" / "consolidated_results_discovery_16iter_4tests_openai_rh.jsonl",
}

# List of plotting scripts and their configurations
PLOT_SCRIPTS = [
    {
        "name": "main_figure.py",
        "description": "Main comparison figure for discovery and remove modes",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "hard_set_performance_plot.py",
        "description": "Performance on hardest problems (85th percentile)",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "model_scaling_plot.py",
        "description": "Model performance scaling analysis",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "regression_analysis_remove_all.py",
        "description": "Regression analysis for remove mode (all features)",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "regression_analysis_remove_interactions.py",
        "description": "Regression analysis with interaction terms",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "compare_tool_usage.py",
        "description": "Compare tool usage across models",
        "args": ["consolidated_results/consolidated_results_remove_all_with_tools_filters_full.jsonl"],  # Uses hardcoded paths
    },
    {
        "name": "compare_tool_usage.py",
        "description": "Compare tool usage across models",
        "args": ["consolidated_results/consolidated_results_discovery_all_with_tools_filters.jsonl", "discovery_16iter_4tests"],  # Uses hardcoded paths
    },
    {
        "name": "implicit_pass_at_1.py",
        "description": "Implicit pass@1 analysis",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "plot_pass_n.py",
        "description": "Pass@N analysis for different N values",
        "args": ["consolidated_results/consolidated_results_remove_all_with_tools_filters_full.jsonl"],  # Uses hardcoded paths
    },
    {
        "name": "plot_pass_n.py",
        "description": "Pass@N analysis for different N values",
        "args": ["consolidated_results/consolidated_results_discovery_all_with_tools_filters.jsonl", "discovery_16iter_4tests"],  # Uses hardcoded paths
    },
    {
        "name": "plot_n_corruptions.py",
        "description": "Analysis of corruption count effects",
        "args": ["final_results/multi_corrupt_1.jsonl", "final_results/multi_corrupt_2.jsonl", "final_results/multi_corrupt_4.jsonl"],  # Uses hardcoded paths
    },
    {
        "name": "improved_logistic_curve_analysis.py",
        "description": "Logistic curve fitting for model performance",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "right_function_success.py",
        "description": "Analysis of right function identification success",
        "args": [],  # Uses hardcoded paths
    },
    {
        "name": "contrastive_analysis_new_new.py",
        "description": "Contrastive analysis between model pairs",
        "args": [
            "consolidated_results/consolidated_results_remove_all_with_tools_filters_full.jsonl",
            "o4-mini:remove_32iter_8tests",
            "o4-mini:remove_8iter_2tests",
        ],
    },
    {
        "name": "contrastive_analysis_new_new.py",
        "description": "Contrastive analysis between model pairs",
        "args": [
            "consolidated_results/consolidated_results_remove_all_with_tools_filters_full.jsonl",
            "o4-mini:remove_16iter_4tests",
            "gpt-4o:remove_16iter_4tests"
        ],
    }
]

# Contrastive analysis script requires special handling
CONTRASTIVE_SCRIPT = {
    "name": "contrastive_analysis_new_new.py",
    "description": "Contrastive analysis between model pairs",
    "runs": [
        {
            "args": [
                str(CONSOLIDATED_FILES["remove_all_with_tools_filters"]),
            ],
            "description": "Compare o4-mini vs GPT-4o"
        },
    ]
}


def run_script(script_path, args=None, description=""):
    """Run a single plotting script."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    if description:
        print(f"Description: {description}")
    print('='*60)
    
    cmd = [str(VENV_PYTHON), str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ Success: {script_path.name}")
        if result.stdout:
            print("Output:")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {script_path.name}")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        raise
        return False
    
    return True


def main():
    """Run all plotting scripts."""
    print("Running All Final Plots")
    print("="*60)
    
    # Check that all required files exist
    print("\nChecking required files...")
    missing_files = []
    
    for name, path in DATA_FILES.items():
        if not path.exists():
            missing_files.append(f"Data file '{name}': {path}")
    
    for name, path in CONSOLIDATED_FILES.items():
        if not path.exists():
            missing_files.append(f"Consolidated file '{name}': {path}")
    
    if missing_files:
        print("WARNING: Some required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nSome plots may fail due to missing data.")
    
    # Run all standard plotting scripts
    success_count = 0
    total_count = 0
    
    for script_config in PLOT_SCRIPTS:
        script_path = FINAL_PLOTTERS_DIR / script_config["name"]
        if not script_path.exists():
            print(f"\nSkipping {script_config['name']} - file not found")
            continue
            
        total_count += 1
        if run_script(
            script_path, 
            script_config.get("args", []), 
            script_config.get("description", "")
        ):
            success_count += 1
    
    # Run contrastive analysis with multiple configurations
    contrastive_path = FINAL_PLOTTERS_DIR / CONTRASTIVE_SCRIPT["name"]
    if contrastive_path.exists():
        for run_config in CONTRASTIVE_SCRIPT["runs"]:
            total_count += 1
            if run_script(
                contrastive_path,
                run_config["args"],
                run_config["description"]
            ):
                success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total scripts run: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    # Check output directory
    output_dir = BASE_DIR / "final_plots"
    if output_dir.exists():
        plot_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf"))
        print(f"\nGenerated {len(plot_files)} plot files in {output_dir}")
    


if __name__ == "__main__":
    # Activate virtual environment is handled by running with venv python
    main()