"""
Regression analysis to predict model success based on code complexity metrics with interaction terms.

This script fits logistic regression models to predict whether a model will 
successfully solve a problem in REMOVE mode, comparing models with and without 
interaction terms and evaluating them using BIC.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


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


def fit_models_with_and_without_interactions(df):
    """Fit logistic regression models with and without interaction terms and compare using BIC."""
    # Ensure we have enough data points
    if len(df) < 10:
        print(f"Not enough data points: {len(df)}")
        return None

    # Add model's average performance
    model_perf = calculate_model_performance(df)
    df = pd.merge(df, model_perf, on=["model", "test_config"])

    # Create interaction term
    df["code_lines_x_harmonic"] = df["code_lines"] * df["harmonic"]

    # Model 1: No interactions
    X1 = df[["code_lines", "harmonic", "avg_performance"]]
    X1 = sm.add_constant(X1)
    y = df["success"]

    # Model 2: With code_lines × harmonic interaction
    X2 = df[["code_lines", "harmonic", "avg_performance", "code_lines_x_harmonic"]]
    X2 = sm.add_constant(X2)

    # Fit models
    results = {}
    model_names = {
        "base_model": "Without Interactions",
        "interaction_model": "With code_lines × harmonic Interaction",
    }

    try:
        # Fit base model
        model1 = sm.Logit(y, X1)
        result1 = model1.fit()
        results["base_model"] = {
            "result": result1,
            "bic": result1.bic,
            "aic": result1.aic,
            "params": result1.params.to_dict(),
            "pvalues": result1.pvalues.to_dict(),
        }

        # Fit interaction model
        model2 = sm.Logit(y, X2)
        result2 = model2.fit()
        results["interaction_model"] = {
            "result": result2,
            "bic": result2.bic,
            "aic": result2.aic,
            "params": result2.params.to_dict(),
            "pvalues": result2.pvalues.to_dict(),
        }

        # Compare models
        print("\n===== Model Comparison =====")
        print(f"Number of observations: {len(df)}")

        for model_key, model_data in results.items():
            print(f"\n--- {model_names[model_key]} ---")
            print(f"BIC: {model_data['bic']:.2f}")
            print(f"AIC: {model_data['aic']:.2f}")
            print("Parameters:")
            for param, value in model_data["params"].items():
                p_value = model_data["pvalues"][param]
                star = "*" if p_value < 0.05 else " "
                stars = (
                    "***"
                    if p_value < 0.001
                    else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                )
                print(f"  {param}: {value:.4f} (p={p_value:.4f}) {stars}")

        # Recommend the better model based on BIC
        bic_diff = results["interaction_model"]["bic"] - results["base_model"]["bic"]
        if bic_diff < 0:
            print(
                f"\nThe interaction model has a BETTER fit (BIC difference: {abs(bic_diff):.2f})"
            )
            if abs(bic_diff) > 10:
                print("The evidence for the interaction model is VERY STRONG")
            elif abs(bic_diff) > 6:
                print("The evidence for the interaction model is STRONG")
            else:
                print(
                    "The evidence for the interaction model is POSITIVE but not conclusive"
                )
        elif bic_diff > 0:
            print(
                f"\nThe base model has a BETTER fit (BIC difference: {abs(bic_diff):.2f})"
            )
            if abs(bic_diff) > 10:
                print("The evidence against the interaction term is VERY STRONG")
            elif abs(bic_diff) > 6:
                print("The evidence against the interaction term is STRONG")
            else:
                print(
                    "The evidence against the interaction term is POSITIVE but not conclusive"
                )
        else:
            print("\nBoth models have EQUIVALENT fit")

        # Save model names for later use
        results["model_names"] = model_names

        return results

    except Exception as e:
        print(f"Error fitting regression models: {e}")
        return None


def main():
    # Configuration
    file_path = "/home/ubuntu/breakpoint/consolidated_results/consolidated_results_remove_all_normalized.jsonl"

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

    # Fit and compare models
    results = fit_models_with_and_without_interactions(df)

    if results:
        print("\n===== Additional Analysis =====")

        # Check if interaction is significant
        p_interaction = results["interaction_model"]["pvalues"].get(
            "code_lines_x_harmonic", 1.0
        )
        if p_interaction < 0.05:
            print(
                f"The interaction term is statistically significant (p={p_interaction:.4f})"
            )
            coef_interaction = results["interaction_model"]["params"].get(
                "code_lines_x_harmonic", 0.0
            )
            if coef_interaction > 0:
                print(
                    "The positive coefficient suggests that high code line count AND high harmonic centrality TOGETHER"
                )
                print(
                    "have a stronger negative effect on success than would be predicted from their individual effects"
                )
            else:
                print(
                    "The negative coefficient suggests that high code line count AND high harmonic centrality TOGETHER"
                )
                print(
                    "have a weaker negative effect on success than would be predicted from their individual effects"
                )
        else:
            print(
                f"The interaction term is NOT statistically significant (p={p_interaction:.4f})"
            )

        # Print the summary of the better model
        better_model = (
            "interaction_model"
            if results["interaction_model"]["bic"] < results["base_model"]["bic"]
            else "base_model"
        )
        model_names = results["model_names"]
        print(f"\n===== Best Model: {model_names[better_model]} =====")
        print(results[better_model]["result"].summary())


if __name__ == "__main__":
    main()
