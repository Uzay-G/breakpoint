"""Test the improved bootstrap error handling with edge cases."""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_plotters.improved_logistic_curve_analysis import bootstrap_prediction_intervals
import statsmodels.api as sm


def test_perfect_separation():
    """Test with perfect separation (should cause convergence issues)."""
    print("\n=== Testing with perfect separation ===")
    
    # Create perfectly separated data
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    try:
        result = bootstrap_prediction_intervals(X_with_const, y, n_bootstrap=100)
        print(f"Success! {result['n_successful']} successful, {result['n_failed']} failed")
    except Exception as e:
        print(f"Failed with error: {e}")


def test_all_same_outcome():
    """Test with all same outcome (should cause issues)."""
    print("\n=== Testing with all same outcome ===")
    
    # Create data with all same outcome
    X = np.random.randn(20, 1)
    y = np.ones(20)  # All 1s
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    try:
        result = bootstrap_prediction_intervals(X_with_const, y, n_bootstrap=100)
        print(f"Success! {result['n_successful']} successful, {result['n_failed']} failed")
    except Exception as e:
        print(f"Failed with error: {e}")


def test_small_sample():
    """Test with very small sample size."""
    print("\n=== Testing with small sample size ===")
    
    # Create very small dataset
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 1])
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    try:
        result = bootstrap_prediction_intervals(X_with_const, y, n_bootstrap=100)
        print(f"Success! {result['n_successful']} successful, {result['n_failed']} failed")
    except Exception as e:
        print(f"Failed with error: {e}")


def test_normal_case():
    """Test with normal data."""
    print("\n=== Testing with normal data ===")
    
    # Create normal dataset
    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = (X.flatten() + np.random.randn(50) * 0.5 > 0).astype(int)
    
    # Add constant
    X_with_const = sm.add_constant(X)
    
    try:
        result = bootstrap_prediction_intervals(X_with_const, y, n_bootstrap=100)
        print(f"Success! {result['n_successful']} successful, {result['n_failed']} failed")
    except Exception as e:
        print(f"Failed with error: {e}")


if __name__ == "__main__":
    test_perfect_separation()
    test_all_same_outcome()
    test_small_sample()
    test_normal_case()