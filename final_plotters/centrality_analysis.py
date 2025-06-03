"""
Correlation analysis between all code complexity and centrality metrics.

This script creates a correlation matrix heatmap showing relationships between
all metrics including binary success scores.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scienceplots

# Try to use scienceplots style if available
plt.style.use(["science"])
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
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
    }
)


def load_data(filepath):
    """Load and parse the consolidated results file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_correlation_data(data):
    """
    Prepare data for correlation analysis, extracting all metrics and binary success scores.
    
    Args:
        data: List of dictionaries containing problem data
        
    Returns:
        pandas DataFrame with all metrics and success scores
    """
    correlation_data = []
    
    for item in data:
        function_id = item.get("function_id", "unknown")
        
        # Skip entries without required data
        if "complexity_metrics" not in item or "model_performance" not in item:
            continue
            
        # Initialize metrics dictionary
        metrics = {"function_id": function_id}
        
        # Extract basic complexity metrics
        complexity = item["complexity_metrics"]
        metrics["code_line_count"] = complexity.get("code_line_count", complexity.get("line_count", None))
        metrics["cyclomatic"] = complexity.get("cyclomatic", None)
        metrics["halstead_difficulty"] = complexity.get("halstead_difficulty", None)
        metrics["halstead_volume"] = complexity.get("halstead_volume", None)
        
        # Extract centrality metrics
        if "centrality" in complexity:
            centrality = complexity["centrality"]
            metrics["harmonic"] = centrality.get("harmonic", None)
            metrics["betweenness"] = centrality.get("betweenness", None)
            metrics["degree"] = centrality.get("degree", None)
            metrics["distance_discount"] = centrality.get("distance_discount", None)
            metrics["in_degree"] = centrality.get("in_degree", None)
            metrics["out_degree"] = centrality.get("out_degree", None)
            metrics["pagerank"] = centrality.get("pagerank", None)
        
        # Calculate average success rate across all models and remove mode configs
        success_scores = []
        for model_name, model_data in item["model_performance"].items():
            for test_config, success in model_data.items():
                # Only include REMOVE mode test configs
                if test_config.startswith("remove_"):
                    # Binary success (1.0 if perfect, 0.0 otherwise)
                    success_scores.append(1.0 if success == 1.0 else 0.0)
        
        if success_scores:
            metrics["avg_success"] = np.mean(success_scores)
            correlation_data.append(metrics)
    
    return pd.DataFrame(correlation_data)


def create_correlation_heatmap(df, output_path=None):
    """Create and save a correlation matrix heatmap."""
    # Select only numeric columns and drop any with all NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols].dropna(axis=1, how='all')
    
    # Drop function_id if it's numeric (it shouldn't be used in correlation)
    if 'function_id' in df_numeric.columns:
        df_numeric = df_numeric.drop('function_id', axis=1)
    
    # Define metric groups for better organization
    complexity_metrics = ['code_line_count', 'cyclomatic', 'halstead_difficulty', 'halstead_volume']
    degree_metrics = ['degree', 'in_degree', 'out_degree']
    centrality_metrics = ['pagerank', 'harmonic', 'betweenness', 'distance_discount']
    success_metrics = ['avg_success']
    
    # Organize columns in logical groups
    ordered_cols = []
    for group in [complexity_metrics, degree_metrics, centrality_metrics, success_metrics]:
        for col in group:
            if col in df_numeric.columns:
                ordered_cols.append(col)
    
    # Reorder dataframe columns
    df_numeric = df_numeric[ordered_cols]
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure with extra space for labels
    fig, ax = plt.subplots(figsize=(15, 13))
    fig.subplots_adjust(left=0.15, bottom=0.15)  # Add margins for group labels
    
    # Create custom colormap (diverging from blue to red)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        annot=True,
        fmt='.2f',
        annot_kws={'size': 8},
        ax=ax
    )
    
    # Add lines to separate metric groups
    n_complexity = len([col for col in complexity_metrics if col in ordered_cols])
    n_degree = len([col for col in degree_metrics if col in ordered_cols])
    n_centrality = len([col for col in centrality_metrics if col in ordered_cols])
    
    # Draw separator lines
    positions = []
    current_pos = 0
    if n_complexity > 0:
        current_pos += n_complexity
        positions.append(current_pos)
    if n_degree > 0:
        current_pos += n_degree
        positions.append(current_pos)
    if n_centrality > 0:
        current_pos += n_centrality
        positions.append(current_pos)
    
    # Draw all separator lines (including before success metrics)
    for pos in positions:
        if pos < len(ordered_cols):  # Don't draw line at the very end
            ax.axhline(y=pos, color='black', linewidth=2, alpha=0.7)
            ax.axvline(x=pos, color='black', linewidth=2, alpha=0.7)
    
    # Customize the plot
    ax.set_title("Correlation Matrix of Code Complexity and Centrality Metrics", fontsize=18, pad=20)
    
    # Add group labels
    group_positions = []
    current_pos = 0
    if n_complexity > 0:
        group_positions.append((current_pos + n_complexity/2, "Complexity"))
        current_pos += n_complexity
    if n_degree > 0:
        group_positions.append((current_pos + n_degree/2, "Degree"))
        current_pos += n_degree
    if n_centrality > 0:
        group_positions.append((current_pos + n_centrality/2, "Centrality"))
        current_pos += n_centrality
    group_positions.append((current_pos + len([col for col in success_metrics if col in ordered_cols])/2, "Success"))
    
    # Add text labels for groups at medium distance from plot
    for pos, label in group_positions:
        ax.text(pos, 0, label, ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.text(-1.65, pos, label, ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Make labels more readable
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        text = label.get_text()
        # Replace underscores with spaces and capitalize
        new_text = text.replace('_', ' ').title()
        label.set_text(new_text)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {output_path}")
    
    plt.show()
    
    # Print highly correlated pairs (excluding diagonal)
    print("\nHighly correlated metric pairs (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")


def create_scatter_matrix(df, output_path=None):
    """Create a scatter plot matrix for selected important metrics."""
    # Select key metrics for scatter matrix
    key_metrics = [
        'code_line_count', 'cyclomatic', 'harmonic', 'betweenness', 
        'pagerank', 'avg_success'
    ]
    
    # Filter to only existing columns
    available_metrics = [col for col in key_metrics if col in df.columns]
    df_subset = df[available_metrics].dropna()
    
    # Create scatter matrix
    fig = plt.figure(figsize=(12, 10))
    
    # Use pandas scatter_matrix
    pd.plotting.scatter_matrix(
        df_subset,
        alpha=0.5,
        figsize=(12, 10),
        diagonal='kde',
        grid=True,
        color='#268bd2'
    )
    
    # Adjust layout and add title
    fig.suptitle("Scatter Matrix of Key Metrics", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Scatter matrix saved to {output_path}")
    
    plt.show()


def calculate_metric_importance(df):
    """Calculate and display importance of each metric in predicting success."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['function_id', 'avg_success', 'max_success', 'min_success']]
    X = df[feature_cols].dropna()
    
    # Create binary success variable (using avg_success > 0.5 as threshold)
    y = (df.loc[X.index, 'avg_success'] > 0.5).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit random forest to get feature importances
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importances
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importances['feature'], importances['importance'], color='#268bd2')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Relative Importance of Metrics in Predicting Success')
    ax.invert_yaxis()
    
    # Make labels more readable
    labels = [label.replace('_', ' ').title() for label in importances['feature']]
    ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.show()
    
    print("\nFeature Importance Rankings:")
    for _, row in importances.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")


def main():
    # Configuration
    file_path = "/home/ubuntu/breakpoint/consolidated_results/consolidated_results_remove_all_normalized.jsonl"
    output_dir = "/home/ubuntu/breakpoint/final_plots/"
    
    # Load data
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    print(f"Loaded {len(data)} entries")
    
    # Prepare data for correlation analysis
    df = prepare_correlation_data(data)
    print(f"Prepared {len(df)} data points for correlation analysis")
    
    # Data summary
    print("\nData Summary:")
    print(f"Number of metrics: {len(df.columns) - 1}")  # Excluding function_id
    print(f"Metrics included: {', '.join([col for col in df.columns if col != 'function_id'])}")
    print(f"\nMissing value counts:")
    print(df.isnull().sum())
    
    # Create correlation heatmap
    output_path = f"{output_dir}correlation_heatmap_all_metrics.png"
    create_correlation_heatmap(df, output_path)
    
    # Create scatter matrix for key metrics
    scatter_output_path = f"{output_dir}scatter_matrix_key_metrics.png"
    create_scatter_matrix(df, scatter_output_path)
    
    # Calculate and display metric importance
    calculate_metric_importance(df)


if __name__ == "__main__":
    main()