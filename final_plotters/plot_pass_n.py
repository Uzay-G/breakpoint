import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Try to use scienceplots for better styling
try:
    import scienceplots
    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.prop_cycle': plt.cycler('color', ['#6C71C4', '#268bd2', '#dc322f', '#859900', '#d33682', '#2aa198', '#b58900', '#6c71c4', '#cb4b16']),
        'font.size': 26,
        'axes.titlesize': 36,
        'axes.labelsize': 34,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
    })
except ImportError:
    print("For better plots, install scienceplots: pip install scienceplots")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.prop_cycle': plt.cycler('color', ['#6C71C4', '#268bd2', '#dc322f', '#859900', '#d33682', '#2aa198', '#b58900', '#6c71c4', '#cb4b16']),
        'font.size': 26,
        'axes.titlesize': 36,
        'axes.labelsize': 34,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
    })

# Define consistent Solarized colors
SOLARIZED_COLORS = {
    "o4-mini": "#268bd2",  # Blue
    "gpt-4o": "#dc322f",   # Red
    "o3-mini": "#6C71C4",  # Green
    "claude-3-7-sonnet-latest": "#d33682"  # Magenta
}

def process_jsonl_file(file_path):
    """Process a JSONL file containing multiple JSON objects."""
    all_data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "model_performance" in data:
                    all_data.append(data)
            except json.JSONDecodeError:
                continue
    return all_data

def extract_test_budget(config_name):
    """Extract the test budget from the config name."""
    # Format is like "remove_16iter_4tests"
    parts = config_name.split('_')
    for part in parts:
        if part.endswith('tests'):
            return int(part.replace('tests', ''))
    return 4  # Default if not found

def calculate_aggregated_pass_at_n(all_data, config_name="remove_16iter_4tests"):
    """Calculate aggregated pass@n metrics across multiple problems."""
    # Extract the test budget from the config name
    test_budget = extract_test_budget(config_name)
    
    # Initialize collection structure
    model_successes = defaultdict(lambda: defaultdict(int))
    model_totals = defaultdict(int)
    
    # Process each problem
    for data in all_data:
        for model_name, model_configs in data.get("model_performance", {}).items():
            if config_name in model_configs:
                model_totals[model_name] += 1
                
                # Get tool calls for this model and config
                if config_name in data.get("tool_call_lists", {}).get(model_name, {}):
                    tool_calls = data["tool_call_lists"][model_name][config_name]
                    
                    # Filter to only submit attempts
                    submit_indices = [i for i, tool in enumerate(tool_calls) if tool == "submit_attempt" or tool == "replace_function"]
                    total_submissions = len(submit_indices)
                    
                    # Check if the problem was solved
                    score = model_configs[config_name]
                    if score == 1.0:
                        # If solved, record as success at attempt number
                        # Note: submit_indices are 0-based, but we want 1-based attempts
                        success_attempt = len(submit_indices)  # Last submit attempt
                        
                        # Record success at this attempt and all higher attempts
                        for n in range(success_attempt, test_budget + 1):
                            model_successes[model_name][n] += 1
                    else:
                        if len(submit_indices) != 0:
                            print(len(submit_indices), "submit attempts found but not solved")
    
    # Calculate pass@n for each model
    pass_at_n_by_model = {}
    
    for model_name in model_totals:
        model_pass_at_n = {}
        total = model_totals[model_name]
        
        # Calculate pass@n for each n from 1 to test_budget
        for n in range(1, test_budget + 1):
            successes = model_successes[model_name][n]
            model_pass_at_n[n] = successes / total if total > 0 else 0
            
        pass_at_n_by_model[model_name] = model_pass_at_n
    
    return pass_at_n_by_model, model_totals

def plot_multi_model_pass_at_n(pass_at_n_by_model, model_totals, config_name, output_file='pass_at_n_multi_model.png'):
    """Plot the pass@n curve for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Extract test budget
    test_budget = extract_test_budget(config_name)
    
    for i, (model_name, pass_at_n) in enumerate(pass_at_n_by_model.items()):
        # Sort by attempt number
        attempts = sorted(pass_at_n.keys())
        success_rates = [pass_at_n[n] * 100 for n in attempts]  # Convert to percentage
        
        color = SOLARIZED_COLORS.get(model_name, f"C{i}")
        
        # Plot the line
        ax.plot(
            attempts, 
            success_rates, 
            marker='o', 
            markersize=5,
            linestyle='-', 
            color=color,
            label=f'{model_name}'
        )
        
        # Annotate the final point
        final_n = max(attempts)
        final_rate = pass_at_n[final_n] * 100
        ax.annotate(
            f'{final_rate:.1f}%',
            xy=(final_n, final_rate),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=30,
            color=color,
            weight='bold'
        )
    
    # Add labels and title
    ax.set_xlabel("Number of Test Submissions (N)")
    ax.set_ylabel("Solve Rate")
    ax.set_title(f"Pass@N Scaling for {config_name.split("_")[0].capitalize()}")
    
    # Set the x-axis to have integer ticks from 1 to test_budget
    ax.set_xticks(range(1, test_budget + 1))
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format the y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    
    # Adjust y-axis
    ax.set_ylim(0, 105)  # Add headroom for annotations
    
    # Add legend with a nice layout
    ax.legend(loc='lower right', frameon=True)
    
    # Ensure proper layout and save with high resolution
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    
    # Optionally show the plot
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python pass_at_n_multi_model.py <path_to_jsonl_file> [config_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    config_name = sys.argv[2] if len(sys.argv) > 2 else "remove_16iter_4tests"
    
    # Ensure the output directory exists
    os.makedirs("final_plots", exist_ok=True)
    
    # Process the data
    all_data = process_jsonl_file(file_path)
    
    if not all_data:
        print("No valid data found in the file.")
        sys.exit(1)
    
    pass_at_n_by_model, model_totals = calculate_aggregated_pass_at_n(all_data, config_name)
    
    # Print statistics
    print(f"Configuration: {config_name}")
    print(f"Test budget: {extract_test_budget(config_name)}")
    print("\nPass@N metrics by model:")
    
    for model_name, pass_at_n in pass_at_n_by_model.items():
        print(f"\n{model_name} (Total problems: {model_totals[model_name]}):")
        for n in sorted(pass_at_n.keys()):
            print(f"  Pass@{n}: {pass_at_n[n]:.4f} ({pass_at_n[n]*100:.2f}%)")
    
    # Generate the plot
    output_file = f"final_plots/pass_at_n_{config_name}.png"
    plot_multi_model_pass_at_n(pass_at_n_by_model, model_totals, config_name, output_file)

if __name__ == "__main__":
    main()