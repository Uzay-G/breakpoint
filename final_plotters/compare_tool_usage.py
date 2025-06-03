import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import sys
from pathlib import Path
import time
import pandas as pd
import seaborn as sns

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

class ModelToolComparer:
    def __init__(self, file_path: str, config_name: str = "remove_16iter_4tests"):
        """Initialize with a JSONL file path and config name."""
        self.file_path = file_path
        self.config_name = config_name
        self.data = []
        self.models = set()
        self.test_budget = self.extract_test_budget(config_name)
        self.model_problems = defaultdict(list)
        self.output_dir = Path("final_plots")
        self.output_dir.mkdir(exist_ok=True)
        self.load_data()
        
    def extract_test_budget(self, config_name):
        """Extract the test budget from the config name."""
        parts = config_name.split('_')
        for part in parts:
            if part.endswith('tests'):
                return int(part.replace('tests', ''))
        return 4  # Default if not found
        
    def load_data(self):
        """Load and parse the JSONL data file with the new format."""
        print(f"Loading data from {self.file_path}...")
        
        with open(self.file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "model_performance" in entry and "tool_call_lists" in entry:
                        problem_id = entry.get("function_id", entry.get("problem", "unknown"))
                        
                        # Process each model separately
                        for model_name, configs in entry["model_performance"].items():
                            if self.config_name in configs:
                                self.models.add(model_name)
                                score = configs[self.config_name]
                                
                                # Get tool calls for this model and config
                                tool_calls = []
                                if model_name in entry["tool_call_lists"] and self.config_name in entry["tool_call_lists"][model_name]:
                                    tool_calls = entry["tool_call_lists"][model_name][self.config_name]
                                
                                # Format tool usage like the original format
                                tool_usage = []
                                for tool_name in tool_calls:
                                    tool_usage.append({"tool": tool_name})
                                
                                # Create problem data structure
                                problem_data = {
                                    "problem_name": problem_id,
                                    "score": score,
                                    "model": model_name,
                                    "tool_usage": tool_usage
                                }
                                
                                self.data.append(problem_data)
                                self.model_problems[model_name].append(problem_data)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        print(f"Loaded {len(self.data)} problem-model combinations for config '{self.config_name}'")
        print(f"Models found: {sorted(self.models)}")
        
        # Count problems per model
        model_problem_counts = Counter([p["model"] for p in self.data])
        for model, count in model_problem_counts.items():
            print(f"{model}: {count} problems")

    def analyze_tool_usage_distribution(self):
            """Analyze how frequently each model uses different tools."""
            model_tool_counts = {model: Counter() for model in self.models}
            model_problem_counts = Counter()
            
            for problem in self.data:
                model = problem["model"]
                model_problem_counts[model] += 1
                
                for tool_use in problem["tool_usage"]:
                    tool = tool_use.get("tool")
                    if tool:
                        model_tool_counts[model][tool] += 1
            
            # Normalize by problem count
            normalized_counts = {}
            for model, counts in model_tool_counts.items():
                total_problems = model_problem_counts[model]
                normalized_counts[model] = {tool: count/total_problems for tool, count in counts.items()}
            
            return normalized_counts

    def plot_tool_usage_heatmap(self, normalized_counts):
        """Plot a heatmap of tool usage across models."""
        print("Generating tool usage heatmap...")
        
        try:
            # Convert to DataFrame for easier plotting
            print(normalized_counts)
            models = sorted(self.models)
            tools = sorted(normalized_counts[models[0]].keys())
            
            data = []
            for model in models:
                row = [normalized_counts[model].get(tool, 0) for tool in tools]
                data.append(row)
            
            df = pd.DataFrame(data, index=models, columns=tools)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, cmap="PuRd", fmt=".2f", cbar_kws={'label': 'Average Usage Per Problem'})
            plt.title(f"Tool Usage Distribution Across Models for {self.config_name.split("_")[0].capitalize()}")
            plt.ylabel("Model")
            plt.xlabel("Tool Type")
            plt.tight_layout()
            
            output_file = self.output_dir / f"tool_usage_heatmap_{self.config_name}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Error generating tool usage heatmap: {e}")

    def analyze_average_info_and_submissions(self):
        """Calculate the average number of submissions and info gathering calls per model."""
        print("Analyzing average info gathering and submission counts...")
        
        model_stats = {}
        
        for model in self.models:
            submission_counts = []
            info_gathering_counts = []
            
            for problem in self.model_problems[model]:
                # Count submissions
                submit_count = sum(1 for t in problem["tool_usage"] if t.get("tool") == "submit_attempt" or t.get("tool") == "replace_function")
                submission_counts.append(min(submit_count, self.test_budget))  # Cap at test budget
                
                # Count info gathering calls
                info_count = sum(1 for t in problem["tool_usage"] if t.get("tool") != "submit_attempt" and t.get("tool") != "replace_function")
                info_gathering_counts.append(info_count)
            
            # Calculate averages
            if submission_counts:  # If the model has any problems
                avg_submissions = sum(submission_counts) / len(submission_counts)
                avg_info = sum(info_gathering_counts) / len(info_gathering_counts)
                
                model_stats[model] = {
                    "avg_submissions": avg_submissions,
                    "avg_info_gathering": avg_info,
                    "problem_count": len(submission_counts)
                }
        
        return model_stats

    def plot_average_info_and_submissions(self, model_stats):
        """Plot the average number of submissions and info gathering calls per model."""
        print("Generating average info and submissions plot...")
        
        if not model_stats:
            print("No data available for plotting")
            return
        
        # Sort models by avg_submissions
        models = sorted(model_stats.keys(), key=lambda m: model_stats[m]["avg_submissions"])
        
        # Extract data for plotting
        avg_submissions = [model_stats[model]["avg_submissions"] for model in models]
        avg_info = [model_stats[model]["avg_info_gathering"] for model in models]
        problem_counts = [model_stats[model]["problem_count"] for model in models]
        
        # Set up the figure for a grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bars side by side
        x = np.arange(len(models))
        width = 0.35
        
        # Use purple colors
        submissions_bars = ax.bar(x - width/2, avg_submissions, width, label='Average Submissions', color='#6C71C4')
        info_bars = ax.bar(x + width/2, avg_info, width, label='Average Info Gathering', color="#D33682")
        
        # Add annotations with exact values
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=30)
        
        autolabel(submissions_bars)
        autolabel(info_bars)
        
        # Add labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Count')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        plt.title(f"Information Gathering Tool Usage vs Submissions for {self.config_name.split("_")[0].capitalize()}")
        
        # Add legend
        ax.legend()
        
        # Add grid lines for readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        output_file = self.output_dir / f"avg_info_and_submissions_{self.config_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
        plt.close()
        
        # Also create a ratio plot (info gathering per submission)
        self.plot_info_to_submission_ratio(model_stats)
    
    def plot_info_to_submission_ratio(self, model_stats):
        """Plot the ratio of info gathering calls per submission."""
        print("Generating info-to-submission ratio plot...")
        
        # Calculate ratios and prepare data
        models = sorted(model_stats.keys())
        ratios = []
        problem_counts = []
        
        for model in models:
            stats = model_stats[model]
            # Avoid division by zero
            ratio = stats["avg_info_gathering"] / stats["avg_submissions"] if stats["avg_submissions"] > 0 else 0
            ratios.append(ratio)
            problem_counts.append(stats["problem_count"])
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by ratio
        sorted_indices = np.argsort(ratios)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_ratios = [ratios[i] for i in sorted_indices]
        sorted_counts = [problem_counts[i] for i in sorted_indices]
        
        # Plot bars
        bars = ax.barh(sorted_models, sorted_ratios, color='#6C71C4')
        
        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.3, 
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f} (n={sorted_counts[i]})',
                va='center'
            )
        
        ax.set_xlabel('Info Gathering Calls per Submission')
        ax.set_title(f'Information Efficiency by Model ({self.config_name})')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        output_file = self.output_dir / f"info_per_submission_{self.config_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Ratio plot saved to {output_file}")
        plt.close()

    

    def run_analysis(self):
        """Run the simple analysis."""
        print(f"\n=== Analyzing Tool Usage for {self.config_name} ===\n")
        
        # Generate the average info and submissions plot
        model_stats = self.analyze_average_info_and_submissions()
        self.plot_average_info_and_submissions(model_stats)

        tool_usage = self.analyze_tool_usage_distribution()
        self.plot_tool_usage_heatmap(tool_usage)
        
        print(f"\n=== Analysis Complete ===")


def main():
    if len(sys.argv) < 2:
        print("Usage: python avg_tool_usage.py <path_to_jsonl_file> [config_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    config_name = sys.argv[2] if len(sys.argv) > 2 else "remove_16iter_4tests"
    
    try:
        analyzer = ModelToolComparer(file_path, config_name)
        analyzer.run_analysis()
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == "__main__":
    main()