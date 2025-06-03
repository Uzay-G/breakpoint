import os
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path("/home/ubuntu/breakpoint")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "final_results"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILE = OUTPUT_DIR / "consolidated_2.jsonl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model name mapping to handle inconsistencies
MODEL_NAME_MAPPING = {
    "o3mini": ["o3mini", "o3-mini"],
    "4o": ["4o"],
    "claude": ["claude"],
    "deepseek": ["deepseek"],
}

# Reverse mapping for normalization
MODEL_NAME_REVERSE = {}
for norm_name, variants in MODEL_NAME_MAPPING.items():
    for variant in variants:
        MODEL_NAME_REVERSE[variant] = norm_name


def read_json_or_jsonl(file_path: str) -> List[Dict]:
    """
    Read a file that could be either JSON or JSONL and return a list of dictionaries.
    Handles the different formats found in the data files.

    Args:
        file_path: Path to the JSON or JSONL file

    Returns:
        List of dictionaries parsed from the file
    """
    # Handle the case where the file doesn't exist
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []

    # First, try to read as a single JSON document (e.g., easy.jsonl is actually a JSON array)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            # If it's a list, return it directly
            if isinstance(data, list):
                logger.info(
                    f"Successfully loaded {file_path} as a JSON array with {len(data)} records"
                )
                return data
            # If it's a single object, wrap it in a list
            logger.info(f"Successfully loaded {file_path} as a single JSON document")
            return [data]
    except json.JSONDecodeError:
        # If that fails, try reading line by line as JSONL
        data = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Error parsing JSON line in {file_path}: {e}")
        logger.info(
            f"Successfully loaded {file_path} as JSONL with {len(data)} records"
        )
        return data
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []


def extract_problem_metadata() -> Dict[str, Dict]:
    """
    Extract function information and complexity metrics from problem data files.

    Returns:
        Dictionary mapping function IDs to their complexity metrics
    """
    logger.info("Extracting problem metadata from data files...")

    # Dictionary to store function metadata
    function_metadata = {}
    # Set to track all unique function IDs
    unique_function_ids = set()

    # Use only the specific data files needed - excluding easy.jsonl as requested
    data_files = ["hardest_final.jsonl", "less_hard_v2_no_repeats.jsonl"]
    logger.info(f"Using {len(data_files)} data files: {', '.join(data_files)}")

    for file_name in data_files:
        file_path = os.path.join(DATA_DIR, file_name)
        logger.info(f"Processing {file_path}")

        # Determine difficulty level from filename
        difficulty_level = "unknown"
        if "hardest" in file_name:
            difficulty_level = "hardest"
        elif "hard" in file_name:  # This will also match "less_hard"
            difficulty_level = "hard"

        # Read the JSON/JSONL file using the format-aware reader
        problems = read_json_or_jsonl(file_path)
        logger.info(f"  Found {len(problems)} problems in {file_name}")

        # Track the number of functions processed in this file
        processed_count = 0

        # Process each problem
        for problem in problems:
            # Skip if there's no complexity info
            if not problem.get("complexity_info"):
                continue

            # Construct a unique function ID
            if "repo" in problem and "fpath" in problem and "function_name" in problem:
                # Format: repo_name/file_path/function_name
                repo_name = problem.get("repo", {}).get("name", "")
                if not repo_name and "path" in problem.get("repo", {}):
                    repo_path = problem.get("repo", {}).get("path", "")
                    repo_name = os.path.basename(repo_path)

                function_id = (
                    f"{repo_name}/{problem['fpath']}/{problem['function_name']}"
                )
            elif "problem" in problem:  # In case of different format
                function_id = problem["problem"]
            else:
                # Skip if we can't identify the function
                continue

            # Skip if we've already processed this function in another file
            if function_id in unique_function_ids:
                logger.debug(f"Skipping duplicate function ID: {function_id}")
                continue

            # Add to the set of unique function IDs
            unique_function_ids.add(function_id)
            processed_count += 1

            # Extract complexity metrics
            complexity_info = problem.get("complexity_info", {})

            # Store function metadata
            function_metadata[function_id] = {
                "function_id": function_id,
                "complexity_metrics": {
                    "line_count": complexity_info.get("line_count", 0),
                    "code_line_count": complexity_info.get("code_line_count", 0),
                    "cyclomatic": complexity_info.get("cyclomatic", 1),
                },
                "difficulty_level": difficulty_level,
            }

            # Extract centrality metrics if available
            if "centrality" in complexity_info:
                centrality = complexity_info["centrality"]
                function_metadata[function_id]["complexity_metrics"]["centrality"] = {
                    "degree": centrality.get("degree", 0),
                    "in_degree": centrality.get("in_degree", 0),
                    "out_degree": centrality.get("out_degree", 0),
                    "betweenness": centrality.get("betweenness", 0.0),
                    "pagerank": centrality.get("pagerank", 0.0),
                    "harmonic": centrality.get("harmonic", 0.0),
                    "distance_discount": centrality.get("distance_discount", 0.0),
                }
            elif (
                "degree" in complexity_info
            ):  # Some files might have centrality directly in complexity_info
                function_metadata[function_id]["complexity_metrics"]["centrality"] = {
                    "degree": complexity_info.get("degree", 0),
                    "in_degree": complexity_info.get("in_degree", 0),
                    "out_degree": complexity_info.get("out_degree", 0),
                    "betweenness": complexity_info.get("betweenness", 0.0),
                    "pagerank": complexity_info.get("pagerank", 0.0),
                    "harmonic": complexity_info.get("harmonic", 0.0),
                    "distance_discount": complexity_info.get("distance_discount", 0.0),
                }

            # Extract Halstead metrics if available
            if "halstead_volume" in complexity_info:
                function_metadata[function_id]["complexity_metrics"][
                    "halstead_volume"
                ] = complexity_info.get("halstead_volume", 0.0)
            if "halstead_difficulty" in complexity_info:
                function_metadata[function_id]["complexity_metrics"][
                    "halstead_difficulty"
                ] = complexity_info.get("halstead_difficulty", 0.0)

            # Extract repository information if available
            if "repo" in problem and "repo_stats" in problem["repo"]:
                function_metadata[function_id]["repo_stats"] = problem["repo"][
                    "repo_stats"
                ]

        logger.info(f"  Processed {processed_count} unique functions from {file_name}")

    logger.info(
        f"Extracted metadata for {len(function_metadata)} unique functions total"
    )
    return function_metadata


def extract_problem_metadata_with_repo_info() -> Dict[str, Dict]:
    """
    Extract function information including repository statistics from repo_info files.

    Returns:
        Dictionary mapping function IDs to their metadata including repo stats
    """
    logger.info("Extracting problem metadata with repository information...")

    # Dictionary to store function metadata
    function_metadata = {}
    # Set to track all unique function IDs
    unique_function_ids = set()

    # Use repo_info files
    data_files = [
        "hardest_final_repo_info.jsonl",
        "less_hard_v2_no_repeats_repo_info.jsonl",
    ]
    logger.info(f"Using {len(data_files)} repo info files: {', '.join(data_files)}")

    for file_name in data_files:
        file_path = os.path.join(DATA_DIR, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Repo info file not found: {file_path}")
            # Try to use the original files instead
            original_file = file_name.replace("_repo_info", "")
            logger.info(f"Falling back to original file: {original_file}")
            file_path = os.path.join(DATA_DIR, original_file)
            if not os.path.exists(file_path):
                logger.error(f"Original file not found either: {file_path}")
                continue

        logger.info(f"Processing {file_path}")

        # Determine difficulty level from filename
        difficulty_level = "unknown"
        if "hardest" in file_name:
            difficulty_level = "hardest"
        elif "hard" in file_name:  # This will also match "less_hard"
            difficulty_level = "hard"

        # Read the JSON/JSONL file using the format-aware reader
        problems = read_json_or_jsonl(file_path)
        logger.info(f"  Found {len(problems)} problems in {file_name}")

        # Track the number of functions processed in this file
        processed_count = 0

        # Process each problem
        for problem in problems:
            # Skip if there's no complexity info
            if not problem.get("complexity_info"):
                continue

            # Construct a unique function ID
            if "repo" in problem and "fpath" in problem and "function_name" in problem:
                # Format: repo_name/file_path/function_name
                repo_name = problem.get("repo", {}).get("name", "")
                if not repo_name and "path" in problem.get("repo", {}):
                    repo_path = problem.get("repo", {}).get("path", "")
                    repo_name = os.path.basename(repo_path)

                function_id = (
                    f"{repo_name}/{problem['fpath']}/{problem['function_name']}"
                )
            elif "problem" in problem:  # In case of different format
                function_id = problem["problem"]
            else:
                # Skip if we can't identify the function
                continue

            # Skip if we've already processed this function in another file
            if function_id in unique_function_ids:
                logger.debug(f"Skipping duplicate function ID: {function_id}")
                continue

            # Add to the set of unique function IDs
            unique_function_ids.add(function_id)
            processed_count += 1

            # Extract complexity metrics
            complexity_info = problem.get("complexity_info", {})

            # Store function metadata
            function_metadata[function_id] = {
                "function_id": function_id,
                "complexity_metrics": {
                    "line_count": complexity_info.get("line_count", 0),
                    "code_line_count": complexity_info.get("code_line_count", 0),
                    "cyclomatic": complexity_info.get("cyclomatic", 1),
                },
                "difficulty_level": difficulty_level,
            }

            # Extract centrality metrics if available
            if "centrality" in complexity_info:
                centrality = complexity_info["centrality"]
                function_metadata[function_id]["complexity_metrics"]["centrality"] = {
                    "degree": centrality.get("degree", 0),
                    "in_degree": centrality.get("in_degree", 0),
                    "out_degree": centrality.get("out_degree", 0),
                    "betweenness": centrality.get("betweenness", 0.0),
                    "pagerank": centrality.get("pagerank", 0.0),
                    "harmonic": centrality.get("harmonic", 0.0),
                    "distance_discount": centrality.get("distance_discount", 0.0),
                }
            elif (
                "degree" in complexity_info
            ):  # Some files might have centrality directly in complexity_info
                function_metadata[function_id]["complexity_metrics"]["centrality"] = {
                    "degree": complexity_info.get("degree", 0),
                    "in_degree": complexity_info.get("in_degree", 0),
                    "out_degree": complexity_info.get("out_degree", 0),
                    "betweenness": complexity_info.get("betweenness", 0.0),
                    "pagerank": complexity_info.get("pagerank", 0.0),
                    "harmonic": complexity_info.get("harmonic", 0.0),
                    "distance_discount": complexity_info.get("distance_discount", 0.0),
                }

            # Extract Halstead metrics if available
            if "halstead_volume" in complexity_info:
                function_metadata[function_id]["complexity_metrics"][
                    "halstead_volume"
                ] = complexity_info.get("halstead_volume", 0.0)
            if "halstead_difficulty" in complexity_info:
                function_metadata[function_id]["complexity_metrics"][
                    "halstead_difficulty"
                ] = complexity_info.get("halstead_difficulty", 0.0)

            # Extract repository information if available
            if "repo" in problem:
                if "repo_stats" in problem["repo"]:
                    function_metadata[function_id]["repo_stats"] = problem["repo"][
                        "repo_stats"
                    ]
                elif "name" in problem["repo"]:
                    # Store the repository name for later lookup
                    function_metadata[function_id]["repo_name"] = problem["repo"][
                        "name"
                    ]

            # Extract test information if available
            if "test_info" in problem and isinstance(problem["test_info"], dict):
                test_info = problem["test_info"]

                # Store test information
                function_metadata[function_id]["test_info"] = {
                    "failed": test_info.get("failed", 0),
                    "total": test_info.get("passed", 0)
                    + test_info.get("failed", 0)
                    + test_info.get("deselected", 0),
                }

        logger.info(f"  Processed {processed_count} unique functions from {file_name}")

    logger.info(
        f"Extracted metadata for {len(function_metadata)} unique functions total"
    )
    return function_metadata


def extract_model_performance() -> Dict[str, Dict]:
    """
    Extract model performance scores from result files.

    Returns:
        Dictionary mapping function IDs to their model performance scores
    """
    logger.info("Extracting model performance from result files...")

    # Dictionary to store model performance for each function
    function_performance = {}

    # Process only result files that start with 'hard_' or 'hardest_'
    result_files = [
        f
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".jsonl") and (f.startswith("hard_") or f.startswith("hardest_"))
    ]
    logger.info(f"Found {len(result_files)} relevant result files")

    # Pattern to extract info from result filenames
    # Format: {difficulty}_{mode}_{model}_budget_{budget}.jsonl
    pattern = re.compile(r"(hard|hardest)_(discovery|remove)_(.+)_budget_(\d+)\.jsonl")

    # Store sample function IDs for debugging
    sample_original_ids = []

    for file_name in result_files:
        match = pattern.match(file_name)
        if not match:
            logger.warning(f"Skipping {file_name} - doesn't match expected pattern")
            continue

        difficulty, mode, model_name, budget = match.groups()

        # Normalize model name
        if model_name in MODEL_NAME_REVERSE:
            normalized_model_name = MODEL_NAME_REVERSE[model_name]
        else:
            logger.warning(f"Unknown model name: {model_name}, using as is")
            normalized_model_name = model_name

        file_path = os.path.join(RESULTS_DIR, file_name)
        logger.info(f"Processing {file_path}")

        # Read the JSONL file
        results = read_json_or_jsonl(file_path)
        if not results:
            continue

        # Remove the first line if it contains summary statistics
        if results and "total_problems" in results[0]:
            results = results[1:]

        logger.info(f"  Found {len(results)} function results in {file_name}")

        # Process each result
        for result in results:
            # Skip if no problem or score info
            if "problem" not in result or "score" not in result:
                continue

            # The original function ID from the result file
            original_function_id = result["problem"]

            # Store sample for debugging
            if (
                len(sample_original_ids) < 5
                and original_function_id not in sample_original_ids
            ):
                sample_original_ids.append(original_function_id)

            score = result["score"]

            # Initialize the function entry if it doesn't exist
            if original_function_id not in function_performance:
                function_performance[original_function_id] = {
                    "function_id": original_function_id,
                    "model_performance": {},
                }

            # Initialize the model entry if it doesn't exist
            if (
                normalized_model_name
                not in function_performance[original_function_id]["model_performance"]
            ):
                function_performance[original_function_id]["model_performance"][
                    normalized_model_name
                ] = {}

            # Store the performance score
            performance_key = f"{mode}_{budget}"
            function_performance[original_function_id]["model_performance"][
                normalized_model_name
            ][performance_key] = score

            # We're now handling test_info directly in extract_problem_metadata_with_repo_info

    logger.info(
        f"Extracted performance data for {len(function_performance)} unique functions"
    )
    logger.info("Sample original function IDs from performance data:")
    for i, id in enumerate(sample_original_ids):
        logger.info(f"  {i+1}. {id}")

    return function_performance


def extract_repo_stats() -> Dict[str, Dict]:
    """
    Extract repository statistics from repo_info files.

    Returns:
        Dictionary mapping repo names to their statistics
    """
    logger.info("Extracting repository statistics...")

    # Dictionary to store repo stats
    repo_stats = {}

    # Use repo_info files
    data_files = [
        "hardest_final_repo_info.jsonl",
        "less_hard_v2_no_repeats_repo_info.jsonl",
    ]

    for file_name in data_files:
        file_path = os.path.join(DATA_DIR, file_name)

        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Repo info file not found: {file_path}")
            continue

        logger.info(f"Processing {file_path}")

        # Read the file
        problems = read_json_or_jsonl(file_path)
        logger.info(f"  Found {len(problems)} problems in {file_name}")

        # Extract repo stats from each problem
        for problem in problems:
            if "repo" not in problem or "name" not in problem["repo"]:
                continue

            repo_name = problem["repo"]["name"]

            # Skip if we've already processed this repo
            if repo_name in repo_stats:
                continue

            # Check if repo_stats is available
            if "repo_stats" in problem["repo"]:
                repo_stats[repo_name] = problem["repo"]["repo_stats"]

    logger.info(f"Extracted statistics for {len(repo_stats)} repositories")
    return repo_stats


def merge_data(
    function_metadata: Dict[str, Dict],
    function_performance: Dict[str, Dict],
    repo_stats: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    Merge function metadata with performance data and repository statistics.

    Args:
        function_metadata: Dictionary mapping function IDs to their complexity metrics
        function_performance: Dictionary mapping function IDs to their model performance scores
        repo_stats: Dictionary mapping repo names to their statistics (optional)

    Returns:
        List of dictionaries with merged data for each function
    """
    logger.info("Merging metadata and performance data...")

    # Debug: print sample of function IDs from both dictionaries
    logger.info("Sample metadata function IDs:")
    for i, id in enumerate(list(function_metadata.keys())[:5]):
        logger.info(f"  {i+1}. {id}")

    logger.info("Sample performance function IDs:")
    for i, id in enumerate(list(function_performance.keys())[:5]):
        logger.info(f"  {i+1}. {id}")

    # Create a simple function name to full ID mapping
    function_name_to_metadata_id = {}
    for metadata_id in function_metadata:
        # Get just the function name (last part after last slash)
        parts = metadata_id.split("/")
        if len(parts) >= 1:
            function_name = parts[-1]
            function_name_to_metadata_id[function_name] = metadata_id

    logger.info(
        f"Created mapping for {len(function_name_to_metadata_id)} function names"
    )

    # Count matches
    matched_count = 0
    for perf_id in function_performance:
        if perf_id in function_name_to_metadata_id:
            matched_count += 1

    logger.info(f"Found {matched_count} functions with name-based matches")

    merged_data = []
    model_performance_count = 0

    # Process functions that have metadata information
    for function_id in function_metadata:
        # Get metadata for this function
        metadata = function_metadata[function_id]

        # Get just the function name for matching with performance data
        function_name = function_id.split("/")[-1]

        # Try to get performance data using the simple function name
        perf_id = None
        if function_name in function_performance:
            perf_id = function_name

        # Get performance data for this function if it exists
        if perf_id:
            performance = function_performance[perf_id]
            has_performance = True
            model_performance_count += 1
            if model_performance_count <= 5:
                logger.info(
                    f"Function {function_id} matched with performance data for {perf_id}: {list(performance['model_performance'].keys())}"
                )
        else:
            performance = {"function_id": function_id, "model_performance": {}}
            has_performance = False

        # Create merged entry
        merged_entry = {
            "function_id": function_id,
            "complexity_metrics": metadata.get("complexity_metrics", {}),
            "model_performance": performance.get("model_performance", {}),
            "difficulty_level": metadata.get("difficulty_level", "unknown"),
        }

        # Add repo stats if available
        if "repo_stats" in metadata:
            merged_entry["repo_stats"] = metadata["repo_stats"]
        elif repo_stats and "repo_name" in metadata:
            # Look up repo stats by repo name
            repo_name = metadata["repo_name"]
            if repo_name in repo_stats:
                merged_entry["repo_stats"] = repo_stats[repo_name]

        # Add test information if available
        if "test_info" in metadata:
            merged_entry["test_info"] = metadata["test_info"]

        merged_data.append(merged_entry)

    logger.info(f"Merged data for {len(merged_data)} functions")
    logger.info(
        f"Functions with actual model performance data: {model_performance_count}"
    )
    return merged_data


def save_to_jsonl(data: List[Dict], output_file: str) -> None:
    """
    Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        output_file: Path to the output file
    """
    logger.info(f"Saving data to {output_file}...")

    with open(output_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Saved {len(data)} entries to {output_file}")
