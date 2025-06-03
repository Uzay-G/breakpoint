#!/usr/bin/env python
import subprocess
import os
import asyncio
import argparse
from datetime import datetime

# Configuration
MODELS = ["o4-mini"]
REMOVE_DATA = "/home/ubuntu/breakpoint/data/new_all_functions_cleaner.json"
DISCOVERY_DATA = "/home/ubuntu/breakpoint/data/o4_corruptions.json"
OUTPUT_DIR = "/home/ubuntu/breakpoint/new_final_results/"
WORKERS = 30  # Default number of concurrent workers


async def run_experiment(
    mode, data_file, iterations, test_budget, n_problems=1, models=MODELS
):
    """Run evaluation experiment with specified parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/{mode}_{iterations}iter_{test_budget}tests_{timestamp}_{models}.jsonl"

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "runners.eval",
        "--data",
        data_file,
        "--output",
        output_file,
        "--mode",
        mode,
        "--max_iterations",
        str(iterations),
        "--test_budget",
        str(test_budget),
        "--n_problems",
        str(n_problems),
        "--workers",
        str(WORKERS),
        "--model",
    ] + MODELS

    print(f"Running: {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="/home/ubuntu/breakpoint",
    )

    stdout, stderr = await process.communicate()
    print(f"Experiment completed: {output_file}")
    print(stdout.decode())
    if stderr:
        with open(
            f"errors_{mode}_{iterations}iter_{test_budget}tests_{timestamp}.txt", "w"
        ) as f:
            print(stderr.decode())
            f.write(f"Errors: {stderr.decode()}")

    return output_file


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation experiments")
    parser.add_argument(
        "--test", action="store_true", help="Run a test with only 1 problem"
    )
    parser.add_argument(
        "--long", action="store_true", help="Run a test with only 1 problem"
    )
    args = parser.parse_args()

    n_problems = -1  # Default is already 1 problem per model

    print("Starting experiment series...")

    if args.test:
        # Just run one test experiment in remove mode
        print("\n=== Running Test Remove Mode Experiment ===")
        await run_experiment("discovery", DISCOVERY_DATA, 5, 1, n_problems=1)
        return

    elif args.long:
        await run_experiment("remove", REMOVE_DATA, 50, 5, 10)
        return

    # Run full set of experiments
    # Run remove mode experiments
    print("\n=== Running Remove Mode Experiments ===")
    await run_experiment("remove", REMOVE_DATA, 32, 8, n_problems)
    # await run_experiment("remove", REMOVE_DATA, 8, 2, n_problems)    # Run discovery mode experiments
    # await run_experiment("remove", REMOVE_DATA, 4, 1, n_problems)
    print("\n=== Running Discovery Mode Experiments ===")
    # await run_experiment("discovery", DISCOVERY_DATA, 32, 8, n_problems)
    # await run_experiment("discovery", DISCOVERY_DATA, 4, 1, n_problems)
    print("\nAll experiments completed!")


if __name__ == "__main__":
    asyncio.run(main())
