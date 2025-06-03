import asyncio
import logging
import json
import datetime

from lib.agents import CodeAgent
from lib.problem_generator import Problem, load_problems_from_json
from lib.code_benchmark import CorruptionBenchmark

from typing import List
from dataclasses import dataclass
import argparse
from pathlib import Path
import random


def dump_results(results, output_path, config):
    # Summarize results
    success_count = sum(r.score for r in results)
    total_count = len(results)
    success_rate = success_count / total_count if total_count > 0 else 0

    # Calculate additional statistics
    avg_score = sum(r.score for r in results) / total_count if total_count > 0 else 0
    perfect_solves = sum(1 for r in results if r.score == 1.0)

    # Create summary dictionary
    summary = {
        "total_problems": total_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "perfect_solves": perfect_solves,
        "perfect_solve_rate": perfect_solves / total_count if total_count > 0 else 0,
        "avg_score": avg_score,
        "config": config,
    }

    with open(output_path, "a") as f:
        # Write summary at the top
        f.write(json.dumps(summary) + "\n")
        # Write individual results
        for r in results:
            result_data = {
                "problem": r.problem_spec,
                "score": r.score,
                "metadata": r.metadata,
                "config": config,
            }
            f.write(json.dumps(result_data) + "\n")


@dataclass
class EvalConfig:
    model_names: List[str]
    # models to evaluate on
    problems: List[Problem]
    output: str  # e.g. "eval_results.jsonl"
    num_workers: int = 15
    max_iterations: int = 8
    test_budget: int = 3
    mode: str = "all"  # or one of ["remove", "corrupt", "discovery"]
    tool_use: bool = True
    n_problems: int = 1
    log_file: str = ""
    thinking_budget: int = 10000
    multi: int = 1


async def run_eval(config: EvalConfig):
    """
    For a list of functions (and optionally their corruption data),
    get their info and eval results, and store it in a jsonl file,
    one line per function.
    """

    benchmark = CorruptionBenchmark(
        problems=config.problems, num_workers=config.num_workers, multi=args.multi
    )

    results = {}
    modes = []
    if config.mode == "all":
        modes = ["remove", "discovery"]
    else:
        modes = [config.mode]

    for mode in modes:
        mode_results = {}
        repair_mode = "target" if mode in ["remove", "corrupt"] else "discovery"

        for model_name in config.model_names:
            agent = CodeAgent(
                model_name=model_name,
                max_iterations=config.max_iterations,
                test_budget=config.test_budget,
                repair_mode=repair_mode,
                tool_use=config.tool_use,
                thinking_budget=config.thinking_budget,
            )

            logging.info("Starting evaluation")
            print(config.num_workers, "FROM EVAL")
            mode_results[model_name] = await benchmark.run_eval(agent, mode=mode)

            info_dict = {
                "model_name": model_name,
                "mode": mode,
                "max_iterations": config.max_iterations,
                "thinking_budget": config.thinking_budget,
                "test_budget": config.test_budget,
                "tools_allowed": [x["function"]["name"] for x in agent._get_tools()],
                "log_file": config.log_file,
            }

            dump_results(mode_results[model_name], config.output, info_dict)
        results[mode] = mode_results


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/eval-{timestamp}.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    parser = argparse.ArgumentParser(description="Run code benchmark evaluation")

    parser.add_argument(
        "--data",
        type=str,
        default="/home/ubuntu/breakpoint/data/new_all_functions_cleaner.json",
        help="Path to the evaluation data json file",
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"results/gpt-nano-budget_10_test_budget_3_debug.jsonl",
        help="Output file for results",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=8, help="Max agent iterations"
    )
    parser.add_argument("--test_budget", type=int, default=3)
    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--n_problems", default=-1, type=int)
    parser.add_argument("--thinking_budget", default=7500)
    parser.add_argument(
        "--function_name",
        type=str,
        help="Target a specific function (only works when n_problems=1)",
    )
    parser.add_argument(
        "--multi",
        type=int,
        default=1,
        help="For discovery, apply multiple corruptions at once",
    )
    parser.add_argument(
        "--model",
        default=["gpt-4.1-nano"],
        type=str,
        nargs="+",
        help="Model name(s) to evaluate. Can provide multiple models.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logging.info(f"Error: Data file {data_path} not found")
        exit(1)

    with open(args.output, "w") as f:
        f.write("")

    problems = load_problems_from_json(args.data)
    random.shuffle(problems)

    if args.n_problems == 1 and args.function_name:
        matching_problems = [
            x for x in problems if x.function_name == args.function_name
        ]
        if matching_problems:
            problems = matching_problems
        else:
            logging.info(
                f"Warning: No function named '{args.function_name}' found in problems"
            )

    if args.n_problems != -1:
        problems = problems[: args.n_problems]

    if args.function_name:
        problems = [x for x in problems if x.function_name == args.function_name]

    config = EvalConfig(
        model_names=args.model,
        problems=problems,
        output=args.output,
        mode=args.mode,
        max_iterations=args.max_iterations,
        test_budget=args.test_budget,
        num_workers=args.workers,
        n_problems=args.n_problems,
        log_file=log_file,
        thinking_budget=args.thinking_budget,
        multi=args.multi,
        tool_use=True,
    )

    asyncio.run(run_eval(config))
