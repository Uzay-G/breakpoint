import time
import asyncio
import threading
import json
from typing import Optional

from lib.codeparsing_utils import *
from lib.agents import *
from lib.problem_generator import Problem, ProblemEnv
import shutil
import sys
import datetime


class CorruptionBenchmark:
    def get_human_eval_instructions(self, problem, agent, worker_dir, mode="remove"):
        """
        Generate instructions for human evaluators.

        Args:
            problem: The problem to be evaluated
            agent: The agent with evaluation parameters
            worker_dir: The working directory where the problem is prepared
            mode: The evaluation mode ("remove", "corrupt", or "discovery")

        Returns:
            A formatted string with human evaluation instructions
        """
        function_def_with_docstring = get_source_code_from_name(
            os.path.join(worker_dir, problem.fpath), problem.function_name
        )

        # For discovery mode, don't reveal which function has the issue
        if mode == "discovery":
            instructions = f"""# Human Evaluation Task - Discovery Mode

You are an expert Python programmer tasked with diagnosing and fixing an issue in a codebase.

## The Problem

The test suite is currently failing with several test errors. Your task is to:
1. Explore the codebase to understand its structure
2. Identify which function is causing the test failures
3. Fix the implementation of the problematic function so that all tests pass

This is a discovery task - you need to identify which function is causing the problem first,
then repair it to make the tests pass.

## Test Failures

Here's information about some of the failing tests:

"""
            for fail in problem.test_info.get("failures_info", [])[:3]:
                instructions += f"```\n{fail[:2000]}\n```\n\n"

        # For target/remove or corrupt mode, provide function details
        elif agent.repair_mode == "target":
            instructions = f"""# Human Evaluation Task

You are an expert Python programmer tasked with repairing broken code.

## The Problem

Your goal is to fix the implementation of '{problem.function_name}' in '{problem.fpath}' so that all tests pass.

When this function was removed from the codebase, several test errors occurred.

## Function Information

Here's the function definition and docstring:

```python
{function_def_with_docstring}
```

## Test Failures

Here's information about some of the failing tests:

"""
            for fail in problem.test_info.get("failures_info", [])[:3]:
                instructions += f"```\n{fail[:2000]}\n```\n\n"
        else:
            instructions = f"""# Human Evaluation Task

You are an expert Python programmer tasked with diagnosing and fixing an issue in a codebase.

## The Problem

The test suite is currently failing with several test errors. Your task is to:
1. Explore the codebase to understand its structure
2. Analyze the code and test failures to find the root cause
3. Fix the issue by modifying the appropriate file(s)

## Function Information

The function that needs attention is '{problem.function_name}' in '{problem.fpath}'.

Here's the function definition and docstring:

```python
{function_def_with_docstring}
```

## Test Failures

Here's information about some of the failing tests:

"""
            for fail in problem.test_info.get("failures_info", [])[:3]:
                instructions += f"```\n{fail[:2000]}\n```\n\n"

        # Add common information for all modes
        instructions += f"""
## Test Budget

You have a test budget of {agent.test_budget} test runs.

## Running the Tests

To run the tests, use the following command in this directory:

```
{problem.repo.test_command}
```

## Submission

After you've fixed the code, please record your solution and process for comparison with AI model performance.
"""

        return instructions

    def __init__(
        self,
        problems: list[Problem],
        multi=1,
        test_command="source venv/bin/activate && ./venv/bin/pytest",
        num_workers=8,
    ):
        self.problems = problems
        self.total_problems = len(problems)
        self.current_index = 0
        self.results = {}
        self.successful_solves = 0
        self.num_workers = num_workers
        self.semaphore = asyncio.Semaphore(num_workers)
        self.test_command = test_command
        self.multi = multi

    def __len__(self):
        """Return the number of problems in the benchmark."""
        return self.total_problems

    def __iter__(self):
        """Make the benchmark iterable."""
        self.current_index = 0
        return self

    def __next__(self):
        """Get the next problem when iterating."""
        if self.current_index < self.total_problems:
            problem = self.problems[self.current_index]
            self.current_index += 1
            return problem
        raise StopIteration

    def get_success_rate(self):
        """Return the success rate of solved problems."""
        if not self.results:
            return 0.0
        return self.successful_solves / len(self.results)

    def get_summary(self):
        """Return a summary of benchmark results."""
        return {
            "total_problems": self.total_problems,
            "problems_attempted": len(self.results),
            "successful_solves": self.successful_solves,
            "success_rate": self.get_success_rate(),
            "results": self.results,
        }

    def reset(self):
        """Reset the benchmark results."""
        self.results = {}
        self.successful_solves = 0
        self.current_index = 0

    def prepare_env_for_problem(self, problem, mode):
        """
        Prepare the environment for a specific problem.

        This method:
        1. Creates a clean working directory for the problem
        2. Copies the repository files
        3. Sets up a virtual environment if needed

        Args:
            problem: The problem object containing repository information

        Returns:
            The path to the working directory for this problem
        """
        import os

        # Create a temporary working directory
        repo_path = problem.repo.path
        worker_dir = prepare_directory(problem.repo)
        abs_path = os.path.join(worker_dir, problem.fpath)
        removal_info = remove_functions_in_file(abs_path, problem.function_name)

        if mode != "remove":
            insert_function_code(
                problem.corruption["code"],
                removal_info["func_start"],
                removal_info["func_def_end"],
                removal_info["indent"],
                abs_path,
            )

        return worker_dir

    async def eval_solution(self, wdir, problem):
        score = 0.0

        test_output = await run_tests(wdir, self.test_command, log_id=wdir)
        if test_output.get("had_execution_error"):
            return score
        else:
            fails = test_output.get("failed", 0)
            score = 1 - min(fails / problem.errors_wo, 1)
            return score

    async def eval_problem(self, problem, agent, mode):
        async with self.semaphore:
            # Acquire a worker directory from the queue.

            worker_dir = self.prepare_env_for_problem(problem, mode)
            attempt = None

            if agent.model_interface.model_name == "human":
                # Create a README file for human evaluators with our factored-out function
                # Pass the current mode to the instructions generator
                instructions = self.get_human_eval_instructions(
                    problem, agent, worker_dir, mode=mode
                )

                # Write the instructions to a README.md file in the worker directory
                readme_path = os.path.join(worker_dir, "HUMAN_EVAL_README.md")
                with open(readme_path, "w") as readme_file:
                    readme_file.write(instructions)

                print(f"Problem instance stored in {worker_dir}")
                print(f"Human evaluation instructions provided in {readme_path}")
                print(f"Mode: {mode}")
                sys.exit(0)

            # Offload the blocking agent call.
            # Create a ProblemEnv object to pass to the agent

            problem_id = f"{problem.function_name}_{agent.model_interface.model_name}_{agent.repair_mode}_{agent.max_iterations}_{self.multi}_{agent.test_budget}"

            # Create a directory for storing problem data if it doesn't exist
            temp_dir = os.path.join(os.getcwd(), "temp_problems")
            os.makedirs(temp_dir, exist_ok=True)

            # Create a file path for the problem data
            problem_file_path = os.path.join(temp_dir, f"{problem_id}.json")

            # Check if the file exists and load the attempt if it does
            loaded = False

            if os.path.exists(problem_file_path):
                loaded = True
                logging.info(f"Loading saved attempt from {problem_file_path}")
                try:
                    with open(problem_file_path, "r") as f:
                        saved_data = json.load(f)
                        attempt = StudentAttempt(
                            problem_spec=saved_data["problem"],
                            student_solution="",  # This field isn't saved in the file
                            actual_solution="",
                            score=saved_data["score"],
                            metadata=saved_data["metadata"],
                        )
                        if saved_data["score"] is None:
                            attempt.score = 0
                            loaded = False

                        tool_calls = [
                            (x["args"]["file_path"], x["args"]["func_name"])
                            for x in attempt.metadata["tool_usage"]
                            if x["tool"] == "replace_function"
                        ]

                        right_function = tool_calls and tool_calls[-1] == (
                            problem.fpath,
                            problem.function_name,
                        )
                        attempt.metadata["right_function"] = right_function

                except Exception as e:
                    logging.info(
                        f"Failed to load saved attempt: {e}. Rerunning problem."
                    )
                    loaded = False

            if not loaded:
                # If file doesn't exist, run the agent
                problem_env = ProblemEnv(problem=problem, execution_dir=worker_dir)
                attempt = await agent(problem_env)

                tool_calls = [
                    (x["args"]["file_path"], x["args"]["func_name"])
                    for x in attempt.metadata["tool_usage"]
                    if x["tool"] == "replace_function"
                ]

                right_function = tool_calls and tool_calls[-1] == (
                    problem.fpath,
                    problem.function_name,
                )
                attempt.metadata["right_function"] = right_function

                result_data = {
                    "problem": attempt.problem_spec,
                    "score": attempt.score,
                    "metadata": attempt.metadata,
                }

                # Save the problem data to the file
                with open(problem_file_path, "w") as f:
                    json.dump(result_data, f)

            shutil.rmtree(worker_dir)

            return attempt

    async def run_eval(self, agent, mode="remove"):
        """
        Run the evaluation for a single agent on all problems concurrently.

        Args:
            agent: The agent to evaluate.
            num_workers: Maximum number of concurrent evaluations.

        Returns:
            A dictionary containing the evaluation results.
        """
        # Set up the semaphore to limit concurrency.
        if self.multi != 1:
            results = await self.run_eval_with_combined_corruptions(
                agent, mode, corruptions_per_group=self.multi
            )
            return results
        else:
            tasks = [
                asyncio.create_task(self.eval_problem(problem, agent, mode))
                for problem in self.problems
            ]
            results = await asyncio.gather(*tasks)

        return results

    async def run_eval_with_combined_corruptions(
        self, agent, mode="remove", corruptions_per_group=4
    ):
        """
        Run evaluation where corruptions with related functions in the same codebase
        are combined and applied successively for the model to solve.

        Args:
            agent: The agent to evaluate.
            mode: The mode for applying corruptions ("remove" or another mode).
            corruptions_per_group: Maximum number of corruptions to combine in a single group.

        Returns:
            A list of evaluation results for each corruption group.
        """
        # Group problems by repository and related functions
        corruption_groups = self._group_problems_by_function_relationship(
            corruptions_per_group
        )

        # Create tasks for each group of problems
        tasks = []

        for group in corruption_groups:
            task = asyncio.create_task(self._eval_combined_problems(group, agent, mode))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results

    def _group_problems_by_function_relationship(
        self, corruptions_per_group=4, correlated=False
    ):
        """
        Group problems based on function relationships within each repository.

        Args:
            corruptions_per_group: Maximum number of corruptions to combine in a single group.

        Returns:
            A list of lists, where each inner list contains problems with related functions.
        """

        # Group problems by repository
        problems_by_repo = {}
        for problem in self.problems:
            repo_path = os.path.join(problem.repo.path, problem.repo.code_path)
            if repo_path not in problems_by_repo:
                problems_by_repo[repo_path] = []
            problems_by_repo[repo_path].append(problem)

        # For each repository, build function graph and group related problems
        grouped_problems = []

        for repo_path, repo_problems in problems_by_repo.items():
            # Build function graph for this repository
            try:
                function_graph, _ = build_function_graph(repo_path)

                # Process each problem in this repository
                remaining_problems = repo_problems.copy()

                while remaining_problems:
                    # Take first problem as seed
                    base_problem = remaining_problems.pop(0)
                    current_group = [base_problem]

                    # Get the function key for the base problem
                    base_function_key = (
                        base_problem.fpath.replace(
                            base_problem.repo.code_path + "/", ""
                        ),
                        base_problem.function_name,
                    )

                    # Identify related functions based on graph
                    related_functions = self._get_related_functions(
                        base_function_key, function_graph
                    )

                    if correlated and len(related_functions) < corruptions_per_group:
                        continue

                    # Find other problems with functions related to the base problem
                    i = 0
                    while (
                        i < len(remaining_problems)
                        and len(current_group) < corruptions_per_group
                    ):
                        problem = remaining_problems[i]
                        problem_function_key = (
                            problem.fpath.replace(problem.repo.code_path + "/", ""),
                            problem.function_name,
                        )

                        # Check if this problem's function is related to the base problem
                        if problem_function_key in related_functions or (
                            not correlated
                        ):

                            current_group.append(remaining_problems.pop(i))
                        else:
                            i += 1

                    if len(current_group) == corruptions_per_group:
                        grouped_problems.append(current_group)

            except Exception as e:
                # If we can't build the function graph, create individual groups
                print(f"Error building function graph for {repo_path}: {str(e)}")
                continue

        return grouped_problems

    def _get_related_functions(self, function_key, function_graph, max_distance=3):
        """
        Get a set of functions related to the given function within a certain distance in the graph.

        Args:
            function_key: Tuple of (file_path, function_name) for the function
            function_graph: The function dependency graph
            max_distance: Maximum distance to consider functions related

        Returns:
            Set of related function keys
        """
        if function_key not in function_graph:
            return set()

        # BFS to find related functions within max_distance
        visited = {function_key}
        queue = [(function_key, 0)]  # (function_key, distance)
        related = set()

        while queue:
            current_key, distance = queue.pop(0)

            # If we've reached the max distance, don't explore further
            if distance >= max_distance:
                continue

            # Get neighbors (both callers and callees)
            neighbors = function_graph.get(current_key, set())
            callers = {
                caller
                for caller in function_graph
                if current_key in function_graph[caller]
            }

            all_neighbors = neighbors.union(callers)

            for neighbor in all_neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
                    related.add(neighbor)

        return related

    async def _eval_combined_problems(self, problem_group, agent, mode):
        """
        Evaluate a group of problems by applying corruptions successively.

        Args:
            problem_group: A list of problems with related functions.
            agent: The agent to evaluate.
            mode: The mode for applying corruptions.

        Returns:
            The evaluation result.
        """
        if not problem_group:
            return {}

        async with self.semaphore:
            base_problem = problem_group[0]
            worker_dir = self.prepare_env_for_problem(base_problem, mode)

            for problem in problem_group[1:]:
                self._apply_additional_corruption(worker_dir, problem)

            base_problem.test_info = await run_tests(
                worker_dir,
                base_problem.repo.test_command,
                base_problem.function_name + problem_group[1].function_name,
            )

            if agent.model_interface.model_name == "human":
                # Create a README file for human evaluators with our factored-out function
                # Pass the current mode to the instructions generator
                instructions = self.get_human_eval_instructions(
                    base_problem, agent, worker_dir, mode=mode
                )

                # Write the instructions to a README.md file in the worker directory
                readme_path = os.path.join(worker_dir, "HUMAN_EVAL_README.md")
                with open(readme_path, "w") as readme_file:
                    readme_file.write(instructions)

                print(f"Problem instance stored in {worker_dir}")
                print(f"Human evaluation instructions provided in {readme_path}")
                print(f"Mode: {mode}")
                sys.exit(0)

            problem_env = ProblemEnv(problem=base_problem, execution_dir=worker_dir)

            # Run the agent
            attempt = await agent(problem_env)

            # Add metadata about the combined corruption
            attempt.metadata["num_corruptions"] = len(problem_group)

            attempt.metadata["corruption_functions"] = [
                p.function_name for p in problem_group
            ]

            submit_calls = [
                (x["args"]["file_path"], x["args"]["func_name"])
                for x in attempt.metadata["tool_usage"]
                if x["tool"] == "replace_function"
            ]

            right_function = True

            for p in problem_group:
                if (p.fpath, p.function_name) not in submit_calls:
                    right_function = False

            attempt.metadata["right_function"] = right_function

            shutil.rmtree(worker_dir)

            return attempt

    def _apply_additional_corruption(self, worker_dir, problem):
        """
        Apply an additional corruption to the working directory.

        Args:
            worker_dir: The working directory.
            problem: The problem containing the corruption to apply.
        """
        import os

        # Get the file path
        file_path = os.path.join(worker_dir, problem.fpath)

        # Find and remove the function, then insert the corrupted version
        try:
            removal_info = remove_functions_in_file(file_path, problem.function_name)
            insert_function_code(
                problem.corruption["code"],
                removal_info["func_start"],
                removal_info["func_def_end"],
                removal_info["indent"],
                file_path,
            )
        except:
            print(f"Could not insert on {problem.function_name}")
