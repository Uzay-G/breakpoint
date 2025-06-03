import asyncio
import os
import random
import pickle
import logging
import argparse
import json
import datetime
import shutil
from pydantic import BaseModel

from lib.codeparsing_utils import *
from lib.agents import *


class Repo(BaseModel):
    path: str
    code_path: str
    name: Optional[str] = ""
    test_command: Optional[str] = "source venv/bin/activate && ./venv/bin/pytest"
    url: Optional[str] = ""
    stats: Optional[Dict] = None


class Problem(BaseModel):
    repo: Repo
    fpath: str
    function_name: str
    test_info: Optional[Dict] = None
    complexity_info: Optional[Dict] = None
    corruption: Optional[Dict] = None


class MultiProblem(BaseModel):
    problems: List[Problem]


class ProblemEnv(BaseModel):
    problem: Problem
    execution_dir: str


class CodebaseCache:
    def __init__(
        self,
        repo,
        num_workers=30,
        test_command="source venv/bin/activate && ./venv/bin/pytest",
        centrality_alpha=0.8,
    ):
        self.repo = repo
        self.num_workers = num_workers
        self.test_command = test_command

        self.repo_functions = parse_repo_for_functions(
            self.repo.path + self.repo.code_path, build_dependency_info=False
        )

        self.repo.stats = get_repo_info(
            os.path.join(self.repo.path, self.repo.code_path)
        )

        # count total number of functions in repo
        num_functions = sum(len(v) for v in self.repo_functions.values())
        logging.info(f"Loaded {num_functions} functions from repo {self.repo.path}")

        self.function_graph, self.fn_complexity = build_function_graph(
            os.path.join(self.repo.path, self.repo.code_path), alpha=centrality_alpha
        )
        self.semaphore = asyncio.Semaphore(self.num_workers)
        self.processing_queue = set()
        self.banned_functions = set()
        self.fn_info_cache = {}

        self.viable_functions = []

        self.find_good_functions()

    def find_good_functions(
        self,
        complexity_metric="code_line_count",
        min_complexity=1,
        max_complexity=10000,
    ):
        """
        Find functions suitable for testing based on complexity and centrality.

        Args:
            alpha: Discount factor for centrality calculation
            complexity_metric: Which complexity metric to use ('line_count' or 'cyclomatic')
            min_complexity: Minimum acceptable complexity
            max_complexity: Maximum acceptable complexity
        """

        candidates = []

        for x in self.function_graph.keys():
            metrics = self.fn_complexity[x]
            # Default to line_count if the specified metric is not available
            if complexity_metric not in metrics:
                complexity = metrics.get("code_line_count", 0)
            else:
                complexity = metrics[complexity_metric]

            if min_complexity < complexity < max_complexity:
                candidates.append(x)

        viable_functions = []

        for cand in candidates:
            rel_file, func_name = cand
            # Check if the file is allowed and does not belong to tests
            if "test" in rel_file:
                continue

            viable_functions.append((rel_file, func_name))

        self.viable_functions = viable_functions
        logging.info(f"Found {len(self.viable_functions)} important functions")

    async def assess_function(self, function):
        """
        Given a function removal, run tests to assess its quality.
        """

        self.processing_queue.add(function)

        if function in self.fn_info_cache or function in self.banned_functions:
            return False

        # get worker dir, run tests
        worker_dir = prepare_directory(self.repo)

        test_output, _ = await test_without_function(
            function,
            worker_dir,
            self.repo.code_path,
            self.repo.test_command,
            restore=True,
        )

        shutil.rmtree(worker_dir)

        return test_output

    async def seed_functions(self, n: Optional[int], hardmode=False):
        print(n, hardmode)
        self.processing_queue = set()

        if hardmode:
            failures_lower_bound = 2
            complexity_lower_bound = 20
            centrality_lower_bound = 4
        else:
            failures_lower_bound = 5
            complexity_lower_bound = 5
            centrality_lower_bound = 1

        random.shuffle(self.viable_functions)

        queue = asyncio.Queue()
        for func in self.viable_functions:
            queue.put_nowait(func)

        async def worker(worker_id, stop_event):
            while not stop_event.is_set() and not queue.empty():
                try:
                    func = queue.get_nowait()  # Get next function without blocking
                    complexity = self.fn_complexity[func].get("code_line_count", 0)
                    centrality = self.fn_complexity[func]["centrality"].get("degree", 0)

                    if (
                        complexity >= complexity_lower_bound
                        and centrality >= centrality_lower_bound
                    ):
                        test_output = await self.assess_function(func)
                        num_failures = test_output["failed"]

                        logging.info(
                            f"Worker {worker_id}: After removing function '{func}', {test_output['failed']} tests failed."
                        )

                        if num_failures >= failures_lower_bound:
                            self.fn_info_cache[func] = test_output

                            logging.info(f"LENGTH: {len(self.fn_info_cache)}")

                            if n is not None and len(self.fn_info_cache) >= n:
                                stop_event.set()
                                return

                except asyncio.QueueEmpty:
                    return

        stop_event = asyncio.Event()
        await asyncio.gather(*[worker(i, stop_event) for i in range(self.num_workers)])

    def dump(self, filename):
        """
        Serializes selected attributes of the instance into a pickle file.
        Non-picklable fields (e.g., semaphore, num_workers) are omitted.
        """
        data = {
            "repo_path": self.repo.path,
            "code_path": self.repo.code_path,
            "fn_complexity": self.fn_complexity,
            "banned_functions": self.banned_functions,
            "fn_info_cache": self.fn_info_cache,
            "viable_functions": self.viable_functions,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Dumped instance data to {filename}")

    @classmethod
    def from_dump(cls, filename, num_workers=8):
        """
        Loads the instance from a previously dumped pickle file.
        The num_workers parameter is used to reinitialize the semaphore.
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance.repo = Repo(path=data["repo_path"], code_path=data["code_path"])
        instance.fn_complexity = data["fn_complexity"]
        instance.processing_queue = set()
        instance.banned_functions = data["banned_functions"]
        instance.fn_info_cache = data["fn_info_cache"]
        instance.viable_functions = data["viable_functions"]

        instance.num_workers = num_workers
        instance.semaphore = asyncio.Semaphore(instance.num_workers)
        logging.info(f"Loaded instance data from {filename}")
        return instance

    def get_problems(self):
        problems = []
        for fpath, fname in self.fn_info_cache.keys():
            problem = Problem(
                fpath=os.path.join(self.repo.code_path, fpath),
                function_name=fname,
                test_info=self.fn_info_cache[(fpath, fname)],
                repo=self.repo,
                complexity_info=self.fn_complexity[(fpath, fname)],
            )
            problems.append(problem)
        return problems


class SubtleInverseGenerator:
    def __init__(
        self,
        problems: dict,
        cutoff=50,
        model="claude-3-7-sonnet-latest",
        test_budget=3,
        max_iterations=5,
        num_workers=5,
        iterate_with_tests=False,
        num_corruptions=-1,
    ):
        self.cutoff = cutoff
        self.model = model
        self.test_budget = test_budget
        self.max_iterations = max_iterations
        self.problems = problems
        self.iterate_with_tests = iterate_with_tests
        self.semaphore = asyncio.Semaphore(num_workers)
        self.num_corruptions = num_corruptions

    async def generate_dataset(self):
        """
        Generate a dataset of subtle inversions in parallel.

        Args:
            n (int): Number of functions to process.

        Returns:
            A list of dictionaries with keys "file_path", "fn", and "corrupted_code".
        """

        # Process at most n functions from the list of important functions
        logging.info(
            f"Selected {self.num_corruptions} functions from {len(self.problems)} available functions\n"
        )
        if self.num_corruptions == -1:
            selected_problems = random.sample(self.problems, len(self.problems))
        else:
            selected_problems = random.sample(self.problems, self.num_corruptions)

        # Create tasks for all function inversions
        logging.info(
            f"\n{'='*80}\nStarting corruption generation for {self.num_corruptions} functions\n{'='*80}"
        )
        tasks = [self._generate(problem) for problem in selected_problems]

        results = await asyncio.gather(*tasks)
        results = [x for x in results if x]

        logging.info(f"\n{'='*80}\nCORRUPTION GENERATION SUMMARY\n{'='*80}")

        avg_score = (
            sum(res.corruption["score"] for res in results) / len(results)
            if results
            else 0
        )
        logging.info(
            f"Average score: {avg_score:.2f} (lower is better for corruptions)"
        )

        return results

    async def _generate(self, problem: Problem):
        """
        For the given function (tuple of relative file and function name),
        read its code, extract the function definition, and call the model to produce
        a subtle inversion (i.e. a minimal modification that will break some tests).

        Args:
            fn (tuple): (relative file path, function name)

        Returns:
            The attempt object from the agent containing the modified function code.
        """
        # Prepare working directory (helper assumed)
        async with self.semaphore:

            worker_dir = prepare_directory(problem.repo)
            if not worker_dir:
                return False

            file_path = os.path.join(problem.repo.path, problem.fpath)

            # Extract the target function's code (helper assumed)
            fn_info = extract_function_info(
                open(file_path, "r").read(), problem.function_name
            )
            try:
                func_code = fn_info["func_code"]
            except:
                logging.info(
                    f"Could not extract info of function {problem.function_name} from file {file_path}"
                )
                return None

            # Create a system prompt instructing the model to introduce a subtle bug.
            test_examples = ""
            failed_tests = problem.test_info["failed_tests"]

            if len(failed_tests) == 0 or not "failures_info" in problem.test_info:
                return False

            for test_file, test_function in failed_tests[:5]:
                test_file_path = os.path.join(problem.repo.path, test_file)
                print(test_file_path)
                test_fn_info = extract_function_info(
                    open(test_file_path, "r").read(), test_function
                )

                if test_fn_info is None:
                    logging.info(
                        f"Failed to retrieve test code for {test_file_path, test_function}"
                    )
                    continue

                test_examples += f"Test function {test_function} in {test_file}:\n{test_fn_info['func_code']}\n\n"

            instructions = f"""
                You are an expert Python programmer tasked with introducing a bug in a function located in {os.path.join(problem.repo.code_path, problem.fpath)}.

                Your goal is to modify the following function so that some of the test cases fail. DO NOT INDICATE WITH COMMENTS OR GIVE ANY CLUE AS TO WHERE THE BUG IS, OR WHERE YOU MODIFIED THE FUNCTION, as your modified function will be a test for another agent.

                Use the following heuristic: if another agent were to get this codebase without knowing where the corruption is, and were to see the new test errors, it would be hard to find the function where it was caused and it would be hard to then, given the function, fix the problem with it.
                Your goal is to make it subtle, not necessarily to make all the original tests fail, although at least 2 tests should fail.

                Inserting some esoteric add-on or string that will show up in tests and can easily be found via searching the code would also not work well with this heuristic.

                For example, your corruption should involve the actual behavior of the function, a bad idea for this heuristic would just be a syntax error or making the function empty or whatever.

                A good idea using this heuristic would be to see how the function interacts with the rest of the code and identify an area where you can subtly damage the code.

                Motivate your corruptions by looking at how the tests work and what can make them fail.

                Do not change the function definition, just copy it over. 

                Keep asking yourself: If I saw this error in this codebase, would I easily fix it? If yes, iterate and do better.

                Function to corrupt:
                {func_code}

                To help guide you to what is being tested, here are some relevant tests that you can try to make fail.

                {test_examples}

                You have access to tool that allow you to do the following:
                
                1. Explore the repo structure
                2. Read and analyze code files
                3. Search for patterns
                4. Submit attempted patches for the function, and then get test feedback (max {self.test_budget} times). Reflect on the errors and if they are subtle/hard to debug. You can call tools at most {self.max_iterations} times.
                
                Remember that the only way to submit your code is via the submit_attempt tool. Call one tool at a time. Again, DO NOT INDICATE WITH COMMENTS OR GIVE ANY CLUE AS TO WHERE THE BUG IS, OR WHERE YOU MODIFIED THE FUNCTION.

                IMPORTANT: You will use only the submit_attempt tool to provide your corrupted function.
                After each submission, you'll receive direct feedback about which tests passed and failed.
                Your goal is to create a subtle corruption that causes tests to fail (score of 0.0 is perfect).
            """

            # Initialize an agent (using the CodeAgent class) repurposed for inversion generation.
            agent = CodeAgent(
                model_name=self.model,
                test_budget=self.test_budget,
                max_iterations=self.max_iterations,
                mode="break",
                instructions=instructions,
                tool_use=False,
                iterate_with_tests=self.iterate_with_tests,
            )

            env = ProblemEnv(problem=problem, execution_dir=worker_dir)

            attempt = await agent._run_agent(env)

            if (
                not attempt.metadata["test_info"]["had_execution_error"]
                and attempt.score != 1.0
                and attempt.student_solution != ""
            ):
                problem.corruption = {
                    "code": attempt.student_solution,
                    "score": attempt.score,
                }

                problem.test_info = attempt.metadata["test_info"]
                return problem
            else:
                return None


def load_problems_from_json(json_file_path):
    """
    Load repository and function information from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing repository information.

    Returns:
        dict: A dictionary of repository information and functions.
    """

    with open(json_file_path, "r") as f:
        data = json.load(f)

    problems = []
    for problem in data:
        repo = Repo(
            path=problem["repo"]["path"],
            name=problem["repo"]["name"],
            code_path=problem["repo"].get("name", ""),
            url=problem["repo"].get("url", ""),
            test_command=problem["repo"].get(
                "test_command", "source venv/bin/activate && ./venv/bin/pytest"
            ),
            stats=problem["repo"].get("stats", {}),
        )
        problem = Problem(
            repo=repo,
            fpath=problem["fpath"],
            function_name=problem.get("function_name"),
            test_info=problem.get("test_info", {}),
            complexity_info=problem.get("complexity_info", {}),
            corruption=problem.get("corruption", {}),
        )
        problems.append(problem)

    return problems


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/corruption_debugger_{timestamp}.log"
    print(log_filename)

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w", 
    )

    parser = argparse.ArgumentParser(
        description="Run different generator tasks on demand."
    )
    parser.add_argument(
        "--dump_file",
        type=str,
        default=None,
        help="Path to the json dump file to save the corruptions",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "corrupt",
        help="Generate a dataset with LLM generated corruptions.",
    )
    dataset_parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to the pickle dump file for the generator (e.g., eval_beartype.pkl)",
    )
    dataset_parser.add_argument(
        "--num_corruptions",
        type=int,
        required=True,
    )
    dataset_parser.add_argument(
        "--iterate_with_tests",
        action="store_true",
        help="If set, allows the model to iterate on test feedback without using tools",
    )
    # Subparser for the cache generation command using CodebaseCache
    cache_parser = subparsers.add_parser(
        "cache", help="Run CodebaseCache tasks on demand."
    )

    cache_parser.add_argument(
        "--repo_path",
        type=str,
        default="/home/ubuntu/repos/beartype/",
        help="Repository path (default: /home/ubuntu/repos/beartype/)",
    )
    cache_parser.add_argument(
        "--code_path",
        type=str,
        default="beartype/",
        help="Code path relative to the repo (default: beartype/)",
    )
    cache_parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of concurrent workers (default: 8)",
    )
    cache_parser.add_argument(
        "--load_file",
        type=str,
        default=None,
        help="Path to the pickle dump file to load an existing instance",
    )
    cache_parser.add_argument(
        "--num_functions",
        type=int,
        default=0,
        help="Number of functions to seed (default: 0, meaning no seeding)",
    )

    cacheall_parser = subparsers.add_parser(
        "cacheall",
        help="Process multiple repositories and create a combined function cache",
    )
    cacheall_parser.add_argument(
        "--repos_dir",
        type=str,
        required=True,
        help="Directory containing multiple repositories",
    )
    cacheall_parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file with combined function information",
    )
    cacheall_parser.add_argument(
        "--config_json",
        type=str,
        help="Optional JSON file with repository-specific configurations",
    )
    cacheall_parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of concurrent workers per repository (default: 8)",
    )
    cacheall_parser.add_argument(
        "--functions_per_repo",
        type=int,
        default=None,
        help="Number of functions to select per repository. If None, other selection methods are used.",
    )
    cacheall_parser.add_argument(
        "--hard", action="store_true", help="Use hard mode for function selection"
    )
    corruptall_parser = subparsers.add_parser(
        "corruptall",
        help="Generate corruptions for functions from multiple repositories",
    )
    corruptall_parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the JSON file generated by the cacheall command",
    )
    corruptall_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file to save the corruptions (pickle format)",
    )
    corruptall_parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-latest",
        help="Model to use for corruption generation",
    )
    corruptall_parser.add_argument(
        "--iterate_with_tests",
        action="store_true",
        help="If set, allows the model to iterate on test feedback without using tools",
    )
    corruptall_parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of concurrent workers for corruption generation",
    )
    corruptall_parser.add_argument(
        "--num_corruptions",
        type=int,
        default=-1,
        help="Number of functions to corrupt. Default is -1 (all functions).",
    )
    args = parser.parse_args()

    async def main():
        if args.command == "corrupt":
            problems = load_problems_from_json(args.cache_path)
            generator = SubtleInverseGenerator(
                problems=problems,
                iterate_with_tests=args.iterate_with_tests,
                num_workers=2,
            )

            dataset = await generator.generate_dataset(args.num_corruptions)

            if dataset:
                logging.info("\n\n" + "=" * 100)
                logging.info("GENERATED CORRUPTIONS")
                logging.info("=" * 100)

                for i, corruption in enumerate(dataset):
                    logging.info(f"\n\nCORRUPTION #{i+1}:")
                    logging.info(f"File: {corruption['file_path']}")
                    logging.info(f"Function: {corruption['fn']}")
                    logging.info(
                        f"Score: {corruption['score']} (0.0 is perfect corruption)"
                    )
                    logging.info("\nCorrupted Code:")
                    logging.info("-" * 80)
                    logging.info(corruption["corrupted_code"])
                    logging.info("-" * 80)
            else:
                logging.info("No corruptions were generated.")

            if args.dump_file:
                with open(args.dump_file, "w") as f:
                    json.dumps(dataset, f)

        elif args.command == "cache":

            print(args.repo_path, args.code_path)
            repo = Repo(path=args.repo_path, code_path=args.code_path)
            repo.url = get_origin_url(args.repo_path)
            crg = CodebaseCache(repo=repo, num_workers=args.num_workers)
            logging.info("Created a new CodebaseCache instance.")

            if args.num_functions:
                await crg.seed_functions(args.num_functions)
                logging.info(f"Seeded {args.num_functions} functions.")

            problems = crg.get_problems()
            with open(args.dump_file, "w") as f:
                f.write(json.dumps([x.model_dump() for x in problems]))
                logging.info(f"Dumped instance state to {args.dump_file}")

        elif args.command == "cacheall":

            problems = []
            if os.path.exists(args.output_json):
                problems = json.load(open(args.output_json, "r"))

            repos = set(["MCDReforged"])
            for problem in problems:
                repos.add(problem["repo"]["name"])

            repo_dirs = [
                d
                for d in os.listdir(args.repos_dir)
                if os.path.isdir(os.path.join(args.repos_dir, d))
            ]

            for repo_name in repo_dirs:
                if repo_name in repos:
                    continue
                repo_path = os.path.join(args.repos_dir, repo_name)

                if not repo_path.endswith("/"):
                    repo_path += "/"

                logging.info(
                    f"\n{'='*80}\nProcessing repository: {repo_name}\n{'='*80}"
                )
                venv_path = os.path.join(repo_path, "venv")

                # Get repository configuration or use default
                install_cmd = f"source {venv_path}/bin/activate && ./venv/bin/pip install pytest-reportlog"
                logging.info("Installing pytest-reportlog...")
                process = await asyncio.create_subprocess_shell(
                    install_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    executable="/bin/bash",
                    cwd=repo_path,
                )
                await process.communicate()

                code_path = None
                test_command = "source venv/bin/activate && ./venv/bin/pytest"
                n_functions = args.functions_per_repo

                # If code_path is not specified, try to detect it
                if not code_path:
                    try:
                        potential_code_dirs = [
                            d
                            for d in os.listdir(repo_path)
                            if os.path.isdir(os.path.join(repo_path, d))
                            and not d.startswith(".")
                            and d
                            not in ["test", "tests", "venv", "env", "docs", "examples"]
                        ]

                        if len(potential_code_dirs) == 1:
                            code_path = potential_code_dirs[0] + "/"
                            logging.info(f"Detected code path: {code_path}")
                        elif repo_name.lower() in os.listdir(repo_path):
                            code_path = repo_name.lower() + "/"
                            logging.info(
                                f"Using repository name as code path: {code_path}"
                            )
                        else:
                            # Look for directory with most Python files
                            py_file_counts = {}
                            for d in potential_code_dirs:
                                dir_path = os.path.join(repo_path, d)
                                py_files = []
                                for root, _, files in os.walk(dir_path):
                                    py_files.extend(
                                        [f for f in files if f.endswith(".py")]
                                    )
                                py_file_counts[d] = len(py_files)

                            # Also count python files in the root directory
                            root_py_files = [
                                f for f in os.listdir(repo_path) if f.endswith(".py")
                            ]
                            py_file_counts[""] = len(
                                root_py_files
                            )  # Empty string represents root

                            # Find directory with most Python files
                            if py_file_counts:
                                best_dir = max(
                                    py_file_counts.items(), key=lambda x: x[1]
                                )
                                if (
                                    best_dir[1] > 0
                                ):  # If there's at least one Python file
                                    code_path = best_dir[0] + "/" if best_dir[0] else ""
                                    logging.info(
                                        f"Using directory with most Python files ({best_dir[1]}): {code_path}"
                                    )
                                else:
                                    print(
                                        f"Skipping {repo_path} for lack of Python files"
                                    )
                                    continue
                            else:
                                print(
                                    f"Skipping {repo_path} for lack of clear source directory"
                                )
                                continue

                    except Exception as e:
                        logging.error(f"Error detecting code path: {str(e)}")
                        continue

                repo_info = Repo(
                    name=repo_name,
                    path=repo_path,
                    code_path=code_path,
                    test_command=test_command,
                )

                repo_info.url = get_origin_url(repo_path)
                repo_code_info = get_repo_info(os.path.join(repo_path, code_path))

                print(repo_code_info)
                if repo_code_info["functions_count"] < 100 or not repo_info.url:
                    continue

                try:
                    logging.info(
                        f"Attempting test-based function selection for {repo_name}"
                    )

                    cache = CodebaseCache(
                        repo=repo_info,
                        num_workers=args.num_workers,
                        test_command=test_command,
                    )

                    await cache.seed_functions(n_functions, hardmode=args.hard)

                    problems += [x.model_dump() for x in cache.get_problems()]

                    logging.info(f"Test-based selection succeeded for {repo_name}.")

                except Exception as e:
                    logging.info(f"Error creating codebase cache for {repo_name}")

                # Write the combined data to a JSON file
                with open(args.output_json, "w") as f:
                    json.dump(problems, f, indent=2)

            # Log summary
            logging.info(f"\n{'='*80}\nSUMMARY\n{'='*80}")
            logging.info(f"Processed {len(repo_dirs)} repositories.")
            logging.info(f"Combined data written to {args.output_json}")

        elif args.command == "corruptall":
            if os.path.exists(args.input_json):
                problems = load_problems_from_json(args.input_json)

            corruptor = SubtleInverseGenerator(
                problems=problems,
                model=args.model,
                iterate_with_tests=args.iterate_with_tests,
                num_workers=args.num_workers,
                num_corruptions=args.num_corruptions,
            )

            results = await corruptor.generate_dataset()

            with open(args.output, "w") as f:
                f.write(json.dumps([x.model_dump() for x in results]))

        else:
            parser.error("Invalid command provided.")

    asyncio.run(main())
