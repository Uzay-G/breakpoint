from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import SandboxEnvironmentSpec

import breakpoint_eval.problem
from breakpoint_eval.inspect.docker.utils import get_compose_file_path


def get_single_fn_discovery_problems(
    repo: str = "uzpg/breakpoint",
    data_file: str = "data/discovery-data.json",
    revision: str = "810c549be55ccbeba026a21fd41add4c5c4f0a71",
) -> list[breakpoint_eval.problem.Problem]:
    problems = breakpoint_eval.problem.get_problems_from_hf(
        repo=repo, data_file=data_file, revision=revision
    )

    for problem in problems:
        assert problem.mode == "discovery", (
            f"Problem mode must be 'discovery', got {problem.mode}"
        )

    return problems


def get_single_fn_discovery_dataset(
    problems: list[breakpoint_eval.problem.Problem] | None = None,
) -> MemoryDataset:
    """Get the OG breakpoint discovery dataset from Hugging Face."""
    if problems is None:
        problems = get_single_fn_discovery_problems()

    samples = []
    for problem in problems:
        prompt = f"""\
Your task is to fix a bug in a Python codebase. Your goal is to make all tests in the codebase pass. Follow these instructions carefully:

1. The codebase you will be working with is named `{problem.repo.name_from_url}`. A copy of the codebase is available at `/root/code`. This codebase is a git repository, so you can use git commands to navigate and manage the code if you would like. You can read and modify any files in this directory.

2. Currently, the tests in the codebase are failing. It is guaranteed that there is a single function in the codebase that can be edited to fix all the failing tests. Your task is to discover such a function and provide a fixed implementation that will make all tests pass.

3. To run tests, you can run the `pytest` command in the `/root/code` directory. This command will provide you with detailed feedback on which tests pass and which tests fail.

4. When you believe you have a working implementation, use the `submit` tool to submit your solution (see tool documentation for usage details). The `submit` tool takes as input a target function and a new implementation of the target function, and returns a pass/fail status with no detailed feedback. Success requires ALL tests to pass and there is no partial credit. If your submission fails, please revise your implementation and resubmit.

5. Important constraints:
   - Your implementation must be general-purpose, not hardcoded to specific test cases.
   - Do not modify the test framework or interfere with `pytest`.
   - You do not have internet access. Work only with the provided codebase and tools.
""".strip()

        samples.append(
            Sample(
                input=prompt,
                id=problem.id,
                metadata=dict(
                    problem=problem.model_dump(),
                    orig_impl=problem.get_orig_impl(),
                    json_sha256_hash=problem.json_sha256_hash,
                ),
                sandbox=SandboxEnvironmentSpec(
                    type="docker",
                    config=str(get_compose_file_path(problem)),
                ),
            )
        )

    return MemoryDataset(
        samples=samples,
        name="hf://uzpg/breakpoint/data/discovery-data.json",
    )
