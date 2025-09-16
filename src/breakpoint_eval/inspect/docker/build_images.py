import base64
import functools
import logging
import os
import subprocess
from typing import Literal

import click
import tqdm.contrib.concurrent

from breakpoint_eval.problem import Problem

from ..single_fn_discovery.data import get_single_fn_discovery_problems
from ..single_fn_remove.data import get_single_fn_remove_problems
from .constants import BREAKPOINT_INSPECT_DOCKER_DIR

logger = logging.getLogger(__name__)


def build_image_for_problem(problem: Problem, verbose: bool = False) -> None:
    assert problem.repo.commit is not None, "Repository commit hash is required."

    if problem.mode not in {"remove", "discovery"}:
        raise NotImplementedError(f"Unsupported problem mode: {problem.mode}")

    diff_base64 = base64.b64encode(problem.get_diff().encode("utf-8")).decode("ascii")

    # Run docker build
    try:
        subprocess.run(
            [
                "docker",
                "build",
                str(BREAKPOINT_INSPECT_DOCKER_DIR),
                "-t",
                problem.docker_image_name,
                "--build-arg",
                f"REPO_URL={problem.repo.url}",
                "--build-arg",
                f"REPO_COMMIT_HASH={problem.repo.commit}",
                "--build-arg",
                f"DIFF_BASE64={diff_base64}",
            ],
            check=True,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        logger.error(f"Failed to build Docker image for problem {problem.id}.")
        raise


@click.command()
@click.option(
    "--task",
    type=click.Choice(["single_fn_discovery", "single_fn_remove"]),
    help="Name of the task to check images for.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit on number of problems to process. If None, all problems are used.",
)
@click.option(
    "--problem-id",
    "problem_ids",
    type=str,
    multiple=True,
    default=[],
    help="If provided, only build the images for the problems with the specified problem ids. Can be specified multiple times (e.g. --problem-id pid1 --problem-id pid2). These problem ids are used as sample ids in inspect. (Optional, default: [])",
)
@click.option(
    "--n-workers",
    type=int,
    default=max(os.cpu_count() // 2, 1),
    help="Number of worker processes to use for building images. (default: half of CPU cores)",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def main(
    task: Literal["single_fn_discovery", "single_fn_remove"],
    limit: int | None,
    problem_ids: list[str],
    n_workers: int,
    verbose: bool,
) -> None:
    """Build images for the specified task."""
    problems = dict(
        single_fn_discovery=get_single_fn_discovery_problems,
        single_fn_remove=get_single_fn_remove_problems,
    )[task]()

    if limit is not None:
        assert limit <= len(problems), (
            f"Requested {limit} problems, but only {len(problems)} available."
        )
        problems = problems[:limit]

    if len(problem_ids) > 0:
        matched_problems = [p for p in problems if p.id in problem_ids]
        assert len(matched_problems) == len(problem_ids), (
            f"One or more problem IDs not found in dataset: {problem_ids}."
        )

        problems = matched_problems

    tqdm.contrib.concurrent.process_map(
        functools.partial(build_image_for_problem, verbose=verbose),
        problems,
        max_workers=n_workers,
        desc="Building Docker images for problems",
        chunksize=1,
    )


if __name__ == "__main__":
    main()
