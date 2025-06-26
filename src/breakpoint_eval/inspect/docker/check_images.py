import os
from typing import Literal

import click
import inspect_ai

from ..single_fn_discovery.ref_soln import single_fn_discovery_ref_soln
from ..single_fn_remove.ref_soln import single_fn_remove_ref_soln


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
    help="Limit the number of problems to process. (Optional, default: None)",
)
@click.option(
    "--problem-id",
    "problem_ids",
    type=str,
    multiple=True,
    default=[],
    help="If provided, only check the images for the problems with the specified problem ids. Can be specified multiple times (e.g. --problem-id pid1 --problem-id pid2). These problem ids are used as sample ids in inspect. (Optional, default: [])",
)
def main(
    task: Literal["single_fn_discovery", "single_fn_remove"],
    limit: int | None,
    problem_ids: list[str],
):
    """Check images for the specified task."""
    if limit is not None and len(problem_ids) > 0:
        raise ValueError(
            "Cannot specify both --limit and --problem-id. Use one or the other."
        )

    n_cpus = os.cpu_count()
    assert n_cpus is not None, "Could not determine number of CPUs."

    inspect_ai.eval(
        tasks=dict(
            single_fn_discovery=single_fn_discovery_ref_soln,
            single_fn_remove=single_fn_remove_ref_soln,
        )[task](),
        limit=limit,
        sample_id=None if len(problem_ids) == 0 else problem_ids,
        model="mockllm/model",
        max_connections=max(n_cpus // 4, 10),
    )


if __name__ == "__main__":
    main()
