"""
To run the reference solution for the Breakpoint task, use the following command:
    python -m breakpoint_eval.inspect.single_fn_discovery.ref_soln
"""

import textwrap

from inspect_ai import Task, task
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import sandbox, store

import breakpoint_eval.problem
from breakpoint_eval.inspect.single_fn_discovery import data
from breakpoint_eval.inspect.single_fn_discovery.task import (
    single_fn_discovery_pytest_scorer,
    single_fn_discovery_setup,
    submit,
)


@task
def single_fn_discovery_ref_soln() -> Task:
    """Reference solution for the breakpoint task."""
    dataset = data.get_single_fn_discovery_dataset()

    return Task(
        dataset=dataset,
        scorer=[single_fn_discovery_pytest_scorer(), tests_fail_initially()],
        solver=[single_fn_discovery_setup(), breakpoint_ref_soln_solver()],
    )


@scorer(metrics=[accuracy(), stderr()])
def tests_fail_initially():
    async def score(state: TaskState, target: Target) -> Score:
        res = await sandbox().exec(["pytest", "-q"])

        # Score of 1 if tests fail initially.
        return Score(value=0 if res.returncode == 0 else 1)

    return score


@solver
def breakpoint_ref_soln_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        problem: breakpoint_eval.problem.Problem = store().get("problem")

        # Submit the original implementation of the function.
        await submit()(
            file_path=problem.fpath,
            function_name=problem.function_name,
            code=textwrap.indent(
                text=problem.get_orig_impl(),
                prefix=3
                * " ",  # We use a random prefix to test the prefix removal logic.
            ),
        )

        return state

    return solve
