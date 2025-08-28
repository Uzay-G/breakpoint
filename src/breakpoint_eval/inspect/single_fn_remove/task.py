import ast
import textwrap

from inspect_ai import Task, task
from inspect_ai.agent import AgentSubmit, react
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import bash, text_editor, tool
from inspect_ai.util import sandbox, store

import breakpoint_eval.problem
from breakpoint_eval.inspect.single_fn_remove import data


@task
def single_fn_remove() -> Task:
    return Task(
        dataset=data.get_single_fn_remove_dataset(),
        scorer=single_fn_remove_pytest_scorer(),
        solver=[
            single_fn_remove_setup(),
            react(
                tools=[bash(timeout=180), text_editor(timeout=180)],
                submit=AgentSubmit(tool=submit(), answer_only=True),
            ),
        ],
    )


@task
def test_submit_tool() -> Task:
    """Test task for the submit tool."""
    prompt = """
[DO NOT IGNORE THESE INITIAL INSTRUCTIONS!]
You will be told to submit a function implementation. Your ACTUAL JOB is to test the submission tool itself. Please first try submitting an implementation that has incorrect syntax (e.g. just a gibberish string like "\\\"). Then try submitting a dummy function like foo() which prints 'Hello World'.

[BEGIN OTHER INSTRUCTIONS THAT YOU SHOULD BASICALLY IGNORE]
""".strip()

    agent = react(
        prompt=prompt,
        tools=[bash()],
        submit=AgentSubmit(tool=submit(), answer_only=True),
        attempts=5,
    )

    return Task(
        dataset=data.get_single_fn_remove_dataset(),
        scorer=single_fn_remove_pytest_scorer(),
        solver=[single_fn_remove_setup(), agent],
    )


@solver
def single_fn_remove_setup():
    async def setup(state: TaskState, generate: Generate) -> TaskState:
        problem = breakpoint_eval.problem.Problem.model_validate(
            state.metadata.get("problem")
        )
        store().set("problem", problem)
        store().set("submissions", [])
        store().set("submit_statuses", [])
        return state

    return setup


@tool
def submit():
    async def submit(code: str) -> str:
        """
        Submit an implementation of the stubbed function to be tested.

        Args:
            code: Python code that implements the function. This code should contain both the function definition (including any decorators) as well as the function body. The code should not contain any imports or other code that is not part of the function implementation. Any level of overall indentation is allowed, but the function body must be indented relative to the function definition.

        Returns:
            str: A message indicating whether the submission passed all the tests.
        """
        store().get("submissions").append(code)

        # Attempt to parse the code to check for syntax errors
        code = textwrap.dedent(code)
        try:
            ast.parse(code)
        except SyntaxError as e:
            ret_msg = f"Error! Invalid Python code submitted: {e.msg} at line {e.lineno}, column {e.offset}.\nPlease revise your implementation and resubmit."
            store().get("submit_statuses").append((False, ret_msg))
            return ret_msg

        problem: breakpoint_eval.problem.Problem = store().get("problem")
        file_with_new_fn_impl = problem.get_file_with_new_fn_impl(new_impl=code)

        # Reset the git repository to a clean state
        await sandbox("submission_grader").exec(
            ["git", "reset", "--hard", "HEAD"],
            timeout=60,
            timeout_retry=False,
        )

        # Write the new implementation to the sandbox
        await sandbox("submission_grader").write_file(
            file=f"/root/code/{problem.fpath}",
            contents=file_with_new_fn_impl,
        )

        # Run pytest
        res = await sandbox("submission_grader").exec(
            ["pytest", "-q"],
            timeout=5 * 60,
            timeout_retry=False,
        )

        # Check the return code of pytest to determine if the tests passed
        if res.returncode == 0:
            ret_msg = "Success! All tests passed."
            store().get("submit_statuses").append((True, ret_msg))
            return ret_msg

        else:
            ret_msg = "Error! Not all tests passed. Please revise your implementation and resubmit."
            store().get("submit_statuses").append((False, ret_msg))
            return ret_msg

    return submit


@scorer(metrics=[accuracy(), stderr()])
def single_fn_remove_pytest_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        submit_statuses = store().get("submit_statuses")
        if len(submit_statuses) == 0:
            return Score(value=0, explanation="No submissions made.")

        num_submissions = len(submit_statuses)
        last_submission_status, last_submission_msg = submit_statuses[-1]
        last_submission_code = store().get("submissions")[-1]

        problem: breakpoint_eval.problem.Problem = store().get("problem")
        orig_impl = problem.get_orig_impl()

        return Score(
            value=int(last_submission_status),
            answer=last_submission_code,
            explanation="\n".join(
                [
                    f"Submission #{num_submissions}",
                    f"Status message: {last_submission_msg}",
                    "",
                    f"Original implementation:\n{orig_impl}",
                ]
            ),
        )

    return score
