from __future__ import annotations

import json
import textwrap
from pathlib import Path
import sys

from inspect_ai import task, Task
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.dataset._dataset import Sample
from inspect_ai.scorer import Scorer, scorer, accuracy, stderr, Score
from inspect_ai.util import sandbox, SandboxEnvironmentSpec
from inspect_ai.solver import system_message, generate

COMPOSE_FILE = str(Path(__file__).parent.parent / "docker-compose.yml")
DEFAULT_DATASET = str(Path(__file__).parent.parent / "data/breakpoint-inspect.jsonl")

@scorer(metrics=[accuracy(), stderr()])
def tester():
    async def score(self, state):
        res = await sandbox().exec(["pytest", "-q"])
        passed = res.returncode == 0
        return Score(value=passed)
    return score


@task
def breakpoint(task_args={}) -> Task:
    """
    Breakpoint, single-function **removal** variant:
    the target function body is stubbed out; solver must rebuild it.
    """

    path = task_args.get("dataset", DEFAULT_DATASET)

    breakpoint_ds = json_dataset(
        path,
        FieldSpec(
            id="id",
            files="files",
            setup="setup",              # the exporter put a generic setup here
            metadata=[
                "repo",
                "function_name",
                "fpath",
                "corruption",           # → dict(code=..., other stuff…)
                "test_info",
            ],
        ),
    )

    
    for sample in breakpoint_ds:
        sample.sandbox = SandboxEnvironmentSpec(
            type="docker",
            config=COMPOSE_FILE
        )

    return Task(
        dataset=breakpoint_ds,
        scorer=tester(),
        epochs=1,
        metadata={"benchmark": "Breakpoint-Remove"},
        solver=[system_message("_")]
    )

