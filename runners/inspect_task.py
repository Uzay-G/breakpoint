from __future__ import annotations

import json
import textwrap
from pathlib import Path

from inspect_ai import task, Task, Sample
from inspect_ai.dataset import json_dataset, FieldSpec
from inspect_ai.scorer import Scorer
from inspect_ai.util import sandbox, SandboxEnvironmentSpec


# ---------------------------------------------------------------------
# 1.  Load the raw dataset ------------------------------------------------
# ---------------------------------------------------------------------
DATASET_PATH = Path("datasets/breakpoint.jsonl")

breakpoint_ds = json_dataset(
    DATASET_PATH,
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


# ---------------------------------------------------------------------
# 2.  Small helper that patches the generic setup script ---------------
# ---------------------------------------------------------------------
def add_patch_to_setup(sample: Sample, *, mode: str) -> Sample:
    # prepend the patch before the original venv / pip / pytest part
    sample.setup = patch_snippet + "\n" + (sample.setup or "")
    # ensure sandboxing; we can stay with Inspect's default image
    sample.sandbox = sample.sandbox or SandboxEnvironmentSpec(type="docker")
    return sample


def slice_and_patch(mode: str):
    """Return a lazily-mapped dataset that patches each sample on the fly."""
    return breakpoint_ds.map(lambda s: add_patch_to_setup(s, mode=mode), lazy=True)


# ---------------------------------------------------------------------
# 3.  Minimal scorer: pytest exit code ---------------------------------
# ---------------------------------------------------------------------
class BreakpointPytestScorer(Scorer):
    async def score(self, state):
        res = await sandbox().exec(["pytest", "-q"])
        passed = res.returncode == 0
        return passed, {"stdout": res.stdout[:10_000]}


# ---------------------------------------------------------------------
# 4.  Public tasks ------------------------------------------------------
# ---------------------------------------------------------------------

@task
def breakpoint_remove() -> Task:
    """
    Breakpoint, single-function **removal** variant:
    the target function body is stubbed out; solver must rebuild it.
    """
    return Task(
        dataset=slice_and_patch(mode="remove"),
        scorer=BreakpointPytestScorer(),
        epochs=1,
        metadata={"benchmark": "Breakpoint-Remove"},
    )


@task
def breakpoint_corrupt() -> Task:
    """
    Breakpoint, single-function **corruption** variant:
    we overwrite the original implementation with the corrupted one
    shipped in sample.metadata["corruption"]["code"].
    """
    return Task(
        dataset=slice_and_patch(mode="corrupt"),
        scorer=BreakpointPytestScorer(),
        epochs=1,
        metadata={"benchmark": "Breakpoint-Corrupt"},
    )
