import json
from pathlib import Path
from typing import Iterable

from lib.agents import *
from lib.problem_generator import Problem, load_problems_from_json                  # your class
from lib.codeparsing_utils import remove_functions_in_file # already in repo
from lib.code_benchmark import CorruptionBenchmark
import re
import sys
import os, tarfile

# ---------------------------------------------------------------------
# Config – edit these to taste
OUT = Path("data/breakpoint-inspect.jsonl")
VENVS = True              # bundle a per‑sample venv setup script
CHECK_FAIL_FIRST = True   # fail the sample if initial tests unexpectedly pass
# ---------------------------------------------------------------------

def export(problems: Iterable[Problem], *, out_file: Path = OUT) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    bm = CorruptionBenchmark([])

    with out_file.open("w") as fh:
        for i, pb in enumerate(problems):
            # 1. Copy the repo snapshot to a tarball so Inspect can untar quickly
            mode = "remove"
            if pb.corruption:
              mode = "discovery"

            escaped_fpath = re.sub(r'[<>:"/\\|?*\s]', '_', pb.fpath)
            tar_name = f"{pb.repo.name}-{pb.function_name}-{escaped_fpath}-{mode}.tar.gz"
            tar_path = Path(pb.repo.path) / tar_name # or wherever you prefer

            wdir = bm.prepare_env_for_problem(pb, mode)

            if not tar_path.exists():
                # build the archive once
                with tarfile.open(tar_path, "w:gz") as tar:
                    tar.add(os.fspath(wdir), arcname=".")

            # 3. Inline setup shell
            setup = f"""#!/bin/bash
set -euo pipefail
tar -xf repo.tar.gz
cd {pb.repo.code_path}
python -m venv venv
source venv/bin/activate
[ -f requirements.txt ] && pip install -r requirements.txt
[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt
pip install pytest pytest-reportlog
{pb.repo.test_command} -q || true
"""

            sample = {
                "id": pb.function_name,
                "input": "Make the tests pass",
                "files"   : {"repo.tar.gz": str(tar_path)},
                "setup"   : setup if VENVS else "",
                "sandbox": "docker",
                "metadata": {
                    "repo"          : pb.repo.name,
                    "function_name" : pb.function_name,
                    "fpath"         : pb.fpath,
                    "corruption": pb.corruption,
                    "test_info": pb.test_info
                },
            }
            fh.write(json.dumps(sample) + "\n")

    print(f"Wrote {i+1} samples → {out_file}")

if __name__ == "__main__":
    export(load_problems_from_json(sys.argv[1]))
