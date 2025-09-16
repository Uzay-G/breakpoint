import hashlib
import json
import logging
from typing import Literal

import datasets
from pydantic import BaseModel

from breakpoint_eval import codeparsing_utils_v2

logger = logging.getLogger(__name__)


class Repo(BaseModel):

    name: str | None = ""
    path: str | None = None
    code_path: str | None = None
    test_command: str | None = "source venv/bin/activate && ./venv/bin/pytest"
    stats: dict | None = None
    url: str | None = None
    commit: str | None = None

    def model_post_init(self, __context):
        if (self.url is not None and
            self.url.startswith("https://github.com/") and
            self.url.count("/") == 4 and 
            not self.url.endswith(".git")
        ):
            # Auto add .git suffix if not present
            logger.info(
                f"Automatically adding .git suffix to repo URL: {self.url} -> {self.url}.git"
            )
            self.url = self.url + ".git"

    @property
    def name_from_url(self) -> str | None:
        if self.url is None:
            return None
        # example url: https://github.com/UKGovernmentBEIS/inspect_ai.git
        assert self.url.endswith(".git"), "URL must end with .git"
        return self.url.split("/")[-1].removesuffix(".git")


class Problem(BaseModel):
    repo: Repo
    fpath: str
    function_name: str

    test_info: dict | None = None
    complexity_info: dict | None = None
    corruption: dict | None = None

    @property
    def json_sha256_hash(self) -> str:
        """
        Generate a unique hash for the problem based on its attributes.
        This is useful for caching and identifying problems.
        """
        instance_dict = self.model_dump()
        json_str = json.dumps(instance_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @property
    def mode(self) -> Literal["remove", "discovery"]:
        """
        Determine the mode of the problem based on the presence of corruption.
        """
        return "remove" if self.corruption is None else "discovery"

    @property
    def id(self) -> str:
        """
        Generate a unique identifier for the problem based on its attributes.
        """
        return "-".join(
            [
                self.mode,
                self.repo.name_from_url,
                self.function_name.split(".")[
                    -1
                ],  # Use the last part of the function name
                self.json_sha256_hash[:16],
            ]
        )

    @property
    def docker_image_name(self) -> str:
        """
        Generate a unique Docker image name based on the problem's attributes.
        """
        return f"breakpoint:{self.id}"

    def get_orig_impl(self) -> str:
        """
        Get the original implementation of the function from the repository at the specified commit.
        """
        return codeparsing_utils_v2.get_orig_fn_impl(
            repo_url=self.repo.url,
            commit_hash=self.repo.commit,
            file_path=self.fpath,
            function_name=self.function_name,
        )

    def get_diff(self, reverse: bool = False) -> str:
        """
        Generate a diff string for the problem, which can be used to apply changes
        to the codebase.

        reverse: If True, the diff will transform the modified version to the original
        version. This is useful as a reference solution.
        """

        match self.mode:
            case "remove":
                return codeparsing_utils_v2.get_remove_diff(
                    repo_url=self.repo.url,
                    commit_hash=self.repo.commit,
                    file_path=self.fpath,
                    function_to_remove=self.function_name,
                    reverse=reverse,
                )

            case "discovery":
                assert self.corruption is not None, (
                    "Corruption must be defined for discovery mode"
                )
                assert "code" in self.corruption, "Corruption must contain 'code' key"

                return codeparsing_utils_v2.get_replace_diff(
                    repo_url=self.repo.url,
                    commit_hash=self.repo.commit,
                    file_path=self.fpath,
                    function_to_replace=self.function_name,
                    new_impl=self.corruption["code"],
                    reverse=reverse,
                )

            case _:
                raise NotImplementedError(f"Unsupported problem mode: {self.mode}")

    def get_file_with_new_fn_impl(
        self,
        new_impl: str,
        file_path: str | None = None,
        function_name: str | None = None,
    ) -> str:
        """
        Get the file content with the new implementation of the function.
        """
        return codeparsing_utils_v2.get_file_with_new_fn_impl(
            repo_url=self.repo.url,
            commit_hash=self.repo.commit,
            file_path=self.fpath if file_path is None else file_path,
            function_name=self.function_name
            if function_name is None
            else function_name,
            new_impl=new_impl,
        )


class MultiProblem(BaseModel):
    problems: list[Problem]


class ProblemEnv(BaseModel):
    problem: Problem
    execution_dir: str


BROKEN_REPOS = [
    # pytm has an issue with `pip install -e .` in the Dockerfile
    # unclear what exactly the root cause is but removing these problems for now
    "pytm",
    # This repo requires internet access for tests
    "deep-translator",
    # simpleai repo has a test that is flaky (sometimes passes, sometimes fails)
    "simpleai",
    # Requires internet access for at least one test
    "ai-vocabulary-builder",
]

BROKEN_PROBLEM_IDS = [
    # These problems have tests that pass even when the function is removed / corrupted
    "remove-beartype-reduce_hint_pep695_unsubscripted-",
    "discovery-beartype-unpack_hint_or_sane-0105456d6470ab39",
    "discovery-beartype-_get_hint_pep695_parameterizable_typeparams-553a3c16dcffea99",
    "discovery-beartype-_resolve_func_scope_pep695-51319fcb76b7e0c7",
]


def get_problems_from_hf(
    repo: str,
    data_file: str,
    revision: str,
    split: str = "train",
) -> list[Problem]:
    """
    Load problems from a Hugging Face dataset.

    Args:
        repo (str): The Hugging Face repository name.
        data_file (str): The specific data file to load.

    Returns:
        list[breakpoint_eval.problem.Problem]: A list of Problem instances.
    """
    ds = datasets.load_dataset(
        repo,
        data_files=data_file,
        split=split,
        revision=revision,
    ).to_list()

    all_problems = [Problem.model_validate(p) for p in ds]

    problems = []
    for problem in all_problems:
        if problem.repo.name_from_url in BROKEN_REPOS:
            logger.info(
                f"Skipping problem {problem.id} from broken repo {problem.repo.name_from_url}."
            )
            continue

        if any(broken_id in problem.id for broken_id in BROKEN_PROBLEM_IDS):
            logger.info(f"Skipping broken problem {problem.id}.")
            continue

        problems.append(problem)

    return problems
