import pathlib

import git

BREAKPOINT_ROOT = pathlib.Path(
    git.Repo(
        __file__,
        search_parent_directories=True,
    ).working_dir
)
