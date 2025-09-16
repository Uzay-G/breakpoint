import difflib
import textwrap

import requests

from breakpoint_eval.codeparsing_utils import extract_function_info


# XXX: Make this function async since it has network calls
def get_file_content(
    repo_url: str,
    commit_hash: str,
    file_path: str,
) -> str:
    """
    Fetches the content of a file from a GitHub repository at a specific commit.

    Args:
        repo_url (str): The URL of the GitHub repository.
        commit_hash (str): The commit hash to fetch the file from.
        file_path (str): The path to the file in the repository.

    Returns:
        str: The content of the file.
    """
    assert "github.com" in repo_url, "Only GitHub repositories are supported."
    raw_url = "/".join(
        [
            repo_url.replace("github.com", "raw.githubusercontent.com").removesuffix(
                ".git"
            ),
            commit_hash,
            file_path,
        ]
    )

    return requests.get(raw_url).text


def get_git_formatted_diff(
    orig_file_content: str,
    new_file_content: str,
    file_path: str,
):
    """
    Returns a git-formatted diff between the original and new content of a file.

    Args:
        orig_content (str): The original content of the file.
        new_content (str): The new content of the file.
        file_path (str): The path to the file in the repository.

    Returns:
        str: The git-formatted diff.
    """
    diff_lines = list(
        difflib.unified_diff(
            orig_file_content.splitlines(keepends=True),
            new_file_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
        )
    )

    if all(d.endswith("\n") for d in diff_lines):
        return "".join(diff_lines)

    # We need to handle the special case that some of the lines in the diff do not have a newline.
    assert not orig_file_content.endswith("\n") or not new_file_content.endswith("\n")
    assert sum(int(not d.endswith("\n")) for d in diff_lines) <= 2

    diff_lines = [
        (d if d.endswith("\n") else d + "\n\\ No newline at end of file\n")
        for d in diff_lines
    ]
    return "".join(diff_lines)


def get_remove_diff(
    repo_url: str,
    commit_hash: str,
    file_path: str,
    function_to_remove: str,
    reverse: bool = False,
) -> str:
    """
    Returns the patch that would need to be applied to the repo to remove the specified function or class method from the specified file.

    The returned patch can be applied using `git apply` or similar tools.

    reverse: If True, the diff will transform the modified version to the original
    version. This is useful as a reference solution.
    """
    orig_file_content = get_file_content(
        repo_url=repo_url, commit_hash=commit_hash, file_path=file_path
    )
    lines = orig_file_content.splitlines(keepends=True)

    # Get metadata about the target function without modifying the file.
    info = extract_function_info(orig_file_content, function_to_remove)
    assert info is not None

    def_end = info["func_def_end"]

    # Modify the definition end line to include a "pass".
    # We preserve any trailing comment if present.
    lines[def_end - 1] = lines[def_end - 1].rstrip() + "\n"
    new_indent = " " * (info["indent"] + 4)
    pass_line = f"{new_indent}pass\n"

    # Create new file content properly (lines already have newlines)
    new_file_content = "".join(
        lines[:def_end] + [pass_line] + lines[info["node_end_lineno"] :]
    )

    if reverse:
        new_file_content, orig_file_content = (orig_file_content, new_file_content)

    return get_git_formatted_diff(
        orig_file_content=orig_file_content,
        new_file_content=new_file_content,
        file_path=file_path,
    )


def get_replace_diff(
    repo_url: str,
    commit_hash: str,
    file_path: str,
    function_to_replace: str,
    new_impl: str,
    reverse: bool = False,
) -> str:
    """
    Returns the patch that would need to be applied to the repo to replace the specified function or class method from the specified file with the given new implementation.

    The returned patch can be applied using `git apply` or similar tools.

    reverse: If True, the diff will transform the modified version to the original
    version. This is useful as a reference solution.
    """
    orig_file_content = get_file_content(
        repo_url=repo_url, commit_hash=commit_hash, file_path=file_path
    )
    new_file_content = get_file_with_new_fn_impl(
        repo_url=repo_url,
        commit_hash=commit_hash,
        file_path=file_path,
        function_name=function_to_replace,
        new_impl=new_impl,
    )

    if reverse:
        new_file_content, orig_file_content = (orig_file_content, new_file_content)

    return get_git_formatted_diff(
        orig_file_content=orig_file_content,
        new_file_content=new_file_content,
        file_path=file_path,
    )


def get_orig_fn_impl(
    repo_url: str,
    commit_hash: str,
    file_path: str,
    function_name: str,
) -> str:
    """
    Fetches the original implementation of a function or class method from a GitHub repository
    at a specific commit.
    """
    orig_file_content = get_file_content(
        repo_url=repo_url, commit_hash=commit_hash, file_path=file_path
    )
    lines = orig_file_content.splitlines(keepends=True)

    # Get metadata about the target function without modifying the file.
    info = extract_function_info(orig_file_content, function_name)
    assert info is not None

    func_start = info["func_start"]
    func_end = info["node_end_lineno"]

    # This is the original function definition with decorators and docstring
    return "".join(lines[func_start:func_end])


def get_file_with_new_fn_impl(
    repo_url: str,
    commit_hash: str,
    file_path: str,
    function_name: str,
    new_impl: str,
) -> str:
    """
    Replaces the implementation of a function or class method in a GitHub repository
    at a specific commit with a new implementation.
    """
    orig_file_content = get_file_content(
        repo_url=repo_url, commit_hash=commit_hash, file_path=file_path
    )
    lines = orig_file_content.splitlines(keepends=True)

    # Get metadata about the target function without modifying the file.
    info = extract_function_info(orig_file_content, function_name)
    assert info is not None

    func_start = info["func_start"]
    func_end = info["node_end_lineno"]
    indent = " " * info["indent"]

    # Replace the function body with the new implementation
    lines = (
        lines[:func_start]
        + [
            indent + new_impl_line
            for new_impl_line in textwrap.dedent(new_impl).splitlines(keepends=True)
        ]
        + lines[func_end:]
    )

    return "".join(lines)
