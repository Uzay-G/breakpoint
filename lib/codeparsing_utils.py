import os
import ast
import json
import re
import shutil
import io
import tokenize
import warnings
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import deque
import asyncio
import tempfile
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
import subprocess
import textwrap
import code2flow
import numpy as np


def unindent_code(source_code):
    """
    Remove common leading whitespace from all lines of code.
    This makes pre-indented code blocks suitable for ast.parse().
    """
    # Use textwrap.dedent to remove common leading whitespace
    dedented_code = textwrap.dedent(source_code)
    cleaned_code = dedented_code.strip()

    return cleaned_code


def copy_repo(src_dir: str, dst_dir: str) -> None:
    """
    Recursively copy the entire repository from src_dir to dst_dir.
    If dst_dir already exists, it will be removed first.
    """
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)


def get_origin_url(directory):
    try:
        # Extract repo name from directory path
        repo_name = os.path.basename(os.path.normpath(directory))

        # Read saved_repos.txt to find matching repo
        if os.path.exists("saved_repos.txt"):
            with open("saved_repos.txt", "r", encoding="utf-8") as f:
                for line in f:
                    user_repo = line.strip()
                    if "/" in user_repo and repo_name == user_repo.split("/")[1]:
                        # Construct GitHub URL from username/repo_name
                        return f"https://github.com/{user_repo}.git"

        # If no match found in saved_repos.txt, fall back to git command
        result = subprocess.check_output(
            ["git", "-C", directory, "config", "--get", "remote.origin.url"],
            stderr=subprocess.STDOUT,
        )
        return result.decode("utf-8").strip()

    except (subprocess.CalledProcessError, FileNotFoundError, IOError) as e:
        print(f"Failed to get origin url for {directory}: {str(e)}")
        return ""


def gther_python_files(repo_dir: str) -> list:
    """
    Recursively collect all Python (.py) files from the repository.
    """
    python_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


class FunctionDefInfo:
    """
    Holds metadata about a function or class definition:
      - name: function/class name (string)
      - start_line: the line number where the def starts
      - end_line: the line number where the def ends (exclusive)
      - depends_on: optional set of function/class names it calls (for a dependency graph)
      - docstring: optional
    """

    def __init__(
        self,
        name: str,
        start_line: int,
        end_line: int,
        depends_on: Optional[Set[str]] = None,
        docstring: Optional[str] = None,
    ):
        self.name = name
        self.wandb_enabled = True
        self.start_line = start_line
        self.end_line = end_line
        self.depends_on = depends_on if depends_on else set()
        self.docstring = docstring if docstring else None

    def __repr__(self):
        return (
            f"FunctionDefInfo(name={self.name}, lines=[{self.start_line},"
            f"{self.end_line}), depends_on={self.depends_on}, docstring={self.docstring})"
        )


def parse_node(node: ast.AST, build_dependency_info: bool = False) -> FunctionDefInfo:
    depends_on = set()
    if build_dependency_info:
        depends_on = gather_function_calls(node)

    method_info = FunctionDefInfo(
        # Prefix with class name for clarity
        name=node.name,
        start_line=node.lineno,
        end_line=node.end_lineno + 1 if hasattr(node, "end_lineno") else node.lineno,
        depends_on=depends_on,
        docstring=ast.get_docstring(node),
    )

    return method_info


def parse_python_file(
    file_path: str, build_dependency_info: bool = False
) -> List[FunctionDefInfo]:
    """
    Parse a single Python file (file_path), returning a list of FunctionDefInfo
    for each top-level function definition in that file.

    If build_dependency_info is True, we also attempt a naive
    "which functions does this function call?" analysis by scanning AST calls.

    Note: This only captures *top-level* (module-level) function defs,
    as well as top-level methods in classes.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Use Python's ast to find function definitions
        tree = ast.parse(source_code)

        func_infos = []
        # We only look for top-level functions and classes
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # check the number of lines to decide
                if (
                    isinstance(node, ast.ClassDef)
                    and node.end_lineno - node.lineno > 50
                ):
                    # process class methods separately
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = parse_node(item, build_dependency_info)
                            # prefix class name
                            method_info.name = f"{node.name}.{method_info.name}"
                            func_infos.append(method_info)

                method_info = parse_node(node, build_dependency_info)
                func_infos.append(method_info)

        # Sort them by start line for easier processing
        func_infos.sort(key=lambda x: x.start_line)
        return func_infos

    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        # Log the error and return an empty list
        print(f"Error parsing {file_path}: {e}")
        return []


def gather_function_calls(func_def_node: ast.AST) -> Set[str]:
    """
    Given an AST node (typically FunctionDef, AsyncFunctionDef, or ClassDef),
    gather function names that appear in calls within that node's body.

    Returns a set of function names that this node depends on.

    This approach captures:
    - Direct function calls: func()
    - Method calls: obj.method(), self.method()
    - Class methods: ClassName.method()
    """
    calls = set()

    for child in ast.walk(func_def_node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                # Direct function call: func()
                calls.add(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                # Method call: obj.method()
                if isinstance(child.func.value, ast.Name):
                    obj_name = child.func.value.id
                    method_name = child.func.attr

                    # For self.method() calls, store just the method name
                    if obj_name == "self":
                        calls.add(method_name)
                    # For super().method() calls
                    elif obj_name == "super":
                        calls.add(method_name)
                    # For Class.method() or module.function() calls
                    else:
                        # Store as qualified name
                        calls.add(f"{obj_name}.{method_name}")

                # Handle chained calls like obj.attr.method()
                elif isinstance(child.func.value, ast.Attribute):
                    # At minimum record the method name
                    calls.add(child.func.attr)

    return calls


def gather_python_files(repo_dir: str) -> list:
    """
    Recursively collect all Python (.py) files from the repository.
    """
    python_files = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def build_function_graph(
    repo_dir: str, alpha=0.5
) -> Tuple[Dict[Tuple[str, str], Set[Tuple[str, str]]], Dict[Tuple[str, str], dict]]:
    """
    Build a function dependency graph across the entire repository.
    """

    # Get function info using existing parser
    repo_functions = parse_repo_for_functions(repo_dir)

    # Create function name to path mapping
    name_to_path = {}
    for rel_path, func_infos in repo_functions.items():
        for func_info in func_infos:
            name_to_path[func_info.name] = rel_path

    # Run code2flow
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Find Python files
        python_files = []
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        print(repo_dir, python_files)
        code2flow.code2flow(python_files, temp_path)

        # Load code2flow output
        with open(temp_path, "r") as f:
            flow_data = json.load(f)

        nodes = flow_data.get("graph", {}).get("nodes", {})
        edges = flow_data.get("graph", {}).get("edges", [])

        # Extract function names from nodes
        node_names = {}  # node_id -> qualified_name
        for node_id, data in nodes.items():
            name = data.get("name", "")
            qualified_name = name.split("::")[1] if "::" in name else name
            node_names[node_id] = qualified_name

        # Build function graph with our keys
        fgraph = {}

        for qualified_name in set(node_names.values()):
            rel_path = name_to_path.get(qualified_name, "unknown.py")
            fgraph[(rel_path, qualified_name)] = set()

        # Add edges
        for edge in edges:
            source_name = node_names.get(edge.get("source"))
            target_name = node_names.get(edge.get("target"))

            if source_name and target_name:
                if source_name == target_name:
                    continue

                source_path = name_to_path.get(source_name, "unknown.py")
                target_path = name_to_path.get(target_name, "unknown.py")

                source_key = (source_path, source_name)
                target_key = (target_path, target_name)

                if source_key in fgraph:
                    fgraph[source_key].add(target_key)

        # Calculate centrality
        centrality = compute_centrality(fgraph, alpha=alpha)

        # Compute complexity metrics
        complexity_metrics = {}
        for key in fgraph.keys():
            rel_path, qualified_name = key
            metrics = {"centrality": centrality.get(key, 0)}

            # Try to find function info and compute complexity
            if rel_path in repo_functions:
                for func_info in repo_functions[rel_path]:
                    if (
                        func_info.name == qualified_name
                    ):  # ignores duplicated functions so this is fine
                        # Add line count
                        metrics["line_count"] = (
                            func_info.end_line - func_info.start_line
                        )

                        # Add code line count
                        source_code = get_source_code_from_name(
                            os.path.join(repo_dir, rel_path), qualified_name
                        )
                        metrics["code_line_count"] = count_lines(
                            source_code, code_only=True
                        )

                        # Try to get complexity metrics
                        try:
                            file_path = os.path.join(repo_dir, rel_path)
                            source = get_source_code_from_name(
                                file_path, qualified_name
                            )
                            if source:
                                metrics.update(calculate_function_complexity(source))
                        except Exception:
                            pass
                        break

            complexity_metrics[key] = metrics

        return fgraph, complexity_metrics

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def parse_repo_for_functions(
    repo_dir: str, build_dependency_info: bool = False
) -> Dict[str, List[FunctionDefInfo]]:
    """
    Recursively parse all .py files in `repo_dir`,
    returning a dict:
       { relative_file_path: [FunctionDefInfo, ...], ... }
    """
    repo_functions = {}
    for root, _, files in os.walk(repo_dir):
        for filename in files:
            if filename.endswith(".py"):
                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, repo_dir)
                func_infos = parse_python_file(abs_path, build_dependency_info)
                if func_infos:
                    repo_functions[rel_path] = func_infos
    return repo_functions


def extract_function_info(source: str, target_fn: str) -> Optional[Dict[str, Any]]:
    """
    Parse the source code and extract metadata for the specified function or class.
    This helper does not modify the code; it only returns info.

    Args:
        source (str): The source code of the file.
        target_fn (str): Name of the function, class, or class method to process.
                         Format can be 'function_name', 'class_name', or 'class_name.method_name'.

    Returns:
        dict: A dictionary containing metadata about the function/class or None if not found.
    """
    lines = source.splitlines(keepends=True)
    try:
        tree = ast.parse(source)
    except:
        warnings.warn("Failed to parse the source code")
        return None

    # Helper function to extract common information from nodes
    def extract_node_info(node, display_name=None):
        """Extract common metadata from an AST node"""
        # Get indentation from the definition line
        def_line = lines[node.lineno - 1]
        indent = len(def_line) - len(def_line.lstrip(" "))

        # Get decorator start line (if any)
        dec_start = (
            min([d.lineno for d in node.decorator_list])
            if node.decorator_list
            else node.lineno
        )
        dec_start -= 1  # Convert to 0-based indexing

        # Check for docstring and determine the body start
        has_docstring = (
            len(node.body) > 0
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        )

        if has_docstring:
            doc_end = node.body[0].end_lineno
        else:
            doc_end = node.body[0].lineno - 1 if node.body else node.end_lineno

        return {
            "func_name": display_name or node.name,
            "indent": indent,
            "func_start": dec_start,
            "func_def_body": "".join(lines[dec_start:doc_end]),
            "func_code": "".join(lines[node.lineno - 1 : node.end_lineno]),
            "func_def_end": doc_end,
            "node_end_lineno": node.end_lineno,  # needed for later file modification
        }

    # Parse the target name
    parts = target_fn.split(".")

    # Handle class method case
    if len(parts) == 2:
        class_name, method_name = parts
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method_node in node.body:
                    if (
                        isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and method_node.name == method_name
                    ):
                        return extract_node_info(
                            method_node, f"{class_name}.{method_name}"
                        )
        warnings.warn(f"Method {method_name} not found in class {class_name}")

    # Handle regular function or class case
    elif len(parts) == 1:
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                and node.name == target_fn
            ):
                return extract_node_info(node)
        warnings.warn(f"Definition '{target_fn}' not found")

    # Invalid format
    else:
        warnings.warn(
            "Invalid function name format. Expected 'function_name', 'class_name', or 'class_name.method_name'"
        )


def remove_functions_in_file(file_path: str, function_to_remove: str) -> dict:
    """
    Modifies `file_path` in-place, removing the body of the specified function
    (replacing it with a "pass") while keeping its definition line (including
    multi-line arguments), decorators, and docstring.

    This function calls a helper that extracts metadata about the function without
    modifying the code, and then uses that info to remove the body.

    Returns:
        dict: Metadata about the modified function.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    lines = source.splitlines(keepends=True)

    # Get metadata about the target function without modifying the file.
    info = extract_function_info(source, function_to_remove)

    if not info:
        return None

    def_end = info["func_def_end"]

    # Modify the definition end line to include a "pass".
    # We preserve any trailing comment if present.
    target_line = lines[def_end - 1].rstrip()
    new_indent = " " * (info["indent"] + 4)
    pass_line = f"{new_indent}pass\n"

    # if len(split_line) > 1:
    #     lines[def_end - 1] = f"{split_line[0]} pass # {split_line[1]}\n"
    # else:3. uyes
    #     lines[def_end - 1] = f"{split_line[0]} pass\n"

    # Write the file back:
    # Keep all lines up to the modified definition, then skip the original body lines.
    new_lines = lines[:def_end] + [pass_line] + lines[info["node_end_lineno"] :]
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return info


def strip_ansi(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def parse_pytest_json_report(
    log: str,
    stderr: str = None,
    detailed_failures: int = 2,
) -> Dict:
    """
    Parse a pytest JSON report log file and return test execution statistics.
    """
    failed = 0
    passed = 0
    failures_info = []
    had_execution_error = False
    error_message = ""
    failed_tests = []
    n_lines_summary = 50

    if stderr:
        had_execution_error = True
        error_message = strip_ansi(stderr)

    for line in log.splitlines():
        if not line.strip():
            continue

        try:
            entry = json.loads(line)

            if entry.get("$report_type") == "TestReport":
                if entry["outcome"] == "failed":
                    failed += 1

                    if failed <= detailed_failures:
                        test_name = entry["nodeid"]
                        failure_detail = f"\n{'='*60}\n"
                        failure_detail += f"Failure #{failed}: {test_name}\n"

                        parts = entry["nodeid"].split("::")
                        test_file = parts[0] if len(parts) > 0 else ""
                        test_function = parts[-1] if len(parts) > 1 else ""
                        failed_tests.append((test_file, test_function))

                        if "longrepr" in entry:
                            longrepr = entry["longrepr"]
                            if isinstance(longrepr, dict):
                                # Get the crash information
                                crash_info = longrepr.get("reprcrash", {})
                                error_path = crash_info.get("path", "")
                                error_line = crash_info.get("lineno", "")
                                error_msg = crash_info.get("message", "")

                                failure_detail += (
                                    f"Location: {error_path}:{error_line}\n"
                                )
                                failure_detail += f"Error: {error_msg}\n\n"
                            else:
                                # Fallback for string representation

                                failure_detail += "\n".join(
                                    longrepr.split("\n")[-n_lines_summary:]
                                )

                        failures_info.append(failure_detail)
                    else:
                        # Simple summary for remaining failures
                        failures_info.append(f"Failure #{failed}: {entry['nodeid']}")

                elif entry["outcome"] == "passed":
                    passed += 1

            elif entry.get("$report_type") == "CollectReport":
                if entry["result"] == "error" or entry["outcome"] == "failed":
                    with open("error.log", "w") as f:
                        f.write(json.dumps(log))
                    had_execution_error = True
                    if "longrepr" in entry:
                        if type(entry["longrepr"]) is dict:
                            error_message = entry["longrepr"]["reprcrash"]["message"]
                        else:
                            error_message = entry["longrepr"]

        except json.JSONDecodeError:
            continue

    success = failed == 0 and not had_execution_error

    result = {
        "success": success,
        "failed": failed,
        "passed": passed,
        "deselected": 0,
        "failures_info": failures_info,
        "had_execution_error": had_execution_error,
        "error_message": error_message,
        "failed_tests": failed_tests,
    }

    if type(result["error_message"]) is dict:
        result["error_message"] = error_message.get("message", "")

    return result


def apply_code_with_indentation(
    insertion_code: str, insertion_indent: int
) -> List[str]:
    """
    Takes a block of 'insertion_code', which might have unknown or inconsistent indentation,
    and reflows it so that the *first 'def' line* is aligned with 'insertion_indent' spaces,
    and subsequent lines are relative to that.

    Returns a new list of lines to append or insert.
    """
    # 1) Split the insertion_code into lines
    code_lines = insertion_code.splitlines()

    # 2) Detect the indentation in the code's actual first line
    #    We'll guess how many leading spaces it has:
    if not code_lines:
        return ""

    user_first_line = code_lines[0]
    user_indent_count = len(user_first_line) - len(user_first_line.lstrip())

    # 3) If we want the entire block to have 'insertion_indent' as the base,
    #    we compute the difference (delta).
    delta = insertion_indent - user_indent_count

    # 4) We'll re-indent each line by 'delta'. If delta is positive, we add spaces. If negative, we remove spaces.
    adjusted_lines = []
    for line in code_lines:
        # measure leading spaces of the user line
        leading = len(line) - len(line.lstrip())
        new_indent = leading + delta
        if new_indent < 0:
            new_indent = 0
        # re-strip and re-add new_indent spaces
        stripped_line = line.lstrip()
        adjusted_line = (" " * new_indent) + stripped_line
        adjusted_lines.append(adjusted_line)

    return "\n".join(adjusted_lines)


def insert_function_code(
    new_function_code: str, start_line: int, end_skip_range: int, indent: int, path: str
) -> str:
    """
    Splices `new_function_code` lines into the original file content
    at the same range (start_line to end_line) from which the original
    function was removed.

    Args:
        original_content: the file's text (before removal).
        new_function_code: text of the new function body.
                          (We assume 'def foo(...)' is included.)
        start_line, end_line: 1-based indices specifying the removed range
                              in the original content. [start_line, end_line)
        indent: how many spaces were on the original 'def' line.
                We'll re-indent new_function_code to match this.

    Returns:
        A string with the new function code spliced in at the same location,
        with indentation adjusted.
    """

    adjusted_fn = apply_code_with_indentation(new_function_code, indent) + "\n"
    old_lines = open(path, "r", encoding="utf-8").readlines()
    new_lines = []

    i = 0
    while i < len(old_lines):
        if i == start_line:
            new_lines.append(adjusted_fn)
        elif i >= start_line and i < end_skip_range + 1:
            pass
        else:
            new_lines.append(old_lines[i])
        i += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(new_lines))


def get_venv_env(venv_dir):
    """
    Return a copy of the current environment with the virtual environment activated.
    """
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = venv_dir
    env["PATH"] = os.path.join(venv_dir, "bin") + os.pathsep + env.get("PATH", "")
    return env


def setup_repo_env(repo_dir):

    venv_dir = os.path.join(repo_dir, "venv")
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        try:
            subprocess.run(["python3", "-m", "venv", venv_dir], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment {e}")
            return False
    else:
        print("Virtual environment already exists; skipping creation.")

    venv_env = get_venv_env(venv_dir)

    # Install dependencies from requirements.txt if it exists.
    requirements_path = os.path.join(repo_dir, "requirements.txt")
    dev_req = os.path.join(repo_dir, "requirements-dev.txt")
    dev_req2 = os.path.join(repo_dir, "requirements/requirements_dev.txt")
    if os.path.exists(requirements_path):
        print(f"Installing dependencies from requirements.txt...")
        try:
            subprocess.run(
                ["pip3", "install", "-r", requirements_path], check=True, env=venv_env
            )
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies for  {e}")
            return False
    try:
        subprocess.run(
            ["pip3", "install", "-e", "."], check=True, env=venv_env, cwd=repo_dir
        )
    except:
        pass
    try:
        subprocess.run(["pip3", "install", "-r", dev_req], check=True, env=venv_env)
    except:
        pass
    try:
        subprocess.run(["pip3", "install", "-r", dev_req2], check=True, env=venv_env)
    except:
        pass

    try:
        subprocess.run(["pip", "install", "pytest"], check=True, env=venv_env)
    except subprocess.CalledProcessError as e:
        print(f"Error installing pytest for {e}")
        return False
    return venv_env


def prepare_directory(repo):
    # Create a temporary worker directory
    worker_dir = tempfile.mkdtemp(prefix="benchmark_")

    # Attempt to clean .pyc files from the worker directory
    command = 'find . -type f -name "*.pyc" -delete'
    result = subprocess.run(command, cwd=worker_dir, shell=True, stderr=subprocess.PIPE)

    if result.stderr:
        raise Exception(
            "Error while cleaning worker directory: " + result.stderr.decode("utf-8")
        )

    # Determine the source repository directory
    if repo.path and os.path.exists(repo.path):
        original_path = repo.path
    else:
        # Determine the repository directory name.
        # Use the basename of repo.path (if provided) or derive it from repo.url.
        repo_dir_name = os.path.basename(repo.path.rstrip(os.sep))

        tmp_repos_dir = os.path.expanduser("/tmp/repos")
        candidate_path = os.path.join(tmp_repos_dir, repo_dir_name)

        if os.path.exists(candidate_path):
            original_path = candidate_path
        elif hasattr(repo, "url") and repo.url:
            # Ensure the parent directory exists
            os.makedirs(tmp_repos_dir, exist_ok=True)

            # Clone the repository into the candidate_path
            clone_command = ["git", "clone", repo.url, candidate_path]
            clone_result = subprocess.run(
                clone_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if clone_result.returncode != 0:
                error_msg = clone_result.stderr.decode("utf-8")
                raise Exception("Failed to clone repository: " + error_msg)
            setup_repo_env(candidate_path)
            original_path = candidate_path
        else:
            logging.info(
                f"ERROR: Repository path not found: {repo.path} and no URL provided"
            )
            return ""

    # Remove the worker directory (created above) and copy the repository files into it
    if os.path.exists(worker_dir):
        shutil.rmtree(worker_dir)
    shutil.copytree(original_path, worker_dir)

    return worker_dir


async def run_tests(repo_dir: str, test_command: str = "pytest", log_id: str = ""):
    """
    Run tests and parse pytest output more comprehensively.
    """
    # pass to reportlog file, tmpfile
    reportlog = "/tmp/pytest-report" + log_id + ".jsonl"
    if log_id:
        test_command += f" --report-log={reportlog}"

    # count how long it takes
    process = await asyncio.create_subprocess_shell(
        test_command,
        cwd=repo_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        executable="/bin/bash",
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=240  # 60 second timeout
        )
    except asyncio.TimeoutError:
        try:
            process.terminate()
            await asyncio.sleep(0.5)
            process.kill()  # Force kill if not terminated
        except:
            pass

        return {
            "had_execution_error": True,
            "error_message": "Test execution timed out after 120 seconds",
            "failed": 0,  # Assuming at least one test failed
            "passed": 0,
            "score": 0,
        }

    # read reportlog
    try:
        with open(reportlog, "r") as f:
            reportlog_content = f.read()
    except:
        reportlog_content = ""
        print(f"Errored on {repo_dir}")
        print(stdout, stderr)

    # parse stderr into string
    stderr = stderr.decode("utf-8") if stderr else None
    return parse_pytest_json_report(reportlog_content, stderr=stderr)


async def test_without_function(
    removal_location, worker_dir, code_path, test_command, restore=False
):
    """
    Removes the target function from one worker_dir copy of the repo.
    Runs the test suite and returns the number of failed tests.
    """
    rel_path, func_name = removal_location
    abs_path = os.path.join(worker_dir, code_path, rel_path)
    content_after_deleted = open(abs_path, "r", encoding="utf-8").read()
    executed_removal_info = remove_functions_in_file(abs_path, func_name)
    test_output = await run_tests(
        worker_dir, test_command, log_id=worker_dir.split("/")[-1]
    )

    if restore:
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content_after_deleted)

    return test_output, executed_removal_info


def get_source_code_from_name(file_path, function_name):
    # file_path: absolute file path
    # print(f"GETTING CODE FROM PATH {file_path}")
    info = extract_function_info(open(file_path, "r").read(), function_name)
    if info:
        return info["func_code"]
    else:
        return None


def count_lines(source_code: str, code_only=True) -> int:
    """
    Count the number of lines in the provided source code.
    If code_only, only count lines that contain tokens
    other than comments or newlines.

    Note: If a line contains code and an inline comment, it is counted.
    """
    if code_only:
        code_lines = set()
        source_io = io.StringIO(source_code)

        # Iterate over tokens in the source code.
        for token in tokenize.generate_tokens(source_io.readline):
            tok_type, tok_string, (srow, _), _, _ = token

            # Exclude tokens that are comments or just newline markers.
            if tok_type not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.ENCODING,
            ):
                code_lines.add(srow)
        return len(code_lines)
    else:
        return len(source_code.splitlines())


def calculate_function_complexity(source_code: str) -> dict:
    """
    Calculate multiple complexity metrics for a function.

    Args:
        source_code: The source code of the function as a string

    Returns:
        Dictionary with different complexity measures:
        - line_count: Number of lines in the function
        - cyclomatic: Cyclomatic complexity (McCabe)
        - halstead_volume: Halstead volume metric
        - halstead_difficulty: Halstead difficulty metric
        - maintainability: Maintainability index
    """
    metrics = {"line_count": count_lines(source_code)}

    try:
        source_code = unindent_code(source_code)
        # Use Radon's cyclomatic complexity
        cc_results = cc_visit(source_code)
        assert cc_results
        # Store the cyclomatic complexity of the first function found
        metrics["cyclomatic"] = cc_results[0].complexity
    except:
        metrics["cyclomatic"] = 1  # Default value

        # Calculate Halstead metrics
    h_results = h_visit(source_code)
    assert h_results and h_results.total
    # Get the first function's metrics or use total
    if h_results.functions:
        # Use the metrics of the first function
        func_name, func_report = h_results.functions[0]
        metrics["halstead_volume"] = func_report.volume
        metrics["halstead_difficulty"] = func_report.difficulty
    else:
        # Use total metrics
        metrics["halstead_volume"] = h_results.total.volume
        metrics["halstead_difficulty"] = h_results.total.difficulty

    return metrics


def compute_centrality(function_graph, alpha=0.5):
    """
    Compute multiple centrality scores for functions.

    Args:
        function_graph: dict{ (file, func) -> set[(file, func)] }
            Function call graph where keys are callers and values are callees
        alpha: Discount factor for distance-discount method

    Returns:
        A dict { (file, func): {"metric_name": score, ...} } with all centrality metrics
    """
    # Initialize results dictionary - each function will have all metrics
    results = {func: {} for func in function_graph.keys()}

    # Calculate and store each centrality metric

    # Original distance-discount (outgoing influence) centrality
    dd_scores = distance_discount_scores(function_graph, alpha)
    for func, score in dd_scores.items():
        results[func]["distance_discount"] = score

    # Simple in-degree: how many functions directly call this one
    in_scores = in_degree_centrality(function_graph)
    for func, score in in_scores.items():
        results[func]["in_degree"] = score

    # True in-degree with decay: weighted sum of all incoming paths
    true_in_scores = true_in_degree_centrality(function_graph, alpha)
    for func, score in true_in_scores.items():
        results[func]["true_in_degree"] = score

    # Out-degree: how many functions this one calls directly
    out_scores = out_degree_centrality(function_graph)
    for func, score in out_scores.items():
        results[func]["out_degree"] = score

    # Betweenness: measures bridge functions between code modules
    btw_scores = betweenness_centrality(function_graph)
    for func, score in btw_scores.items():
        results[func]["betweenness"] = score

    # PageRank: recursive importance based on being called by important functions
    pr_scores = pagerank_centrality(function_graph)
    for func, score in pr_scores.items():
        results[func]["pagerank"] = score

    # Harmonic: weighted importance based on distance from all other functions
    harm_scores = harmonic_centrality(function_graph)
    for func, score in harm_scores.items():
        results[func]["harmonic"] = score

    # Bidirectional harmonic: importance when treating all call relationships as bidirectional
    bi_harm_scores = bidirectional_harmonic_centrality(function_graph)
    for func, score in bi_harm_scores.items():
        results[func]["bidirectional_harmonic"] = score

    # Degree centrality: total connections (both incoming and outgoing)
    deg_scores = degree_centrality(function_graph)
    for func, score in deg_scores.items():
        results[func]["degree"] = score

    return results


def in_degree_centrality(function_graph):
    """
    Calculate in-degree centrality - how many other functions call this function.
    Higher values indicate commonly used utility functions.
    """
    scores = {func: 0 for func in function_graph.keys()}

    # Count incoming edges by iterating through all edges
    for caller, callees in function_graph.items():
        for callee in callees:
            if callee in scores:  # The callee might be external/not in our graph
                scores[callee] += 1

    return scores


def true_in_degree_centrality(function_graph, alpha=0, max_iter=3):
    """
    Calculate true in-degree centrality with decay factor.

    This centrality measure considers both direct and indirect incoming function calls:
    - Direct callers contribute a score of 1
    - Callers of callers contribute alpha
    - Callers at distance 3 contribute alpha^2, and so on...

    This is similar to PageRank but focused on incoming paths and using distance-based decay.

    Args:
        function_graph: dict{ (file, func) -> set[(file, func)] }
            Function call graph where keys are callers and values are callees
        alpha: Decay factor (0-1) for longer paths
        max_iter: Maximum number of iterations for convergence

    Returns:
        Dict mapping functions to their true in-degree centrality scores
    """
    # First, build the reverse graph (incoming calls)
    reverse_graph = {}
    for func in function_graph:
        reverse_graph[func] = set()

    for caller, callees in function_graph.items():
        for callee in callees:
            if callee in reverse_graph:  # The callee might be external/not in our graph
                reverse_graph[callee].add(caller)

    # Now compute distance-discounted scores on the reverse graph
    scores = {}
    for func in reverse_graph:
        # BFS from this function in the reverse graph to find all incoming paths
        visited = set()
        queue = deque()
        # We'll store (node, dist)
        queue.append((func, 0))
        visited.add(func)
        total_score = 0.0

        while queue:
            current, dist = queue.popleft()
            # We skip the start node itself, but any parent is included
            if dist > 0:
                total_score += alpha ** (
                    dist - 1
                )  # dist-1 so direct callers get score 1

            # If we've reached maximum iteration depth, stop expanding
            if dist >= max_iter:
                continue

            # Expand incoming calls (parents in the call graph)
            for caller in reverse_graph.get(current, []):
                if caller not in visited:
                    visited.add(caller)
                    queue.append((caller, dist + 1))

        scores[func] = total_score

    return scores


def out_degree_centrality(function_graph):
    """
    Calculate out-degree centrality - how many other functions this function calls.
    Higher values indicate controller functions that coordinate many operations.
    """
    return {func: len(callees) for func, callees in function_graph.items()}


def degree_centrality(function_graph):
    """
    Calculate degree centrality - the total number of connections (both incoming and outgoing).
    Higher values indicate functions that are highly connected overall within the codebase.

    This is simply the sum of in-degree and out-degree centrality, representing the total
    number of direct connections a function has with other functions.
    """
    in_scores = in_degree_centrality(function_graph)
    out_scores = out_degree_centrality(function_graph)

    return {
        func: in_scores.get(func, 0) + out_scores.get(func, 0)
        for func in set(in_scores) | set(out_scores)
    }


def betweenness_centrality(function_graph):
    """
    Calculate betweenness centrality - how often a function appears on shortest
    paths between other functions. Identifies "bridge" functions connecting
    different parts of the codebase.
    """
    # Implementation uses BFS to find shortest paths
    scores = {func: 0.0 for func in function_graph.keys()}
    functions = list(function_graph.keys())
    n = len(functions)

    for i in range(n):
        # BFS from each source node
        source = functions[i]
        distances = {}
        predecessors = {func: [] for func in function_graph.keys()}
        queue = deque([(source, 0)])
        distances[source] = 0

        while queue:
            node, dist = queue.popleft()
            for neighbor in function_graph.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    predecessors[neighbor].append(node)
                    queue.append((neighbor, dist + 1))
                elif distances[neighbor] == dist + 1:
                    predecessors[neighbor].append(node)

        # Compute dependency
        dependency = {func: 0.0 for func in functions}

        for target in reversed(functions):
            if target not in distances or target == source:
                continue

            num_paths = len(predecessors[target])
            if num_paths == 0:
                continue

            for pred in predecessors[target]:
                dependency[pred] += (1.0 + dependency[target]) / num_paths

        for func, dep in dependency.items():
            if func != source:
                scores[func] += dep

    # Normalize

    for func in scores:
        if np.isnan(scores[func]):
            scores[func] = 0.0  # Replace NaN with 0

    normalization = (n - 1) * (n - 2)
    if normalization > 0:
        for func in scores:
            scores[func] /= normalization

    return scores


def pagerank_centrality(function_graph, damping=0.85, max_iter=100, tol=1e-6):
    """
    Calculate PageRank centrality - importance based on being called by other
    important functions. Identifies core functions that many key parts rely on.
    """
    functions = list(function_graph.keys())
    n = len(functions)
    scores = {func: 1.0 / n for func in functions}

    for _ in range(max_iter):
        new_scores = {func: (1 - damping) / n for func in functions}

        # Build transpose graph (incoming calls)
        incoming = {func: set() for func in functions}
        for caller, callees in function_graph.items():
            for callee in callees:
                if callee in incoming:
                    incoming[callee].add(caller)

        # Update based on incoming links
        for func in functions:
            callers = incoming[func]
            for caller in callers:
                outgoing_count = len(function_graph[caller])
                if outgoing_count > 0:  # Avoid division by zero
                    new_scores[func] += damping * scores[caller] / outgoing_count

        # Check for convergence
        diff = sum(abs(new_scores[func] - scores[func]) for func in functions)
        if diff < tol:
            break

        scores = new_scores

    return scores


def harmonic_centrality(function_graph):
    """
    Calculate harmonic centrality - sum of reciprocals of distances.
    Handles disconnected code modules better than other centrality measures.
    """
    scores = {}
    functions = list(function_graph.keys())
    n = len(functions)

    for func in functions:
        # BFS to compute distances
        distances = {func: 0}
        queue = deque([func])

        while queue:
            node = queue.popleft()
            for neighbor in function_graph.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        # Sum of reciprocals of distances (1/d)
        harmonic_sum = 0
        for other in functions:
            if other != func:
                if other in distances:
                    harmonic_sum += 1.0 / distances[other]

        scores[func] = harmonic_sum / (n - 1) if n > 1 else 0

    return scores


def bidirectional_harmonic_centrality(function_graph):
    """
    Calculate harmonic centrality on a bidirectional version of the function call graph.

    This treats all function call relationships as bidirectional, measuring how
    close a function is to all other functions when edges can be traversed in either direction.

    Useful for finding functions that are central to the codebase's structure
    regardless of call direction, revealing functions that connect different
    parts of the codebase together.

    Args:
        function_graph: dict{ (file, func) -> set[(file, func)] }
            Function call graph where keys are callers and values are callees

    Returns:
        Dict mapping functions to their bidirectional harmonic centrality scores
    """
    # First create a bidirectional version of the graph
    bidirectional_graph = {}
    for func in function_graph:
        bidirectional_graph[func] = set(function_graph.get(func, set()))

    # Add reverse edges to make it bidirectional
    for caller, callees in function_graph.items():
        for callee in callees:
            if callee in bidirectional_graph:
                bidirectional_graph[callee].add(caller)

    # Now compute harmonic centrality on the bidirectional graph
    scores = {}
    functions = list(bidirectional_graph.keys())
    n = len(functions)

    for func in functions:
        # BFS to compute distances in the bidirectional graph
        distances = {func: 0}
        queue = deque([func])

        while queue:
            node = queue.popleft()
            for neighbor in bidirectional_graph.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        # Sum of reciprocals of distances (1/d)
        harmonic_sum = 0
        for other in functions:
            if other != func:
                if other in distances:
                    harmonic_sum += 1.0 / distances[other]
                # If other is not reachable, it contributes 0 to the sum

        # Normalize by the maximum possible sum (n-1)
        scores[func] = harmonic_sum / (n - 1) if n > 1 else 0

    return scores


def distance_discount_scores(function_graph, alpha=0.5):
    """
    Compute a "distance-discounted" dependency measure for each function:

    score(f) = sum( alpha^dist(f, g) )  for all g reachable from f,
    where dist(f,g) is the shortest path distance (f->...->g).

    Args:
        function_graph: dict{ str -> set[str] }
            e.g. { 'funcA': {'funcB','funcC'}, 'funcB': {'funcC'}, 'funcC': set() }
        alpha: discount factor in (0,1).
               If alpha=0.5, direct callees add 0.5, distance-2 nodes add 0.25, etc.

    Returns:
        A dict { function_name: discounted_score }, sorted by name ascending.
        You can re-sort by value if desired.
    """
    scores = {}
    for f in function_graph.keys():
        scores[f] = _distance_discount_for_single(function_graph, f, alpha)
    return scores


def _distance_discount_for_single(graph, start, alpha):
    """
    BFS from `start` to find all reachable nodes and their distances.
    Then sum alpha^distance over them.
    Exclude the start node itself from the sum.
    """

    visited = set()
    queue = deque()
    # We'll store (node, dist)
    queue.append((start, 0))
    visited.add(start)
    total_score = 0.0

    while queue:
        current, dist = queue.popleft()
        # We skip the start node itself, but any child or further is included
        if dist > 0:
            total_score += alpha**dist

        # Expand children
        for callee in graph.get(current, []):
            if callee not in visited:
                visited.add(callee)
                queue.append((callee, dist + 1))

    return total_score


def _count_file_lines(path: str) -> int:
    """Return the number of lines in *path* (utf‑8)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception as exc:  # noqa: BLE001
        logging.error("Counting lines failed for %s: %s", path, exc)
        return 0


def get_repo_info(code_path: str) -> Dict[str, float]:
    """
    Compute summary statistics for the repository (or sub‑tree) at *code_path*.

    Returns a dict like:
        {
            "exists":            bool,
            "files_count":       int,   # python files
            "functions_count":   int,
            "total_lines":       int,
            "avg_lines_per_file":float, # 0 if no files
            "avg_lines_per_func":float, # 0 if no functions
        }
    """
    info = {
        "exists": False,
        "files_count": 0,
        "functions_count": 0,
        "total_lines": 0,
        "avg_lines_per_file": 0.0,
        "avg_lines_per_func": 0.0,
    }

    if not os.path.exists(code_path):
        return info

    info["exists"] = True

    try:
        python_files = gather_python_files(code_path)
        info["files_count"] = len(python_files)

        # Functions keyed by file → list[str]
        fn_map = parse_repo_for_functions(code_path)
        info["functions_count"] = sum(len(fns) for fns in fn_map.values())

        info["total_lines"] = sum(_count_file_lines(fp) for fp in python_files)

        if info["files_count"]:
            info["avg_lines_per_file"] = info["total_lines"] / info["files_count"]
        if info["functions_count"]:
            info["avg_lines_per_func"] = info["total_lines"] / info["functions_count"]

    except Exception as exc:  # noqa: BLE001
        logging.exception("Error analysing repo at %s: %s", code_path, exc)
        print(e)
        info["error"] = str(exc)

    return info
