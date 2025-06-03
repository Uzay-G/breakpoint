import json
import logging
import os
import time
from typing import List, Dict, Any

import tiktoken
from lib.chat import ModelInterface, StudentAttempt, Chat
from lib.codeparsing_utils import *
from lib.problem_generator import ProblemEnv


class CodeAgent:
    def __init__(
        self,
        model_name: str = "o3-mini",
        test_budget: int = 3,
        max_iterations: int = 8,
        max_tokens: int = 40000, 
        mode: str = "fix",
        instructions: str = "",
        repair_mode: str = "target",
        tool_use=False,
        iterate_with_tests=True,
        thinking_budget=2000,
        file_token_ratio=0.1,  # Fraction of context to use for file reading
    ):
        """
        Initialize the code repair agent with configurable capabilities.

        Args:
            model_name: OpenAI model to use
            api_key: Optional API key
            max_tokens: Maximum tokens for model responses (GPT-4o will use a lower limit automatically)
            test_budget: Number of test runs allowed
            max_iterations: Maximum agent-environment interactions
            verbose: Whether to logging.info detailed logs
        """
        self.model_interface = ModelInterface(
            model_name=model_name, max_tokens=max_tokens
        )
        self.test_budget = test_budget
        self.max_iterations = max_iterations
        self.instructions = instructions
        self.mode = mode
        self.repair_mode = repair_mode
        self.tool_use = tool_use
        self.iterate_with_tests = iterate_with_tests
        self.thinking_budget = thinking_budget
        self.file_token_ratio = file_token_ratio


    def __call__(self, env: ProblemEnv) -> str:
        """
        Fix a broken function using repo exploration and testing.

        Args:
            problem: Problem details including repo path, file path, and function name

        Returns:
            The fixed function code
        """

        return self._run_agent(env)

    async def _run_agent(self, env: ProblemEnv) -> StudentAttempt:
        """Run the agent to fix the broken function."""
        self.tests_remaining = self.test_budget
        problem = env.problem
        has_submitted = False 

        test_info = problem.test_info


        logging.info(
            f"Starting repair of {problem.function_name} in {problem.fpath}"
        )
        logging.info(f"Test budget: {self.tests_remaining} runs")

        system_prompt = "You are an expert Python programmer modifying a codebase."

        function_def_with_docstring = get_source_code_from_name(
            os.path.join(env.execution_dir, problem.fpath), problem.function_name
        )

        if self.mode == "fix":
            if self.repair_mode == "target":
                instructions = f"""You are an expert Python programmer tasked with repairing broken code.

                Your goal is to fix the implementation of '{problem.function_name}' in '{problem.fpath}' so that all tests pass.

                When this function was removed from the codebase, several test errors occurred.

                You have access to these tools:
                1. list_directory - explore the repo structure
                2. read_file - view file contents
                3. search_code - search for patterns across the codebase
                4. submit_attempt - Submit attempted patches for the function, and then get test feedback (max {self.test_budget} times)

                When applying patches, be precise and target only the necessary changes.  All the failing tests are due to this one failing function implementation. You can only call one tool at a time.

                Here's the function definition and docstring:
                {function_def_with_docstring}
                """
            else:
                instructions = f"""You are an expert Python programmer tasked with diagnosing and fixing an issue in a codebase.

                The test suite is currently failing with several test errors that were inserted by someone. Your task is to:

                1. Explore the codebase to understand its structure
                2. Analyze the code and test failures to find the root cause
                3. Fix the issue by modifying the appropriate file(s)
                
                You have access to these tools:
                1. list_directory - explore the repo structure
                2. read_file - view file contents
                3. search_code - search for patterns across the codebase
                4. replace_function - replace a function in the codebase with a function with the same definition but a different body, to try and patch the bug. You can do call this at {self.test_budget} attempts)

                You *need* to call the replace_function function to make any progress and you need to do it while you can. You need to pass the tests, but you also need to find the right code to modify. If you make them pass by changing files that are not the right ones, it will be counted as a failure and you will get a 0 score, ending your attempt.
                When applying patches, be precise and target only the necessary functions.  You can call one tool at a time.
                """

            instructions += "\nInfo on some of the failing tests:\n"

            for fail in problem.test_info.get("failures_info", [])[:3]:
                instructions += fail[:2000] + "\n"
        else:
            instructions = self.instructions

        chat = Chat(self.model_interface, system_prompt=system_prompt)

        function_code = ""
        score = 0.0
        solution_found = False
        tool_usage = []

        logging.info(f"System prompt: {system_prompt}")
        logging.info(f"Instructions: {self.instructions}")

        # Main agent loop
        i = 0
        while i < self.max_iterations:
            logging.info(f"Iteration {i+1}/{self.max_iterations}")

            # Determine the message to send
            iterations_left = self.max_iterations - i
            message = ""

            current_tools = self._get_tools()

            if i == 0:
                message = instructions

            if iterations_left <= 1 and not has_submitted and self.tests_remaining > 0:
                message += f"""IMPORTANT: You have only {iterations_left} tool calls remaining and haven't submitted any attempts yet.
                You MUST submit a repair attempt NOW.
                Test runs left: {self.tests_remaining}"""

                current_tools = [
                    tool
                    for tool in self._get_tools()
                    if tool["function"]["name"] == "submit_attempt"
                    or tool["function"]["name"] == "replace_function"
                ]

            elif (
                self.mode == "break"
                and has_submitted
            ):
                # For test iteration mode, provide helpful feedback about the test results
                message += f"""Your previous corruption attempt received a score of {score}.
                
                Remember, in corruption mode:
                - Score of 0.0 means all tests fail (perfect corruption)
                - Score of 1.0 means all tests pass (no corruption)
                
                Test runs left: {self.tests_remaining}
                
                Please provide a revised implementation using the submit_attempt tool."""
            else:
                message += f"Tool uses left: {self.max_iterations - i}, Test runs left: {self.tests_remaining}"

            logging.info(f"Sending message {message}")

            response = await chat.send_message(
                message=message,
                tools=current_tools,
                thinking_budget=self.thinking_budget,
            )

            if response.thinking:
                logging.info(
                    f"Thinking for function {problem.function_name}: {response.thinking.thinking}"
                )

            if response.tool_calls:
                n_new_calls = min(self.max_iterations - i, len(response.tool_calls))

                i += n_new_calls

                for tool_call in response.tool_calls[:n_new_calls]:
                    tool_name = tool_call.function.name
                    args = tool_call.function.arguments

                    logging.info(
                        f"Calling tool {tool_name} with args {args} at iteration {i}"
                    )

                    # Execute the tool
                    if (
                        tool_name == "submit_attempt" or tool_name == "replace_function"
                    ) and self.tests_remaining == 0:
                        chat.add_tool_response(
                            tool_call.id,
                            "You have no test runs left.",
                        )
                        continue

                    result = await self._execute_tool(tool_name, args, env)

                    tool_usage.append(
                        {
                            "tool": tool_name,
                            "args": args,
                        }
                    )

                    logging.info(f"Response: {result}")

                    if tool_name == "submit_attempt" or tool_name == "replace_function":
                        has_submitted = True  # Mark that the agent has submitted
                        self.tests_remaining -= 1

                        if result["success"]:
                            # Check if we've found a solution
                            if result.get("score", 0.0) == 1.0 and self.mode == "fix":
                                solution_found = True

                        if "score" in result:
                            score = result["score"]

                        function_code = args.get("func_code", "")
                        test_info = result.get("test_info", {})

                    # Add tool result to the conversation
                    result_str = ""
                    for k, v in result.items():
                        result_str += f"{k}: {str(v)}\n"
                    logging.info(
                        f"Response for function {problem.function_name}: {result_str}"
                    )
                    chat.add_tool_response(tool_call.id, result_str)

                if solution_found:
                    break

            if self.tests_remaining <= 0:
                break

        metadata = {
            "repo_path": problem.repo.path,
            "file_path": problem.fpath,
            "tool_usage": tool_usage,
            "test_info": test_info,
        }

        attempt = StudentAttempt(
            problem_spec=problem.function_name,
            student_solution=function_code,
            actual_solution=None,
            metadata=metadata,
            score=score,
        )
        return attempt

    def _get_tools(self) -> List[Dict[str, Any]]:
        """Define the tools available to the agent."""
        tools = []

        if self.tool_use:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files and directories at the specified path. Truncates at 100 files.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Relative path to list (empty for repo root)",
                                }
                            },
                            "required": [],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "description": "Search for a pattern across Python files",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {
                                    "type": "string",
                                    "description": "Text or regex pattern to search for",
                                }
                            },
                            "required": ["pattern"],
                        },
                    },
                },
            ]

            # Always add read_file
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read the contents of a file. If the file is too large (exceeds 40% of model context), it will automatically return a list of functions in the file instead. You can use this extra tool use for anything, but typically you'll want to use read_function to read specific functions from the large file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Relative path to the file",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                }
            )

            tools.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_file_functions",
                            "description": "Return a list of top-level function definitions in a file. Truncates at 100 functions.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Relative path to the file",
                                    }
                                },
                                "required": ["file_path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "read_function",
                            "description": "Read a specific function definition in a requested file. Please make sure the function exists before you request it, otherwise it will throw an error. Commonly used after read_file returns a list of functions when a file is too large.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "function_name": {
                                        "type": "string",
                                        "description": "Name of the function to be read",
                                    },
                                    "file_path": {
                                        "type": "string",
                                        "description": "Relative path to the file",
                                    },
                                },
                                "required": ["function_name", "file_path"],
                            },
                        },
                    },
                ]
            )

        if self.repair_mode == "target":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "submit_attempt",
                        "description": "Submit the new code for the target function. You can use this several times depending on your test budget",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "func_code": {
                                    "type": "string",
                                    "description": "Function code (with function definition, function body, nothing else)",
                                }
                            },
                            "required": ["func_code"],
                        },
                    },
                }
            )
        else:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "replace_function",
                        "description": "Submit the new code for the target function. Replace the function def and the body, but don't go beyond that.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "func_code": {
                                    "type": "string",
                                    "description": "Function code (with function definition, function body, nothing else)",
                                },
                                "func_name": {
                                    "type": "string",
                                    "description": "Name of function",
                                },
                                "file_path": {
                                    "type": "string",
                                    "description": "Relative file path in repo where the function is",
                                },
                            },
                            "required": ["func_code", "func_name", "file_path"],
                        },
                    },
                }
            )
        return tools

    async def _execute_tool(
        self, tool_name: str, args: Dict[str, Any], env: ProblemEnv
    ) -> Dict[str, Any]:
        """Execute the specified tool with given arguments."""
        problem = env.problem
        worker_dir = env.execution_dir

        if tool_name == "list_directory":
            path = args.get("path", "")
            try:
                target_path = os.path.join(worker_dir, path)
                if not os.path.exists(target_path):
                    return {"success": False, "message": f"Path does not exist: {path}"}

                items = os.listdir(target_path)
                result = []

                for item in items[:100]:
                    item_path = os.path.join(target_path, item)
                    item_type = "directory" if os.path.isdir(item_path) else "file"
                    item_rel_path = os.path.join(path, item)

                    result.append(
                        {"name": item, "type": item_type, "path": item_rel_path}
                    )

                return {
                    "success": True,
                    "message": f"Listed {len(items)} items",
                    "items": result,
                }

            except Exception as e:
                print(e)
                return {
                    "success": False,
                    "message": f"Error listing directory: {str(e)}",
                }

        elif tool_name == "read_file":

            file_path = args.get("file_path", "")
            full_path = os.path.join(worker_dir, file_path)

            if not os.path.exists(full_path):
                return {"success": False, "message": f"File not found: {file_path}"}

            if not os.path.isfile(full_path):
                return {"success": False, "message": f"Path is not a file: {file_path}"}

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            model_token_limit = self.model_interface.get_token_limit()

            # Use a percentage of the model's context window for file content
            token_limit = int(model_token_limit * self.file_token_ratio)

            # create dummy message to get token count
            if content == "":
                content_token_count = 0
            else:
                content_token_count = self.model_interface.count_tokens(
                    [{"role": "user", "content": content}]
                )

            if content_token_count > token_limit:
                # File is too large, return function list instead
                function_infos = parse_python_file(full_path)

                def truncate_docstring(func_def: FunctionDefInfo):
                    return (
                        {"name": func_def.name, "description": func_def.docstring[:100]}
                        if func_def.docstring
                        else {"name": func_def.name, "description": "None"}
                    )

                function_list = [
                    truncate_docstring(func_def) for func_def in function_infos
                ]

                return {
                    "success": True,
                    "message": f"File is too large ({content_token_count} tokens). Here is a list of functions in the file that you can read individually.",
                    "content": f"Functions in {file_path}:\n{json.dumps(function_list, indent=2)}",
                    "functions": function_list,
                }

            return {
                "success": True,
                "message": f"File read successfully",
                "content": content,
            }

        elif tool_name == "list_file_functions":
            file_path = args.get("file_path", "")
            try:
                full_path = os.path.join(worker_dir, file_path)

                if (
                    not os.path.exists(full_path)
                    and not os.path.isfile(full_path)
                    and not file_path.endswith(".py")
                ):
                    return {
                        "success": False,
                        "message": f"Invalid file read: {file_path}",
                    }

                function_infos = parse_python_file(full_path)[
                    :200
                ]

                def truncate_docstring(func_def: FunctionDefInfo):
                    return (
                        {"name": func_def.name, "description": func_def.docstring[:100]}
                        if func_def.docstring
                        else {"name": func_def.name, "description": "None"}
                    )

                return {
                    "success": True,
                    "message": f"File read successfully",
                    "content": [
                        truncate_docstring(func_def) for func_def in function_infos
                    ],
                }

            except Exception as e:
                logging.info(f"Error reading file: {e}")
                return {"success": False, "message": f"Error reading file: {str(e)}"}

        elif tool_name == "read_function":
            file_path = args.get("file_path", "")
            function_name = args.get("function_name", "")

            try:
                full_path = os.path.join(worker_dir, file_path)
                if not os.path.exists(full_path):
                    return {"success": False, "message": f"File not found: {file_path}"}

                function_infos = parse_python_file(full_path)

                for func_def in function_infos:
                    if func_def.name == function_name:
                        start_line = func_def.start_line
                        end_line = func_def.end_line
                        assert start_line >= 1 and end_line >= start_line
                        content = []

                        with open(full_path, "r", encoding="utf-8") as file:
                            for current_line_number, line in enumerate(file, start=1):
                                if current_line_number < start_line:
                                    continue
                                if current_line_number > end_line:
                                    break
                                content.append(line)

                        return {
                            "success": True,
                            "message": f"Function read successfully",
                            "content": "".join(content),
                        }

                return {
                    "success": False,
                    "message": f"Function not found in the requested file",
                }

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error reading function: {str(e)}",
                }

        elif tool_name == "submit_attempt":
            fn_code = args.get("func_code", "")
            full_path = os.path.join(worker_dir, problem.fpath)
            content_before = open(full_path, "r").read()

            try:
                fn_info = remove_functions_in_file(full_path, problem.function_name)
                with open(full_path, "r") as f:
                    insert_function_code(
                        fn_code,
                        fn_info["func_start"],
                        fn_info["func_def_end"],
                        fn_info["indent"],
                        full_path,
                    )

            except:
                logging.info(
                    f"Parsing error on function {problem.function_name}, UH OH"
                )
                return {
                    "success": False,
                    "message": "Parsing error",
                    "score": 0,
                    "test_info": {},
                }

            test_output = await run_tests(
                worker_dir, problem.repo.test_command, log_id=worker_dir.split("/")[-1]
            )
            test_output["failures_info"] = test_output.get("failures_info", [])[:1]

            if test_output["passed"] == 0 and test_output["failed"] == 0:
                logging.info(f"UH OH {full_path}:\n{test_output}")

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content_before)

            if test_output.get("had_execution_error"):
                logging.info(f"Had error on {problem.function_name}, {test_output}")
                return {
                    "success": False,
                    "message": f"Failed to execute tests. Error message: {test_output['error_message']}",
                    "score": 0,
                    "test_info": test_output,
                }

            else:
                fails = test_output.get("failed", 0)
                if fails == 0:
                    final_score = 1.0
                else:
                    final_score = 1 - min(fails / problem.test_info["failed"], 1)

                passes = test_output.get("passed", 0)
                test_feedback = f"{passes} passed, {fails} failed.\n"

                if (
                    self.iterate_with_tests
                    and self.mode == "break"
                ):
                    current_failed_tests = test_output.get("failed_tests", [])
                    fails_test_names = [test[1] for test in current_failed_tests]
                    test_feedback += f"List of failed tests: {fails_test_names}\n"
                    tests_originally_failed = [
                        test[1] for test in problem.test_info["failed_tests"]
                    ]
                    success_tests = [
                        test
                        for test in tests_originally_failed
                        if test not in fails_test_names
                    ]
                    test_feedback += f"List of successes: {success_tests}\n"

                logging.info(f"Score: {final_score}")

                return {
                    "success": True,
                    "message": f"Test results, ie % of test coverage now passing: {final_score}. Feedback:\n{test_feedback}",
                    "score": final_score,
                    "test_info": test_output,
                }

        elif tool_name == "replace_function":

            fn_code = args.get("func_code", "")
            full_path = os.path.join(worker_dir, args.get("file_path", ""))

            if "test" in full_path:
                logging.info("WARNING TRYING TO MODIFY TESTS")
                return {"success": False, "message": "Cannot modify test files"}

            try:
                fn_info = remove_functions_in_file(full_path, args["func_name"])
                insert_function_code(
                    fn_code,
                    fn_info["func_start"],
                    fn_info["func_def_end"],
                    fn_info["indent"],
                    full_path,
                )
            except:
                logging.info(
                    f"Parsing error on function {problem.function_name}, UH OH"
                )
                return {
                    "success": False,
                    "message": "Parsing error",
                    "score": 0,
                    "test_info": {},
                }

            test_output = await run_tests(
                worker_dir, problem.repo.test_command, log_id=worker_dir.split("/")[-1]
            )

            test_output["failures_info"] = test_output["failures_info"][:1]

            if test_output.get("had_execution_error"):
                # code didn't even run
                return {
                    "success": False,
                    "message": f"Failed to execute tests. Error message: {test_output['error_message']}",
                    "score": 0,
                    "test_info": test_output,
                }
            else:
                fails = test_output.get("failed", 0)

                if fails == 0:
                    final_score = 1.0
                else:
                    final_score = 1 - min(fails / problem.test_info["failed"], 1)

                passes = test_output.get("passed", 0)
                test_feedback = f"{passes} passed, {fails} failed.\n"
                for failure in test_output.get("failures_info", [])[:4]:
                    test_feedback += f"Failure: {failure}\n"

                logging.info(f"Score: {final_score}")
                return {
                    "success": True,
                    "message": f"Test results, ie % of test coverage now passing: {final_score}. Feedback:\n{test_feedback}",
                    "score": final_score,
                    "test_info": test_output,
                }

        elif tool_name == "search_code":
            pattern = args.get("pattern", "")
            try:
                results = []
                for root, _, files in os.walk(worker_dir):
                    for file in files:
                        if file.endswith(".py"):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, worker_dir)

                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()

                            if pattern in content:
                                # Get context around the match
                                lines = content.split("\n")
                                matches = []
                                for i, line in list(enumerate(lines))[
                                    :10
                                ]:  # think about
                                    if pattern in line:
                                        start = max(0, i - 2)
                                        end = min(len(lines), i + 3)
                                        context = "\n".join(lines[start:end])
                                        matches.append(
                                            {"line": i + 1, "context": context}
                                        )

                                results.append({"file": rel_path, "matches": matches})
                return {
                    "success": True,
                    "message": f"Found pattern in {len(results)} files",
                    "results": results,
                }
            except Exception as e:
                return {"success": False, "message": f"Error searching code: {str(e)}"}
        else:
            return {"success": False, "message": f"Unknown tool: {tool_name}"}
