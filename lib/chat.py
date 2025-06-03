import os
import asyncio
from typing import Optional, List, Any, Tuple, Type, Union, Dict
import json
import logging
import tiktoken

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass



@dataclass
class Message:
    thinking: Any
    tool_calls: Any
    content: Any


load_dotenv()

from openai import AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic


class FunctionInfo:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, id, name, inputs):
        self.id = id
        self.function = FunctionInfo(name, inputs)


class ModelInterface:
    """
    A unified, asynchronous interface for calling language models with the
    structured parse method. Supports OpenAI, Anthropic, and HuggingFace models.
    """

    # Define maximum token limits for different models
    MODEL_TOKEN_LIMITS = {
        "gpt-4.1": 1_000_000,  # 1 M‑token context window
        "gpt-4o": 128000,  # 128 K
        "o3-mini": 200_000,  # 200 K
        "o4-mini": 200_000,  # 200 K
        "deepseek-chat": 64_000,  # 64 K
        "deepseek-reasoner": 64_000,  # 64 K
        "claude-3-7-sonnet-latest": 200_000,  # 200 K
        "gpt-4.1-nano": 1_000_000,
    }

    def __init__(
        self, model_name: str, api_key: Optional[str] = None, max_tokens: int = None
    ):
        """
        Args:
            model_name: e.g. "gpt-4o", "claude-3-opus", or huggingface like "DeepSeek/.."
            api_key: optional override of environment API key
            max_tokens: max tokens for completions (OpenAI/Anthropic) or max_new_tokens (HuggingFace)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens

        # OpenAI models
        if model_name in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-nano",
            "o1",
            "o1-mini",
            "o3-mini",
            "o4-mini",
            "o3",
        ]:
            self.client = AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_KEY"),
                organization=os.environ.get("OPENAI_ORG"),
            )
            self.model_provider = "openai"

        # Anthropic models
        elif model_name.startswith("claude-") or model_name in [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
            "claude-2",
            "claude-3-5-sonnet",
            "claude-3-7-sonnet-latest",
        ]:
            self.client = AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
            self.model_provider = "anthropic"

        # Deepseek
        elif model_name.startswith("deepseek"):
            self.client = AsyncOpenAI(
                api_key=os.environ.get("DEEPSEEK_KEY"),
                base_url="https://api.deepseek.com",
            )
            self.model_provider = "deepseek"
            self.max_tokens = 75000

    def _format_tools_for_model(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format tools to match the expected format for each model provider.
        """
        if self.model_provider == "openai":
            # OpenAI expects tools in its own format
            return tools
        elif self.model_provider == "anthropic":
            # Convert to Anthropic tool format
            anthropic_tools = []

            for tool in tools:
                if tool.get("type") == "function":
                    function = tool.get("function", {})

                    anthropic_tool = {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "input_schema": function.get("parameters", {}),
                    }

                    anthropic_tools.append(anthropic_tool)

            return anthropic_tools

        # HuggingFace doesn't support tools directly
        return []

    def tool_response(self, id, content):
        if self.model_provider == "anthropic":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": json.dumps(content),
                    }
                ],
            }
        else:
            return {"role": "tool", "tool_call_id": id, "content": json.dumps(content)}

    def parse_openai_tools(self, message_obj):
        formatted_tool_calls = []
        if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
            for tool_call in message_obj.tool_calls:
                try:
                    # Parse the function arguments into a Python dict
                    inputs = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    inputs = {}  # Default to empty dict if parsing fails

                # Create standardized tool call object
                formatted_tool_calls.append(
                    ToolCall(
                        id=tool_call.id, name=tool_call.function.name, inputs=inputs
                    )
                )
        return formatted_tool_calls

    async def call(
        self,
        messages: List[dict],
        system_prompt: str = "You are a helpful assistant.",
        output_type: Optional[Type[BaseModel]] = None,
        max_retries: int = 20,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = 1024,
        max_completion_tokens: Optional[int] = None,
    ) -> Tuple[Union[Message, BaseModel], Any]:
        """
        Calls the LLM using OpenAI, Anthropic, or HuggingFace,
        returning either a standardized Message object or a Pydantic-validated object.

        Args:
            messages: List of conversation messages
            system_prompt: System prompt for the model
            output_type: Optional Pydantic model for structured output
            max_retries: Number of retries on failure
            tools: Optional list of tools/functions
            thinking_budget: Token budget for extended thinking (Anthropic)

        Returns:
            Standardized Message object or Pydantic model if output_type is specified
        """
        # Determine the maximum tokens to use - either user-specified or model default

        messages_to_add = []

        # For OpenAI models
        if self.model_provider in ("openai", "deepseek"):
            formatted_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            last_exception = None
            for attempt in range(max_retries):
                try:

                    request_args = {
                        "messages": formatted_messages,
                        "model": self.model_name,
                        "tools": tools,
                    }
                    if output_type is None:
                        try:
                            # Prepare request arguments

                            # Add max_completion_tokens only if specified
                            if max_completion_tokens is not None:
                                request_args["max_completion_tokens"] = (
                                    max_completion_tokens
                                )

                            response = await asyncio.wait_for(
                                self.client.chat.completions.create(**request_args),
                                timeout=120,
                            )

                            # Process the response into our standard Message format
                            message_obj = response.choices[0].message
                            messages_to_add = [message_obj]
                            tool_calls = self.parse_openai_tools(message_obj)
                            result = message_obj.content
                        except asyncio.TimeoutError:
                            logging.info(
                                f"Request to {self.model_name} timed out after 600 seconds"
                            )
                            continue
                    else:
                        request_args["output_type"] = output_type
                        try:
                            response = await asyncio.wait_for(
                                self.client.beta.chat.completions.parse(**request_args),
                                timeout=120.0,
                            )
                        except asyncio.TimeoutError:
                            logging.info(
                                f"Request to {self.model_name} timed out after 600 seconds"
                            )
                            continue
                        messages_to_add = [response.choices[0].message]
                        parsed_obj = response.choices[0].message.parsed
                        result = output_type.model_validate(parsed_obj)
                        tool_calls = self.parse_openai_tools(
                            response.choices[0].message
                        )
                    return (
                        Message(
                            thinking="",  # OpenAI doesn't have thinking output
                            tool_calls=tool_calls,
                            content=result,
                        ),
                        messages_to_add,
                    )

                except Exception as e:
                    logging.info(
                        f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}"
                    )
                    last_exception = e
                    await asyncio.sleep(1.0 * (attempt + 1))

            raise last_exception

        elif self.model_provider == "anthropic":
            # Format messages for Anthropic

            last_exception = None
            for attempt in range(max_retries):
                try:
                    # Create request parameters with thinking enabled
                    request_params = {
                        "model": self.model_name,
                        "system": system_prompt,
                        "messages": messages,
                        "stream": False,
                        "timeout": 600.0,
                        "max_tokens": max_completion_tokens or 63000,
                    }

                    # Add thinking budget if specified
                    if thinking_budget:
                        request_params["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": thinking_budget,
                        }

                    # Add tools if provided
                    if tools:
                        request_params["tools"] = self._format_tools_for_model(tools)

                    # If output_type is provided, add formatting instructions
                    if output_type is not None:
                        request_params["system"] = (
                            f"{system_prompt}\n\nPlease format your response as a valid JSON object matching the following schema: {output_type.schema_json()}"
                        )

                    response = await self.client.messages.create(**request_params)
                    messages_to_add = [
                        {"role": "assistant", "content": response.content}
                    ]

                    return (
                        self._extract_anthropic_message(response, output_type),
                        messages_to_add,
                    )

                except Exception as e:
                    logging.info(
                        f"[DEBUG] Attempt {attempt+1}/{max_retries} failed with exception: {e}"
                    )
                    last_exception = e
                    await asyncio.sleep(1.0 * (attempt + 1))

            raise last_exception

    def _extract_anthropic_message(self, response, output_type=None) -> Message:
        """
        Extract content from Anthropic response into the standard Message format.
        """

        content = None
        thinking = None
        tool_calls = []

        # Process each content block
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "tool_use":
                    tool_calls.append(block)
                elif block.type == "thinking":
                    thinking = block
                elif block.type == "text":
                    if output_type:
                        try:
                            json_str = block.text
                            if "```json" in json_str:
                                json_str = (
                                    json_str.split("```json")[1].split("```")[0].strip()
                                )
                            elif "```" in json_str:
                                json_str = (
                                    json_str.split("```")[1].split("```")[0].strip()
                                )

                            parsed_data = json.loads(json_str)
                            content = output_type.model_validate(parsed_data)
                        except Exception as parse_error:
                            logging.error(
                                f"Failed to parse Anthropic response into {output_type}: {parse_error}"
                            )
                            logging.error(f"Raw response: {block}")
                            raise
                    else:
                        content = block.text

        formatted_tool_calls = []

        for tc in tool_calls:
            formatted_tool_calls.append(
                ToolCall(id=tc.id, name=tc.name, inputs=tc.input)
            )

        return Message(
            thinking=thinking, tool_calls=formatted_tool_calls, content=content
        )

    def get_message_tool_id(self, message):
        tool_call_id = None

        if type(message) is dict:
            if self.model_provider == "anthropic":
                if type(message["content"]) is list:
                    for block in message["content"]:
                        if hasattr(block, "type") and block.type == "tool_use":
                            tool_call_id = block.id

            if message.get("type", "") == "tool_result":
                tool_call_id = message["tool_use_id"]
            elif message["role"] == "tool":
                tool_call_id = message["tool_call_id"]

        elif "tool_calls" in message.__dict__ and message.tool_calls:
            tool_call_id = message.tool_calls[0].id
        return tool_call_id

    def message_to_string(self, msg: Any) -> str:

        # 0. Primitive fast paths
        if isinstance(msg, str):
            return msg

        if hasattr(msg, "content"):
            content = msg.content
            # Anthropic: content is a list of blocks
            if isinstance(content, list):
                text = "".join(getattr(b, "text", str(b)) for b in content)
            else:
                text = str(content)

            # OpenAI: if tool_calls exist, include them in the count
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                text += json.dumps(tool_calls[0], ensure_ascii=False, default=str)

            return text

        if isinstance(msg, dict):
            return str(msg.get("content", msg))

        # 4. Fallback – whatever is left
        return str(msg)

    def count_tokens(self, messages):
        if self.model_provider == "anthropic":
            # Use Anthropic's token counting method
            client = Anthropic()
            return client.messages.count_tokens(
                messages=messages,
                model=self.model_name,
                thinking={"type": "enabled", "budget_tokens": 10000},
            ).input_tokens

        else:
            # Get the appropriate tokenizer for the model
            message_str = "\n".join(
                [self.message_to_string(message) for message in messages]
            )
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fallback to cl100k_base for newer models not explicitly supported
                encoding = tiktoken.encoding_for_model("gpt-4o")
            return len(encoding.encode(message_str))

    def get_token_limit(self):
        """
        Returns the maximum token limit for the current model.

        Returns:
            int: The maximum token limit for the model or default if not specified
        """
        return self.MODEL_TOKEN_LIMITS.get(self.model_name, 50000)


class Chat:
    """
    Manages a conversation with an AI model, handling message history and model interactions.
    Provides a unified interface across different model providers (OpenAI, Anthropic, etc.).
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        system_prompt: str = "You are a helpful assistant.",
    ):
        """
        Initialize a new chat session.

        Args:
            model_interface: An instance of ModelInterface to use for generating responses
            system_prompt: System prompt to guide the model's behavior
        """
        self.model_interface = model_interface
        self.system_prompt = system_prompt
        self.tool_call_ids = []
        self.messages = []

    async def recursive_summarize(self, messages: List[dict]) -> str:
        """
        Use the language model itself to summarize the provided messages.
        This function concatenates the contents of the given messages and calls the model
        with a summarization prompt.
        """
        # Concatenate the content of messages to form the text to be summarized.
        combined_text = "\n".join(
            self.model_interface.message_to_string(msg) for msg in messages
        )

        # Formulate a summarization prompt instructing the model to be concise yet preserve key details.
        prompt = (
            "Summarize the following part of the trajectory concisely while preserving all key details, keeping in mind the objective:\n\n"
            f"{combined_text}\n\nSummary:"
        )

        # Use self.call to ask the model for a summary.
        _, response = await self.model_interface.call(
            messages=[self.messages[0], {"role": "user", "content": prompt}],
            system_prompt="You are a summarizer. You are given an agent trajectory on a coding task, whose description you are provided with. Retrieve all key facts and information that is useful to solve the coding problem and output a summary of the info, that will then replace the original messages in the conversation",
            max_retries=3,
            output_type=None,
            max_completion_tokens=3000,
        )
        if response is None:
            return {"role": "user", "content": "Content truncated"}

        # Strip any extraneous whitespace from the generated summary.
        return response[0]

    async def trim_conversation(self) -> None:
        """
        Shrink the conversation so its *total* token footprint stays
        ≤ `model_interface.get_token_limit()`.

        1.  Starting *after* the system prompt, gather the oldest messages
            until (naïvely) *deleting* them would take us under the limit.
        2.  Expand that slice so every message that shares a `tool_call_id`
            with it is also included – this keeps tool‑request / tool‑reply
            pairs together.
        3.  Ask the model for a short summary of that slice, insert the
            summary back into the same chronological position, and update
            **all three** bookkeeping structures:
            `self.messages`, `self.counted_history`, and `self.token_count`.
        4.  Repeat until we fit.
        """
        limit = self.model_interface.get_token_limit()

        # Index of the *first* user/assistant message
        first_real_idx = 1

        if self.messages:
            token_count = self.model_interface.count_tokens(self.messages)
        else:
            token_count = 0

        MAX_PASSES = 5
        passes = 0
        # ----------------------------------------------------------------

        while token_count > limit * 0.95:
            passes += 1
            if passes > MAX_PASSES:
                logging.warning(
                    "trim_conversation: exceeded %d shrink attempts – "
                    "falling back to brute-force removal.",
                    MAX_PASSES,
                )

                # --- brute-force: drop oldest real messages -------------
                while token_count > limit * 0.95 and len(self.messages) > 2:
                    # never remove the system prompt
                    self.messages.pop(first_real_idx)
                    self.tool_call_ids.pop(first_real_idx)
                    token_count = self.model_interface.count_tokens(self.messages)

                logging.info(
                    "trim_conversation: after brute-force removal token_count=%d",
                    token_count,
                )
                return

            logging.info(f"Recursively trimming conversation")
            logging.info(f"Token limit: {limit}")
            logging.info(f"Token count: {token_count}")

            # ── 1. pick an initial slice of oldest messages ─────────────────
            cumulative = 0
            slice_end = None  # exclusive index

            for i in range(first_real_idx + 1, len(self.messages), 2):
                if self.model_interface.model_provider == "anthropic" and any(
                    [
                        hasattr(b, "type") and b.type == "tool_use"
                        for b in self.messages[i]["content"]
                    ]
                ):
                    continue  # stupid anthropic token counting problems
                cumulative = self.model_interface.count_tokens(self.messages[: i + 1])

                if token_count - cumulative <= limit * 0.95 - 3000:
                    slice_end = i + 1
                    break

            # Safety: if we never found a stopping point, summarise *all*
            # but the two newest messages
            if slice_end is None:
                slice_end = len(self.messages)

            slice_idxs = list(range(first_real_idx, slice_end))

            removed = []
            for idx in slice_idxs:
                #
                try:
                    up_to_first = self.messages[: idx + 1]
                    top_count = self.model_interface.count_tokens(up_to_first)
                except:
                    up_to_first = self.messages[: idx + 2]
                    top_count = self.model_interface.count_tokens(up_to_first)
                try:
                    up_to_zeroeth = self.messages[:idx]
                    bottom_count = self.model_interface.count_tokens(up_to_zeroeth)
                except:
                    up_to_zeroeth = self.messages[: idx - 1]
                    bottom_count = self.model_interface.count_tokens(up_to_zeroeth)

                message_len = (
                    top_count - bottom_count
                )  # i am annoyed at the anthropic api design

                if message_len > limit / 2:
                    # crude but effective: chop the string roughly in half
                    logging.info(
                        f"Removing huge messages: {str(self.messages[idx])[:1000]}"
                    )
                    removed.append(idx)

            # ── 2. keep tool‑call groups intact ────────────────────────────
            slice_tool_ids = {
                self.tool_call_ids[idx]
                for idx in slice_idxs + removed
                if self.tool_call_ids[idx]
            }

            if slice_tool_ids:
                for j in range(slice_end, len(self.messages)):
                    if self.tool_call_ids[j] in slice_tool_ids:
                        slice_idxs.append(j)

            slice_idxs = sorted(set(slice_idxs))

            # ── 3. summarise that slice ────────────────────────────────────

            msgs_to_summarise = [
                self.messages[idx] for idx in slice_idxs if not idx in removed
            ]
            logging.info("Msgs to summarize " + str(msgs_to_summarise))
            summary_msg = await self.recursive_summarize(msgs_to_summarise)

            logging.info("Summary" + str(summary_msg))

            # ── 4. remove the old messages (backwards to keep indices valid)
            for idx in reversed(slice_idxs):
                self.messages.pop(idx)
                self.tool_call_ids.pop(idx)

            # insert the summary where the first removed message used to be
            insert_at = slice_idxs[0]
            self.messages.insert(insert_at, summary_msg)
            self.tool_call_ids.insert(insert_at, None)

            token_count = self.model_interface.count_tokens(self.messages)

    def append_message(self, message):
        self.messages.append(message)
        tool_call_id = self.model_interface.get_message_tool_id(message)
        self.tool_call_ids.append(tool_call_id)

    def add_user_message(self, text, role, caching=False):
        if caching:
            content = {"type": "text", "text": text}
            content["cache_control"] = {"type": "ephemeral"}
            self.append_message({"role": role, "content": [content]})
        else:
            self.append_message({"role": role, "content": text})

    async def send_message(
        self,
        message: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        thinking_budget: Optional[int] = 1024,
    ) -> Any:
        """
        Send a message to the model and get a response.

        Args:
            message: The user message to send
            tools: Optional list of tools/functions the model can use
            output_type: Optional Pydantic model for structured output
            thinking_budget: Token budget for model thinking (Anthropic)

        Returns:
            The model's response, which could be a string, Pydantic model, or tool call
        """
        # Add user message to history

        await self.trim_conversation()

        self.add_user_message(message, "user", caching=True)

        # Call the model
        message, history = await self.model_interface.call(
            messages=self.messages,
            system_prompt=self.system_prompt,
            tools=tools,
            output_type=output_type,
            thinking_budget=thinking_budget,
        )

        del self.messages[-1]["content"][0]["cache_control"]

        for m in history:
            self.append_message(m)

        return message

    def add_tool_response(self, tool_call_id: str, content: Any) -> None:
        """
        Add a tool response to the conversation history.

        Args:
            tool_call_id: ID of the tool call this response is for
            content: Response content from the tool
        """
        self.append_message(self.model_interface.tool_response(tool_call_id, content))

    def reset(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        self.tool_call_ids = []


class StudentAttempt(BaseModel):
    """
    Container for a single problem or episode attempt by the student.
    """

    problem_spec: Any
    student_solution: Any
    actual_solution: Any
    metadata: Dict = Field(default_factory=dict)
    moves_taken: Optional[List[Any]] = None
    score: Optional[float] = None

    def to_failure_string(self) -> str:
        """
        Summarize if the attempt failed, including chain-of-thought.
        """
        if self.score is not None and self.score >= 1.0:
            return ""

        chain_of_thought_str = ""
        if self.moves_taken is not None:
            for i, move in enumerate(self.moves_taken):
                chain_of_thought_str += f"  Move {i+1}:\n"
                for j, step in enumerate(move.steps):
                    chain_of_thought_str += f"    Step {j+1}:\n"
                    chain_of_thought_str += f"      Explanation: {step.explanation}\n"
                    chain_of_thought_str += f"      Output: {step.output}\n"
                chain_of_thought_str += f"    Final decision: {move.final_answer}\n"

        path_str = ""
        if "path" in self.metadata:
            path_str = f"Path taken: {self.metadata['path']}\n"

        return (
            f"FAILURE on problem:\n"
            f"  {self.problem_spec}\n"
            f"Expected: {self.student_solution}\n"
            f"Got: {self.actual_solution}\n"
            f"{path_str}"
            f"Student's moves:\n{chain_of_thought_str}\n"
        )
