from pathlib import Path


def load_world_model(world_model_path: str | None) -> str | None:
    """Load world model text from a file path."""
    if world_model_path is None:
        return None
    path = Path(world_model_path)
    if not path.exists():
        raise FileNotFoundError(f"World model file not found: {world_model_path}")
    return path.read_text()


def build_world_model_section(world_model: str | None) -> str:
    """Build the world model section for prompts."""
    if not world_model:
        return ""
    return f"""\
## Codebase Guidance

The following guidance has been provided about this codebase to help you understand its patterns, conventions, and structure:

<world_model>
{world_model}
</world_model>

---

"""


def build_message_limit_info(message_limit: int | None, item_number: int = 3) -> str:
    """Build the message limit info for prompts."""
    if message_limit is None:
        return ""
    return f"""
{item_number}. You have a maximum of {message_limit} message turns to complete this task. Plan your approach carefully and use your turns efficiently. Prioritize understanding the codebase structure and test requirements before implementing.
"""
