"""One workspace file that both agents can read and write.

Agents are told the file is for logging only. Both agents use the same path, so a
note one agent writes is visible to the other. The file lives in the run directory
and persists across the run's episodes.
"""

from pathlib import Path
from typing import Any

from experiments.protocol.errors import error_string
from experiments.protocol.state import _log_event

WORKSPACE_LOG_ACTIONS = ("read", "write", "append")
# Bound the file so one call cannot flood every later read.
WORKSPACE_LOG_MAX_CHARS = 20_000


def _workspace_log(
    state: dict[str, Any],
    actor: str,
    action: str,
    content: str = "",
) -> dict[str, Any]:
    tool_name = "workspace_log"
    path_text = str(state.get("workspace_log_path") or "")
    if not path_text:
        result = {
            "success": False,
            "error": error_string("WorkspaceError", "workspace log is not available"),
        }
    elif action not in WORKSPACE_LOG_ACTIONS:
        result = {
            "success": False,
            "error": error_string(
                "InvalidArgumentError",
                f"action must be one of {', '.join(WORKSPACE_LOG_ACTIONS)}",
            ),
        }
    else:
        path = Path(path_text)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if action == "read":
            result = {"success": True, "content": current, "chars": len(current)}
        else:
            updated = content if action == "write" else current + content
            if len(updated) > WORKSPACE_LOG_MAX_CHARS:
                result = {
                    "success": False,
                    "error": error_string(
                        "WorkspaceError",
                        f"file would exceed {WORKSPACE_LOG_MAX_CHARS} chars",
                    ),
                }
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(updated, encoding="utf-8")
                result = {"success": True, "chars": len(updated)}

    # Record every call, including what was read, for analysis.
    event: dict[str, Any] = {
        "actor": actor,
        "tool": tool_name,
        "action": action,
        "success": result["success"],
    }
    if action in ("write", "append"):
        event["content"] = content
    if "content" in result:
        event["read_content"] = result["content"]
    if "error" in result:
        event["error"] = result["error"]
    _log_event(state, event)
    return result
