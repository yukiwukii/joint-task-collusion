"""Inter-agent messaging and per-agent raw-log retrieval."""

import json
from typing import Any

from experiments.protocol.errors import error_string
from experiments.protocol.state import _log_event, agent_state, peer_id


def _send_message(
    state: dict[str, Any],
    actor: str,
    content: str,
    message_type: str = "other",
    tool_name: str = "send_message",
) -> dict[str, Any]:
    slot = agent_state(state, actor)
    if slot["delivered"]:
        return _reject_message(
            state=state,
            actor=actor,
            content=content,
            message_type=message_type,
            tool_name=tool_name,
            error=error_string(
                "MessageError",
                f"{actor} already sent its one message for this round",
            ),
        )

    if (
        state["throttled"]
        and len(content) > state["char_limit"]
    ):
        return _reject_message(
            state=state,
            actor=actor,
            content=content,
            message_type=message_type,
            tool_name=tool_name,
            error=error_string(
                "MessageError", f"exceeds {state['char_limit']} char limit"
            ),
        )

    slot["delivered"] = True
    recipient = peer_id(actor)
    # Keep the originating 0-based round; display tags add one.
    payload: dict[str, Any] = {
        "from": actor,
        "to": recipient,
        "content": content,
        "round": state["round"],
    }
    payload["message_type"] = message_type
    # Deliver immediately so the peer can reply within the same round.
    agent_state(state, recipient)["inbox"].append(payload)

    result = {"success": True, "chars_sent": len(content)}
    event = {
        "actor": actor,
        "recipient": recipient,
        "tool": tool_name,
        "success": True,
        "content": content,
    }
    event["message_type"] = message_type
    _log_event(state, event)
    return result


def _reject_message(
    *,
    state: dict[str, Any],
    actor: str,
    content: str,
    message_type: str,
    error: str,
    tool_name: str = "send_message",
) -> dict[str, Any]:
    result = {"success": False, "error": error, "chars_sent": 0}
    # Record the complete attempted message, including refused deliveries.
    event = {
        "actor": actor,
        "recipient": peer_id(actor),
        "tool": tool_name,
        "success": False,
        "error": error,
        "content": content,
    }
    event["message_type"] = message_type
    _log_event(state, event)
    return result


def _send_message_strict(
    state: dict[str, Any],
    actor: str,
    content: str,
) -> dict[str, Any]:
    """Deliver ``content`` only if it is a verbatim substring of the sender's raw log.

    A deliberately crippled channel: anything that is not copied exactly from the
    get_log string is refused. The type label is fixed, so no free text reaches the
    peer. The char limit applies in every episode, throttled or not.
    """
    tool_name = "send_message_strict"
    message_type = "raw_log_excerpt"
    raw_log = _format_private_raw_log(agent_state(state, actor))
    error = None
    if not content:
        error = "content is empty"
    elif len(content) > state["char_limit"]:
        error = f"exceeds {state['char_limit']} char limit"
    elif content not in raw_log:
        error = "content is not a verbatim excerpt of your raw log"
    if error is not None:
        return _reject_message(
            state=state,
            actor=actor,
            content=content,
            message_type=message_type,
            tool_name=tool_name,
            error=error_string("MessageError", error),
        )
    return _send_message(
        state=state,
        actor=actor,
        content=content,
        message_type=message_type,
        tool_name=tool_name,
    )


def _format_private_raw_log(slot: dict[str, Any]) -> str:
    return json.dumps(
        slot.get("private_raw_log", []),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _get_log(state: dict[str, Any], actor: str) -> dict[str, Any]:
    """Return only the calling agent's raw log.

    The dispatch gate permits one retrieval per agent per episode.
    """
    slot = agent_state(state, actor)
    raw_log = _format_private_raw_log(slot)
    slot["private_raw_log_retrieved"] = True
    result = {
        "success": True,
        "raw_log": raw_log,
        "raw_log_chars": len(raw_log),
    }
    _log_event(
        state,
        {
            "actor": actor,
            "tool": "get_log",
            "success": True,
        },
    )
    return result
