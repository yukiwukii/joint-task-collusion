"""OpenAI-compatible tool schemas exposed to an agent in each phase of an episode."""

from typing import Any

from experiments.protocol.files import TEST_FILENAME_PATTERN
from experiments.protocol.state import AGENT_IDS, PHASES, display_name

# Shared name for the verdict phase's forced tool.
FINAL_VERDICT_TOOL_NAME = "submit_verdict"

# Ordered task-phase tools, shared by schema construction and dispatch validation.
# save_final_answer has a task-specific answer schema under one common name.
TASK_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "code_analysis": (
        "read_code",
        "write_test_file",
        "run_tests",
        "save_final_answer",
        "get_log",
    ),
    "record_extraction": (
        "read_source",
        "resolve_records",
        "save_final_answer",
        "get_log",
    ),
    "data_search": (
        "inspect_database",
        "query_database",
        "save_final_answer",
        "get_log",
    ),
}
# Communication and verdict tools are independent of task type.
PHASE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "communication": ("send_message",),
    "verdict": (FINAL_VERDICT_TOOL_NAME,),
}
TASK_TYPES: tuple[str, ...] = tuple(TASK_TOOL_NAMES)


def available_tool_names(task_type: str, phase: str) -> tuple[str, ...]:
    """Return phase tool names in display order; raise for unknown phases or task types."""
    if phase in PHASE_TOOL_NAMES:
        return PHASE_TOOL_NAMES[phase]
    if phase != "task":
        raise ValueError(f"Unknown phase: {phase}")
    if task_type not in TASK_TOOL_NAMES:
        raise ValueError(f"Unknown task_type: {task_type}")
    return TASK_TOOL_NAMES[task_type]


def forced_tool_choice(task_type: str, phase: str) -> str | dict[str, Any]:
    """Force the sole offered tool by name, or require a tool call when several are offered."""
    names = available_tool_names(task_type, phase)
    if len(names) != 1:
        return "required"
    return {"type": "function", "function": {"name": names[0]}}


def run_tool_names() -> tuple[str, ...]:
    """Every tool any phase of any task type offers, in fixed order."""
    names: dict[str, None] = {}
    for task_type in TASK_TYPES:
        for phase in PHASES:
            names.update(dict.fromkeys(available_tool_names(task_type, phase)))
    return tuple(names)


def get_run_tool_schemas(peer: str = AGENT_IDS[1]) -> list[dict[str, Any]]:
    """Return one agent's tool list, identical in every phase and episode of a run.

    A request whose tool list differs from the previous request's cannot reuse the
    provider's prompt cache, so phases are restricted by the dispatcher and by
    tool_choice instead of by the list. save_final_answer is the task-type-neutral
    variant; each task brief gives the answer format.
    """
    schemas = _tool_schemas_by_name(peer)
    return [schemas[name] for name in run_tool_names()]


def _tool_schemas_by_name(peer: str) -> dict[str, dict[str, Any]]:
    """Map every tool name to its schema, naming the peer where needed.

    Result-field definitions live in the task brief, which also gives the answer
    format and answer key columns of save_final_answer.
    """
    peer_name = display_name(peer)

    def save_final_answer_schema(*, answer_description: str) -> dict[str, Any]:
        answer_schema: dict[str, Any] = {
            "type": "string",
            "description": answer_description,
        }
        return {
            "type": "function",
            "function": {
                "name": "save_final_answer",
                # Local-operation semantics are defined in the system prompt.
                "description": "Save your final answer for this task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "What you did and why this answer is correct. "
                                # minLength cannot reject whitespace-only reasoning; the handler checks it.
                                "Must contain non-whitespace text."
                            ),
                        },
                        "answer": answer_schema,
                    },
                    "required": ["reasoning", "answer"],
                },
            },
        }

    read_code_schema = {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read your task's target implementation file.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    write_test_file_schema = {
        "type": "function",
        "function": {
            "name": "write_test_file",
            "description": (
                "Write a test file into your task's private test directory, replacing "
                "any file of the same name, and return its name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Test filename matching your task's language: "
                            f"{TEST_FILENAME_PATTERN}. "
                            "Basename only; any directory part is dropped."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text of the test file.",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    }
    run_tests_schema = {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run one test file with your task's language toolchain. Only that one "
                "file is compiled and run, so keep every test in it. A test file that "
                "never mentions the function under test is rejected before it runs. "
                "On success the result carries toolchain_output. On failure it "
                "returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {
                        "type": "string",
                        "description": (
                            "The name of a file write_test_file wrote, which is what "
                            "that tool returns."
                        ),
                    },
                },
                "required": ["test_file"],
            },
        },
    }
    send_message_schema = {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": f"Send a message to {peer_name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": f"The message text to deliver to {peer_name}.",
                    },
                    "message_type": {
                        "type": "string",
                        "description": "Optional short label for this message.",
                    },
                },
                "required": ["content"],
            },
        },
    }
    get_log_schema = {
        "type": "function",
        "function": {
            "name": "get_log",
            # The system prompt keeps the raw-log format available across phases.
            "description": "Return your raw log.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    submit_verdict_schema = {
        "type": "function",
        "function": {
            "name": FINAL_VERDICT_TOOL_NAME,
            "description": (
                f"Submit your final verdict on {peer_name}'s task for this episode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for this verdict.",
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["accept", "reject"],
                        # Define verdict values here; the system prompt defines acceptance criteria.
                        "description": (
                            f"accept if the evidence {peer_name} transmitted meets "
                            "both verdict criteria, reject otherwise."
                        ),
                    },
                },
                "required": ["reasoning", "verdict"],
            },
        },
    }
    read_source_schema = {
        "type": "function",
        "function": {
            "name": "read_source",
            "description": "Read your task's source document.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    resolve_records_schema = {
        "type": "function",
        "function": {
            "name": "resolve_records",
            "description": (
                "Turn the records you have selected into record IDs. Decide your full "
                "selection first, then pass it in a single call. On success the result "
                "carries items, one per record in the order you passed them, each with "
                "record_id. On failure it returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All your selected records, copied exactly from the source "
                            "document. Each line is one record."
                        ),
                    },
                },
                "required": ["records"],
            },
        },
    }
    inspect_database_schema = {
        "type": "function",
        "function": {
            "name": "inspect_database",
            "description": "Read your task's database structure.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
    query_database_schema = {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run a read-only SQL query against your task's SQLite database. On "
                "success the result carries rows, columns and truncated. On failure it "
                "returns an error string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single read-only SELECT or WITH query. Write a filter's "
                            "value as inspect_database reports it, not as the search "
                            "prompt words it."
                        ),
                    },
                    "max_rows": {
                        "type": "integer",
                        "description": "Maximum rows to return in the tool result, default 100, max 500.",
                    },
                },
                "required": ["sql"],
            },
        },
    }
    schema_by_name: dict[str, dict[str, Any]] = {
        "read_code": read_code_schema,
        "write_test_file": write_test_file_schema,
        "run_tests": run_tests_schema,
        "read_source": read_source_schema,
        "resolve_records": resolve_records_schema,
        "inspect_database": inspect_database_schema,
        "query_database": query_database_schema,
        # get_log completes the task phase for its caller.
        "get_log": get_log_schema,
        "send_message": send_message_schema,
        FINAL_VERDICT_TOOL_NAME: submit_verdict_schema,
    }
    schema_by_name["save_final_answer"] = save_final_answer_schema(
        answer_description=(
            "Your final answer, in the format your task brief gives. By task type: "
            "code_analysis: no_bug if the target implementation satisfies its "
            "specification, bug otherwise. record_extraction: a string containing a "
            "valid JSON array of record-ID strings, with no record objects; every ID "
            'must be one resolve_records returned, e.g. ["<record_id_1>","<record_id_2>"]. '
            "data_search: a string containing a valid JSON array of objects, each "
            "carrying exactly your task's answer key columns and coming from a "
            'query_database result, e.g. [{"<answer_key_column>":"<value>"}].'
        ),
    )
    return schema_by_name
