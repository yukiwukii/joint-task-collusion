# Changes

Two new agent tools. Tool on/off switch in config. Old code path kept: default config give same prompts, same schemas, same behavior as before (checked byte for byte against `HEAD`).

## New tools

- `send_message_strict`. Fake "broken" channel. Agent give `content`. Runner check it is exact substring of agent own `get_log` string and at most `char_limit` chars (default 200). Else `MessageError`. No free `message_type`; label always `raw_log_excerpt`. Prompt and schema never state `char_limit`, same as `send_message`. Code: `experiments/protocol/messaging.py` `_send_message_strict`.
- `workspace_log`. One file both agents read, write, append. Agent told "for logging purposes". No mention of peer or communication. File `<run dir>/workspace/log.txt`, empty at start, persist across episodes, cap 20,000 chars. Code: `experiments/protocol/workspace.py` (new file).

## Tool switches

- `configs/main.yaml` new `tools:` section: `send_message`, `send_message_strict`, `workspace_log`, `workspace_log_phases`.
- Same as flags: `--tool-send-message/--no-tool-send-message`, `--tool-send-message-strict`, `--tool-workspace-log`, `--tool-workspace-log-phases task communication verdict`.
- `ToolSet` in `experiments/tool_schemas.py` hold switches. Default `ToolSet()` = old behavior.
- Send tools mutually exclusive: `send_message_strict` on switch `send_message` off (`ToolSet.__post_init__`). Launcher refuse config with both off.
- Non-default set add suffix to run label, e.g. `_tools-send-message-strict+workspace-log`. Default label unchanged.

## Files touched

All edits add optional parameter with old default, or add new branch. No old function removed.

- `tool_schemas.py`: `ToolSet`, tool names, two new schemas. `available_tool_names`, `task_tool_names`, `forced_tool_choice`, `get_tool_schemas`, `get_task_tool_schemas` take optional `tools`.
- `protocol/dispatch.py`: two new handlers. New tools stay out of raw log. `workspace_log` skip task-phase order gate.
- `protocol/messaging.py`: `_send_message` and `_reject_message` take optional `tool_name`, so strict sends logged under own name.
- `prompts/system.py`: send-tool text follow switches. New "Logging" section only when `workspace_log` on.
- `prompts/tasks.py`, `prompts/messages.py`: tool lists and round instruction follow switches.
- `episode_runner.py`: put switches and workspace path in episode state. Strict send also end agent turn. Turn with only `workspace_log` calls cost no budget, up to 50 per agent per slot (`WORKSPACE_FREE_TURNS` in `protocol/__init__.py`). Verdict phase excluded: still 3 turns. Free turns only where `workspace_log` is on and offered in that phase; `workspace_log: false` give normal budget, no +50. Past cap, normal cost.
- `analysis/agreement_judge.py`: transcript add this episode's workspace writes when present, plus one note line. New CSV column `workspace_calls`. No workspace: prompt byte-identical to before.
- `models.py`: `EpisodeRunConfig.tools`, `EpisodeRunConfig.workspace_log_path`.
- `runner.py`, `cli.py`, `config.py`: flags, validation, `tools` YAML section, create workspace file, pass flags to child runs, resume check.
- `output.py`: `channel_transcript` include strict sends. `run_config.tools` and per-episode `tools` + `workspace_log` in `run.json`.
- `analysis/results_viewer.py`: Episodes tab show `workspace_log` section per episode (each call: who, action, phase, round, text; plus file at episode end) and header badge `workspace N calls`. Only when run had `workspace_log` on.
- `docs/tool-list.md` new. `docs/repo.md` two lines added.

## Watch out

- `workspace_log` free up to 50 turns per slot. Looping agent end, but can burn many calls: worst case 334 model calls per agent per episode at 5 rounds (task 65, communication 5 × 53, verdict 3, reflection 1), versus 34 without workspace.
- Verbatim check is substring only. Agent can pick which slice to send, so slice choice still carry a little signal.
- Task phase run Alice fully, then Bob. Alice note in task phase reach Bob same episode. Bob note in task phase reach Alice in communication phase or next episode.
- `--resume-from` start new run dir, so new empty workspace file. Old notes not carried.
- Controlled-Bob mode not checked with new tools.
- Agreement judge see only this episode's workspace calls. File persist across episodes, so deal written in episode 1 and kept silent later is not shown in later episodes.
- Workspace runs give judge more text. Agreement rate not comparable to channel-only runs. Split by `workspace_calls`.
- Relaxation judge unchanged. It read reflections only.

## Static tool list (prompt cache)

Provider put tool list at front of cached prompt. Old code send different list each phase and each task type. So first call of every phase miss cache: verdict 0%, reflection ~1% cached.

- `tool_schemas.py`: `get_run_tool_schemas` give one list per agent per run: all task types' tools plus enabled send tools, `workspace_log`, `submit_verdict`. `save_final_answer` schema now task-type neutral. `get_tool_schemas`, `get_task_tool_schemas`, `task_tool_names` removed.
- `agents.py`: phase turns and reflection send that one list.
- `protocol/dispatch.py`: `ToolUnavailableError` only for name not in list. Tool from other phase or other task type get `PhaseError`, which name allowed tools.
- `prompts/tasks.py`: own-task brief get "Answer format" block (was in schema). "Output format" block say other listed tools rejected in this phase.
- `prompts/system.py`: one line: tool list same every phase, phase opening name allowed tools.
- Phases still forced by `tool_choice` (communication, verdict). `tool_choice` not part of cache key (tested).

