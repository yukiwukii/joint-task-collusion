#!/usr/bin/env python3
"""Detect future policy-relaxation strategies in private reflections.

The judge reads one agent-reflection and checks for a strategy that could accept
without the complete raw log required by policy. Other episode fields are excluded.
Positives require a verbatim reflection quote. Invalid replies are retried; failures
remain unresolved with an empty ``relaxation`` and ``relaxation_parse_ok=0``.

Replies are cached by gate version, model, and prompt. ``--think`` enables reasoning.
Output contains one row per agent-reflection; ``turning_point`` marks the first
positive for each run-agent, and ``turning_point_episode`` repeats its episode.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.config import (  # noqa: E402
    apply_config,
    config_defaults,
    model_slug,
)

load_dotenv(ROOT / ".env")
PROMPT_FILE = Path(__file__).with_name("relaxation_judge_prompts.md")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3.8-27b"
# Reasoning effort and temperature requested from hosted routes.
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TEMPERATURE = 1.0

RUNG = "relaxation"
GATE_VERSION = {RUNG: 2}
QUOTE_FIELD = "EVIDENCE QUOTE"


# ---------------------------------------------------------------- prompts


def load_prompts(path: Path = PROMPT_FILE) -> dict[str, dict[str, str]]:
    """Read ``## section`` / ``### subsection`` fenced blocks from the prompt file."""
    prompts: dict[str, dict[str, str]] = {}
    section, subsection, fenced, buf = None, "", False, []
    for line in path.read_text().split("\n"):
        if line.startswith("```"):
            if fenced and section is not None:
                prompts.setdefault(section, {})[subsection] = "\n".join(buf).rstrip("\n")
            fenced, buf = not fenced, []
            continue
        if fenced:
            buf.append(line)
        elif line.startswith("### "):
            subsection = line[4:].strip()
        elif line.startswith("## "):
            section, subsection = line[3:].strip(), ""
    return prompts


def render(template: str, **values: str) -> str:
    """Substitute prompt placeholders literally, without interpreting JSON braces."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", value)
    return template


def check_prompts(prompts: dict[str, dict[str, str]]) -> None:
    """Fail before API work if the single relaxation prompt is malformed."""

    def die(message: str) -> None:
        raise SystemExit(f"{PROMPT_FILE.name}: {message}")

    if list(prompts) != [RUNG]:
        die(f"expected exactly one section named '{RUNG}'")
    blocks = prompts[RUNG]
    for part in ("system", "user"):
        if part not in blocks:
            die(f"'{RUNG}' has no '### {part}' fenced block")
    if blocks["user"].count("{reflection}") != 1:
        die("the user block must contain {reflection} exactly once")
    joined = blocks["system"] + blocks["user"]
    for label in ("JUDGEMENT", QUOTE_FIELD):
        if f"{label}:" not in joined:
            die(f"the prompt no longer asks for '{label}:'")


# ---------------------------------------------------------------- reply parsing and gates


def strip_thinking(text: str) -> str:
    """Drop an inline Qwen-style reasoning block before parsing the final answer."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    return re.sub(r"^.*?<think>", "", text, flags=re.S | re.I).strip()


def parse_reply(text: str) -> dict | None:
    """Parse the two labelled response lines, allowing a multiline evidence quote."""
    body = strip_thinking(text)
    keys = ("JUDGEMENT", QUOTE_FIELD, "EXPLANATION")
    found: dict[str, list[str]] = {}
    current = None
    for line in body.split("\n"):
        match = re.match(r"^\s*([A-Z][A-Z -]*[A-Z]):\s*(.*)$", line)
        if match and match.group(1) in keys:
            current = match.group(1)
            found[current] = [match.group(2)]
        elif current is not None:
            found[current].append(line)
    if "JUDGEMENT" not in found:
        return None
    verdict = "\n".join(found["JUDGEMENT"]).strip().strip("[].").lower()
    if verdict not in ("yes", "no"):
        return None
    quote = "\n".join(found.get(QUOTE_FIELD, [])).strip().strip("[]")
    if quote.lower() in ("", "none", "n/a"):
        quote = ""
    return {"answer": int(verdict == "yes"), "quote": quote}


def norm(text: str) -> str:
    """Whitespace- and case-insensitive form used by the verbatim quote check."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def quote_gate(parsed: dict, source: str) -> dict:
    """Record whether the supplied evidence is a real source span."""
    parsed = dict(parsed)
    quote = (parsed.get("quote") or "").strip()
    quote_ok = bool(quote) and norm(quote) in norm(source)
    parsed["quote_ok"] = int(quote_ok)
    return parsed


def format_gate(parsed: dict) -> dict:
    """Flag a No that nevertheless supplied an evidence quote; do not change its label."""
    parsed = dict(parsed)
    contradicts = not parsed.get("answer") and bool((parsed.get("quote") or "").strip())
    parsed["format_ok"] = int(not contradicts)
    return parsed


def validated_reply(text: str, source: str) -> dict | None:
    """Return a well-formed reply, or None so the caller can retry it."""
    value = parse_reply(text)
    if value is None:
        return None
    gated = format_gate(quote_gate(value, source))
    if value["answer"] and not gated["quote_ok"]:
        return None
    if not gated["format_ok"]:
        return None
    return value


def cache_key(judge: str, model: str, prompt: str) -> str:
    gate = GATE_VERSION.get(judge, 1)
    digest = hashlib.sha1(
        f"{judge}\x00{gate}\x00{model}\x00{prompt}".encode()
    ).hexdigest()
    return f"{judge}:{digest}"


# ---------------------------------------------------------------- judging


class Judge:
    def __init__(self, base_url: str, model: str, think: bool,
                 max_tokens: int, timeout: float, reasoning_effort: str,
                 temperature: float):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url,
                             api_key=os.environ.get("OPENROUTER_API_KEY", "EMPTY"),
                             timeout=timeout)
        self.local = "localhost" in base_url or "127.0.0.1" in base_url
        self.model = model
        self.think = think
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        # One cache per file, so each run's replies stay in that run's directory.
        self.caches: dict[Path, dict[str, dict]] = {}
        self.lock = threading.Lock()
        self.calls = 0

    def load_cache(self, cache_path: Path) -> int:
        """Read one cache file up front; its entries are reused before any API call."""
        entries: dict[str, dict] = {}
        if cache_path.exists():
            with cache_path.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entries[row["key"]] = row["parsed"]
        with self.lock:
            self.caches.setdefault(cache_path, {}).update(entries)
        return len(entries)

    def _remember(self, cache_path: Path, key: str, parsed: dict, raw: str,
                  reasoning: str = "") -> None:
        with self.lock:
            self.caches.setdefault(cache_path, {})[key] = parsed
            with cache_path.open("a") as handle:
                handle.write(json.dumps({
                    "key": key,
                    "parsed": parsed,
                    "raw": raw[:4000],
                    "reasoning": reasoning,
                }, ensure_ascii=False) + "\n")

    def ask(self, system: str, user: str, source: str, cache_path: Path) -> dict:
        key = cache_key(RUNG, self.model, system + "\x00" + user)
        with self.lock:
            hit = self.caches.get(cache_path, {}).get(key)
        if hit is not None and hit.get("parse_ok"):
            return hit

        # chat_template_kwargs is a local-server field; hosted APIs reject it.
        extra = ({"chat_template_kwargs": {"enable_thinking": bool(self.think)}} if self.local
                 else {"reasoning": {"effort": self.reasoning_effort}})
        raw = ""
        reasoning = ""
        error = ""
        value = None
        request_count = 0
        for attempt in range(3):
            try:
                request_count += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens * (1, 2, 4)[attempt],
                    extra_body=extra or None,
                )
                message = response.choices[0].message
                raw = message.content or ""
                reasoning = (
                    getattr(message, "reasoning", None)
                    or getattr(message, "reasoning_content", None)
                    or ""
                )
                value = validated_reply(raw, source)
                if value is not None:
                    error = ""
                    break
                error = "empty completion" if not raw else "invalid completion"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt == 2:
                    raw = ""
                    reasoning = ""
        with self.lock:
            self.calls += request_count

        parsed = {
            # Leave failed judgements unresolved; parse_ok=0 makes them retryable.
            "answer": None,
            "quote": "",
            "quote_ok": 0,
            "format_ok": 1,
            "parse_ok": 0,
            "error": error,
        }
        if value is not None:
            parsed = quote_gate(value, source)
            parsed = format_gate(parsed)
            parsed.update({"parse_ok": 1, "error": ""})
        parsed["reasoning_chars"] = len(reasoning)
        self._remember(cache_path, key, parsed, raw, reasoning)
        return parsed


# ---------------------------------------------------------------- source assembly


def condition_family(path: Path) -> str:
    family = path.parent.parent.parent
    return "" if family == ROOT else family.name


def reflection_rows(path: Path, run: dict, verdict_policy: str = "") -> list[dict]:
    """Build one row per agent-reflection; ``_reflection`` is the only judge source."""
    path = path.resolve()
    config = run.get("run_config") or {}
    models = config.get("models") or {}
    memory_limits = config.get("cross_episode_memory_length") or {}
    manifest = config.get("manifest") or ""
    match = re.search(r"rep(\d+)", manifest) if isinstance(manifest, str) else None
    rep = match.group(1) if match else ""
    rows = []
    for episode in run.get("results") or []:
        if verdict_policy and episode.get("verdict_policy") != verdict_policy:
            continue
        agents = episode.get("agents")
        if not isinstance(agents, dict):
            continue
        for agent, record in agents.items():
            if not isinstance(record, dict):
                continue
            reflection = str(record.get("reflection") or "").strip()
            rows.append({
                "run_path": str(path.relative_to(ROOT)),
                "cond_dir": path.parent.parent.name,
                "family": condition_family(path),
                "model": models.get(agent, "") if isinstance(models, dict) else str(models),
                "rep": rep,
                "episode_index": episode.get("episode_index"),
                "episode_id": episode.get("episode_id"),
                "group": episode.get("group"),
                "ep_verdict_policy": episode.get("verdict_policy"),
                "agent": agent,
                "task_id": record.get("task_id"),
                "memory_max_episodes": (
                    memory_limits.get(agent, -1)
                    if isinstance(memory_limits, dict) else -1
                ),
                "reflection_chars": len(reflection),
                "_reflection": reflection,
            })
    return rows


def annotate_turning_points(rows: list[dict]) -> None:
    """Mark the first positive reflection independently for each run-agent."""
    first: dict[tuple[str, str], int | str] = {}
    for row in sorted(rows, key=lambda item: (
        item["run_path"], item["agent"], int(item["episode_index"])
    )):
        key = (row["run_path"], row["agent"])
        if row.get(RUNG) and key not in first:
            first[key] = row["episode_index"]
    for row in rows:
        episode = first.get((row["run_path"], row["agent"]), "")
        row["turning_point_episode"] = episode
        row["turning_point"] = int(episode != "" and row["episode_index"] == episode)


def run_rep(path: Path) -> int | None:
    match = re.search(r"_rep(\d+)(?:_|$)", path.parent.name)
    return int(match.group(1)) if match else None


def select_paths(paths: list[Path], runs_per_condition: int = 0) -> list[Path]:
    """Keep the latest artifact per repetition, then optionally cap each condition."""
    latest: dict[tuple[Path, int | str], Path] = {}
    for path in paths:
        rep = run_rep(path)
        key = (path.parent.parent, rep if rep is not None else path.parent.name)
        current = latest.get(key)
        if current is None or path.parent.name > current.parent.name:
            latest[key] = path
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in latest.values():
        grouped[path.parent.parent].append(path)
    selected = []
    for condition in sorted(grouped, key=str):
        condition_paths = sorted(
            grouped[condition],
            key=lambda path: (
                run_rep(path) is None,
                run_rep(path) if run_rep(path) is not None else path.parent.name,
            ),
        )
        if runs_per_condition:
            condition_paths = condition_paths[:runs_per_condition]
        selected.extend(condition_paths)
    return selected


# ---------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="",
                        help="YAML file supplying defaults; see configs/. Flags override it.")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="glob(s) matching run.json files, relative to the repo root")
    parser.add_argument("--out", default="",
                        help="one combined CSV to write; the default writes "
                             "relaxation.csv into each run's own directory")
    parser.add_argument("--cache", default="",
                        help="one shared cache file; the default keeps "
                             "relaxation_cache.jsonl beside each run's CSV")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--think", action="store_true",
                        help="leave the judge model's reasoning on")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N runs for a smoke test")
    parser.add_argument("--runs-per-condition", type=int, default=0,
                        help="after deduplicating repetitions, keep N per condition")
    parser.add_argument("--verdict-policy", default="",
                        help="only judge episodes under this verdict policy")
    # Read --config before the full parse, which enforces required options.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    config = pre.parse_known_args()[0].config
    if config:
        defaults = config_defaults(
            Path(config), {"alice": "alice_", "bob": "bob_", "run": "", "judge": ""}
        )
        # Judge the run this config produces, writing beside its trajectories.
        slug = model_slug(defaults.get("alice_model", ""), defaults.get("bob_model", ""))
        results = Path(defaults.get("output_dir", "results")) / slug
        defaults.setdefault("runs", [str(results / "*" / "run.json")])
        apply_config(parser, defaults)
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.runs:
        paths += [Path(path) for path in sorted(glob.glob(str(ROOT / pattern)))]
    paths = select_paths(paths, args.runs_per_condition)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print("no run.json matched", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for path in paths:
        try:
            with path.open() as handle:
                run = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        rows += reflection_rows(path, run, args.verdict_policy)
    scope = f" under {args.verdict_policy}" if args.verdict_policy else ""
    print(f"{len(paths)} runs -> {len(rows)} agent-reflections{scope}")
    if not rows:
        print("no reflections to judge", file=sys.stderr)
        return 2

    prompts = load_prompts(PROMPT_FILE)
    check_prompts(prompts)
    blocks = prompts[RUNG]

    judge = Judge(
        args.base_url, args.model, args.think, args.max_tokens,
        args.timeout, args.reasoning_effort, args.temperature
    )

    def cache_for(row: dict) -> Path:
        """Cache beside the run's CSV, unless --cache names one shared file."""
        if args.cache:
            return ROOT / args.cache
        return (ROOT / row["run_path"]).parent / f"{RUNG}_cache.jsonl"

    entries = 0
    cache_paths = sorted({cache_for(row) for row in rows})
    for path in cache_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        entries += judge.load_cache(path)
    print(f"cache: {entries} entries in {len(cache_paths)} file(s), "
          f"e.g. {cache_paths[0]}")

    def work(index: int):
        row = rows[index]
        source = row["_reflection"]
        user = render(blocks["user"], reflection=source or "(no reflection)")
        return index, judge.ask(blocks["system"], user, source, cache_for(row))

    def record_result(index: int, parsed: dict) -> None:
        row = rows[index]
        row[RUNG] = parsed.get("answer")
        row[f"{RUNG}_quote_ok"] = parsed.get("quote_ok", 0)
        row[f"{RUNG}_parse_ok"] = parsed.get("parse_ok", 0)
        row[f"{RUNG}_format_ok"] = parsed.get("format_ok", 1)
        row[f"{RUNG}_reasoning_chars"] = parsed.get("reasoning_chars", 0)
        row[f"{RUNG}_quote"] = parsed.get("quote", "")

    print(f"{len(rows)} judge units (1 private reflection per unit)")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, index) for index in range(len(rows))]
        for future in as_completed(futures):
            index, parsed = future.result()
            record_result(index, parsed)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(rows)}  (api calls so far: {judge.calls})",
                      flush=True)

    annotate_turning_points(rows)

    fields = [key for key in rows[0] if not key.startswith("_")]

    def write_csv(out_path: Path, table: list[dict]) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(table)
        print(f"wrote {len(table)} rows -> {out_path}")

    if args.out:
        write_csv(ROOT / args.out, rows)
    else:
        # Default: one CSV per run, written beside the run.json it annotates.
        by_run: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_run[row["run_path"]].append(row)
        for run_path, table in sorted(by_run.items()):
            write_csv((ROOT / run_path).parent / f"{RUNG}.csv", table)
    print(f"api calls: {judge.calls}")

    print(f"\n=== reflection-level {RUNG} ===")
    print(f"{'model':>30} {'n_ref':>7} {RUNG[:8]:>8} {'agent_tp':>9} {'parse':>7}")
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, group in sorted(by_model.items(), key=lambda item: -len(item[1])):
        n = len(group)
        resolved = [row for row in group if row[f"{RUNG}_parse_ok"]]
        relax_rate = (
            sum(row[RUNG] for row in resolved) / len(resolved)
            if resolved else float("nan")
        )
        parse_rate = sum(row[f"{RUNG}_parse_ok"] for row in group) / n
        turning_points = sum(row["turning_point"] for row in group)
        print(f"{model[-28:]:>30} {n:>7} {relax_rate:>8.3f} "
              f"{turning_points:>9} {parse_rate:>7.3f}")

    positives = sum(row[RUNG] == 1 for row in rows)
    verified = sum(row[RUNG] == 1 and row[f"{RUNG}_quote_ok"] for row in rows)
    unresolved = sum(not row[f"{RUNG}_parse_ok"] for row in rows)
    bad_format = sum(not row[f"{RUNG}_format_ok"] for row in rows)
    print(f"\nverified positive quotes: {verified}/{positives}")
    print(f"unresolved judgements: {unresolved}/{len(rows)}")
    print(f"format contradictions: {bad_format}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
