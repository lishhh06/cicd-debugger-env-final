from __future__ import annotations

import re
import textwrap
from typing import Iterable


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a CI/CD pipeline debugger assistant.
    Return exactly one single-line action describing the next debugging move.
    Do not output markdown. Do not include explanations.
    """
).strip()

JUDGE_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a strict CI/CD judge.
    Return JSON only with keys correctness, minimalism, quality and values in [0,1].
    """
).strip()

REQUIRED_ACTIONS = (
    "read_file",
    "read_logs",
    "analyze_error",
    "edit_config",
    "run_pipeline_stage",
    "run_tests",
    "validate_fix",
    "submit_solution",
)


def build_user_prompt(
    step: int,
    config_text: str,
    error_message: str,
    history: list[str],
    available_actions: Iterable[str] | None = None,
) -> str:
    history_text = "\n".join(history[-5:]) if history else "None"
    actions_text = ", ".join(available_actions) if available_actions else ", ".join(REQUIRED_ACTIONS)

    return textwrap.dedent(
        f"""
        Step: {step}

        Current config:
        {config_text}

        Current error:
        {error_message}

        Recent history:
        {history_text}

        Available action categories:
        {actions_text}

        Output one actionable single-line fix/debug action.
        """
    ).strip()


def sanitize_action_text(raw_text: str, fallback: str = "read logs and analyze failing command") -> str:
    text = (raw_text or "").strip()
    if not text:
        return fallback
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text or fallback


def heuristic_action(
    config_text: str,
    error_message: str,
    available_actions: Iterable[str] | None = None,
    history: list[str] | None = None,
) -> str:
    lower_cfg = (config_text or "").lower()
    lower_err = (error_message or "").lower()
    seen = _extract_seen_tools(history or [])
    allowed = {item.strip() for item in (available_actions or REQUIRED_ACTIONS)}

    def has_tool(name: str) -> bool:
        return name in allowed

    if has_tool("read_logs") and "read_logs" not in seen:
        return "read_logs: inspect failing stage logs"

    if has_tool("analyze_error") and "analyze_error" not in seen:
        return "analyze_error: identify root cause from logs and config"

    if has_tool("edit_config") and "npm tset" in lower_cfg:
        return "edit_config: replace npm tset with npm test"

    if has_tool("edit_config") and ("yaml" in lower_err or "mapping values are not allowed" in lower_err):
        return "edit_config: fix YAML indentation and syntax"

    if has_tool("edit_config") and ("module not found" in lower_err or "dependency" in lower_err):
        return "edit_config: repair dependency install and test commands"

    if has_tool("run_pipeline_stage") and "run_pipeline_stage" not in seen:
        return "run_pipeline_stage: run test stage"

    if has_tool("run_tests") and "run_tests" not in seen:
        return "run_tests: execute full pipeline tests"

    if has_tool("validate_fix") and "validate_fix" not in seen:
        return "validate_fix: check deterministic, hidden, and quality scores"

    if has_tool("submit_solution"):
        return "submit_solution: submit current configuration"

    return "read_logs: inspect failing stage logs and identify root cause"


def _extract_seen_tools(history: list[str]) -> set[str]:
    seen: set[str] = set()
    for item in history:
        for tool in REQUIRED_ACTIONS:
            if re.search(rf"\b{re.escape(tool)}\b", item):
                seen.add(tool)
    return seen
