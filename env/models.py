from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Observation(BaseModel):
    task_id: str
    difficulty: str
    failure_stage: str
    actual_bug: str
    config: str
    logs: str
    error_message: str
    available_tools: list[str]
    progress_flags: dict[str, bool]
    file_modification_count: int
    hidden_test_pass_rate: float
    step_count: int
    last_action_error: str | None = None


class Action(BaseModel):
    tool: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_input(cls, raw_action: Any) -> "Action":
        if isinstance(raw_action, cls):
            return raw_action

        if isinstance(raw_action, str):
            raw = raw_action.strip()
            if not raw:
                return cls(tool="", payload={})

            if ":" in raw:
                tool_part, payload_part = raw.split(":", 1)
                return cls(tool=tool_part.strip().lower(), payload={"raw": payload_part.strip()})

            parts = raw.split(maxsplit=1)
            tool = parts[0].strip().lower() if parts else ""
            payload = {"raw": parts[1].strip()} if len(parts) > 1 else {}
            return cls(tool=tool, payload=payload)

        if isinstance(raw_action, dict):
            tool = str(raw_action.get("tool") or raw_action.get("action_type") or "").strip().lower()
            incoming_payload = raw_action.get("payload")

            if isinstance(incoming_payload, dict):
                payload: dict[str, Any] = dict(incoming_payload)
            elif incoming_payload is not None:
                payload = {"raw": str(incoming_payload)}
            elif "input" in raw_action:
                payload = {"raw": str(raw_action.get("input") or "").strip()}
            else:
                payload = {}

            return cls(tool=tool, payload=payload)

        return cls(tool="", payload={})


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)


class EnvStateSnapshot(BaseModel):
    initialized: bool
    task_id: str | None = None
    difficulty: str | None = None
    actual_bug: str | None = None
    correct_solution: str | None = None
    failure_stage: str | None = None
    step_count: int = 0
    done: bool = False
    progress_flags: dict[str, bool] = Field(default_factory=dict)
    file_modification_count: int = 0
    total_changed_lines: int = 0
    hidden_test_pass_rate: float = 0.0
    stage_results: dict[str, bool] = Field(default_factory=dict)
    failed_validations: int = 0
    last_action_error: str | None = None
    last_error: str | None = None