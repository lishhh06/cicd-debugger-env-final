from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class CICDTask:
    """Represents a single CI/CD debugging scenario."""

    task_id: str
    title: str
    description: str
    difficulty: str
    failure_stage: str
    broken_config: str
    expected_config: str
    logs: str
    error_message: str
    actual_bug: str
    metadata: dict[str, Any] = field(default_factory=dict)
    deterministic_grader: Callable | None = None
    llm_grader: Callable | None = None
