from __future__ import annotations

from env.tasks.easy import EASY_TASKS
from env.tasks.hard import HARD_TASKS
from env.tasks.medium import MEDIUM_TASKS
from env.tasks.task_types import CICDTask


def get_all_tasks() -> list[CICDTask]:
    return [*EASY_TASKS, *MEDIUM_TASKS, *HARD_TASKS]


def get_tasks_by_difficulty(difficulty: str | None) -> list[CICDTask]:
    if not difficulty:
        return get_all_tasks()

    normalized = difficulty.strip().lower()
    return [task for task in get_all_tasks() if task.difficulty.lower() == normalized]


def get_task_by_id(task_id: str | None) -> CICDTask | None:
    if not task_id:
        return None

    for task in get_all_tasks():
        if task.task_id == task_id:
            return task
    return None


__all__ = [
    "CICDTask",
    "get_all_tasks",
    "get_task_by_id",
    "get_tasks_by_difficulty",
]
