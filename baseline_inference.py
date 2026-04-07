from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


BASELINE_TASKS: list[tuple[str, str]] = [
    ("easy-command-typo", "easy"),
    ("medium-python-version", "medium"),
    ("hard-needs-order", "hard"),
]

END_PATTERN = re.compile(
    r"^\[END\] success=(true|false) steps=(\d+) score=(\d+\.\d{3}) rewards=(.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference on easy/medium/hard tasks")
    parser.add_argument("--tasks", default=",".join(task for task, _ in BASELINE_TASKS))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "8")))
    parser.add_argument("--policy-mode", choices=["sft", "imp", "direct"], default="imp")
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--benchmark", default=os.getenv("MY_ENV_V4_BENCHMARK", "cicd_debugger_env"))
    parser.add_argument("--offline", action="store_true", default=False)
    parser.add_argument("--force-local-env", action="store_true", default=True)
    parser.add_argument("--output", default="artifacts/baseline_scores.json")
    return parser.parse_args()


def should_run_offline(args: argparse.Namespace) -> bool:
    if args.offline:
        return True

    key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not key:
        return True

    return os.getenv("OFFLINE_INFERENCE", "0") == "1"


def parse_end_line(lines: list[str]) -> dict[str, Any]:
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line.startswith("[END] "):
            continue

        matched = END_PATTERN.match(line)
        if not matched:
            raise RuntimeError(f"Malformed END line: {line}")

        success = matched.group(1) == "true"
        steps = int(matched.group(2))
        score = float(matched.group(3))
        rewards_str = matched.group(4).strip()

        rewards: list[float] = []
        if rewards_str:
            rewards = [float(value) for value in rewards_str.split(",") if value]

        return {
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
            "end_line": line,
        }

    raise RuntimeError("No END line found in inference output")


def run_single_task(
    task_id: str,
    difficulty: str,
    args: argparse.Namespace,
    project_root: Path,
    offline_mode: bool,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "inference.py",
        "--task",
        task_id,
        "--benchmark",
        str(args.benchmark),
        "--max-steps",
        str(max(1, int(args.max_steps))),
        "--policy-mode",
        str(args.policy_mode),
        "--trajectories",
        str(max(1, int(args.trajectories))),
    ]

    if offline_mode:
        command.append("--offline")
    if args.force_local_env:
        command.append("--force-local-env")

    env = os.environ.copy()
    if offline_mode:
        env["OFFLINE_INFERENCE"] = "1"

    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    summary = parse_end_line(lines)

    return {
        "task_id": task_id,
        "difficulty": difficulty,
        "success": summary["success"],
        "steps": summary["steps"],
        "score": summary["score"],
        "rewards": summary["rewards"],
        "start_line": next((line for line in lines if line.startswith("[START] ")), ""),
        "end_line": summary["end_line"],
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    known_difficulties = {task: difficulty for task, difficulty in BASELINE_TASKS}
    requested_tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]

    if not requested_tasks:
        print("No tasks provided for baseline run", file=sys.stderr)
        return 1

    offline_mode = should_run_offline(args)

    print(
        f"[BASELINE] mode={'offline' if offline_mode else 'openai'} tasks={len(requested_tasks)} "
        f"max_steps={max(1, int(args.max_steps))} policy={args.policy_mode}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for task_id in requested_tasks:
        difficulty = known_difficulties.get(task_id, "custom")
        try:
            result = run_single_task(task_id, difficulty, args, project_root, offline_mode)
            results.append(result)
            print(
                f"[BASELINE] task={task_id} difficulty={difficulty} success={str(result['success']).lower()} "
                f"score={result['score']:.3f} steps={result['steps']}",
                flush=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[BASELINE] task={task_id} failed with return code {exc.returncode}", file=sys.stderr)
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
            return exc.returncode or 1
        except Exception as exc:
            print(f"[BASELINE] task={task_id} failed: {exc}", file=sys.stderr)
            return 1

    average_score = sum(item["score"] for item in results) / len(results)
    success_rate = sum(1 for item in results if item["success"]) / len(results)

    payload = {
        "mode": "offline" if offline_mode else "openai",
        "model_name": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
        "api_base_url": os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        "max_steps": max(1, int(args.max_steps)),
        "policy_mode": str(args.policy_mode),
        "trajectories": max(1, int(args.trajectories)),
        "average_score": round(float(average_score), 3),
        "success_rate": round(float(success_rate), 3),
        "results": results,
    }

    output_path = project_root / str(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[BASELINE] average_score={payload['average_score']:.3f} success_rate={payload['success_rate']:.3f}", flush=True)
    print(f"[BASELINE] wrote {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
