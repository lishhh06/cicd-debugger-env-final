from __future__ import annotations

import json
from pathlib import Path


def save_reward_curve(rewards: list[float], output_path: str = "artifacts/reward_curve.csv") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("step,reward\n")
        for idx, reward in enumerate(rewards, start=1):
            handle.write(f"{idx},{float(reward):.4f}\n")

    return str(path)


def save_success_rate_history(success_flags: list[bool], output_path: str = "artifacts/success_rate.csv") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    running = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write("episode,success,success_rate\n")
        for idx, flag in enumerate(success_flags, start=1):
            if flag:
                running += 1
            rate = running / idx
            handle.write(f"{idx},{int(flag)},{rate:.4f}\n")

    return str(path)


def save_metrics_json(metrics: dict, output_path: str = "artifacts/metrics.json") -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    return str(path)
