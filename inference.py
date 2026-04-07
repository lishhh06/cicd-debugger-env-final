from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from openai import OpenAI

from env.environment import CICDDebuggerEnvironment, REQUIRED_TOOLS
from inference.metrics import EpisodeMetrics
from inference.model_wrapper import ModelWrapper, score_action_candidate
from inference.prompts import heuristic_action
from inference.visualize import save_metrics_json, save_reward_curve, save_success_rate_history


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = OPENAI_API_KEY or HF_TOKEN
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
DEFAULT_TASK_ID = os.getenv("MY_ENV_V4_TASK", "easy-command-typo")
DEFAULT_BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "cicd_debugger_env")

MAX_STEPS_DEFAULT = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
OFFLINE_INFERENCE = os.getenv("OFFLINE_INFERENCE", "0") == "1"
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={_single_line(task)} env={_single_line(env_name)} model={_single_line(model)}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(done).lower()
    error_val = _single_line(error) if error else "null"
    action_val = _single_line(action)
    print(f"[STEP] step={step} action={action_val} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _single_line(value: Any) -> str:
    return " ".join(str(value).replace("\n", " ").replace("\r", " ").split())


def _is_hacking_action(action_text: str) -> bool:
    value = (action_text or "").lower()
    patterns = (
        "if: false",
        "when: never",
        "echo \"tests passed\"",
        "echo 'tests passed'",
        "exit 0",
        "force success",
        "status: success",
    )
    return any(token in value for token in patterns)


def _extract_error(info: dict[str, Any] | None) -> str | None:
    if not info:
        return None
    error = info.get("error")
    return str(error) if error else None


def _extract_observation_fields(observation: dict[str, Any]) -> tuple[str, str, list[str]]:
    config_text = str(observation.get("config") or "")
    error_message = str(observation.get("error_message") or "")
    tools = [str(item) for item in (observation.get("available_tools") or REQUIRED_TOOLS)]
    return config_text, error_message, tools


def _tool_from_action(action_text: str) -> str:
    return str(action_text or "").split(":", 1)[0].strip().lower()


def _is_action_allowed(action_text: str, available_tools: list[str]) -> bool:
    return _tool_from_action(action_text) in {tool.lower() for tool in available_tools}


def _normalize_action(action_text: str, available_tools: list[str], fallback: str) -> str:
    action = str(action_text or "").strip()
    if not action:
        return fallback

    aliases = {
        "run_stage": "run_pipeline_stage",
        "validate": "validate_fix",
        "submit": "submit_solution",
        "submit_fix": "submit_solution",
    }
    tool = _tool_from_action(action)
    normalized_tool = aliases.get(tool, tool)
    if normalized_tool != tool:
        suffix = action.split(":", 1)[1].strip() if ":" in action else ""
        action = f"{normalized_tool}: {suffix}" if suffix else normalized_tool

    if _is_action_allowed(action, available_tools):
        return action

    return fallback


def _select_action(
    model_wrapper: ModelWrapper,
    step: int,
    config_text: str,
    error_message: str,
    history: list[str],
    available_actions: list[str],
    policy_mode: str,
    trajectories: int,
) -> str:
    mode = (policy_mode or "imp").lower()
    fallback = heuristic_action(config_text, error_message, available_actions, history)

    if mode == "sft":
        return _normalize_action(fallback, available_actions, fallback)

    if mode == "direct":
        action = model_wrapper.generate_action(
            step=step,
            config_text=config_text,
            error_message=error_message,
            history=history,
            available_actions=available_actions,
        )
        return _normalize_action(action, available_actions, fallback)

    candidates = model_wrapper.generate_candidates(
        step=step,
        config_text=config_text,
        error_message=error_message,
        history=history,
        count=max(1, int(trajectories)),
        available_actions=available_actions,
    )

    if not candidates:
        return _normalize_action(fallback, available_actions, fallback)

    observation_text = f"{config_text}\n{error_message}"
    best = max(candidates, key=lambda item: score_action_candidate(observation_text, item, _is_hacking_action))
    return _normalize_action(best, available_actions, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CI/CD debugger inference loop")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--offline", action="store_true", default=OFFLINE_INFERENCE)
    parser.add_argument("--force-local-env", action="store_true", default=False)
    parser.add_argument("--policy-mode", choices=["sft", "imp", "direct"], default="imp")
    parser.add_argument("--trajectories", type=int, default=3)
    return parser.parse_args()


async def run_episode(args: argparse.Namespace) -> int:
    history: list[str] = []
    steps_taken = 0
    success = False
    episode_completed_cleanly = False
    metrics = EpisodeMetrics()

    env = CICDDebuggerEnvironment(max_steps=max(1, int(args.max_steps)))

    offline_mode = bool(args.offline or not API_KEY)
    client: OpenAI | None = None
    if not offline_mode:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception:
            client = None
            offline_mode = True

    model_wrapper = ModelWrapper(
        client=client,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        offline=offline_mode,
    )

    log_start(task=str(args.task), env_name=str(args.benchmark), model=MODEL_NAME)

    try:
        observation = await env.reset(task_id=str(args.task), difficulty=args.difficulty)

        for step in range(1, max(1, int(args.max_steps)) + 1):
            config_text, error_message, available_tools = _extract_observation_fields(observation)

            action_text = _select_action(
                model_wrapper=model_wrapper,
                step=step,
                config_text=config_text,
                error_message=error_message,
                history=history,
                available_actions=available_tools,
                policy_mode=str(args.policy_mode),
                trajectories=max(1, int(args.trajectories)),
            )

            observation, reward, done, info = await env.step(action_text)
            step_error = _extract_error(info)

            metrics.add_step(action=action_text, reward=float(reward), error=step_error, done=bool(done))
            steps_taken = step

            log_step(step=step, action=action_text, reward=float(reward), done=bool(done), error=step_error)
            history.append(f"step={step} action={_single_line(action_text)} reward={float(reward):.2f}")

            if done:
                episode_completed_cleanly = step_error is None and not _is_hacking_action(action_text)
                break

    except Exception as exc:
        success = False
        if not metrics.rewards:
            metrics.add_step(action="system_error", reward=0.05, error=str(exc), done=True)
    finally:
        score = max(0.01, min(0.99, float(metrics.average_reward)))
        success = episode_completed_cleanly and score >= SUCCESS_SCORE_THRESHOLD

        try:
            save_reward_curve(metrics.rewards)
            save_metrics_json(metrics.summary())
            save_success_rate_history([success])
        except Exception:
            pass

        try:
            await env.close()
        except Exception:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=metrics.rewards)

    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run_episode(args))


if __name__ == "__main__":
    raise SystemExit(main())
