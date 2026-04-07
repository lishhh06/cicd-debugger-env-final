from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

from inference.prompts import REQUIRED_ACTIONS, SYSTEM_PROMPT, build_user_prompt, heuristic_action, sanitize_action_text


@dataclass
class ModelWrapper:
    client: OpenAI | None
    model_name: str
    temperature: float
    max_tokens: int
    offline: bool

    def generate_action(
        self,
        step: int,
        config_text: str,
        error_message: str,
        history: list[str],
        available_actions: Iterable[str] | None = None,
    ) -> str:
        fallback = heuristic_action(config_text, error_message, available_actions, history)
        if self.offline or self.client is None:
            return fallback

        user_prompt = build_user_prompt(
            step=step,
            config_text=config_text,
            error_message=error_message,
            history=history,
            available_actions=available_actions,
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=max(float(self.temperature), 0.0),
                max_tokens=max(16, int(self.max_tokens)),
                stream=False,
            )

            generated = str(completion.choices[0].message.content or "")
            return sanitize_action_text(generated, fallback=fallback)
        except Exception:
            return fallback

    def generate_candidates(
        self,
        step: int,
        config_text: str,
        error_message: str,
        history: list[str],
        count: int,
        available_actions: Iterable[str] | None = None,
    ) -> list[str]:
        candidates = [heuristic_action(config_text, error_message, available_actions, history)]

        for idx in range(max(1, count)):
            action = self.generate_action(
                step=step,
                config_text=config_text,
                error_message=error_message,
                history=history + [f"candidate={idx}"],
                available_actions=available_actions,
            )
            if action:
                candidates.append(action)

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)

        return deduped


def score_action_candidate(observation_text: str, action_text: str, hacking_predicate: Any) -> float:
    obs = (observation_text or "").lower()
    action = (action_text or "").lower()

    score = 0.0

    selected_tool = action.split(":", 1)[0].strip()
    if selected_tool in REQUIRED_ACTIONS:
        score += 0.5
    else:
        score -= 0.5

    if "npm tset" in obs and "edit_config" in action and "npm test" in action:
        score += 2.0
    if "yaml" in obs and "edit_config" in action:
        score += 1.2
    if "missing" in obs and "analyze_error" in action:
        score += 0.8
    if "validate_fix" in action or "submit_solution" in action:
        score += 0.5

    if hacking_predicate(action_text):
        score -= 3.0

    score += min(len(action_text) / 240.0, 0.2)
    return score
