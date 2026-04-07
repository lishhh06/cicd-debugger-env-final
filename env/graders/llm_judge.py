from __future__ import annotations

import json
import re
from typing import Any


class LLMJudge:
    """Scores qualitative fix quality while remaining robust to bad model output."""

    def __init__(self, model: Any):
        self.model = model

    def build_prompt(self, original_config: str, fixed_config: str, error_message: str) -> str:
        return (
            "You are a CI/CD fix quality judge.\n"
            "Return strict JSON with keys correctness, minimalism, quality in [0,1].\n"
            "No prose.\n\n"
            f"Original config:\n{original_config}\n\n"
            f"Fixed config:\n{fixed_config}\n\n"
            f"Error message:\n{error_message}\n"
        )

    def evaluate_fix(self, original_config: str, fixed_config: str, error_message: str) -> dict[str, float]:
        default = {
            "correctness": 0.0,
            "minimalism": 0.0,
            "quality": 0.0,
        }

        if self.model is None:
            return default

        prompt = self.build_prompt(original_config or "", fixed_config or "", error_message or "")

        try:
            raw_output = self.model(prompt, max_length=300)
            text = self._extract_text(raw_output)
        except Exception:
            return default

        if not text.strip():
            return default

        parsed = self._parse_json_with_fallback(text)
        if parsed is None:
            parsed = self._parse_regex_scores(text)

        return {
            "correctness": self._clamp(parsed.get("correctness", 0.0) if parsed else 0.0),
            "minimalism": self._clamp(parsed.get("minimalism", 0.0) if parsed else 0.0),
            "quality": self._clamp(parsed.get("quality", 0.0) if parsed else 0.0),
        }

    def _extract_text(self, raw_output: Any) -> str:
        if isinstance(raw_output, str):
            return raw_output

        if isinstance(raw_output, list) and raw_output:
            first = raw_output[0]
            if isinstance(first, dict):
                for key in ("generated_text", "text", "content"):
                    if key in first and first[key] is not None:
                        return str(first[key])
            return str(first)

        if isinstance(raw_output, dict):
            for key in ("generated_text", "text", "content"):
                if key in raw_output and raw_output[key] is not None:
                    return str(raw_output[key])

        return str(raw_output)

    def _parse_json_with_fallback(self, text: str) -> dict[str, float] | None:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return self._normalize_partial_scores(obj)
        return None

    def _parse_regex_scores(self, text: str) -> dict[str, float]:
        return {
            "correctness": self._extract_score(text, "correctness"),
            "minimalism": self._extract_score(text, "minimalism"),
            "quality": self._extract_score(text, "quality"),
        }

    def _extract_score(self, text: str, key: str) -> float:
        match = re.search(rf"{key}\s*[:=\-]\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
        if not match:
            return 0.0
        return self._clamp(match.group(1))

    def _normalize_partial_scores(self, obj: dict[str, Any]) -> dict[str, float]:
        return {
            "correctness": self._clamp(obj.get("correctness", 0.0)),
            "minimalism": self._clamp(obj.get("minimalism", 0.0)),
            "quality": self._clamp(obj.get("quality", 0.0)),
        }

    def _clamp(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        return max(0.0, min(1.0, parsed))
