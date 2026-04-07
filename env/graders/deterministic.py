from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import yaml


class DeterministicGrader:
    """Deterministic correctness scoring for CI/CD config fixes."""

    COMMAND_KEYS = {
        "script",
        "scripts",
        "run",
        "command",
        "commands",
        "steps",
        "before_script",
        "after_script",
    }

    BROKEN_COMMAND_PATTERNS = (
        r"\bnpm\s+tset\b",
        r"\bpyhton\b",
        r"\bpip\s+isntall\b",
        r"\bgo\s+tset\b",
    )
    def grade(self, current_config, expected_config, metadata=None):
        metadata = metadata or {}

        score = self._compute_score(current_config, expected_config, metadata)

        is_valid = (
             current_config.strip() == expected_config.strip()
        )

       # ensure strictly between 0 and 1
        epsilon = 0.01

        safe_score = max(epsilon, min(score, 0.99))

        return {
             "reward": float(safe_score),
             "is_valid": bool(is_valid),
            }

    def _compute_score(self, current_config, expected_config, metadata=None):
    
        metadata = metadata or {}
        current_config = current_config or ""
        expected_config = expected_config or ""

        syntax_score = self._syntax_score(current_config)
        functional_score = self._functional_score(current_config, expected_config, metadata)
        similarity_score = self._similarity_score(current_config, expected_config)

        total = (0.20 * syntax_score) + (0.60 * functional_score) + (0.20 * similarity_score)

        if syntax_score == 0.0:
            total = min(total, 0.30)

        return round(self._clamp_01(total), 4)

    def _syntax_score(self, config_text: str) -> float:
        if not (config_text or "").strip():
            return 0.05

        try:
            yaml.safe_load(config_text)
            return 0.95
        except yaml.YAMLError:
            return 0.05

    def _functional_score(self, current_config: str, expected_config: str, metadata: dict[str, Any]) -> float:
        expected_commands = self._extract_commands(expected_config)
        current_commands = self._extract_commands(current_config)

        if expected_commands:
            matched = 0
            for expected in expected_commands:
                if any(self._commands_match(expected, current) for current in current_commands):
                    matched += 1
            command_score = matched / len(expected_commands)
        else:
            command_score = self._similarity_score(current_config, expected_config)

        issue_score = self._issue_resolution_score(current_config, metadata)
        broken_penalty = 0.35 if self._has_known_broken_command(current_config) else 0.0

        combined = (0.80 * command_score) + (0.20 * issue_score) - broken_penalty
        return self._clamp_01(combined)

    def _issue_resolution_score(self, current_config: str, metadata: dict[str, Any]) -> float:
        broken_token = self._normalize_text(str(metadata.get("broken_token", "")))
        fixed_token = self._normalize_text(str(metadata.get("fixed_token", "")))
        current_normalized = self._normalize_text(current_config)

        if not broken_token and not fixed_token:
            return 0.95

        if broken_token and broken_token in current_normalized:
            return 0.05

        if fixed_token and fixed_token not in current_normalized:
            return 0.05

        return 0.95

    def _extract_commands(self, config_text: str) -> list[str]:
        commands: list[str] = []

        try:
            parsed = yaml.safe_load(config_text)
        except yaml.YAMLError:
            parsed = None

        if parsed is not None:
            self._walk_yaml(parsed, commands)

        if not commands:
            commands.extend(self._extract_commands_from_text(config_text))

        deduped: list[str] = []
        seen: set[str] = set()
        for command in commands:
            normalized = self._normalize_text(command)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)

        return deduped

    def _walk_yaml(self, node: Any, commands: list[str]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_name = str(key).lower()
                if key_name in self.COMMAND_KEYS:
                    commands.extend(self._extract_string_values(value))
                self._walk_yaml(value, commands)
        elif isinstance(node, list):
            for item in node:
                self._walk_yaml(item, commands)

    def _extract_string_values(self, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, str)]
        if isinstance(value, dict):
            output: list[str] = []
            for nested in value.values():
                output.extend(self._extract_string_values(nested))
            return output
        return []

    def _extract_commands_from_text(self, config_text: str) -> list[str]:
        commands: list[str] = []

        for raw_line in (config_text or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line and not line.startswith("-") and line.endswith(":"):
                continue

            line = line.lstrip("-").strip()
            if any(token in line.lower() for token in ("npm", "pytest", "python", "yarn", "pnpm", "go test", "mvn test")):
                commands.append(line)

        return commands

    def _has_known_broken_command(self, config_text: str) -> bool:
        return any(re.search(pattern, config_text or "", flags=re.IGNORECASE) for pattern in self.BROKEN_COMMAND_PATTERNS)

    def _commands_match(self, expected: str, current: str) -> bool:
        expected_normalized = self._normalize_text(expected)
        current_normalized = self._normalize_text(current)

        if expected_normalized == current_normalized:
            return True

        if expected_normalized in current_normalized:
            return True

        if current_normalized in expected_normalized and len(current_normalized) > 6:
            return True

        return False

    def _similarity_score(self, current_config: str, expected_config: str) -> float:
        left = self._normalize_text(current_config)
        right = self._normalize_text(expected_config)

        if not left and not right:
            return 0.95
        if not left or not right:
            return 0.05

        return self._clamp_01(SequenceMatcher(None, left, right).ratio())

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", (value or "")).strip().lower()

    def _clamp_01(self, value: float) -> float:
        return max(0.01, min(0.99, float(value)))
