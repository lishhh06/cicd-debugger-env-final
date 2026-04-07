from __future__ import annotations

from typing import Any

from env.graders.deterministic import DeterministicGrader


class HiddenTestRunner:
    """Evaluates whether a fix generalizes across deterministic CI variants."""

    def __init__(self, grader: DeterministicGrader | None = None, pass_threshold: float = 0.65):
        self.grader = grader or DeterministicGrader()
        self.pass_threshold = pass_threshold

    def generate_variants(self, config_text: str) -> list[str]:
        base = config_text or ""
        variants: list[str] = []

        for replacements in self._variant_replacement_sets():
            variant = self._apply_replacements(base, replacements)
            if variant not in variants:
                variants.append(variant)

        return variants

    def evaluate_fix(
        self,
        fixed_config: str,
        task: dict[str, Any] | None = None,
        expected_config: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        fixed_config = fixed_config or ""
        task = task or {}
        metadata = metadata or {}
        expected = expected_config or str(task.get("expected_config", ""))

        if not fixed_config.strip() or not expected.strip():
            return 0.0

        total = 0
        passed = 0

        for replacements in self._variant_replacement_sets():
            fixed_variant = self._apply_replacements(fixed_config, replacements)
            expected_variant = self._apply_replacements(expected, replacements)
            score = self.grader.grade(fixed_variant, expected_variant, metadata)
            total += 1
            if score >= self.pass_threshold:
                passed += 1

        if total == 0:
            return 0.0

        return round(passed / total, 4)

    def _variant_replacement_sets(self) -> list[tuple[tuple[str, str], ...]]:
        return [
            tuple(),
            (("ubuntu-latest", "windows-latest"),),
            (("windows-latest", "ubuntu-latest"),),
            (("node-version: 16", "node-version: 18"),),
            (("node-version: \"16\"", "node-version: \"18\""),),
            (("python-version: \"3.10\"", "python-version: \"3.12\""),),
            (("NODE_ENV=production", "NODE_ENV=development"),),
        ]

    def _apply_replacements(self, text: str, replacements: tuple[tuple[str, str], ...]) -> str:
        output = text
        for old, new in replacements:
            output = output.replace(old, new)
        return output
