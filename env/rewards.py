from __future__ import annotations

from typing import Any

from env.anti_hacking import AntiHackingDetector
from env.graders.deterministic import DeterministicGrader
from env.hidden_tests import HiddenTestRunner


class RewardCalculator:
    """Composes progress, execution, quality, and anti-hacking penalties."""

    ACTION_PROGRESS_REWARDS = {
        "read_file": 0.02,
        "read_logs": 0.03,
        "analyze_error": 0.05,
        "edit_config": 0.06,
        "run_pipeline_stage": 0.07,
        "run_tests": 0.08,
        "validate_fix": 0.10,
        "submit_solution": 0.12,
    }

    QUALITY_WEIGHTS = {
        "deterministic": 0.40,
        "hidden": 0.25,
        "llm": 0.20,
    }

    def __init__(
        self,
        llm_judge: Any | None = None,
        anti_hacking_detector: AntiHackingDetector | None = None,
        deterministic_grader: DeterministicGrader | None = None,
        hidden_test_runner: HiddenTestRunner | None = None,
    ):
        self.llm_judge = llm_judge
        self.anti_hacking_detector = anti_hacking_detector or AntiHackingDetector()
        self.deterministic_grader = deterministic_grader or DeterministicGrader()
        self.hidden_test_runner = hidden_test_runner or HiddenTestRunner(grader=self.deterministic_grader)

    def calculate_step_reward(
        self,
        state: dict[str, Any] | None,
        action: str,
        result: dict[str, Any] | None,
        original_config: str | None = None,
        fixed_config: str | None = None,
        error_message: str | None = None,
        expected_config: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        state = state or {}
        result = result or {}
        metadata = metadata or {}

        current_config = fixed_config or result.get("fixed_config") or result.get("current_config") or ""
        expected_config = expected_config or result.get("expected_config") or state.get("expected_config") or ""
        original_config = original_config or result.get("original_config") or state.get("original_config") or ""
        error_message = error_message or result.get("error") or state.get("error") or ""

        reward = 0.05
        reward += self._progress_reward(action, result)
        reward += self._execution_reward(result)
        reward += self._quality_reward(
            action=action,
            current_config=current_config,
            expected_config=expected_config,
            original_config=original_config,
            error_message=error_message,
            result=result,
            metadata=metadata,
        )
        reward += self._penalty_reward(state=state, result=result, current_config=current_config)

        return round(self._clamp_01(reward), 4)

    def _progress_reward(self, action: str, result: dict[str, Any]) -> float:
        reward = self.ACTION_PROGRESS_REWARDS.get(action, 0.0)

        if result.get("logs_analyzed"):
            reward += 0.04
        if result.get("error_diagnosed"):
            reward += 0.08
        if result.get("fix_proposed"):
            reward += 0.05

        return reward

    def _execution_reward(self, result: dict[str, Any]) -> float:
        reward = 0.05

        if result.get("pipeline_run"):
            reward += 0.10
        if result.get("tests_passed"):
            reward += 0.20
        if result.get("command_succeeded"):
            reward += 0.06

        return reward

    def _quality_reward(
        self,
        action: str,
        current_config: str,
        expected_config: str,
        original_config: str,
        error_message: str,
        result: dict[str, Any],
        metadata: dict[str, Any],
    ) -> float:
        if not current_config or not expected_config:
            return 0.05

        deterministic_score = result.get("deterministic_score")
        if deterministic_score is None:
            deterministic_score = self.deterministic_grader.grade(current_config, expected_config, metadata)

        if isinstance(deterministic_score, dict):
            deterministic_score = deterministic_score.get("score", 0.05)

        if not isinstance(deterministic_score, (int, float)):
            deterministic_score = 0.05

        hidden_pass_rate = result.get("hidden_test_pass_rate")
        if hidden_pass_rate is None and action in {"validate_fix", "submit_solution"}:
            hidden_pass_rate = self.hidden_test_runner.evaluate_fix(
                fixed_config=current_config,
                expected_config=expected_config,
                metadata=metadata,
            )

        if isinstance(hidden_pass_rate, dict):
            hidden_pass_rate = hidden_pass_rate.get("score", 0.05)

        if not isinstance(hidden_pass_rate, (int, float)):
            hidden_pass_rate = 0.05

        llm_average = 0.05
        judge_scores = result.get("judge_scores")
        if not judge_scores and self.llm_judge and original_config and current_config:
            try:
                judge_scores = self.llm_judge.evaluate_fix(original_config, current_config, error_message)
            except Exception:
                judge_scores = None

        if isinstance(judge_scores, dict):
            correctness = self._clamp_01(judge_scores.get("correctness", 0.0))
            minimalism = self._clamp_01(judge_scores.get("minimalism", 0.0))
            quality = self._clamp_01(judge_scores.get("quality", 0.0))
            llm_average = (correctness + minimalism + quality) / 3.0

        quality_reward = 0.05
        quality_reward += self.QUALITY_WEIGHTS["deterministic"] * self._clamp_01(deterministic_score)
        quality_reward += self.QUALITY_WEIGHTS["hidden"] * self._clamp_01(hidden_pass_rate or 0.0)
        quality_reward += self.QUALITY_WEIGHTS["llm"] * self._clamp_01(llm_average)

        return quality_reward

    def _penalty_reward(self, state: dict[str, Any], result: dict[str, Any], current_config: str) -> float:
        changed_files_count = int(result.get("changed_files_count", state.get("changed_files_count", 0)) or 0)
        changed_lines_count = int(result.get("changed_lines_count", state.get("changed_lines_count", 0)) or 0)
        edit_count = result.get("edit_count", state.get("edit_count", 0))
        step_count = int(state.get("step_count", 0) or 0)
        previous_config = result.get("previous_config") or state.get("previous_config") or ""
        consecutive_edit_actions = int(
            result.get("consecutive_edit_actions", state.get("consecutive_edit_actions", 0)) or 0
        )
        failed_validations = int(result.get("failed_validations", state.get("failed_validations", 0)) or 0)

        penalty = self.anti_hacking_detector.total_penalty(
            current_config=current_config,
            previous_config=previous_config,
            edit_count=edit_count,
            changed_files_count=changed_files_count,
            changed_lines_count=changed_lines_count,
            step_count=step_count,
            consecutive_edit_actions=consecutive_edit_actions,
            failed_validations=failed_validations,
        )

        if result.get("hacking_attempt"):
            penalty -= 0.30

        return penalty

    def _clamp_01(self, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.05
        return max(0.01, min(0.99, parsed))
