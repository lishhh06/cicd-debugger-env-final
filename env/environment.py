from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import random
import re
from typing import Any

import yaml

from env.models import Action, EnvStateSnapshot, Observation, Reward
from env.rewards import RewardCalculator
from env.tasks import get_task_by_id, get_tasks_by_difficulty
from env.tasks.task_types import CICDTask


REQUIRED_TOOLS = [
    "read_file",
    "read_logs",
    "analyze_error",
    "edit_config",
    "run_pipeline_stage",
    "run_tests",
    "validate_fix",
    "submit_solution",
]

MAX_STEPS = 30


@dataclass
class EnvironmentState:
    task: CICDTask
    current_config: str
    previous_config: str
    step_count: int = 0
    done: bool = False
    progress_flags: dict[str, bool] = field(default_factory=dict)
    file_modification_count: int = 0
    total_changed_lines: int = 0
    hidden_test_pass_rate: float = 0.0
    action_history: list[str] = field(default_factory=list)
    stage_results: dict[str, bool] = field(default_factory=dict)
    failed_validations: int = 0
    consecutive_edit_actions: int = 0
    current_logs: str = ""
    last_error: str = ""
    last_action_error: str | None = None
    last_info: dict[str, Any] = field(default_factory=dict)


class CICDDebuggerEnvironment:
    """RL-style CI/CD debugging environment with strict tool-based actions."""

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        seed: int | None = None,
        llm_judge: Any | None = None,
    ) -> None:
        self.max_steps = max(1, int(max_steps))
        self.random = random.Random(seed)
        self.reward_calculator = RewardCalculator(llm_judge=llm_judge)
        self._state: EnvironmentState | None = None

    async def reset(self, task_id: str | None = None, difficulty: str | None = None) -> dict[str, Any]:
        task = self._select_task(task_id=task_id, difficulty=difficulty)

        self._state = EnvironmentState(
            task=task,
            current_config=task.broken_config,
            previous_config=task.broken_config,
            progress_flags={tool: False for tool in REQUIRED_TOOLS},
            current_logs=task.logs,
            last_error=task.error_message,
        )

        return Observation.model_validate(self._build_observation()).model_dump()

    async def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._state.done:
            reward_model = Reward(value=0.0, components={"total": 0.0})
            return Observation.model_validate(self._build_observation()).model_dump(), float(reward_model.value), True, {
                "tool": "none",
                "message": "episode already completed",
                "error": None,
                "reward_model": reward_model.model_dump(),
            }

        parsed_action = Action.from_input(action)
        tool, payload = parsed_action.tool, dict(parsed_action.payload)
        self._state.step_count += 1
        self._state.previous_config = self._state.current_config
        self._state.action_history.append(tool)
        self._state.last_action_error = None

        info: dict[str, Any] = {
            "tool": tool,
            "message": "",
            "error": None,
        }
        changed_lines = 0

        result: dict[str, Any] = {
            "previous_config": self._state.previous_config,
            "current_config": self._state.current_config,
            "fixed_config": self._state.current_config,
            "expected_config": self._state.task.expected_config,
            "error": self._state.last_error,
            "logs_analyzed": False,
            "error_diagnosed": False,
            "fix_proposed": False,
            "pipeline_run": False,
            "tests_passed": False,
            "command_succeeded": False,
            "changed_files_count": 0,
            "changed_lines_count": 0,
            "edit_count": {
                "changed_files_count": self._state.file_modification_count,
                "changed_lines_count": self._state.total_changed_lines,
            },
            "deterministic_score": None,
            "hidden_test_pass_rate": None,
            "judge_scores": None,
            "hacking_attempt": False,
        }

        if tool not in REQUIRED_TOOLS:
            info["message"] = "unsupported action tool"
            info["error"] = f"tool '{tool}' is not allowed"
            self._state.last_action_error = str(info["error"])
        elif tool == "read_file":
            self._state.progress_flags[tool] = True
            result["command_succeeded"] = True
            info["message"] = "returned current workflow config"
            self._state.current_logs = self._state.current_config
            self._state.consecutive_edit_actions = 0
        elif tool == "read_logs":
            self._state.progress_flags[tool] = True
            result["logs_analyzed"] = True
            result["command_succeeded"] = True
            info["message"] = "returned pipeline failure logs"
            self._state.current_logs = self._state.task.logs
            self._state.consecutive_edit_actions = 0
        elif tool == "analyze_error":
            self._state.progress_flags[tool] = True
            result["error_diagnosed"] = True
            result["command_succeeded"] = True
            root_cause = self._detect_root_cause(self._state.current_config, self._state.task)
            info["message"] = f"root cause: {root_cause}"
            self._state.current_logs = f"analysis result: {root_cause}"
            self._state.consecutive_edit_actions = 0
        elif tool == "edit_config":
            self._state.progress_flags[tool] = True
            updated_config, summary = self._apply_edit(self._state.current_config, payload, self._state.task)
            changed_lines = self._count_changed_lines(self._state.current_config, updated_config)

            if changed_lines > 0:
                self._state.current_config = updated_config
                self._state.file_modification_count += 1
                self._state.total_changed_lines += changed_lines
                result["fix_proposed"] = True
                result["command_succeeded"] = True
                info["message"] = summary
                self._state.current_logs = f"edit applied: {summary}"
            else:
                result["command_succeeded"] = False
                info["message"] = "no config changes applied"
                info["error"] = "edit_config did not modify workflow"
                self._state.last_action_error = str(info["error"])
                self._state.current_logs = "edit action produced no changes"

            self._state.consecutive_edit_actions += 1
        elif tool == "run_pipeline_stage":
            self._state.progress_flags[tool] = True
            stage = self._extract_stage(payload, fallback=self._state.task.failure_stage)
            success, stage_logs = self._simulate_stage(self._state.current_config, stage, self._state.task)
            self._state.stage_results[stage] = success
            result["pipeline_run"] = True
            result["command_succeeded"] = success
            info["message"] = f"stage '{stage}' {'passed' if success else 'failed'}"
            if not success:
                info["error"] = stage_logs
                self._state.last_action_error = stage_logs
                self._state.last_error = stage_logs
            self._state.current_logs = stage_logs
            self._state.consecutive_edit_actions = 0
        elif tool == "run_tests":
            self._state.progress_flags[tool] = True
            tests_passed, test_logs = self._run_tests(self._state.current_config, self._state.task)
            result["pipeline_run"] = True
            result["tests_passed"] = tests_passed
            result["command_succeeded"] = tests_passed
            info["message"] = "tests passed" if tests_passed else "tests failed"
            if not tests_passed:
                info["error"] = test_logs
                self._state.last_action_error = test_logs
                self._state.last_error = test_logs
            self._state.current_logs = test_logs
            self._state.consecutive_edit_actions = 0
        elif tool == "validate_fix":
            self._state.progress_flags[tool] = True
            validation = self._validate_current_fix(self._state)
            result.update(validation)
            result["pipeline_run"] = True
            is_valid = bool(validation.get("is_valid"))
            result["command_succeeded"] = is_valid

            if not is_valid:
                self._state.failed_validations += 1
                info["error"] = str(validation.get("summary", "validation failed"))
                self._state.last_action_error = str(info["error"])

            info["message"] = "validation passed" if is_valid else "validation failed"
            self._state.hidden_test_pass_rate = float(validation.get("hidden_test_pass_rate") or 0.0)
            self._state.current_logs = str(validation.get("summary", "validation complete"))
            self._state.consecutive_edit_actions = 0
        elif tool == "submit_solution":
            validation = self._validate_current_fix(self._state)
            result.update(validation)
            result["pipeline_run"] = True
            self._state.progress_flags[tool] = True
            accepted = bool(validation.get("is_valid"))
            result["command_succeeded"] = accepted

            if accepted:
                self._state.done = True
                info["message"] = "solution accepted"
                self._state.current_logs = "submission accepted"
            else:
                self._state.failed_validations += 1
                info["message"] = "solution rejected"
                info["error"] = "submission failed quality checks"
                self._state.last_action_error = str(info["error"])
                self._state.current_logs = str(validation.get("summary", "submission rejected"))

            self._state.hidden_test_pass_rate = float(validation.get("hidden_test_pass_rate") or 0.0)
            self._state.consecutive_edit_actions = 0

        result["hacking_attempt"] = self._detect_hacking_attempt(tool, payload, self._state.current_config)
        result["current_config"] = self._state.current_config
        result["fixed_config"] = self._state.current_config
        result["changed_files_count"] = 1 if changed_lines > 0 else 0
        result["changed_lines_count"] = changed_lines
        result["edit_count"] = {
            "changed_files_count": self._state.file_modification_count,
            "changed_lines_count": self._state.total_changed_lines,
        }

        if info["error"]:
            self._state.last_error = str(info["error"])
        result["error"] = self._state.last_error

        if self._state.step_count >= self.max_steps and not self._state.done:
            self._state.done = True
            if not info["error"]:
                info["error"] = "max_steps_reached"
            info["message"] = "max steps reached"

        reward = self.reward_calculator.calculate_step_reward(
            state={
                "step_count": self._state.step_count,
                "previous_config": self._state.previous_config,
                "expected_config": self._state.task.expected_config,
                "original_config": self._state.task.broken_config,
                "error": self._state.last_error,
                "changed_files_count": self._state.file_modification_count,
                "changed_lines_count": self._state.total_changed_lines,
                "consecutive_edit_actions": self._state.consecutive_edit_actions,
                "failed_validations": self._state.failed_validations,
            },
            action=tool,
            result=result,
            original_config=self._state.task.broken_config,
            fixed_config=self._state.current_config,
            error_message=self._state.last_error,
            expected_config=self._state.task.expected_config,
            metadata=self._state.task.metadata,
        )
      # 🔥 CRITICAL FIX FOR SCALER (FINAL OVERRIDE)
        if tool in ["validate_fix", "submit_solution"]:
            is_correct = bool(result.get("is_valid"))

            if is_correct:
                reward = 1.0
                self._state.done = True
            else:
                reward = 0.0

        reward_model = Reward(value=float(reward), components={"total": float(reward)})
        info["reward_model"] = reward_model.model_dump()

        self._state.last_info = info
        observation = Observation.model_validate(self._build_observation()).model_dump()
        done = bool(self._state.done)

        return observation, float(reward_model.value), done, info

    async def close(self) -> None:
        return None

    def get_state(self) -> dict[str, Any]:
        if self._state is None:
            return EnvStateSnapshot(initialized=False).model_dump()

        snapshot = {
            "initialized": True,
            "task_id": self._state.task.task_id,
            "difficulty": self._state.task.difficulty,
            "actual_bug": self._state.task.actual_bug,
            "correct_solution": self._state.task.expected_config,
            "failure_stage": self._state.task.failure_stage,
            "step_count": self._state.step_count,
            "done": self._state.done,
            "progress_flags": dict(self._state.progress_flags),
            "file_modification_count": self._state.file_modification_count,
            "total_changed_lines": self._state.total_changed_lines,
            "hidden_test_pass_rate": self._state.hidden_test_pass_rate,
            "stage_results": dict(self._state.stage_results),
            "failed_validations": self._state.failed_validations,
            "last_action_error": self._state.last_action_error,
            "last_error": self._state.last_error,
        }
        return EnvStateSnapshot.model_validate(snapshot).model_dump()

    def state(self) -> dict[str, Any]:
        return self.get_state()

    def _build_observation(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Environment not initialized")

        observation = {
            "task_id": self._state.task.task_id,
            "difficulty": self._state.task.difficulty,
            "failure_stage": self._state.task.failure_stage,
            "actual_bug": self._state.task.actual_bug,
            "config": self._state.current_config,
            "logs": self._state.current_logs,
            "error_message": self._state.last_error,
            "available_tools": list(REQUIRED_TOOLS),
            "progress_flags": dict(self._state.progress_flags),
            "file_modification_count": self._state.file_modification_count,
            "hidden_test_pass_rate": self._state.hidden_test_pass_rate,
            "step_count": self._state.step_count,
            "last_action_error": self._state.last_action_error,
        }
        return Observation.model_validate(observation).model_dump()

    def _select_task(self, task_id: str | None, difficulty: str | None) -> CICDTask:
        if task_id:
            task = get_task_by_id(task_id)
            if task is None:
                raise ValueError(f"Unknown task_id: {task_id}")
            return task

        filtered = get_tasks_by_difficulty(difficulty)
        if not filtered:
            raise ValueError(f"No tasks available for difficulty: {difficulty}")

        return self.random.choice(filtered)

    def _parse_action(self, action: Any) -> tuple[str, dict[str, Any]]:
        parsed = Action.from_input(action)
        return parsed.tool, dict(parsed.payload)

    def _extract_stage(self, payload: dict[str, Any], fallback: str) -> str:
        direct_stage = str(payload.get("stage") or "").strip().lower()
        if direct_stage in {"build", "test", "deploy"}:
            return direct_stage

        raw = str(payload.get("raw") or "").lower()
        for stage in ("build", "test", "deploy"):
            if stage in raw:
                return stage

        return fallback

    def _detect_root_cause(self, config_text: str, task: CICDTask) -> str:
        normalized = self._normalize(config_text)
        broken_token = self._normalize(str(task.metadata.get("broken_token", "")))

        if broken_token and broken_token in normalized:
            return task.actual_bug

        if not self._is_yaml_valid(config_text):
            return "workflow YAML is invalid"

        fixed_token = self._normalize(str(task.metadata.get("fixed_token", "")))
        if fixed_token and fixed_token not in normalized:
            return f"missing expected fix token: {task.metadata.get('fixed_token')}"

        return "configuration still deviates from expected pipeline behavior"

    def _apply_edit(self, current_config: str, payload: dict[str, Any], task: CICDTask) -> tuple[str, str]:
        candidate = current_config
        edits: list[str] = []

        new_config = payload.get("new_config")
        if isinstance(new_config, str) and new_config.strip():
            return new_config.strip(), "applied payload new_config"

        raw = str(payload.get("raw") or "")
        raw_lower = raw.lower()

        replace_match = re.search(
            r"replace\s+['\"]?(.+?)['\"]?\s+with\s+['\"]?(.+?)['\"]?\s*$",
            raw,
            flags=re.IGNORECASE,
        )
        if replace_match:
            old = replace_match.group(1).strip()
            new = replace_match.group(2).strip()
            if old and old in candidate:
                candidate = candidate.replace(old, new)
                edits.append(f"replaced '{old}' with '{new}'")

        if "checkout" in raw_lower and "actions/checkout@v4" not in candidate:
            updated = self._ensure_checkout(candidate)
            if updated != candidate:
                candidate = updated
                edits.append("inserted actions/checkout@v4 step")

        if "permissions" in raw_lower or "actions: write" in raw_lower:
            updated = self._ensure_actions_write(candidate)
            if updated != candidate:
                candidate = updated
                edits.append("added actions: write permission")

        if not edits and any(token in raw_lower for token in ("yaml", "indent", "syntax")):
            updated = self._repair_yaml(candidate, task.expected_config)
            if updated != candidate:
                candidate = updated
                edits.append("repaired YAML structure")

        broken_token = str(task.metadata.get("broken_token", ""))
        fixed_token = str(task.metadata.get("fixed_token", ""))
        if not edits and broken_token and fixed_token and broken_token in candidate:
            occurrence_count = candidate.count(broken_token)

            if occurrence_count > 1:
                candidate = task.expected_config
                edits.append("applied canonical fix for ambiguous token")
            elif fixed_token.strip().endswith(":"):
                expected_block = self._extract_expected_block(task.expected_config, fixed_token)
                if expected_block and expected_block not in candidate:
                    candidate = candidate.replace(broken_token, f"{broken_token}\n{expected_block}", 1)
                    edits.append("inserted expected YAML block")
                else:
                    candidate = candidate.replace(broken_token, fixed_token, 1)
                    edits.append("applied metadata token replacement")
            else:
                expected_line = self._find_line_containing(task.expected_config, fixed_token)
                replacement = expected_line.strip() if expected_line else fixed_token
                candidate = candidate.replace(broken_token, replacement, 1)
                edits.append("applied metadata token replacement")

        if not edits and fixed_token and fixed_token not in candidate and not broken_token:
            updated = self._append_missing_token(candidate, fixed_token)
            if updated != candidate:
                candidate = updated
                edits.append("appended expected token")

        if not edits and any(token in raw_lower for token in ("expected config", "apply expected", "canonical fix")):
            candidate = task.expected_config
            edits.append("replaced with expected task config")

        summary = "; ".join(edits) if edits else "no-op edit"
        return candidate, summary

    def _ensure_checkout(self, config_text: str) -> str:
        if "actions/checkout@v4" in config_text:
            return config_text

        marker = "steps:\n"
        insert = "      - uses: actions/checkout@v4\n"
        if marker in config_text:
            return config_text.replace(marker, marker + insert, 1)

        return config_text

    def _ensure_actions_write(self, config_text: str) -> str:
        if "actions: write" in config_text:
            return config_text

        if "permissions:" in config_text:
            lines = config_text.splitlines()
            out: list[str] = []
            inserted = False
            for line in lines:
                out.append(line)
                if line.strip().startswith("permissions:") and not inserted:
                    continue
                if line.strip().startswith("contents:") and not inserted:
                    indent = line[: len(line) - len(line.lstrip(" "))]
                    out.append(f"{indent}actions: write")
                    inserted = True
            if inserted:
                return "\n".join(out)

        return "permissions:\n  actions: write\n" + config_text

    def _append_missing_token(self, config_text: str, token: str) -> str:
        if not token or token in config_text:
            return config_text

        lower_token = token.lower()
        if "actions/checkout@v4" in lower_token:
            return self._ensure_checkout(config_text)
        if "actions: write" in lower_token:
            return self._ensure_actions_write(config_text)

        return config_text + "\n" + token

    def _repair_yaml(self, current_config: str, expected_config: str) -> str:
        if self._is_yaml_valid(current_config):
            return current_config

        if expected_config and self._is_yaml_valid(expected_config):
            return expected_config

        return current_config

    def _find_line_containing(self, config_text: str, token: str) -> str | None:
        target = (token or "").strip()
        if not target:
            return None

        for line in (config_text or "").splitlines():
            if target in line:
                return line

        return None

    def _extract_expected_block(self, config_text: str, token: str) -> str:
        lines = (config_text or "").splitlines()
        target = (token or "").strip()
        if not target:
            return ""

        for idx, line in enumerate(lines):
            if target not in line:
                continue

            base_indent = len(line) - len(line.lstrip(" "))
            block = [line]
            for next_line in lines[idx + 1 :]:
                if not next_line.strip():
                    break
                next_indent = len(next_line) - len(next_line.lstrip(" "))
                if next_indent <= base_indent:
                    break
                block.append(next_line)
            return "\n".join(block)

        return ""

    def _simulate_stage(self, config_text: str, stage: str, task: CICDTask) -> tuple[bool, str]:
        if not self._is_yaml_valid(config_text):
            return False, "invalid workflow YAML"

        expected_has_stage = self._stage_exists(task.expected_config, stage)
        current_has_stage = self._stage_exists(config_text, stage)

        if expected_has_stage and not current_has_stage:
            return False, f"required stage '{stage}' is missing"

        if not expected_has_stage and not current_has_stage:
            return True, f"{stage} stage not required for this task"

        normalized = self._normalize(config_text)
        broken_token = self._normalize(str(task.metadata.get("broken_token", "")))
        fixed_token = self._normalize(str(task.metadata.get("fixed_token", "")))

        if self._contains_hacking_pattern(config_text):
            return False, "unsafe shortcut pattern detected"

        if stage == task.failure_stage and broken_token and broken_token in normalized:
            return False, task.logs

        if stage == task.failure_stage and fixed_token and fixed_token not in normalized:
            return False, task.logs

        commands = self._extract_commands(config_text)

        if stage == "build":
            build_tokens = ("npm ci", "npm install", "pip install", "go build", "mvn", "yarn install", "pnpm install")
            if not any(any(token in cmd for token in build_tokens) for cmd in commands):
                return False, "build stage has no install/build command"

        if stage == "test":
            test_tokens = ("npm test", "pytest", "go test", "mvn test", "yarn test", "pnpm test")
            if not any(any(token in cmd for token in test_tokens) for cmd in commands):
                return False, "test stage has no test command"

        if stage == "deploy":
            deploy_tokens = ("deploy", "publish", "upload-artifact", "release")
            if not any(any(token in cmd for token in deploy_tokens) for cmd in commands):
                return False, "deploy stage has no deployment command"

        return True, f"{stage} stage passed"

    def _run_tests(self, config_text: str, task: CICDTask) -> tuple[bool, str]:
        if self._stage_exists(task.expected_config, "build"):
            build_ok, build_logs = self._simulate_stage(config_text, "build", task)
            if not build_ok:
                return False, build_logs

        if self._stage_exists(task.expected_config, "test"):
            test_ok, test_logs = self._simulate_stage(config_text, "test", task)
            if not test_ok:
                return False, test_logs

        similarity = SequenceMatcher(None, self._normalize(config_text), self._normalize(task.expected_config)).ratio()
        if similarity < 0.45:
            return False, "tests failed: fix diverges significantly from expected pipeline"

        return True, "tests passed"

    def _validate_current_fix(self, state: EnvironmentState) -> dict[str, Any]:
        current = state.current_config
        task = state.task

        deterministic_score = self.reward_calculator.deterministic_grader.grade(
            current,
            task.expected_config,
            metadata=task.metadata,
        )
        hidden_test_pass_rate = self.reward_calculator.hidden_test_runner.evaluate_fix(
            fixed_config=current,
            expected_config=task.expected_config,
            metadata=task.metadata,
        )

        judge_scores = None
        if self.reward_calculator.llm_judge is not None:
            try:
                judge_scores = self.reward_calculator.llm_judge.evaluate_fix(
                    task.broken_config,
                    current,
                    state.last_error,
                )
            except Exception:
                judge_scores = None

        tests_passed, test_logs = self._run_tests(current, task)

        stage_ok, stage_logs = self._simulate_stage(current, task.failure_stage, task)

        broken_token = self._normalize(str(task.metadata.get("broken_token", "")))
        fixed_token = self._normalize(str(task.metadata.get("fixed_token", "")))
        normalized_current = self._normalize(current)

        token_constraints_met = True
        if broken_token and broken_token in normalized_current:
            token_constraints_met = False
        if fixed_token and fixed_token not in normalized_current:
            token_constraints_met = False

        judge_average = 1.0
        if isinstance(judge_scores, dict):
            judge_average = (
                float(judge_scores.get("correctness", 0.0))
                + float(judge_scores.get("minimalism", 0.0))
                + float(judge_scores.get("quality", 0.0))
            ) / 3.0

        is_valid = (
            tests_passed
            and stage_ok
            and token_constraints_met
            and deterministic_score >= 0.72
            and hidden_test_pass_rate >= 0.65
            and judge_average >= 0.5
        )

        summary = (
            f"deterministic={deterministic_score:.2f}; "
            f"hidden={hidden_test_pass_rate:.2f}; "
            f"tests={'pass' if tests_passed else 'fail'}; "
            f"failure_stage={'pass' if stage_ok else 'fail'}"
        )
        if not tests_passed:
            summary = test_logs
        elif not stage_ok:
            summary = stage_logs
        elif not token_constraints_met:
            summary = "validation failed: required bug-fix token constraints not satisfied"

        return {
            "deterministic_score": deterministic_score,
            "hidden_test_pass_rate": hidden_test_pass_rate,
            "judge_scores": judge_scores,
            "tests_passed": tests_passed,
            "is_valid": is_valid,
            "summary": summary,
        }

    def _detect_hacking_attempt(self, tool: str, payload: dict[str, Any], config_text: str) -> bool:
        payload_text = str(payload.get("raw") or "")
        brute_force_detected = bool(
            self._state and (self._state.consecutive_edit_actions >= 6 or self._state.failed_validations >= 3)
        )

        return (
            brute_force_detected
            or self._contains_hacking_pattern(payload_text)
            or self._contains_hacking_pattern(config_text)
        )

    def _contains_hacking_pattern(self, text: str) -> bool:
        if not text:
            return False

        patterns = (
            *self.reward_calculator.anti_hacking_detector.STAGE_SKIP_PATTERNS,
            *self.reward_calculator.anti_hacking_detector.FAKE_SUCCESS_PATTERNS,
            *self.reward_calculator.anti_hacking_detector.HARDCODED_OUTPUT_PATTERNS,
        )
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _extract_commands(self, config_text: str) -> list[str]:
        commands: list[str] = []
        for raw_line in config_text.splitlines():
            line = raw_line.strip().lower()
            if "- run:" in line:
                commands.append(line.replace("- run:", "").strip())
            elif line.startswith("- run "):
                commands.append(line.replace("- run ", "", 1).strip())
        return commands

    def _is_yaml_valid(self, config_text: str) -> bool:
        try:
            parsed = yaml.safe_load(config_text)
        except yaml.YAMLError:
            return False
        return isinstance(parsed, dict)

    def _stage_exists(self, config_text: str, stage: str) -> bool:
        try:
            parsed = yaml.safe_load(config_text)
        except yaml.YAMLError:
            return False

        if not isinstance(parsed, dict):
            return False

        jobs = parsed.get("jobs")
        if isinstance(jobs, dict) and stage in jobs:
            return True

        stages = parsed.get("stages")
        if isinstance(stages, dict) and stage in stages:
            return True
        if isinstance(stages, list) and stage in stages:
            return True

        return False

    def _count_changed_lines(self, previous: str, current: str) -> int:
        prev_lines = previous.splitlines()
        curr_lines = current.splitlines()
        changed = 0

        max_len = max(len(prev_lines), len(curr_lines))
        for idx in range(max_len):
            left = prev_lines[idx] if idx < len(prev_lines) else ""
            right = curr_lines[idx] if idx < len(curr_lines) else ""
            if left != right:
                changed += 1

        return changed

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())
