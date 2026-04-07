import unittest

from env.anti_hacking import AntiHackingDetector
from env.graders.deterministic import DeterministicGrader
from env.hidden_tests import HiddenTestRunner
from env.rewards import RewardCalculator


EXPECTED_CONFIG = """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
"""

WRONG_CONFIG = """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm tset
"""

BROKEN_YAML = """
name CI
jobs:
  test:
    steps
      - run npm test
"""


class FakeJudge:
    def evaluate_fix(self, original, fixed, error):
        return {
            "correctness": 0.9,
            "minimalism": 0.8,
            "quality": 0.9,
        }


class Day2EngineTests(unittest.TestCase):
    def setUp(self):
        self.grader = DeterministicGrader()
        self.detector = AntiHackingDetector()
        self.hidden_runner = HiddenTestRunner(grader=self.grader)
        self.reward_calculator = RewardCalculator(
            llm_judge=FakeJudge(),
            anti_hacking_detector=self.detector,
            deterministic_grader=self.grader,
            hidden_test_runner=self.hidden_runner,
        )

    def test_deterministic_grader_high_for_correct_fix(self):
        score = self.grader.grade(EXPECTED_CONFIG, EXPECTED_CONFIG)
        self.assertGreaterEqual(score, 0.85)
        self.assertLessEqual(score, 1.0)

    def test_deterministic_grader_low_for_wrong_fix(self):
        score = self.grader.grade(WRONG_CONFIG, EXPECTED_CONFIG)
        self.assertLess(score, 0.6)

    def test_deterministic_grader_penalizes_broken_yaml(self):
        score = self.grader.grade(BROKEN_YAML, EXPECTED_CONFIG)
        self.assertLess(score, 0.4)

    def test_deterministic_grader_is_reproducible(self):
        first = self.grader.grade(WRONG_CONFIG, EXPECTED_CONFIG)
        second = self.grader.grade(WRONG_CONFIG, EXPECTED_CONFIG)
        self.assertEqual(first, second)

    def test_anti_hacking_detects_stage_skipping(self):
        config = EXPECTED_CONFIG + "\n    if: false\n"
        penalty = self.detector.penalty_stage_skipping(config)
        self.assertLess(penalty, 0.0)

    def test_anti_hacking_detects_fake_success(self):
        config = EXPECTED_CONFIG.replace("npm test", 'echo "tests passed" && exit 0')
        penalty = self.detector.penalty_fake_success(config)
        self.assertLess(penalty, 0.0)

    def test_anti_hacking_detects_breaking_pipeline(self):
        previous_config = """
stages:
  - build
  - test
jobs:
  build:
    run: npm ci
  test:
    run: npm test
"""
        new_config = """
stages:
  - build
jobs:
  build:
    run: npm ci
"""
        penalty = self.detector.penalty_breaking_pipeline(previous_config, new_config)
        self.assertLess(penalty, 0.0)

    def test_anti_hacking_detects_excessive_edits(self):
        penalty = self.detector.penalty_excessive_edits(changed_files_count=12, changed_lines_count=400)
        self.assertLess(penalty, 0.0)

    def test_anti_hacking_detects_timeout_abuse(self):
        penalty = self.detector.penalty_timeout_abuse(step_count=25)
        self.assertLess(penalty, 0.0)

    def test_hidden_tests_returns_high_pass_rate_for_good_fix(self):
        pass_rate = self.hidden_runner.evaluate_fix(
            fixed_config=EXPECTED_CONFIG,
            expected_config=EXPECTED_CONFIG,
        )
        self.assertGreaterEqual(pass_rate, 0.8)

    def test_hidden_tests_returns_lower_pass_rate_for_bad_fix(self):
        pass_rate = self.hidden_runner.evaluate_fix(
            fixed_config=WRONG_CONFIG,
            expected_config=EXPECTED_CONFIG,
        )
        self.assertLess(pass_rate, 0.8)

    def test_reward_calculator_positive_for_good_behavior(self):
        state = {
            "step_count": 3,
            "previous_config": WRONG_CONFIG,
            "expected_config": EXPECTED_CONFIG,
            "original_config": WRONG_CONFIG,
            "error": "command not found",
        }
        result = {
            "logs_analyzed": True,
            "error_diagnosed": True,
            "fix_proposed": True,
            "pipeline_run": True,
            "tests_passed": True,
            "command_succeeded": True,
            "current_config": EXPECTED_CONFIG,
            "expected_config": EXPECTED_CONFIG,
            "changed_files_count": 1,
            "changed_lines_count": 1,
        }

        reward = self.reward_calculator.calculate_step_reward(
            state=state,
            action="validate_fix",
            result=result,
            original_config=WRONG_CONFIG,
            fixed_config=EXPECTED_CONFIG,
            error_message="command not found",
            expected_config=EXPECTED_CONFIG,
            metadata={"broken_token": "npm tset", "fixed_token": "npm test"},
        )

        self.assertGreater(reward, 0.5)

    def test_reward_calculator_penalizes_hacking(self):
        hacked_config = EXPECTED_CONFIG.replace("npm test", 'echo "tests passed" && exit 0') + "\n    if: false\n"
        state = {
            "step_count": 25,
            "previous_config": EXPECTED_CONFIG,
            "changed_files_count": 15,
            "changed_lines_count": 500,
        }
        result = {
            "current_config": hacked_config,
            "expected_config": EXPECTED_CONFIG,
            "hacking_attempt": True,
            "changed_files_count": 15,
            "changed_lines_count": 500,
        }

        reward = self.reward_calculator.calculate_step_reward(
            state=state,
            action="edit_config",
            result=result,
            original_config=EXPECTED_CONFIG,
            fixed_config=hacked_config,
            error_message="",
            expected_config=EXPECTED_CONFIG,
        )

        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 0.3)


if __name__ == "__main__":
    unittest.main()
