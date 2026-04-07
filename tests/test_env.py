import asyncio
import unittest

from env.environment import CICDDebuggerEnvironment, REQUIRED_TOOLS


class EnvironmentContractTests(unittest.TestCase):
    def test_reset_returns_structured_observation(self):
        env = CICDDebuggerEnvironment(max_steps=10, seed=7)
        observation = asyncio.run(env.reset(task_id="easy-command-typo"))

        self.assertIn("config", observation)
        self.assertIn("logs", observation)
        self.assertIn("error_message", observation)
        self.assertIn("progress_flags", observation)
        self.assertEqual(observation["task_id"], "easy-command-typo")
        self.assertEqual(observation["available_tools"], REQUIRED_TOOLS)
        self.assertEqual(observation["step_count"], 0)

    def test_step_returns_obs_reward_done_info(self):
        env = CICDDebuggerEnvironment(max_steps=10, seed=3)
        asyncio.run(env.reset(task_id="easy-command-typo"))

        observation, reward, done, info = asyncio.run(env.step("read_logs: inspect failing stage logs"))

        self.assertIsInstance(observation, dict)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("tool", info)

    def test_action_space_rejects_extra_tools(self):
        env = CICDDebuggerEnvironment(max_steps=10, seed=5)
        asyncio.run(env.reset(task_id="easy-command-typo"))

        observation, reward, done, info = asyncio.run(env.step("propose_fix: force deploy"))

        self.assertIn("error", info)
        self.assertIsNotNone(info["error"])
        self.assertFalse(done)
        self.assertGreaterEqual(reward, 0.0)
        self.assertIn("config", observation)

    def test_action_space_rejects_alias_tools(self):
        env = CICDDebuggerEnvironment(max_steps=10, seed=15)
        asyncio.run(env.reset(task_id="easy-command-typo"))

        _, _, done, info = asyncio.run(env.step("read: workflow file"))

        self.assertIn("error", info)
        self.assertIsNotNone(info["error"])
        self.assertFalse(done)

    def test_submit_solution_path(self):
        env = CICDDebuggerEnvironment(max_steps=12, seed=9)
        asyncio.run(env.reset(task_id="easy-command-typo"))

        asyncio.run(env.step("read_logs: inspect logs"))
        asyncio.run(env.step("analyze_error: identify root cause"))
        asyncio.run(env.step("edit_config: replace npm tset with npm test"))
        asyncio.run(env.step("run_pipeline_stage: run test stage"))
        asyncio.run(env.step("run_tests: execute tests"))
        asyncio.run(env.step("validate_fix: validate score"))
        observation, reward, done, info = asyncio.run(env.step("submit_solution: submit current fix"))

        self.assertTrue(done)
        self.assertGreaterEqual(reward, 0.0)
        self.assertIsNone(info.get("error"))
        self.assertEqual(observation["progress_flags"].get("submit_solution"), True)

    def test_internal_state_tracks_required_fields(self):
        env = CICDDebuggerEnvironment(max_steps=10, seed=11)
        asyncio.run(env.reset(task_id="easy-command-typo"))
        asyncio.run(env.step("read_logs: inspect logs"))

        state = env.get_state()
        self.assertTrue(state.get("initialized"))
        self.assertIn("actual_bug", state)
        self.assertIn("correct_solution", state)
        self.assertIn("progress_flags", state)
        self.assertIn("file_modification_count", state)
        self.assertIn("hidden_test_pass_rate", state)

    def test_yaml_task_is_fixable_via_edit_flow(self):
        env = CICDDebuggerEnvironment(max_steps=12, seed=17)
        asyncio.run(env.reset(task_id="easy-yaml-indentation"))

        asyncio.run(env.step("read_logs: inspect logs"))
        asyncio.run(env.step("analyze_error: identify root cause"))
        observation, _, _, _ = asyncio.run(env.step("edit_config: fix YAML indentation and syntax"))

        self.assertIn("- run: pytest", observation["config"])
        self.assertNotIn("       - run: pytest", observation["config"])

        asyncio.run(env.step("run_tests: execute tests"))
        asyncio.run(env.step("validate_fix: validate score"))
        _, _, done, info = asyncio.run(env.step("submit_solution: submit current fix"))

        self.assertTrue(done)
        self.assertIsNone(info.get("error"))

    def test_hard_needs_order_edit_updates_deploy_dependency(self):
        env = CICDDebuggerEnvironment(max_steps=12, seed=19)
        asyncio.run(env.reset(task_id="hard-needs-order"))

        observation, _, _, _ = asyncio.run(env.step("edit_config: fix deploy dependency ordering"))

        self.assertIn("needs: [build, test]", observation["config"])
        self.assertEqual(observation["config"].count("needs: build"), 1)


if __name__ == "__main__":
    unittest.main()
