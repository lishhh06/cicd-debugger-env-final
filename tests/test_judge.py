import unittest

from env.graders.llm_judge import LLMJudge


class FakeModel:
    def __init__(self, payload, raise_error: bool = False):
        self.payload = payload
        self.raise_error = raise_error

    def __call__(self, prompt, **kwargs):
        if self.raise_error:
            raise RuntimeError("model failure")
        return [{"generated_text": self.payload}]


class LLMJudgeTests(unittest.TestCase):
    def test_good_json_scores_are_parsed(self):
        judge = LLMJudge(FakeModel('{"correctness": 1.0, "minimalism": 0.8, "quality": 0.9}'))
        result = judge.evaluate_fix("npm tset", "npm test", "command not found")

        self.assertGreaterEqual(result["correctness"], 0.9)
        self.assertGreaterEqual(result["minimalism"], 0.7)
        self.assertGreaterEqual(result["quality"], 0.8)

    def test_regex_fallback_for_noisy_output(self):
        noisy = "Correctness: 0.7\nMinimalism: 0.6\nQuality: 0.75"
        judge = LLMJudge(FakeModel(noisy))
        result = judge.evaluate_fix("a", "b", "err")

        self.assertAlmostEqual(result["correctness"], 0.7)
        self.assertAlmostEqual(result["minimalism"], 0.6)
        self.assertAlmostEqual(result["quality"], 0.75)

    def test_partial_fields_default_to_zero(self):
        judge = LLMJudge(FakeModel('{"correctness": 0.8}'))
        result = judge.evaluate_fix("a", "b", "err")

        self.assertAlmostEqual(result["correctness"], 0.8)
        self.assertAlmostEqual(result["minimalism"], 0.0)
        self.assertAlmostEqual(result["quality"], 0.0)

    def test_model_failure_returns_zeroes(self):
        judge = LLMJudge(FakeModel("", raise_error=True))
        result = judge.evaluate_fix("a", "b", "err")

        self.assertEqual(result, {"correctness": 0.0, "minimalism": 0.0, "quality": 0.0})


if __name__ == "__main__":
    unittest.main()
