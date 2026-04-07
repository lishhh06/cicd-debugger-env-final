import os
import re
import subprocess
import sys
from pathlib import Path
import unittest


class InferenceOutputFormatTests(unittest.TestCase):
    def test_inference_prints_required_markers(self):
        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["OFFLINE_INFERENCE"] = "1"

        completed = subprocess.run(
            [sys.executable, "inference.py", "--max-steps", "3", "--offline", "--force-local-env"],
            cwd=project_root,
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[START] "))
        self.assertTrue(lines[-1].startswith("[END] "))

        start_pattern = re.compile(r"^\[START\] task=\S+ env=\S+ model=.+$")
        step_pattern = re.compile(
            r"^\[STEP\] step=\d+ action=.* reward=-?\d+\.\d{2} done=(true|false) error=(null|.+)$"
        )
        end_pattern = re.compile(
            r"^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(-?\d+\.\d{2}(,-?\d+\.\d{2})*)?$"
        )

        self.assertRegex(lines[0], start_pattern)

        step_lines = [line for line in lines if line.startswith("[STEP] ")]
        self.assertTrue(step_lines)
        for line in step_lines:
            self.assertRegex(line, step_pattern)

        self.assertRegex(lines[-1], end_pattern)

        for line in lines:
            self.assertTrue(
                line.startswith("[START] ") or line.startswith("[STEP] ") or line.startswith("[END] "),
                f"Unexpected output line: {line}",
            )


if __name__ == "__main__":
    unittest.main()
