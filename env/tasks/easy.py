from __future__ import annotations

from env.tasks.task_types import CICDTask


def _create_deterministic_grader(expected_config: str):
    """Create a deterministic grader that checks config similarity."""
    def grader(current: str, expected: str, metadata: dict = None) -> float:
        import re
        normalized_current = re.sub(r"\s+", " ", current.strip().lower())
        normalized_expected = re.sub(r"\s+", " ", expected.strip().lower())
        if normalized_current == normalized_expected:
            return 0.95
        elif normalized_current in normalized_expected or normalized_expected in normalized_current:
            return 0.75
        return 0.05
    return grader


EASY_TASKS: list[CICDTask] = [
    CICDTask(
        task_id="easy-command-typo",
        title="Fix test command typo",
        description="A typo in the test command breaks the CI test stage.",
        difficulty="easy",
        failure_stage="test",
        broken_config="""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm tset
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test
""".strip(),
        logs="test stage failed: npm ERR! missing script: tset",
        error_message="command not found: npm tset",
        actual_bug="test step runs npm tset instead of npm test",
        metadata={"broken_token": "npm tset", "fixed_token": "npm test"},
        deterministic_grader=_create_deterministic_grader("""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test
""".strip()),
    ),
    CICDTask(
        task_id="easy-missing-checkout",
        title="Add missing checkout",
        description="Build stage fails because repository checkout is missing.",
        difficulty="easy",
        failure_stage="build",
        broken_config="""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm ci
      - run: npm run build
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
""".strip(),
        logs="build stage failed: package-lock.json not found in workspace",
        error_message="missing checkout step before build commands",
        actual_bug="repository checkout step was removed",
        metadata={"broken_token": "", "fixed_token": "uses: actions/checkout@v4"},
        deterministic_grader=_create_deterministic_grader("""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
""".strip()),
    ),
    CICDTask(
        task_id="easy-yaml-indentation",
        title="Fix YAML indentation",
        description="Pipeline config has malformed YAML indentation.",
        difficulty="easy",
        failure_stage="build",
        broken_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
       - run: pytest
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest
""".strip(),
        logs="yaml parser error: while parsing a block mapping",
        error_message="invalid YAML structure in workflow file",
        actual_bug="test command list item is mis-indented",
        metadata={},
        deterministic_grader=_create_deterministic_grader("""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest
""".strip()),
    ),
]
