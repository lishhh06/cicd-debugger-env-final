from __future__ import annotations

from env.tasks.task_types import CICDTask


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
    ),
]
