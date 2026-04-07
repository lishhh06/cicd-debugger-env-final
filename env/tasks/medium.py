from __future__ import annotations

from env.tasks.task_types import CICDTask


MEDIUM_TASKS: list[CICDTask] = [
    CICDTask(
        task_id="medium-python-version",
        title="Align Python version",
        description="Tests require Python 3.11 but workflow pins an older version.",
        difficulty="medium",
        failure_stage="build",
        broken_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.8"
      - run: pip install -r requirements.txt
      - run: pytest -q
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q
""".strip(),
        logs="build failed: package requires python>=3.11",
        error_message="python interpreter version mismatch",
        actual_bug="workflow pins python-version 3.8 while project requires 3.11",
        metadata={"broken_token": 'python-version: "3.8"', "fixed_token": 'python-version: "3.11"'},
    ),
    CICDTask(
        task_id="medium-cache-key",
        title="Fix cache invalidation key",
        description="Dependency cache key ignores lockfile hash and restores stale dependencies.",
        difficulty="medium",
        failure_stage="test",
        broken_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - uses: actions/cache@v4
        with:
          path: ~/.npm
          key: node-modules-${{ runner.os }}
      - run: npm ci
      - run: npm test
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - uses: actions/cache@v4
        with:
          path: ~/.npm
          key: node-modules-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
      - run: npm ci
      - run: npm test
""".strip(),
        logs="test failed: stale cache restored old dependency tree",
        error_message="cache key misses lockfile fingerprint",
        actual_bug="cache key is too broad and never invalidates on dependency changes",
        metadata={"broken_token": "key: node-modules-${{ runner.os }}", "fixed_token": "hashFiles('**/package-lock.json')"},
    ),
    CICDTask(
        task_id="medium-artifact-permissions",
        title="Repair artifact permissions",
        description="Artifact upload fails due to insufficient token permissions.",
        difficulty="medium",
        failure_stage="deploy",
        broken_config="""
name: CI
on: [push]
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: web-build
          path: dist/
""".strip(),
        expected_config="""
name: CI
on: [push]
permissions:
  contents: read
  actions: write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: web-build
          path: dist/
""".strip(),
        logs="deploy stage failed: Resource not accessible by integration",
        error_message="insufficient permissions for upload-artifact",
        actual_bug="actions:write permission missing from workflow permissions",
        metadata={"broken_token": "permissions:\n  contents: read", "fixed_token": "actions: write"},
    ),
]
