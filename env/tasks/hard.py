from __future__ import annotations

from env.tasks.task_types import CICDTask


HARD_TASKS: list[CICDTask] = [
    CICDTask(
        task_id="hard-matrix-logic",
        title="Fix matrix include-exclude logic",
        description="Matrix includes unsupported versions and causes deterministic CI breakage.",
        difficulty="hard",
        failure_stage="test",
        broken_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pytest -q
""".strip(),
        expected_config="""
name: CI
on: [push]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.13"]
        exclude:
          - os: windows-latest
            python-version: "3.13"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pytest -q
""".strip(),
        logs="test stage failed: wheel build unavailable for windows-latest + python 3.13",
        error_message="matrix includes unsupported runtime combination",
        actual_bug="matrix logic is missing an exclude for unstable runtime pair",
        metadata={"broken_token": "python-version: [\"3.10\", \"3.11\", \"3.13\"]", "fixed_token": "exclude:"},
    ),
    CICDTask(
        task_id="hard-conditional-deploy",
        title="Repair deploy conditional",
        description="Deploy job runs regardless of failed tests due to always() condition.",
        difficulty="hard",
        failure_stage="deploy",
        broken_config="""
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run build
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test
  deploy:
    runs-on: ubuntu-latest
    needs: test
    if: always()
    steps:
      - run: echo deploying
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
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test
  deploy:
    runs-on: ubuntu-latest
    needs: test
    if: success() && github.ref == 'refs/heads/main'
    steps:
      - run: echo deploying
""".strip(),
        logs="deploy stage triggered despite failing tests on non-main branch",
        error_message="unsafe deploy condition bypasses quality gates",
        actual_bug="deploy condition uses always() instead of guarded success check",
        metadata={"broken_token": "if: always()", "fixed_token": "if: success() && github.ref == 'refs/heads/main'"},
    ),
    CICDTask(
        task_id="hard-needs-order",
        title="Fix job dependency ordering",
        description="Deploy depends only on build and can run before tests complete.",
        difficulty="hard",
        failure_stage="deploy",
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
      - run: npm test
  deploy:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - run: echo deploying package
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
  deploy:
    runs-on: ubuntu-latest
    needs: [build, test]
    steps:
      - run: echo deploying package
""".strip(),
        logs="deploy stage started before tests finished, causing regression release",
        error_message="deploy dependency graph skips mandatory test gate",
        actual_bug="deploy job does not depend on test job",
        metadata={"broken_token": "needs: build", "fixed_token": "needs: [build, test]"},
    ),
]
