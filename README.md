---

title: CICD_DEBUGGER
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:

* openenv

---

# CI/CD Pipeline Debugger Environment (OpenEnv)

This project models CI/CD debugging as a reinforcement learning problem where an agent iteratively fixes pipeline failures.

---

## 1. Overview

The environment simulates common CI/CD failures including:

* invalid commands
* version mismatches
* pipeline execution errors

An agent interacts with the system using a structured loop:

Observe → Act → Receive Reward → Repeat

---

## 2. Environment Design

The environment follows the OpenEnv interface:

* `reset()` → returns initial observation
* `step(action)` → returns `(observation, reward, done, info)`
* `state()` → returns current environment state

Typed schemas are defined in `env/models.py`.

---

## 3. Action Space

Supported actions:

* READ_FILE
* READ_LOGS
* ANALYZE_ERROR
* EDIT_CONFIG
* RUN_STAGE
* VALIDATE
* SUBMIT

Each action accepts a structured payload.

---

## 4. Observation

Each observation includes:

* pipeline configuration
* logs and error messages
* current stage
* task-specific metadata

---

## 5. Tasks

The environment includes 13 tasks:

* Easy: command and syntax issues
* Medium: dependency and version issues
* Hard: multi-stage pipeline logic

Each task is designed such that:

* it fails under the initial configuration
* it passes after applying the correct fix

---

## 6. Evaluation and Reward

Reward is computed as:

* `1.0` → correct fix
* `0.0` → incorrect fix

Optional shaping:

* partial reward for progress
* penalties for invalid or premature actions

Tool-specific bonuses:

* `+0.05` → read_logs
* `+0.1` → analyze_error
* `+0.2` → edit_config
* `+0.2` → run_pipeline_stage
* `-0.3` → premature validation (no pipeline executed)
* `-0.2` → invalid tool

All rewards are clamped to `[0.0, 1.0]`.

Evaluation is deterministic and reproducible.

---

## 7. Example Task

Initial:

* Command: `npm tset`
* Expected: `npm test`

Failure:

* Build fails due to invalid command

Fix:

* Replace `tset` → `test`

---

## 8. Example Run

```
[START]
[STEP] action=READ_LOGS reward=0.00
[STEP] action=EDIT_CONFIG reward=0.00
[STEP] action=VALIDATE reward=1.00
[END] success=true
```

---

## 9. Inference

The inference loop:

* queries a model for actions
* executes actions via `env.step()`
* logs output in a fixed format

Output format:

* `[START]`
* `[STEP]`
* `[END]`

---

## 10. API (HuggingFace Space)

The environment is exposed via FastAPI.

Endpoints:

* `GET /`
* `GET /health`
* `POST /reset`
* `POST /step`
* `GET /state`

---

## 11. Setup

```bash
pip install -r requirements.txt
```

Environment variables:

```bash
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=<your_token>
```

---

## 12. Run

```bash
python inference.py
```

---

## 13. Docker

```bash
docker build -t cicd-debugger-env .
docker run -e HF_TOKEN=<your_token> cicd-debugger-env
```

---

## 14. Testing

```bash
python -m unittest discover -s tests -v
```

---

## 15. Validation

```bash
python -m openenv.cli.__main__ validate
```

---

## 16. OpenEnv Compliance

Implements:

* `reset()`
* `step()`
* `state()`

Uses:

* structured observation/action/reward
* deterministic evaluation
* containerized execution

---

## 17. Success Criteria

* pipeline passes after fix
* reward = 1.0
* results are deterministic across runs

---

## 18. Summary

This repository provides a reproducible environment for training and evaluating agents on CI/CD debugging tasks using structured interaction and deterministic evaluation.
