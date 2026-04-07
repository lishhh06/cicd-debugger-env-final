---
title: CICD_DEBUGGER
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# CI/CD Pipeline Debugger Environment (OpenEnv)

## 1. Project Goal

This repository implements an AI training and evaluation environment where an agent learns to debug broken CI/CD pipelines automatically.

The environment targets real-world DevOps failure patterns, including:

- YAML syntax and structure issues
- Incorrect build/test commands (for example, npm tset -> npm test)
- Dependency and setup failures
- Multi-stage pipeline execution errors

This is designed as an RL-style interaction loop:

Observe -> Think -> Act -> Get Reward -> Repeat

## 2. Why This Matters

CI/CD failures are common, repetitive, and often multi-step to resolve. This project turns that workflow into a structured learning environment where agents:

- Read failure context
- Reason about root causes
- Propose and apply fixes
- Get shaped rewards for robust behavior

## 3. System Architecture

High-level flow:

Agent (LLM) -> Action -> Environment.step() -> Reward/Evaluation -> Next step

Core integration path:

Model -> Action -> Environment.step() -> RewardCalculator

RewardCalculator integrates:

- DeterministicGrader
- LLMJudge
- HiddenTestRunner
- AntiHackingDetector

### 3.1 OpenEnv Interface (Typed)

Typed Pydantic models are defined in `env/models.py`:

- `Observation`: strict schema for environment observations
- `Action`: normalized tool + payload action schema
- `Reward`: bounded reward model with components

Environment contract:

- `reset()` returns the initial `Observation` payload
- `step(action)` returns `(observation, reward, done, info)`
- `state()` returns current environment state snapshot

Server/API contract models are exposed in `server/app.py` and use the same typed observation/action/reward structures.

### 3.2 Action and Observation Spaces

Observation fields include:

- `task_id`, `difficulty`, `failure_stage`, `actual_bug`
- `config`, `logs`, `error_message`
- `available_tools`, `progress_flags`
- `file_modification_count`, `hidden_test_pass_rate`, `step_count`, `last_action_error`

Action schema:

- `tool`: one of `read_file`, `read_logs`, `analyze_error`, `edit_config`, `run_pipeline_stage`, `run_tests`, `validate_fix`, `submit_solution`
- `payload`: optional dict (for example `{ "raw": "replace npm tset with npm test" }`)

Reward schema:

- `value`: bounded float in `[0.0, 1.0]`
- `components`: reward breakdown dictionary

## 4. Core Modules

### 4.1 Quality Judge

- File: env/graders/llm_judge.py
- Purpose: quality-aware scoring of fixes
- Output keys: correctness, minimalism, quality (all in [0,1])
- Guarantees:
	- strict JSON parsing attempt
	- robust fallback parsing for messy output
	- no-crash behavior (safe zero scores on failure)

### 4.2 Deterministic Grader

- File: env/graders/deterministic.py
- Purpose: reproducible correctness scoring (0-1)
- Checks:
	- YAML validity
	- command and fix correctness
	- similarity and issue resolution
- Rules:
	- deterministic only
	- same input, same score

### 4.3 Anti-Hacking Detector

- File: env/anti_hacking.py
- Purpose: detect reward-hacking and shortcut behavior
- Penalty detectors:
	- stage skipping (if: false, when: never)
	- fake success (echo tests passed, unsafe exit 0 patterns)
	- pipeline breakage between versions
	- excessive edits
	- timeout abuse via too many steps

### 4.4 Hidden Tests

- File: env/hidden_tests.py
- Purpose: test fix robustness, not just exact-match overfitting
- Method:
	- deterministic variant generation (OS, versions, env shifts)
	- evaluate pass rate across variants

### 4.5 Reward Shaping

- File: env/rewards.py
- Purpose: step-level learning signal
- Components:
	- progress rewards (logs, analysis, fix proposal)
	- execution rewards (pipeline run, tests pass)
	- quality rewards (deterministic + hidden tests + LLM judge)
	- anti-hacking penalties

## 5. Inference and Evaluation

### 5.1 Prompt and Model Layers

- inference/prompts.py: stable prompt templates and fallback action heuristics
- inference/model_wrapper.py: OpenAI client action generation, candidate generation, and safe fallback

Canonical action tools used by environment and inference:

- read_file
- read_logs
- analyze_error
- edit_config
- run_pipeline_stage
- run_tests
- validate_fix
- submit_solution

### 5.2 Metrics and Artifacts

- inference/metrics.py: reward, success-rate, and failure reason tracking
- inference/visualize.py: reward curve and metrics artifact export

### 5.3 Submission-Critical Runtime

- File: inference.py (root)
- Responsibilities:
	- initialize model and environment
	- run step loop
	- calculate rewards
	- emit strict stdout contract
	- always emit END line

Required output format:

- [START] task=... env=... model=...
- [STEP] step=<n> action=... reward=0.00 done=<true|false> error=<msg|null>
- [END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Rules enforced:

- single-line logs only
- reward values with 2 decimals
- lowercase booleans
- no extra runtime log noise

## 6. Task Coverage

The project includes 9 CI-fix tasks spanning:

- easy: syntax and typo fixes
- medium: dependency/env/cache/permissions issues
- hard: matrix logic, conditional flow, orchestration-level failures

Representative baseline tasks (one per difficulty):

- easy: `easy-command-typo` (fix invalid `npm tset` command)
- medium: `medium-python-version` (align workflow Python version)
- hard: `hard-needs-order` (repair deploy job dependency ordering)

## 7. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_openai_compatible_api_key>"
# Optional alias; if set, this takes precedence over HF_TOKEN in inference.py
export OPENAI_API_KEY="<same_token_optional>"
# Optional, only if your inference spins environments from local images.
export LOCAL_IMAGE_NAME="<local_env_image_name>"
```

If you want to use an OpenAI access token directly:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="<your_openai_access_token>"
# Optional alias:
export OPENAI_API_KEY="<same_token_optional>"
```

## 8. Run Inference

Offline/local mode:

```bash
python inference.py --offline --force-local-env --max-steps 8 --policy-mode imp --trajectories 4
```

Model-backed mode:

```bash
python inference.py --max-steps 8 --policy-mode imp --trajectories 4
```

Run baseline across easy/medium/hard tasks:

OpenAI client mode:

```bash
OPENAI_API_KEY="<your_openai_compatible_api_key>" python baseline_inference.py --max-steps 5 --policy-mode imp --trajectories 3 --force-local-env
```

Offline reproducible mode:

```bash
python baseline_inference.py --max-steps 5 --policy-mode imp --trajectories 3 --offline --force-local-env
```

Policy modes:

- sft: deterministic heuristic policy
- direct: single model action per step
- imp: multi-candidate generation and ranking

## 9. Baseline Scores

Reproducible baseline artifact:

- `artifacts/baseline_scores.json`

Latest baseline run (`max_steps=5`, `policy_mode=imp`, `trajectories=3`):

| Task ID | Difficulty | Score | Success |
|---|---|---:|---:|
| easy-command-typo | easy | 0.541 | false |
| medium-python-version | medium | 0.679 | false |
| hard-needs-order | hard | 0.513 | false |

Aggregate:

- average score: `0.578`
- success rate: `0.000`

When `OPENAI_API_KEY` is provided, the same script runs with the OpenAI API client path in `inference.py`.

## 10. Tests

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Coverage includes:

- LLM judge
- deterministic grader
- anti-hacking detectors
- hidden tests
- reward system
- end-to-end inference output format

## 11. Validation and Submission

OpenEnv validation:

```bash
python -m openenv.cli.__main__ validate
```

Pre-submission script:

```bash
./validate-submission.sh <your_hf_space_url>
```

Required environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export OPENAI_API_KEY="<your_openai_compatible_api_key>"
# Optional fallback:
export HF_TOKEN="<your_token>"
```

Docker run (Space/API mode):

```bash
docker build -t cicd-debugger-env .
docker run --rm -p 7860:7860 cicd-debugger-env
```

Server endpoints used by validators:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

## 12. Deploy to Hugging Face Space (OpenAI Token)

This repository is already configured for Docker Spaces (`sdk: docker` in this README front matter).

1. Create a new Hugging Face Space with SDK set to `Docker`.
2. Push this repository to the Space git remote.
3. In Space Settings -> Variables and secrets, add these Secrets:

```text
OPENAI_API_KEY=<your_openai_access_token>
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

4. Optional Secrets:

```text
HF_TOKEN=<optional_fallback_token>
OFFLINE_INFERENCE=0
MAX_STEPS=8
TEMPERATURE=0.2
MAX_TOKENS=120
```

5. Keep the app port as `7860` (already configured).
6. Wait for build completion, then verify:

```bash
curl -sS https://<your-space-name>.hf.space/health
curl -sS -X POST https://<your-space-name>.hf.space/reset -H 'Content-Type: application/json' -d '{}'
```

Notes:

- `.env.example` is for local development reference only. Hugging Face Spaces use Secrets/Variables from Space Settings.
- Runtime code reads `OPENAI_API_KEY` first and falls back to `HF_TOKEN` when `OPENAI_API_KEY` is not provided.

## 13. One-line Presentation Summary

We built an OpenEnv-compliant reinforcement learning environment where AI agents learn to debug real CI/CD pipelines using multi-step reasoning, hybrid grading, anti-hacking safeguards, and robust reward shaping.
