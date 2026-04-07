from __future__ import annotations

from fastapi.responses import HTMLResponse
from dataclasses import dataclass
import os
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
import uvicorn

from env.environment import CICDDebuggerEnvironment, MAX_STEPS
from env.models import Action, Observation, Reward


app = FastAPI(title="CI/CD Debugger OpenEnv Server")


class ResetRequest(BaseModel):
    task_id: str | None = None
    difficulty: str | None = None
    max_steps: int = Field(default=MAX_STEPS, ge=1, le=100)


class StepRequest(BaseModel):
    action: Action | str | dict[str, Any]


class StepResponse(BaseModel):
    task_id: str
    step_count: int
    reward: float
    reward_model: Reward
    done: bool
    observation: Observation
    last_action: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    initialized: bool
    task_id: str | None = None
    step_count: int = 0
    done: bool = False
    last_action: str | None = None
    observation: Observation | None = None
    internal_state: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RuntimeSession:
    env: CICDDebuggerEnvironment
    task_id: str
    step_count: int = 0
    done: bool = False
    last_action: str | None = None
    last_reward: float = 0.0
    last_observation: dict[str, Any] | None = None
    last_info: dict[str, Any] | None = None


runtime_session: RuntimeSession | None = None


def _as_observation_model(observation: dict[str, Any] | Observation) -> Observation:
    if isinstance(observation, Observation):
        return observation
    return Observation.model_validate(observation)


def _build_step_response(session: RuntimeSession) -> StepResponse:
    observation = session.last_observation or {}
    info_payload = session.last_info or {}
    reward_payload = info_payload.get("reward_model")
    if isinstance(reward_payload, dict):
        reward_model = Reward.model_validate(reward_payload)
    else:
        reward_model = Reward(value=float(session.last_reward), components={"total": float(session.last_reward)})

    return StepResponse(
        task_id=session.task_id,
        step_count=int(observation.get("step_count") or session.step_count),
        reward=float(session.last_reward),
        reward_model=reward_model,
        done=bool(session.done),
        observation=_as_observation_model(observation),
        last_action=session.last_action,
        info=info_payload,
    )


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>CI/CD Debugger</title>
    </head>
    <body style="font-family: Arial; padding: 40px; background:#111; color:white;">
        <h1>CI/CD Debugger Environment</h1>

        <button onclick="reset()">Reset</button>

        <br><br>

        <select id="action">
            <option>READ_LOGS</option>
            <option>READ_FILE</option>
            <option>RUN_STAGE</option>
            <option>VALIDATE</option>
        </select>

        <button onclick="step()">Step</button>

        <pre id="output" style="background:#222; padding:20px;"></pre>

        <script>
        async function reset(){
            const res = await fetch('/reset', {method:'POST'});
            const data = await res.json();
            document.getElementById('output').textContent =
                JSON.stringify(data, null, 2);
        }

        async function step(){
            const action = document.getElementById('action').value;

            const res = await fetch('/step', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({
                    action_type: action,
                    payload: {}
                })
            });

            const data = await res.json();
            document.getElementById('output').textContent =
                JSON.stringify(data, null, 2);
        }
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=StepResponse)
async def reset(payload: ResetRequest | None = None) -> StepResponse:
    global runtime_session

    request = payload or ResetRequest()
    env = CICDDebuggerEnvironment(max_steps=int(request.max_steps))
    observation = await env.reset(task_id=request.task_id, difficulty=request.difficulty)

    runtime_session = RuntimeSession(
        env=env,
        task_id=str(observation.get("task_id", request.task_id or "cicd-debugger-task")),
        step_count=0,
        done=False,
        last_action=None,
        last_reward=0.0,
        last_observation=observation,
        last_info={
            "message": "environment reset",
            "tool": "reset",
            "error": None,
            "reward_model": Reward(value=0.0, components={"total": 0.0}).model_dump(),
        },
    )
    return _build_step_response(runtime_session)


@app.post("/step", response_model=StepResponse)
async def step(payload: StepRequest) -> StepResponse:
    global runtime_session

    if runtime_session is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if runtime_session.done:
        return _build_step_response(runtime_session)

    observation, reward, done, info = await runtime_session.env.step(payload.action)

    runtime_session.step_count = int(observation.get("step_count", runtime_session.step_count + 1))
    runtime_session.done = bool(done)
    runtime_session.last_action = payload.action if isinstance(payload.action, str) else str(payload.action)
    runtime_session.last_reward = float(reward)
    runtime_session.last_observation = observation
    runtime_session.last_info = dict(info or {})

    return _build_step_response(runtime_session)


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    if runtime_session is None:
        return StateResponse(initialized=False)

    observation = None
    if runtime_session.last_observation is not None:
        observation = _as_observation_model(runtime_session.last_observation)

    return StateResponse(
        initialized=True,
        task_id=runtime_session.task_id,
        step_count=runtime_session.step_count,
        done=runtime_session.done,
        last_action=runtime_session.last_action,
        observation=observation,
        internal_state=runtime_session.env.state(),
    )


@app.post("/state", response_model=StateResponse)
async def state_post() -> StateResponse:
    return await state()


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
