import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# Import action models for validation for ALL tasks
from .models import Task1Action, Task2Action, Task3Action, Task4Action

# Import all environments
from .task1 import Task1Env
from .task2 import Task2Env
from .task3 import Task3Env
from .task4 import Task4Env

app = FastAPI(
    title="Content Moderation OpenEnv",
    description="Dynamic Content Moderation RL environment for training safety agents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: dict = {}

def _get_env(task: str):
    if task not in _envs:
        if task == "task1-single-post":
            _envs[task] = Task1Env()
        elif task == "task2-user-trajectory":
            _envs[task] = Task2Env()
        elif task == "task3-platform-policy":
            _envs[task] = Task3Env()
        elif task == "task4-appeals":
            _envs[task] = Task4Env()
        else:
            raise HTTPException(status_code=404, detail="Unknown task")
    return _envs[task]


# ─── Standard OpenEnv endpoints ──────────────────────────────────────────────
# Note: Removed strict `response_model` decorators so FastAPI can return 
# any task's specific Pydantic result dynamically without crashing.

@app.post("/reset")
async def reset(task: str = Query(default="task1-single-post")):
    env = _get_env(task)
    return env.reset()


@app.post("/step")
async def step(
    payload: Dict[str, Any],
    task: str = Query(default="task1-single-post"),
):
    env = _get_env(task)
    try:
        # Dynamically validate the incoming payload based on which task is running
        if task == "task1-single-post":
            action_model = Task1Action(**payload)
        elif task == "task2-user-trajectory":
            action_model = Task2Action(**payload)
        elif task == "task3-platform-policy":
            action_model = Task3Action(**payload)
        elif task == "task4-appeals":
            action_model = Task4Action(**payload)
        else:
            raise HTTPException(status_code=400, detail="Unknown task")
            
        return env.step(action_model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
async def state(task: str = Query(default="task1-single-post")):
    env = _get_env(task)
    return env.state()


# ─── Utility endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "name": "task1-single-post",
                "description": "Single post moderation under uncertainty (POMDP)",
                "difficulty": "easy",
                "max_steps": 1,
                "actions": ["allow", "warn", "remove"],
            },
            {
                "name": "task2-user-trajectory",
                "description": "Sequential user moderation over time (True RL)",
                "difficulty": "medium",
                "max_steps": "variable",
                "actions": ["allow", "warn", "restrict", "remove", "ban"],
            },
            {
                "name": "task3-platform-policy",
                "description": "Multi-objective platform strictness optimization",
                "difficulty": "hard",
                "max_steps": 20,
                "actions": ["increase_strictness", "decrease_strictness", "keep_policy_same"],
            },
            {
                "name": "task4-appeals",
                "description": "Appeals system with consistency and correction",
                "difficulty": "hard",
                "max_steps": 1,
                "actions": ["uphold", "reverse", "escalate"],
            }
        ]
    }