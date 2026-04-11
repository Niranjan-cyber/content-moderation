#!/usr/bin/env python3
"""
CascadeGuard — Content Moderation RL Environment
=================================================
FastAPI server implementing the OpenEnv spec.

Architecture:
  Each task lives in its own class (task1.py … task4.py).
  This file loads the dataset once, instantiates all four envs,
  and routes HTTP calls to them. Single-worker (HF Spaces).

Endpoints:
  GET  /             → health check
  GET  /tasks        → list all tasks with metadata
  POST /reset        → start new episode   ?task=<name>
  POST /step         → take one action      ?task=<name>  body: {"action": str}
  GET  /state        → current state        ?task=<name>

Tasks:
  task1-single-post       POMDP,                1 step,  easy
  task2-user-trajectory   Sequential RL,        10 steps, medium
  task3-platform-policy   Multi-objective RL,   20 steps, hard
  task4-appeals           Meta-Decision RL,     10 steps, hard
"""

import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Allow imports from repo root (models.py lives there)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.task1 import Task1Env
from server.task2 import Task2Env
from server.task3 import Task3Env
from server.task4 import Task4Env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.getenv(
    "DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv"),
)
SEED = int(os.getenv("SEED", "42"))

# ---------------------------------------------------------------------------
# Load dataset once at startup
# ---------------------------------------------------------------------------

def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    int_cols = [
        "id", "true_toxicity", "correct_action", "follower_bucket",
        "is_adversarial", "user_id", "day", "violation_history",
        "last_action", "escalation_level", "original_action_taken",
        "should_reverse", "label", "appeal_id",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    for col in ["noisy_toxicity_score", "confidence_level"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    for col in ["text", "modified_text", "content_type", "group"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df


DF: pd.DataFrame = _load_df(DATA_PATH)

# ---------------------------------------------------------------------------
# Env instances (one per task, persisted across requests)
# ---------------------------------------------------------------------------

ENVS = {
    "task1-single-post":     Task1Env(df=DF, seed=SEED),
    "task2-user-trajectory": Task2Env(df=DF, seed=SEED),
    "task3-platform-policy": Task3Env(df=DF, seed=SEED),
    "task4-appeals":         Task4Env(df=DF, seed=SEED),
}

VALID_ACTIONS = {
    "task1-single-post":     {"allow", "warn", "remove"},
    "task2-user-trajectory": {"allow", "warn", "restrict", "remove", "ban"},
    "task3-platform-policy": {"increase_strictness", "decrease_strictness", "keep_policy_same"},
    "task4-appeals":         {"uphold", "reverse", "escalate"},
}

TASK_NAMES = list(VALID_ACTIONS.keys())

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CascadeGuard Content Moderation Environment",
    description=(
        "OpenEnv-compliant RL environment for content moderation research. "
        "Four tasks of increasing complexity grounded in real toxicity data."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class ActionRequest(BaseModel):
    action: str


class StepResponse(BaseModel):
    observation: dict
    reward:      float
    done:        bool
    info:        dict


class ResetResponse(BaseModel):
    observation: dict
    done:        bool
    info:        dict

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/web")
def health():
    return {
        "status":       "ok",
        "environment":  "CascadeGuard Content Moderation Env",
        "version":      "0.3.0",
        "tasks":        TASK_NAMES,
        "dataset_rows": len(DF),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name":        "task1-single-post",
                "description": "Single post moderation under uncertainty (POMDP)",
                "difficulty":  "easy",
                "max_steps":   1,
                "actions":     ["allow", "warn", "remove"],
                "reward_range":[0.0, 1.0],
                "observation_fields": [
                    "post_id", "text", "noisy_toxicity_score",
                    "confidence_level", "follower_bucket", "content_type", "step",
                ],
            },
            {
                "name":        "task2-user-trajectory",
                "description": "Sequential user moderation over time (True RL)",
                "difficulty":  "medium",
                "max_steps":   10,
                "actions":     ["allow", "warn", "restrict", "remove", "ban"],
                "reward_range":[0.0, 1.0],
                "observation_fields": [
                    "post_id", "text", "noisy_toxicity_score", "confidence_level",
                    "follower_bucket", "content_type", "user_id", "day",
                    "violation_history", "last_action", "escalation_level",
                    "post_history", "step",
                ],
            },
            {
                "name":        "task3-platform-policy",
                "description": "Platform-wide policy optimisation (Multi-objective RL)",
                "difficulty":  "hard",
                "max_steps":   20,
                "actions":     ["increase_strictness", "decrease_strictness", "keep_policy_same"],
                "reward_range":[0.0, 1.0],
                "reward_weights": {
                    "safety": 0.35, "engagement": 0.30,
                    "retention": 0.20, "fairness": 0.15,
                },
                "observation_fields": [
                    "current_day", "global_toxicity_rate", "engagement_score",
                    "user_retention_rate", "moderation_strictness_level",
                    "fairness_gap", "group_a_false_positive_rate",
                    "group_b_false_positive_rate", "active_posts",
                    "platform_user_summary", "step",
                ],
            },
            {
                "name":        "task4-appeals",
                "description": "Appeals & consistency system (Meta-Decision RL)",
                "difficulty":  "hard",
                "max_steps":   10,
                "actions":     ["uphold", "reverse", "escalate"],
                "reward_range":[0.0, 1.0],
                "episode_scoring": "0.7 * accuracy + 0.3 * consistency",
                "observation_fields": [
                    "appeal_id", "post_id", "text", "noisy_toxicity_score",
                    "confidence_level", "content_type", "original_action_taken",
                    "original_action_label", "follower_bucket",
                    "precedent_history", "step",
                ],
            },
        ]
    }


@app.post("/reset")
def reset(
    task: str = Query("task1-single-post", description="Task name"),
) -> ResetResponse:
    _validate_task(task)
    result = ENVS[task].reset()
    return ResetResponse(
        observation=result["observation"],
        done=result["done"],
        info=result.get("info", {}),
    )


@app.post("/step")
def step(
    body: ActionRequest,
    task: str = Query("task1-single-post", description="Task name"),
) -> StepResponse:
    _validate_task(task)

    action = body.action.strip().lower()
    if action not in VALID_ACTIONS[task]:
        raise HTTPException(
            400,
            f"Invalid action '{action}' for {task}. "
            f"Valid actions: {sorted(VALID_ACTIONS[task])}",
        )

    env    = ENVS[task]
    result = env.step(action)
    return StepResponse(
        observation=result.get("observation") or {},
        reward=float(result.get("reward", 0.0)),
        done=bool(result.get("done", False)),
        info=result.get("info", {}),
    )


@app.get("/state")
def state(
    task: str = Query("task1-single-post", description="Task name"),
) -> dict:
    _validate_task(task)
    return ENVS[task].state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_task(task: str) -> None:
    if task not in TASK_NAMES:
        raise HTTPException(
            400,
            f"Unknown task '{task}'. Valid tasks: {TASK_NAMES}",
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
