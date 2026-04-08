# Copyright (c) 
# CascadeGuard -- Content Moderation RL Environment
# client.py: OpenEnv WebSocket clients for all four tasks

from __future__ import annotations
from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        Task1Action, Task1Observation,
        Task2Action, Task2Observation,
        Task3Action, Task3Observation,
        Task4Action, Task4Observation,
    )
except ImportError:
    from models import (
        Task1Action, Task1Observation,
        Task2Action, Task2Observation,
        Task3Action, Task3Observation,
        Task4Action, Task4Observation,
    )


# --- Task 1: Single Post Moderation (POMDP) -----------------------------------

class Task1Env(EnvClient[Task1Action, Task1Observation, Dict]):
    """
    WebSocket client for single-post moderation under uncertainty.
    Use reset(task_id='task1-single-post') to start an episode.
    Actions: allow | warn | remove
    """

    def _step_payload(self, action: Task1Action) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Task1Observation]:
        obs_raw = payload.get("observation") or {}
        merged: Dict[str, Any] = {
            **obs_raw,
            "done":     payload.get("done", False),
            "reward":   payload.get("reward"),
            "metadata": obs_raw.get("metadata") or {},
        }
        observation = Task1Observation.model_validate(merged)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict:
        return payload


# --- Task 2: User Trajectory Control (Sequential RL) -------------------------

class Task2Env(EnvClient[Task2Action, Task2Observation, Dict]):
    """
    WebSocket client for multi-step user trajectory moderation.
    Use reset(task_id='task2-user-trajectory') to start an episode.
    Actions: allow | warn | restrict | remove | ban
    """

    def _step_payload(self, action: Task2Action) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Task2Observation]:
        obs_raw = payload.get("observation") or {}
        merged: Dict[str, Any] = {
            **obs_raw,
            "done":     payload.get("done", False),
            "reward":   payload.get("reward"),
            "metadata": obs_raw.get("metadata") or {},
        }
        observation = Task2Observation.model_validate(merged)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict:
        return payload


# --- Task 3: Platform Policy Optimisation (Multi-objective RL) ---------------

class Task3Env(EnvClient[Task3Action, Task3Observation, Dict]):
    """
    WebSocket client for platform-wide moderation policy control.
    Use reset(task_id='task3-platform-policy') to start an episode.
    Actions: increase_strictness | decrease_strictness | keep_policy_same
    """

    def _step_payload(self, action: Task3Action) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Task3Observation]:
        obs_raw = payload.get("observation") or {}
        merged: Dict[str, Any] = {
            **obs_raw,
            "done":     payload.get("done", False),
            "reward":   payload.get("reward"),
            "metadata": obs_raw.get("metadata") or {},
        }
        observation = Task3Observation.model_validate(merged)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict:
        return payload


# --- Task 4: Appeals & Consistency (Meta-Decision RL) ------------------------

class Task4Env(EnvClient[Task4Action, Task4Observation, Dict]):
    """
    WebSocket client for appeals review and consistency enforcement.
    Use reset(task_id='task4-appeals') to start an episode.
    Actions: uphold | reverse | escalate
    """

    def _step_payload(self, action: Task4Action) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Task4Observation]:
        obs_raw = payload.get("observation") or {}
        merged: Dict[str, Any] = {
            **obs_raw,
            "done":     payload.get("done", False),
            "reward":   payload.get("reward"),
            "metadata": obs_raw.get("metadata") or {},
        }
        observation = Task4Observation.model_validate(merged)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Dict:
        return payload