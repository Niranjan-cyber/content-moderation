import os
import random
import pandas as pd
from typing import Optional, Dict, Any

from ..models import (
    Task4Observation,
    Task4Action,
    Task4StepResult,
    Task4ResetResult
)

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv")
)


class Task4Env:

    def __init__(self, data_path: str = DATA_PATH, seed: Optional[int] = None):
        self.df = pd.read_csv(data_path)
        self._rng = random.Random(seed)

        self.current_row = None
        self.step_count = 0
        self.done = True
        self.appeal_id = 0

    # -------------------------------------------------------------

    def reset(self) -> Task4ResetResult:
        idx = self._rng.randint(0, len(self.df) - 1)
        row = self.df.iloc[idx]

        self.current_row = row
        self.step_count = 1
        self.done = False
        self.appeal_id += 1

        return Task4ResetResult(
            observation=self._build_obs(),
            done=False,
            info={"message": "Appeal started"}
        )

    # -------------------------------------------------------------

    def step(self, action_model: Task4Action) -> Task4StepResult:
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        action = action_model.action
        row = self.current_row

        should_reverse = int(row["should_reverse"])

        reward = 0.0

        # [target] Correctness
        if should_reverse == 1:
            if action == "reverse":
                reward += 1.0
            elif action == "uphold":
                reward -= 2.0
        else:
            if action == "uphold":
                reward += 1.0
            elif action == "reverse":
                reward -= 1.0

        # [brain] Consistency bonus
        similarity_score = self._compute_similarity_score(row)
        reward += 0.5 * similarity_score

        # escalate = safe neutral option
        if action == "escalate":
            reward += 0.2

        reward = max(0.0, min(1.0, (reward + 2) / 3))

        self.done = True

        return Task4StepResult(
            observation=self._build_obs(),
            reward=round(reward, 3),
            done=True,
            info={
                "should_reverse": should_reverse,
                "action": action,
                "similarity_score": round(similarity_score, 3)
            }
        )

    # -------------------------------------------------------------

    def _compute_similarity_score(self, row):
        """
        Simple proxy for consistency:
        uses dataset signals instead of embeddings
        """
        similar = self.df[
            (self.df["content_type"] == row["content_type"]) &
            (abs(self.df["noisy_toxicity_score"] - row["noisy_toxicity_score"]) < 0.2)
        ]

        if len(similar) == 0:
            return 0.5

        same_action = (similar["original_action_taken"] == row["original_action_taken"]).mean()
        return float(same_action)

    # -------------------------------------------------------------

    def _build_obs(self) -> Task4Observation:
        row = self.current_row

        sim_cases = len(self.df[self.df["content_type"] == row["content_type"]])
        sim_score = self._compute_similarity_score(row)

        return Task4Observation(
            appeal_id=self.appeal_id,
            original_post_text=row["text"],
            original_action_taken=str(row["original_action_taken"]),
            noisy_toxicity_score=row["noisy_toxicity_score"],
            confidence_level=row["confidence_level"],
            user_id=row["user_id"],
            user_history=row.get("violation_history", 0),
            should_reverse=int(row["should_reverse"]),
            similar_cases=sim_cases,
            similar_agreement=round(sim_score, 3),
            step=self.step_count
        )

    # -------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        return {
            "appeal_id": self.appeal_id,
            "step": self.step_count,
            "done": self.done
        }