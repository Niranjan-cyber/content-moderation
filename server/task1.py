"""
Task 1: Single Post Moderation Under Uncertainty (POMDP)
=========================================================
The agent sees ONE post per episode and must decide: allow / warn / remove.

Key RL challenge:
  - noisy_toxicity_score is an IMPERFECT signal (not the ground truth)
  - true_toxicity is HIDDEN from the agent
  - confidence_level tells the agent how much to trust the noisy signal
  - The agent must learn risk-aware decision making

Episode length: 1 step (single moderation decision)
Score range:    [0.0, 1.0]

Reward table:
  Toxic post  (true_toxicity=1): remove=1.0 | warn=0.5 | allow=0.0
  Safe post   (true_toxicity=0): allow=1.0  | warn=0.6 | remove=0.1
  + Calibration bonus: +/-0.1 based on confidence x correctness
"""

import os
import random
import pandas as pd
from typing import Optional

from ..models import Task1Observation, Task1Action, StepResult, ResetResult
DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv")
)

class Task1Env:

    ACTION_LABELS = {0: "allow", 1: "warn", 2: "remove"}
    ACTION_CODES  = {"allow": 0, "warn": 1, "remove": 2}

    def __init__(self, data_path: str = DATA_PATH, seed: Optional[int] = None):
        self.df = pd.read_csv(data_path)
        self._rng = random.Random(seed)
        self._current_post: Optional[pd.Series] = None
        self._step_count: int = 0
        self._done: bool = True           # Force reset() before first step
        self._episode_id: int = 0
        self._episode_rewards: list = []  # Track rewards across episodes

    # --------------------------------------------------------------------------
    # OpenEnv interface
    # --------------------------------------------------------------------------

    def reset(self) -> ResetResult:
        """Sample a new post, return initial observation. Starts a fresh episode."""
        idx = self._rng.randint(0, len(self.df) - 1)
        self._current_post = self.df.iloc[idx]
        self._step_count = 0
        self._done = False
        self._episode_id += 1

        return ResetResult(
            observation=self._build_observation(step=0),
            done=False,
            info={
                "episode_id": self._episode_id,
                "task": "task1-single-post",
                "hint": "noisy_toxicity_score is imperfect. Use confidence_level to calibrate.",
            },
        )

    def step(self, action: Task1Action) -> StepResult:
        """Execute one moderation decision. Episode ends immediately (single-step)."""
        if self._done:
            raise RuntimeError("Episode already done -- call reset() first.")
        if self._current_post is None:
            raise RuntimeError("No active episode -- call reset() first.")

        self._step_count += 1
        reward = self._compute_reward(action.action)
        self._done = True
        self._episode_rewards.append(reward)

        true_tox = int(self._current_post["true_toxicity"])
        correct_label = self.ACTION_LABELS[int(self._current_post["correct_action"])]

        return StepResult(
            observation=self._build_observation(step=self._step_count),
            reward=reward,
            done=True,
            info={
                "episode_id":     self._episode_id,
                "true_toxicity":  true_tox,
                "correct_action": correct_label,
                "action_taken":   action.action,
                "is_adversarial": bool(self._current_post["is_adversarial"]),
                "false_positive": true_tox == 0 and action.action == "remove",
                "false_negative": true_tox == 1 and action.action == "allow",
                "group":          str(self._current_post["group"]),
            },
        )

    def state(self) -> dict:
        """Return current internal state (for OpenEnv state() endpoint)."""
        return {
            "episode_id":       self._episode_id,
            "step_count":       self._step_count,
            "done":             self._done,
            "current_post_id":  int(self._current_post["id"]) if self._current_post is not None else None,
            "episodes_run":     self._episode_id,
            "mean_reward":      round(
                sum(self._episode_rewards) / len(self._episode_rewards), 4
            ) if self._episode_rewards else 0.0,
        }

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _build_observation(self, step: int) -> Task1Observation:
        row = self._current_post
        return Task1Observation(
            post_id=int(row["id"]),
            text=str(row["text"]),
            noisy_toxicity_score=round(float(row["noisy_toxicity_score"]), 6),
            confidence_level=round(float(row["confidence_level"]), 6),
            follower_bucket=int(row["follower_bucket"]),
            step=step,
        )

    def _compute_reward(self, action: str) -> float:
        """
        Shaped reward providing dense signal in [0.0, 1.0].

        Base reward encodes severity of decision errors:
          - False negatives (missing toxic content) are penalised hardest -> 0.0
          - False positives (removing safe content) are penalised -> 0.1
          - Partial actions (warn on toxic) earn partial credit -> 0.5
          - Unnecessary warn on safe content is a mild penalty -> 0.6
          - Correct actions earn full reward -> 1.0

        Calibration bonus (+/-0.1) rewards agents that use confidence appropriately:
          - High confidence (>0.75) + correct decision  -> +0.1
          - High confidence (>0.75) + wrong decision    -> -0.1
          (low confidence decisions have no bonus/penalty -- uncertainty acknowledged)
        """
        true_tox   = int(self._current_post["true_toxicity"])
        confidence = float(self._current_post["confidence_level"])

        # Base rewards
        if true_tox == 1:
            base = {"remove": 1.0, "warn": 0.5, "allow": 0.0}[action]
        else:
            base = {"allow": 1.0, "warn": 0.6, "remove": 0.1}[action]

        # Calibration signal
        is_correct = (
            (true_tox == 1 and action in ("warn", "remove")) or
            (true_tox == 0 and action in ("allow", "warn"))
        )
        calib = 0.0
        if confidence > 0.75:
            calib = 0.1 if is_correct else -0.1

        # ADD IMPACT-AWARE REWARD
        follower = int(self._current_post["follower_bucket"])

        if true_tox == 1 and action == "allow":
            base -= 0.2 * follower

        if true_tox == 1 and action == "remove":
            base += 0.1 * follower
        return round(min(max(base + calib, 0.0), 1.0), 4)
