"""
Task 1: Single Post Moderation Under Uncertainty (POMDP)
=========================================================
Agent sees ONE post per episode: noisy toxicity signal, confidence, reach, content type.
Must decide: allow / warn / remove.

Ground truth (true_toxicity) is HIDDEN from the agent.
Reward is computed from true_toxicity after the decision.

RL challenge:
  - noisy_toxicity_score is an imperfect signal (not ground truth)
  - confidence_level calibrates how much to trust it
  - content_type encodes severity class (hate_speech/threat >> general)
  - follower_bucket amplifies impact of wrong calls on high-reach posts

Episode length: 1 step
Score range:    [0.0, 1.0]
"""

import os
import random
import pandas as pd
from typing import Optional, Dict, Any

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv"),
)

# Content severity tiers (affect FN penalty magnitude)
HIGH_SEVERITY   = {"hate_speech", "threat", "sexual"}
MEDIUM_SEVERITY = {"political", "personal"}
LOW_SEVERITY    = {"humor", "news_info", "general"}

# Per-bucket impact multiplier (high reach = higher stakes)
REACH_MULT = {0: 1.0, 1: 1.2, 2: 1.5}

# Per-severity FN multiplier (missing harmful content is worse for severe types)
SEVERITY_MULT = {
    **{k: 1.5 for k in HIGH_SEVERITY},
    **{k: 1.0 for k in MEDIUM_SEVERITY},
    **{k: 0.8 for k in LOW_SEVERITY},
}

ACTION_LABELS = {0: "allow", 1: "warn", 2: "remove"}
ACTION_CODES  = {"allow": 0, "warn": 1, "remove": 2}


class Task1Env:
    """Single-post POMDP moderation environment."""

    def __init__(self, df: pd.DataFrame, seed: Optional[int] = None):
        self.df   = df
        self._rng = random.Random(seed)
        self._row: Optional[pd.Series] = None
        self._done       = True
        self._episode_id = 0
        self._rewards: list = []

    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        idx        = self._rng.randint(0, len(self.df) - 1)
        self._row  = self.df.iloc[idx]
        self._done = False
        self._episode_id += 1
        return {
            "observation": self._build_obs(step=0),
            "done":        False,
            "info":        {"episode_id": self._episode_id, "task": "task1-single-post"},
        }

    def step(self, action: str) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first.")

        reward     = self._compute_reward(action)
        self._done = True
        self._rewards.append(reward)

        row      = self._row
        true_tox = int(row["true_toxicity"])

        return {
            "observation": self._build_obs(step=1),
            "reward":      reward,
            "done":        True,
            "info": {
                "true_toxicity":  true_tox,
                "action_taken":   action,
                "false_positive": true_tox == 0 and action == "remove",
                "false_negative": true_tox == 1 and action == "allow",
                "group":          str(row["group"]),
                "content_type":   str(row["content_type"]),
                "follower_bucket":int(row["follower_bucket"]),
                "score":          reward,
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "episode_id":  self._episode_id,
            "done":        self._done,
            "mean_reward": round(sum(self._rewards) / max(1, len(self._rewards)), 4),
        }

    # -------------------------------------------------------------------------

    def _build_obs(self, step: int) -> Dict[str, Any]:
        row = self._row
        return {
            "post_id":             int(row["id"]),
            "text":                str(row["text"]),
            "noisy_toxicity_score":round(float(row["noisy_toxicity_score"]), 4),
            "confidence_level":    round(float(row["confidence_level"]), 4),
            "follower_bucket":     int(row["follower_bucket"]),
            "content_type":        str(row["content_type"]),
            "step":                step,
        }

    def _compute_reward(self, action: str) -> float:
        """
        Reward grounded in true_toxicity (hidden from agent, used only for grading).

        Base rewards:
          Toxic post  (true_tox=1): remove=+1.0 | warn=+0.4 | allow=-1.0
          Safe post   (true_tox=0): allow=+1.0  | warn=+0.4 | remove=-0.8

        Modifiers:
          Severity multiplier — high-severity toxic FNs penalised harder (×1.5)
          Reach multiplier    — large accounts have higher stakes (×1.0–1.5)
          Calibration bonus   — correct at high confidence: +0.1; wrong at high conf: -0.1
          Anti-hedging        — "warn" on a clear-signal post gives less than decisive action

        Normalised to [0, 1]:
          raw range ≈ [-2.25, 1.65] → (raw + 2.25) / 4.0
        """
        row        = self._row
        true_tox   = int(row["true_toxicity"])
        confidence = float(row["confidence_level"])
        follower   = int(row["follower_bucket"])
        ctype      = str(row["content_type"])

        # Base reward
        if true_tox == 1:
            base = {"remove": 1.0, "warn": 0.4, "allow": -1.0}[action]
            # Severity amplification on FN (allowed toxic content)
            if action == "allow":
                sev = SEVERITY_MULT.get(ctype, 1.0)
                base *= sev
        else:
            base = {"allow": 1.0, "warn": 0.4, "remove": -0.8}[action]

        # Reach multiplier (high reach = higher stakes for errors)
        reach = REACH_MULT.get(follower, 1.0)
        base *= reach

        # Calibration bonus/penalty (rewards agents that use confidence correctly)
        is_correct = (
            (true_tox == 1 and action in ("warn", "remove")) or
            (true_tox == 0 and action in ("allow", "warn"))
        )
        calib = 0.0
        if confidence > 0.8:
            calib = 0.1 if is_correct else -0.1

        # Anti-hedging: if signal is unambiguous, "warn" is evasive
        # Unambiguous = high confidence AND (very high OR very low noisy score)
        noisy = float(row["noisy_toxicity_score"])
        is_unambiguous = confidence > 0.85 and (noisy > 0.6 or noisy < 0.15)
        hedge_penalty = -0.15 if (action == "warn" and is_unambiguous) else 0.0

        raw   = base + calib + hedge_penalty
        score = (raw + 2.25) / 4.0
        return round(max(0.0, min(1.0, score)), 4)
