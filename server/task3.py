"""
Task 3: Platform Policy Optimisation (Multi-objective RL)
==========================================================
Agent acts as Head of Trust & Safety, controlling global moderation strictness
over a 20-day simulation. Each day a real batch of posts is processed from
the dataset. Metrics are computed from actual binary labels (true_toxicity,
group A/B fairness).

Must balance four competing objectives:
  safety     (0.35) — don't let toxic content through
  engagement (0.30) — don't over-remove safe content
  retention  (0.20) — high FPR chases users away
  fairness   (0.15) — equal treatment across demographic groups A and B

Key mechanics:
  - Strictness controls auto-mod threshold: threshold = 1.0 - strictness
  - High strictness → removes more → safer but worse engagement/fairness
  - Policy inertia: keep_policy_same penalised slightly to prevent lazy no-ops
  - Episode seed prevents same batch sequence across runs

Episode length: 20 days (steps)
Score range:    [0.0, 1.0]
"""

import os
import random
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv"),
)

BATCH_SIZE = 30   # posts per day
MAX_DAYS   = 20


class Task3Env:
    """Platform-level multi-objective policy optimisation environment."""

    def __init__(self, df: pd.DataFrame, seed: Optional[int] = None):
        self._base_seed = seed if seed is not None else 0
        self._df        = df.copy()
        # Normalise group labels
        self._df["group"] = (
            self._df["group"].astype(str).str.upper().str.strip()
        )
        self._df.loc[~self._df["group"].isin(["A", "B"]), "group"] = "A"
        self._reset_state()

    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        self._episode_seed  = random.randint(0, 999_999)
        self._rng           = random.Random(self._episode_seed)
        self._day           = 1
        self._strictness    = 0.5
        self._done          = False
        self._rewards:  List[float] = []
        self._no_op_streak  = 0
        self._current_batch = self._sample_batch(day=0)
        metrics = self._compute_metrics(self._current_batch, self._strictness)
        obs = self._build_obs(metrics)
        return {
            "observation": obs,
            "done":        False,
            "info":        {
                "message":    "Platform reset to Day 1",
                "strictness": self._strictness,
                "episode_seed": self._episode_seed,
            },
        }

    def step(self, action: str) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first.")

        # 1. Apply policy adjustment
        delta = {"increase_strictness": +0.1, "decrease_strictness": -0.1, "keep_policy_same": 0.0}[action]
        self._strictness = round(max(0.0, min(1.0, self._strictness + delta)), 2)

        # 2. Track inaction streak (anti-hedging)
        if action == "keep_policy_same":
            self._no_op_streak += 1
        else:
            self._no_op_streak = 0

        # 3. Simulate the day
        batch   = self._sample_batch(day=self._day)
        metrics = self._compute_metrics(batch, self._strictness)
        self._current_batch = batch

        # 4. Compute reward
        reward = self._compute_reward(metrics, action)
        self._rewards.append(reward)

        self._day += 1
        self._done = self._day > MAX_DAYS

        obs = {} if self._done else self._build_obs(metrics)
        return {
            "observation": obs,
            "reward":      round(reward, 4),
            "done":        self._done,
            "info": {
                "day":              int(self._day - 1),
                "strictness":       self._strictness,
                "global_toxicity_rate": metrics["toxicity_rate"],
                "engagement_score":     metrics["engagement"],
                "user_retention_rate":  metrics["retention"],
                "fairness_gap":         metrics["fairness_gap"],
                "group_a_fpr":          metrics["a_fpr"],
                "group_b_fpr":          metrics["b_fpr"],
                "mean_episode_reward":  round(
                    sum(self._rewards) / max(1, len(self._rewards)), 4
                ),
                "score": round(reward, 4),
            },
        }

    def state(self) -> Dict[str, Any]:
        m = self._last_metrics
        return {
            "day":        self._day,
            "strictness": self._strictness,
            "done":       self._done,
            "mean_reward": round(
                sum(self._rewards) / max(1, len(self._rewards)), 4
            ) if self._rewards else 0.0,
            "toxicity_rate": round(m.get("toxicity_rate", 0), 4),
            "engagement":    round(m.get("engagement", 0), 4),
            "retention":     round(m.get("retention", 0), 4),
            "fairness_gap":  round(m.get("fairness_gap", 0), 4),
        }

    # -------------------------------------------------------------------------

    def _reset_state(self):
        self._episode_seed   = 0
        self._rng            = random.Random(0)
        self._day            = 1
        self._strictness     = 0.5
        self._done           = True
        self._rewards:  List[float] = []
        self._no_op_streak   = 0
        self._current_batch  = pd.DataFrame()
        self._last_metrics: Dict = {}

    def _sample_batch(self, day: int) -> pd.DataFrame:
        """
        Sample a batch of BATCH_SIZE posts for the current day.
        Uses episode seed + day offset so different episodes see different data,
        but within an episode each day is deterministic (reproducible).
        """
        rng_state = self._episode_seed + day * 31
        return self._df.sample(
            min(BATCH_SIZE, len(self._df)),
            random_state=rng_state,
            replace=False,
        ).reset_index(drop=True)

    def _compute_metrics(self, batch: pd.DataFrame, strictness: float) -> Dict[str, Any]:
        """
        Simulate one day of auto-moderation with current strictness threshold.
        All metrics derived from real dataset labels (true_toxicity, group).

        Auto-mod decision: remove post if noisy_toxicity_score ≥ threshold
        where threshold = 1.0 - strictness
        """
        threshold = round(1.0 - strictness, 2)

        # Add small realistic noise to scores (model is not perfectly calibrated)
        np.random.seed(self._episode_seed + self._day)
        noisy_scores = np.clip(
            batch["noisy_toxicity_score"].values + np.random.normal(0, 0.05, len(batch)),
            0.0, 1.0
        )
        removed = noisy_scores >= threshold

        true_tox    = batch["true_toxicity"].values
        safe        = ~true_tox.astype(bool)

        n_toxic  = true_tox.sum()
        n_safe   = safe.sum()

        # False negatives: toxic posts that were allowed through
        fn = (true_tox.astype(bool) & ~removed).sum()
        # False positives: safe posts incorrectly removed
        fp = (safe & removed).sum()
        # True negatives: safe posts correctly allowed
        tn = (safe & ~removed).sum()

        toxicity_rate = fn / max(1, n_toxic)   # lower = better (safety)
        engagement    = tn / max(1, n_safe)    # higher = better
        fpr_overall   = fp / max(1, n_safe)
        retention     = max(0.0, 1.0 - fpr_overall * 1.5)  # penalise over-removal

        # Fairness: differential false-positive rate between groups A and B
        group_a = batch["group"] == "A"
        group_b = batch["group"] == "B"

        def group_fpr(mask: np.ndarray) -> float:
            safe_in_group = safe & mask
            n = safe_in_group.sum()
            if n == 0:
                return 0.0
            return float((safe_in_group & removed).sum() / n)

        a_fpr = group_fpr(group_a.values)
        b_fpr = group_fpr(group_b.values)
        fairness_gap = abs(a_fpr - b_fpr)

        # Platform user behaviour summary (from full dataset, not just batch)
        full_users = self._df[self._df["user_id"] > 0]
        esc_col    = full_users["escalation_level"] if "escalation_level" in full_users.columns else pd.Series([0])

        self._last_metrics = {
            "toxicity_rate": round(float(toxicity_rate), 4),
            "engagement":    round(float(min(1.0, max(0.0, engagement))), 4),
            "retention":     round(float(min(1.0, max(0.0, retention))), 4),
            "fairness_gap":  round(float(fairness_gap), 4),
            "a_fpr":         round(float(a_fpr), 4),
            "b_fpr":         round(float(b_fpr), 4),
            "safety":        round(float(1.0 - toxicity_rate), 4),
            "fairness_score":round(float(max(0.0, 1.0 - 2.5 * fairness_gap)), 4),
            "_batch":        batch,
            "_platform_summary": {
                "total_active_users": int(self._df["user_id"].nunique()),
                "escalating_users":   int((esc_col >= 2).sum()),
                "improving_users":    int((esc_col == 0).sum()),
                "banned_today":       0,
            },
        }
        return self._last_metrics

    def _compute_reward(self, metrics: Dict[str, Any], action: str) -> float:
        """
        Multi-objective reward. Weights match openenv.yaml spec:
          safety (0.35) + engagement (0.30) + retention (0.20) + fairness (0.15)

        Penalty modifiers:
          - keep_policy_same streak ≥ 3 days: -0.03 (stop being passive)
          - keep_policy_same single step:      -0.01 (small time cost)
          - fairness_gap > 0.3:               additional -0.05
        """
        base = (
            0.35 * metrics["safety"] +
            0.30 * metrics["engagement"] +
            0.20 * metrics["retention"] +
            0.15 * metrics["fairness_score"]
        )

        # Policy inertia penalty
        if action == "keep_policy_same":
            inertia_pen = 0.03 if self._no_op_streak >= 3 else 0.01
            base -= inertia_pen

        # Extra fairness penalty when gap is large
        if metrics["fairness_gap"] > 0.3:
            base -= 0.05

        return float(max(0.0, min(1.0, base)))

    def _build_obs(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Build full platform observation including top-5 urgent posts."""
        batch = metrics.get("_batch", self._current_batch)
        if len(batch) == 0:
            batch = self._current_batch

        # Top 5 most urgent posts: sorted by toxicity × reach
        urgency = batch["noisy_toxicity_score"] * (batch["follower_bucket"] + 1)
        top5    = batch.assign(_urgency=urgency).nlargest(5, "_urgency")

        active_posts = [
            {
                "post_id":             int(r["id"]),
                "text":                str(r["text"])[:150],
                "noisy_toxicity_score":round(float(r["noisy_toxicity_score"]), 4),
                "content_type":        str(r["content_type"]),
                "follower_bucket":     int(r["follower_bucket"]),
                "group":               str(r["group"]),
            }
            for _, r in top5.iterrows()
        ]

        sample = top5.iloc[0] if len(top5) > 0 else batch.iloc[0]

        return {
            "current_day":                 self._day,
            "global_toxicity_rate":        metrics["toxicity_rate"],
            "engagement_score":            metrics["engagement"],
            "user_retention_rate":         metrics["retention"],
            "moderation_strictness_level": round(self._strictness, 4),
            "fairness_gap":                metrics["fairness_gap"],
            "group_a_false_positive_rate": metrics["a_fpr"],
            "group_b_false_positive_rate": metrics["b_fpr"],
            "sample_post_text":            str(sample["text"])[:150],
            "sample_toxicity_score":       round(float(sample["noisy_toxicity_score"]), 4),
            "sample_content_type":         str(sample["content_type"]),
            "active_posts":                active_posts,
            "platform_user_summary":       metrics["_platform_summary"],
            "step":                        self._day,
        }
