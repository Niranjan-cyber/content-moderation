"""
Task 3: Platform Policy Optimization (System-Level RL)
======================================================
The agent acts as Head of Trust & Safety, controlling the platform's global
moderation STRICTNESS THRESHOLD over a 20-day simulation.

Key difference from toy simulations:
  - Every day a real batch of posts is sampled from the dataset.
  - The auto-mod system applies the current threshold: posts whose
    noisy_toxicity_score exceeds the threshold are removed.
  - Real group A / B labels from the dataset drive fairness tracking.
  - The reward function balances four competing objectives:
      safety (low toxicity), engagement, retention, and fairness.

Episode length: 20 days (steps)
Score range:    [0.0, 1.0]

Reward formula per step:
  reward = 0.30 * safety
         + 0.25 * engagement
         + 0.25 * retention
         + 0.20 * fairness
  where each component is in [0, 1].
"""

import os
import random
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List

from .models import Task3Observation, Task3Action, Task3StepResult, Task3ResetResult

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv")
)
# How many posts to sample from the dataset each day
BATCH_SIZE = 50
MAX_DAYS   = 20


class Task3Env:
    """
    Platform-level moderation environment backed by the real dataset.

    State: aggregated platform health metrics computed from a daily batch.
    Action: increase_strictness | decrease_strictness | keep_policy_same

    Strictness (0.0–1.0) controls the auto-mod threshold:
        threshold = 1.0 − strictness
    A post is auto-removed if noisy_toxicity_score > threshold.

    High strictness → removes more content (good for safety, bad for engagement/fairness).
    Low strictness  → allows more content (good for engagement, bad for safety).
    """

    def __init__(self, data_path: str = DATA_PATH, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        np.random.seed(seed if seed is not None else 0)

        raw = pd.read_csv(data_path)

        # Keep only what we need
        self._df = raw[[
            "id", "text", "noisy_toxicity_score", "confidence_level",
            "true_toxicity", "content_type", "group", "is_adversarial",
        ]].copy()
        self._df["group"] = self._df["group"].astype(str).str.upper().str.strip()

        # Sanity: ensure group column only has A / B
        valid_groups = self._df["group"].isin(["A", "B"])
        if not valid_groups.all():
            self._df.loc[~valid_groups, "group"] = "A"

        self._all_indices = list(range(len(self._df)))
        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # OpenEnv interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> Task3ResetResult:
        """Start a new 20-day episode with a clean platform state."""
        self.current_day      = 1
        self.is_done          = False
        self.strictness       = 0.5     # Start at neutral
        self._episode_rewards: List[float] = []

        # Cumulative fairness tracking across the episode
        self._group_a_fp  = 0           # group A false positives (wrongly removed safe posts)
        self._group_a_tot = 0           # group A safe posts seen
        self._group_b_fp  = 0
        self._group_b_tot = 0

        # Seed initial platform state (day-0 snapshot before any action)
        self._last_metrics = self._compute_metrics_for_batch(
            self._sample_batch(), self.strictness
        )

        return Task3ResetResult(
            observation=self._build_obs(self._last_metrics),
            done=False,
            info={"message": "Platform reset to Day 1", "strictness": self.strictness},
        )

    def step(self, action_model: Task3Action) -> Task3StepResult:
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset().")

        # ── 1. Apply agent's policy adjustment ───────────────────────────────
        action = action_model.action
        if action == "increase_strictness":
            self.strictness = min(1.0, round(self.strictness + 0.1, 2))
        elif action == "decrease_strictness":
            self.strictness = max(0.0, round(self.strictness - 0.1, 2))
        # keep_policy_same → no change

        # ── 2. Simulate the day: sample a real batch and apply threshold ──────
        batch    = self._sample_batch()
        metrics  = self._compute_metrics_for_batch(batch, self.strictness)
        self._last_metrics = metrics

        # Update cumulative fairness counters
        self._group_a_fp  += metrics["_a_fp"]
        self._group_a_tot += metrics["_a_safe_total"]
        self._group_b_fp  += metrics["_b_fp"]
        self._group_b_tot += metrics["_b_safe_total"]

        # ── 3. Multi-objective reward ─────────────────────────────────────────
        reward = self._compute_reward(metrics)
        self._episode_rewards.append(reward)

        # ── 4. Episode boundary ───────────────────────────────────────────────
        self.current_day += 1
        if self.current_day > MAX_DAYS:
            self.is_done = True

        # Pick a sample post to show in observation (grounds the agent)
        sample_idx = self._rng.randint(0, len(batch) - 1)
        sample_row = batch.iloc[sample_idx]

        obs = Task3Observation(
            current_day=self.current_day,
            global_toxicity_rate=round(metrics["toxicity_rate"],       4),
            engagement_score     =round(metrics["engagement"],         4),
            user_retention_rate  =round(metrics["retention"],          4),
            moderation_strictness_level=round(self.strictness,         4),
            fairness_gap         =round(metrics["fairness_gap"],       4),
            group_a_false_positive_rate=round(metrics["a_fpr"],        4),
            group_b_false_positive_rate=round(metrics["b_fpr"],        4),
            sample_post_text     =str(sample_row["text"])[:200],
            sample_toxicity_score=round(float(sample_row["noisy_toxicity_score"]), 4),
            sample_content_type  =str(sample_row["content_type"]),
            step=self.current_day - 1,
        )

        return Task3StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self.is_done,
            info={
                "action":          action,
                "strictness":      self.strictness,
                "toxicity_rate":   round(metrics["toxicity_rate"],   4),
                "engagement":      round(metrics["engagement"],      4),
                "retention":       round(metrics["retention"],       4),
                "fairness_gap":    round(metrics["fairness_gap"],    4),
                "threshold_used":  round(1.0 - self.strictness,     4),
            },
        )

    def state(self) -> Dict[str, Any]:
        m = self._last_metrics
        return {
            "day":          self.current_day,
            "strictness":   self.strictness,
            "done":         self.is_done,
            "mean_reward":  round(
                sum(self._episode_rewards) / len(self._episode_rewards), 4
            ) if self._episode_rewards else 0.0,
            "toxicity_rate":   round(m.get("toxicity_rate", 0), 4),
            "engagement":      round(m.get("engagement", 0), 4),
            "retention":       round(m.get("retention", 0), 4),
            "fairness_gap":    round(m.get("fairness_gap", 0), 4),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_batch(self) -> pd.DataFrame:
        """Sample BATCH_SIZE posts uniformly from the dataset."""
        idx = self._rng.choices(self._all_indices, k=BATCH_SIZE)
        return self._df.iloc[idx].reset_index(drop=True)

    def _compute_metrics_for_batch(
        self, batch: pd.DataFrame, strictness: float
    ) -> Dict[str, Any]:
        """
        Simulate a day's auto-moderation and compute platform health metrics.

        Decision rule: auto-remove post if noisy_toxicity_score > threshold
        where threshold = 1.0 − strictness.

        Returns a dict with all metric components plus internal counters
        (prefixed with _) for cumulative fairness tracking.
        """
        threshold = 1.0 - strictness

        # Apply decision
        _noise = np.random.normal(0, 0.18, size=len(batch))
        _effective_scores = np.clip(batch["noisy_toxicity_score"].values + _noise, 0.0, 1.0)
        decisions = (_effective_scores > threshold).astype(int)

        # 0 = allowed,  1 = removed

        true_tox  = batch["true_toxicity"].values

        # ── Safety ───────────────────────────────────────────────────────────
        # Fraction of truly toxic posts that slipped through (false negatives)
        n_toxic = true_tox.sum()
        fn = ((true_tox == 1) & (decisions == 0)).sum()   # toxic + allowed = false neg
        miss_rate     = fn / n_toxic if n_toxic > 0 else 0.0
        safety_score  = 1.0 - miss_rate                   # higher = better

        # ── Engagement ───────────────────────────────────────────────────────
        # Fraction of SAFE posts that were allowed (false positives hurt engagement)
        n_safe  = (true_tox == 0).sum()
        safe_allowed = ((true_tox == 0) & (decisions == 0)).sum()
        engagement = safe_allowed / n_safe if n_safe > 0 else 1.0
        # Penalise slightly for excessive strictness even on correct removals
        fp_rate    = 1.0 - engagement
        engagement = max(0.0, engagement - 0.1 * fp_rate)  # light engagement drag

        # ── Retention ────────────────────────────────────────────────────────
        # High false-positive rate erodes retention; harshness is quadratic
        retention = max(0.0, 1.0 - (fp_rate ** 1.5))

        # ── Fairness — differential false-positive rate across groups ─────────
        group_a = batch["group"] == "A"
        group_b = batch["group"] == "B"

        a_safe = ((true_tox == 0) & group_a.values).sum()
        b_safe = ((true_tox == 0) & group_b.values).sum()
        a_fp   = ((true_tox == 0) & group_a.values & (decisions == 1)).sum()
        b_fp   = ((true_tox == 0) & group_b.values & (decisions == 1)).sum()

        a_fpr  = a_fp / a_safe if a_safe > 0 else 0.0
        b_fpr  = b_fp / b_safe if b_safe > 0 else 0.0

        fairness_gap   = abs(a_fpr - b_fpr)
        fairness_score = max(0.0, 1.0 - 2.5 * fairness_gap)  # gap > 0.4 → score 0

        return {
            "toxicity_rate": float(1.0 - safety_score),   # % toxic that got through
            "safety":        float(safety_score),
            "engagement":    float(min(1.0, max(0.0, engagement))),
            "retention":     float(min(1.0, max(0.0, retention))),
            "fairness_gap":  float(fairness_gap),
            "fairness":      float(max(0.0, fairness_score)),
            "a_fpr":         float(a_fpr),
            "b_fpr":         float(b_fpr),
            # Internal counters for cumulative tracking
            "_a_fp":         int(a_fp),
            "_a_safe_total": int(a_safe),
            "_b_fp":         int(b_fp),
            "_b_safe_total": int(b_safe),
        }

    def _compute_reward(self, metrics: Dict[str, Any]) -> float:
        """
        Multi-objective reward per step.  Weights reflect the spec:
          safety     30%  — toxicity must be controlled
          engagement 25%  — platform must be usable
          retention  25%  — users must not churn
          fairness   20%  — equal treatment across demographic groups
        """
        r = (
            0.30 * metrics["safety"]
          + 0.25 * metrics["engagement"]
          + 0.25 * metrics["retention"]
          + 0.20 * metrics["fairness"]
        )
        return float(min(1.0, max(0.0, r)))

    def _build_obs(self, metrics: Dict[str, Any]) -> Task3Observation:
        """Build initial observation (before any post is sampled for display)."""
        sample_row = self._df.iloc[self._rng.randint(0, len(self._df) - 1)]
        return Task3Observation(
            current_day=1,
            global_toxicity_rate       =round(metrics.get("toxicity_rate", 0.2), 4),
            engagement_score           =round(metrics.get("engagement",    0.8), 4),
            user_retention_rate        =round(metrics.get("retention",     0.9), 4),
            moderation_strictness_level=round(self.strictness,             4),
            fairness_gap               =round(metrics.get("fairness_gap",  0.0), 4),
            group_a_false_positive_rate=round(metrics.get("a_fpr",         0.0), 4),
            group_b_false_positive_rate=round(metrics.get("b_fpr",         0.0), 4),
            sample_post_text           =str(sample_row["text"])[:200],
            sample_toxicity_score      =round(float(sample_row["noisy_toxicity_score"]), 4),
            sample_content_type        =str(sample_row["content_type"]),
            step=1,
        )