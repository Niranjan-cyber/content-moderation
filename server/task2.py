"""
Task 2: User Trajectory Control (Sequential RL)
================================================
Agent moderates the SAME USER across their full posting history (9-10 posts).
Actions have DELAYED consequences: leniency causes escalation, harshness causes churn.

Key design:
  - Real user trajectories from dataset (30 users, user_id 1-30, each 9-10 posts)
  - Dynamic escalation_level tracked from agent actions (not read from dataset)
  - Full post_history returned in every observation
  - Rehabilitation mechanic: user cleans up after a firm warning → bonus
  - Over-punishment mechanic: ban at low escalation → heavy penalty

Episode length: up to 10 steps (or until ban)
Score range:    [0.0, 1.0]
"""

import os
import random
import pandas as pd
from typing import Optional, Dict, Any, List

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv"),
)

ACTION_MAP    = {"allow": 0, "warn": 1, "restrict": 2, "remove": 3, "ban": 4}
ACTION_LABELS = {0: "allow", 1: "warn", 2: "restrict", 3: "remove", 4: "ban"}
MAX_STEPS     = 10


class Task2Env:
    """Sequential user-trajectory moderation environment."""

    def __init__(self, df: pd.DataFrame, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        # Only real user trajectories (user_id > 0 and day > 0)
        real = df[(df["user_id"] > 0) & (df["day"] > 0)].copy()
        counts = real["user_id"].value_counts()
        # Keep users with at least 5 posts for meaningful trajectories
        valid_ids = counts[counts >= 5].index.tolist()
        if not valid_ids:
            raise RuntimeError("No valid Task 2 users found — need user_id>0 with ≥5 posts.")
        self._trajectories: Dict[int, List[Dict]] = {}
        for uid in valid_ids:
            posts = (
                real[real["user_id"] == uid]
                .sort_values("day")
                .to_dict("records")
            )
            self._trajectories[uid] = posts
        self._user_ids = valid_ids
        self._reset_state()

    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        uid = self._rng.choice(self._user_ids)
        self._posts       = list(self._trajectories[uid])  # copy so we never mutate df
        self._uid         = uid
        self._step_idx    = 0
        self._done        = False
        self._action_hist: List[str] = []
        self._episode_rewards: List[float] = []
        # Dynamic user state (tracked by agent actions, not dataset values)
        self._dyn_violation_count = 0
        self._dyn_escalation      = 0
        self._dyn_last_action_int = 0
        self._history_for_obs: List[Dict] = []
        return {
            "observation": self._build_obs(),
            "done":        False,
            "info":        {"user_id": uid, "num_posts": len(self._posts)},
        }

    def step(self, action: str) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first.")

        post     = self._posts[self._step_idx]
        true_tox = int(post["true_toxicity"])
        esc      = self._dyn_escalation
        viol     = self._dyn_violation_count
        prev_action = self._action_hist[-1] if self._action_hist else None
        prev_tox    = int(self._posts[self._step_idx - 1]["true_toxicity"]) if self._step_idx > 0 else None

        reward = self._compute_reward(action, true_tox, esc, viol, prev_action, prev_tox)
        self._episode_rewards.append(reward)
        self._action_hist.append(action)

        # --- Update dynamic user state (NEVER mutates dataset) ---------------
        if true_tox == 1:
            self._dyn_violation_count += 1
            self._dyn_escalation = min(3, esc + 1)
        else:
            # User improves if safe post follows agent action of warn/restrict/remove
            if prev_action in ("warn", "restrict", "remove"):
                self._dyn_escalation = max(0, esc - 1)

        self._dyn_last_action_int = ACTION_MAP[action]

        # Record this post in history for future observations
        self._history_for_obs.append({
            "day":                  int(post["day"]),
            "text":                 str(post["text"]),
            "noisy_toxicity_score": round(float(post["noisy_toxicity_score"]), 4),
            "content_type":         str(post["content_type"]),
            "action_taken":         ACTION_MAP[action],
            "action_label":         action,
        })

        self._step_idx += 1

        # Episode ends on ban or when all posts exhausted
        is_ban = action == "ban"
        no_more = self._step_idx >= len(self._posts)
        self._done = is_ban or no_more

        # Terminal bonus: reward sustained good behavior
        if self._done:
            total_steps = self._step_idx
            viol_rate   = self._dyn_violation_count / max(1, total_steps)
            terminal    = max(0.0, 1.0 - viol_rate) * 0.3
            premature_ban = (action == "ban" and self._dyn_escalation < 2)
            if not premature_ban:
                reward = round(min(1.0, reward + terminal), 4)
                self._episode_rewards[-1] = reward

        obs = {} if self._done else self._build_obs()
        return {
            "observation": obs,
            "reward":      reward,
            "done":        self._done,
            "info": {
                "user_id":          self._uid,
                "step":             self._step_idx,
                "true_toxicity":    true_tox,
                "escalation_level": self._dyn_escalation,
                "violation_count":  self._dyn_violation_count,
                "action_taken":     action,
                "score":            reward,
                "mean_episode_reward": round(
                    sum(self._episode_rewards) / max(1, len(self._episode_rewards)), 4
                ),
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "user_id":         getattr(self, "_uid", None),
            "step":            getattr(self, "_step_idx", 0),
            "done":            getattr(self, "_done", True),
            "escalation_level":getattr(self, "_dyn_escalation", 0),
            "violation_count": getattr(self, "_dyn_violation_count", 0),
        }

    # -------------------------------------------------------------------------

    def _reset_state(self):
        self._posts: List[Dict] = []
        self._uid = None
        self._step_idx = 0
        self._done = True
        self._action_hist: List[str] = []
        self._episode_rewards: List[float] = []
        self._dyn_violation_count = 0
        self._dyn_escalation = 0
        self._dyn_last_action_int = 0
        self._history_for_obs: List[Dict] = []

    def _build_obs(self) -> Dict[str, Any]:
        if self._step_idx >= len(self._posts):
            return {}
        post = self._posts[self._step_idx]
        return {
            "post_id":             int(post["id"]),
            "text":                str(post["text"]),
            "noisy_toxicity_score":round(float(post["noisy_toxicity_score"]), 4),
            "confidence_level":    round(float(post["confidence_level"]), 4),
            "follower_bucket":     int(post["follower_bucket"]),
            "content_type":        str(post["content_type"]),
            "user_id":             int(post["user_id"]),
            "day":                 int(post["day"]),
            "violation_history":   self._dyn_violation_count,
            "last_action":         self._dyn_last_action_int,
            "escalation_level":    self._dyn_escalation,
            "post_history":        list(self._history_for_obs),  # full prior history
            "step":                self._step_idx + 1,
        }

    def _compute_reward(
        self,
        action: str,
        true_tox: int,
        esc: int,
        viol: int,
        prev_action: Optional[str],
        prev_tox: Optional[int],
    ) -> float:
        """
        Shaped per-step reward for sequential user moderation.

        Immediate correctness:
          Toxic post + (remove/restrict/ban) → +0.5 (firm action)
          Toxic post + warn                  → +0.3 (soft action, still signal)
          Toxic post + allow                 → -0.6 (false negative — worst)
          Safe post  + allow                 → +0.5 (correct leniency)
          Safe post  + warn                  → -0.1 (unnecessary caution)
          Safe post  + (remove/restrict)     → -0.6 (false positive)

        Sequential bonuses:
          Rehabilitation: prev was warn+toxic AND now safe → +0.5 (user responded)
          Escalation:     prev was allow+toxic AND now toxic again → -0.4 (problem grew)
          Recidivism:     ban/remove last time AND toxic again → +0.3 (justified escalation)

        Escalation-aware ban judgment:
          Ban at esc_level ≥ 2 with ≥2 violations → +0.6 (justified)
          Ban at esc_level < 2                     → -1.0 (over-punishment)

        Normalised to [0, 1] via (raw + 1.5) / 3.5
        """
        reward = 0.0

        # 1. Immediate correctness
        if true_tox == 1:
            if action in ("remove", "restrict", "ban"):
                reward += 0.5
            elif action == "warn":
                reward += 0.3
            else:  # allow
                reward -= 0.6
        else:
            if action == "allow":
                reward += 0.5
            elif action == "warn":
                reward -= 0.1
            else:  # remove / restrict / ban on safe user
                reward -= 0.6

        # 2. Sequential consequences
        if prev_action is not None and prev_tox is not None:
            # Rehabilitation: warned/restricted last time, user is now safe
            if prev_tox == 1 and prev_action in ("warn", "restrict") and true_tox == 0:
                reward += 0.5

            # Escalation failure: let toxic slide, user posts toxic again
            if prev_tox == 1 and prev_action == "allow" and true_tox == 1:
                reward -= 0.4

            # Correct escalation: firm action last time, user reoffends anyway
            if prev_tox == 1 and prev_action in ("remove", "restrict") and true_tox == 1:
                reward += 0.2  # persisting correctly

        # 3. Ban judgment (terminal action deserves careful scoring)
        if action == "ban":
            if esc >= 2 and viol >= 2:
                reward += 0.6   # ban is justified
            else:
                reward -= 1.0   # ban is premature — very bad

        # 4. Warn escalation awareness
        # Warn at escalation_level >= 3 (repeat offender) is not strong enough
        if action == "warn" and esc >= 3 and true_tox == 1:
            reward -= 0.3  # should have escalated

        raw   = reward
        score = (raw + 1.5) / 3.5
        return round(max(0.0, min(1.0, score)), 4)
