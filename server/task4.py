"""
Task 4: Appeals & Consistency (Meta-Decision RL)
=================================================
Agent reviews contested moderation decisions. Must decide: uphold / reverse / escalate.

Key mechanics:
  - 10 appeals per episode, BALANCED 5 should-reverse + 5 should-uphold
  - Precedent history: agent sees all past decisions — consistency is scored
  - Wrong decisions on similar earlier cases penalise future similar decisions
  - Episode score = 0.7 × accuracy + 0.3 × consistency

Dataset insight:
  - should_reverse=1: post was ALLOWED (original_action=0) but was actually toxic
    → correct appeal outcome = reverse (restore decision to remove)
  - should_reverse=0: original decision was correct
    → correct appeal outcome = uphold

Episode length: 10 appeals
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

MAX_APPEALS          = 10
APPEALS_TO_REVERSE   = 5   # exactly 5 should be reversed, 5 upheld per episode


class Task4Env:
    """Appeals review and consistency environment."""

    def __init__(self, df: pd.DataFrame, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        # Separate pools
        self._reverse_pool = df[df["should_reverse"] == 1].reset_index(drop=True)
        self._uphold_pool  = df[df["should_reverse"] == 0].reset_index(drop=True)
        if len(self._reverse_pool) < APPEALS_TO_REVERSE:
            raise RuntimeError("Not enough should_reverse=1 rows for Task 4.")
        if len(self._uphold_pool) < (MAX_APPEALS - APPEALS_TO_REVERSE):
            raise RuntimeError("Not enough should_reverse=0 rows for Task 4.")
        self._reset_state()

    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        # Build a balanced queue: 5 reverse + 5 uphold, shuffled
        rev_idx = self._rng.sample(range(len(self._reverse_pool)), APPEALS_TO_REVERSE)
        uph_idx = self._rng.sample(
            range(len(self._uphold_pool)), MAX_APPEALS - APPEALS_TO_REVERSE
        )
        appeals = (
            [self._reverse_pool.iloc[i].to_dict() for i in rev_idx] +
            [self._uphold_pool.iloc[i].to_dict()  for i in uph_idx]
        )
        self._rng.shuffle(appeals)

        self._queue:     List[Dict] = appeals
        self._step:      int = 0
        self._done:      bool = False
        self._precedent: List[Dict] = []
        self._correct_count: int = 0
        self._step_rewards: List[float] = []

        return {
            "observation": self._build_obs(),
            "done":        False,
            "info":        {"total_appeals": MAX_APPEALS},
        }

    def step(self, action: str) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode done — call reset() first.")

        appeal       = self._queue[self._step]
        should_rev   = int(appeal["should_reverse"])
        orig_action  = int(appeal["original_action_taken"])
        true_tox     = int(appeal["true_toxicity"])
        content_type = str(appeal["content_type"])

        correct_action = "reverse" if should_rev == 1 else "uphold"
        was_correct    = action == correct_action

        reward = self._compute_reward(
            action, correct_action, was_correct, content_type, orig_action
        )
        self._step_rewards.append(reward)
        if was_correct:
            self._correct_count += 1

        # Record in precedent history (agent can see this in future steps)
        self._precedent.append({
            "appeal_number":   self._step + 1,
            "text":            str(appeal["text"])[:100],
            "content_type":    content_type,
            "original_action": orig_action,
            "agent_decision":  action,
            "was_correct":     was_correct,
            "should_reverse":  bool(should_rev),
        })

        self._step += 1
        self._done = self._step >= MAX_APPEALS

        # Episode-end scoring
        step_score = reward
        if self._done:
            accuracy    = self._correct_count / MAX_APPEALS
            consistency = self._compute_consistency_score()
            final_score = round(0.7 * accuracy + 0.3 * consistency, 4)
            # Override last step reward with episode-level score for grader
            step_score  = final_score

        obs = {} if self._done else self._build_obs()
        return {
            "observation": obs,
            "reward":      round(step_score, 4),
            "done":        self._done,
            "info": {
                "should_reverse":  should_rev,
                "correct_action":  correct_action,
                "action_taken":    action,
                "was_correct":     was_correct,
                "appeals_done":    self._step,
                "correct_count":   self._correct_count,
                "score":           round(step_score, 4),
            },
        }

    def state(self) -> Dict[str, Any]:
        return {
            "step":          self._step,
            "done":          self._done,
            "correct_count": self._correct_count,
            "appeals_total": MAX_APPEALS,
        }

    # -------------------------------------------------------------------------

    def _reset_state(self):
        self._queue: List[Dict] = []
        self._step = 0
        self._done = True
        self._precedent: List[Dict] = []
        self._correct_count = 0
        self._step_rewards: List[float] = []

    def _build_obs(self) -> Dict[str, Any]:
        appeal     = self._queue[self._step]
        orig_int   = int(appeal["original_action_taken"])
        orig_label = {0: "allow", 1: "warn", 2: "remove"}.get(orig_int, "unknown")
        return {
            "appeal_id":            int(appeal.get("appeal_id", self._step + 1)),
            "post_id":              int(appeal["id"]),
            "text":                 str(appeal["text"]),
            "noisy_toxicity_score": round(float(appeal["noisy_toxicity_score"]), 4),
            "confidence_level":     round(float(appeal["confidence_level"]), 4),
            "content_type":         str(appeal["content_type"]),
            "original_action_taken":orig_int,
            "original_action_label":orig_label,
            "follower_bucket":      int(appeal["follower_bucket"]),
            "precedent_history":    list(self._precedent),
            "step":                 self._step,
        }

    def _compute_reward(
        self,
        action: str,
        correct_action: str,
        was_correct: bool,
        content_type: str,
        orig_action: int,
    ) -> float:
        """
        Per-appeal step reward (grader uses episode-end final score, but step
        reward still provides training signal during the episode).

          Correct decision (uphold or reverse): +1.0
          Escalate:                              +0.3 (safe but indecisive)
          Wrong decision:                        -1.0

          Consistency penalty: -0.5 if you decided differently on a similar
          prior case (same content_type, non-escalate decision).

          Normalised to [0,1] via (raw + 1.0) / 2.3
        """
        if action == correct_action:
            raw = 1.0
        elif action == "escalate":
            raw = 0.3
        else:
            raw = -1.0

        # Consistency penalty
        similar = [
            p for p in self._precedent
            if p["content_type"] == content_type and p["agent_decision"] != "escalate"
        ]
        if similar and action != "escalate":
            prior_decisions = {p["agent_decision"] for p in similar}
            if action not in prior_decisions:
                raw -= 0.5

        score = (raw + 1.0) / 2.3
        return round(max(0.0, min(1.0, score)), 4)

    def _compute_consistency_score(self) -> float:
        """
        Episode-level consistency: across all content_type groups, how often
        did the agent make the same decision for similar cases?

        Returns 0.0–1.0 (1.0 = perfectly consistent).
        """
        from collections import defaultdict
        groups: Dict[str, List[str]] = defaultdict(list)
        for p in self._precedent:
            if p["agent_decision"] != "escalate":
                groups[p["content_type"]].append(p["agent_decision"])

        if not groups:
            return 1.0

        consistency_scores = []
        for ctype, decisions in groups.items():
            if len(decisions) < 2:
                continue
            # Fraction of decisions that match the modal decision
            from collections import Counter
            modal_count = Counter(decisions).most_common(1)[0][1]
            consistency_scores.append(modal_count / len(decisions))

        return round(sum(consistency_scores) / max(1, len(consistency_scores)), 4)
