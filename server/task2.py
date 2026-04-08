"""
Task 2: User Trajectory Control (True RL)
=========================================
Agent moderates the SAME user over time.
Delayed rewards: If you are too lenient, the user escalates.
If you are too strict, you lose the user.
"""
import os
import random
import pandas as pd
from typing import Optional, Dict, Any

from ..models import Task2Observation, Task2Action, Task2StepResult, Task2ResetResult

DATA_PATH = os.getenv(
    "DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_final.csv")
)
class Task2Env:
    def __init__(self, data_path: str = DATA_PATH, seed: Optional[int] = None):
        self.df = pd.read_csv(data_path)
        self._rng = random.Random(seed)
        
        # Group by user to get trajectories
        user_counts = self.df['user_id'].value_counts()
        # Only keep users with > 2 posts for sequential learning
        valid_users = user_counts[user_counts > 2].index.tolist()
        self.trajectories = self.df[self.df['user_id'].isin(valid_users)].sort_values(['user_id', 'day'])
        self.user_ids = valid_users
        
        self.reset()

    def reset(self) -> Task2ResetResult:
        # Pick a random user to moderate
        self.current_user = self._rng.choice(self.user_ids)
        self.user_posts = self.trajectories[self.trajectories['user_id'] == self.current_user].to_dict('records')
        
        self.current_step = 0
        self.max_steps = min(len(self.user_posts), 10)
        self.is_done = False
        self.agent_action_history = []
        self._dynamic_last_action = 0       # ADD THIS
        self._dynamic_violation_count = 0
        
        return Task2ResetResult(
            observation=self._get_obs(),
            done=self.is_done,
            info={"message": f"Starting trajectory for user {self.current_user}"}
        )

    def _get_obs(self) -> Task2Observation:
        post = self.user_posts[self.current_step]
        return Task2Observation(
            post_id=post["id"],
            text=post["text"],
            noisy_toxicity_score=post["noisy_toxicity_score"],
            confidence_level=post["confidence_level"],
            follower_bucket=post["follower_bucket"],
            user_id=post["user_id"],
            day=post["day"],
            violation_history=self._dynamic_violation_count,
            last_action=self._dynamic_last_action,
            escalation_level=post["escalation_level"],
            step=self.current_step + 1
        )

    def step(self, action_model: Task2Action) -> Task2StepResult:
        if self.is_done:
            raise RuntimeError("Episode is done. Call reset().")

        action = action_model.action
        self.agent_action_history.append(action)
        
        post = self.user_posts[self.current_step]
        # [fire] ADD DYNAMICS (CRITICAL)
        if self.current_step < len(self.user_posts) - 1:
            next_post = self.user_posts[self.current_step + 1]

            if action == "warn":
                next_post["true_toxicity"] = max(0, next_post["true_toxicity"] - 0.2)

            elif action == "allow":
                next_post["true_toxicity"] = min(1, next_post["true_toxicity"] + 0.2)

            elif action == "restrict":
                next_post["true_toxicity"] = max(0, next_post["true_toxicity"] - 0.1)
        true_tox = int(post["true_toxicity"])
        
        # Track state dynamically so future observations reflect agent's actual actions
        ACTION_CODE = {"allow": 0, "warn": 1, "restrict": 2, "remove": 3, "ban": 4}
        self._dynamic_last_action = ACTION_CODE[action]
        if true_tox == 1:
            self._dynamic_violation_count += 1
        
        # Delayed Reward Logic
        reward = 0.0
        
        # 1. Immediate correctness
        if true_tox == 1:
            if action in ["remove", "restrict", "ban"]: reward += 0.4
            elif action == "warn": reward += 0.2
            else: reward -= 0.5 # Severe penalty for allowing toxic
        else:
            if action == "allow": reward += 0.4
            elif action == "warn": reward -= 0.1
            else: reward -= 0.5 # False positive penalty
            
        # 2. Sequential/Delayed consequences
        if self.current_step > 0:
            prev_tox = int(self.user_posts[self.current_step - 1]["true_toxicity"])
            prev_action = self.agent_action_history[-2]
            
            # Rehabilitation bonus: Warned last time, safe this time
            if prev_tox == 1 and prev_action == "warn" and true_tox == 0:
                reward += 0.6
                
            # Escalation penalty: Allowed toxic last time, user posted toxic again
            if prev_tox == 1 and prev_action == "allow" and true_tox == 1:
                reward -= 0.6

        # Terminal conditions
        if action == "ban":
            self.is_done = True
            # Did they deserve the ban?
            if post["escalation_level"] >= 2: reward += 0.5
            else: reward -= 1.0 # Banned too early
            
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.is_done = True

        # Clamp reward to [0, 1] scale for OpenEnv specs
        normalized_reward = max(0.0, min(1.0, (reward + 1.5) / 3.0))

        return Task2StepResult(
            observation=self._get_obs() if not self.is_done else self._get_empty_obs(),
            reward=round(normalized_reward, 3),
            done=self.is_done,
            info={"raw_reward": reward}
        )

    def _get_empty_obs(self) -> Task2Observation:
        # Returned on the final step
        return Task2Observation(
            post_id=0, text="", noisy_toxicity_score=0.0, confidence_level=0.0,
            follower_bucket=0, user_id=0, day=0, violation_history=0,
            last_action=0, escalation_level=0, step=self.current_step
        )

    def state(self) -> Dict[str, Any]:
        return {
            "current_user": self.current_user,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "done": self.is_done
        }