#!/usr/bin/env python3
"""
environment.py -- CascadeGuard OpenEnv Server
=============================================

FastAPI server implementing the OpenEnv spec for CascadeGuard.

Endpoints:
    GET  /              -> health check
    POST /reset         -> start new episode   ?task=<task_name>
    POST /step          -> take one action      ?task=<task_name>  body: {"action": str}
    GET  /state         -> current state        ?task=<task_name>
    GET  /tasks         -> list all tasks

Tasks:
    task1-single-post       Easy   -- single post, POMDP
    task2-user-trajectory   Medium -- sequential user moderation
    task3-platform-policy   Hard   -- platform-wide policy control
    task4-appeals           Hard   -- appeals & consistency

All columns used come strictly from dataset_final.csv canonical schema:
    id, appeal_id, text, modified_text, content_type,
    true_toxicity, correct_action,
    noisy_toxicity_score, confidence_level,
    follower_bucket, group, is_adversarial,
    user_id, day,
    violation_history, last_action, escalation_level,
    original_action_taken, should_reverse,
    label
"""

import os
import random
from typing import Any, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Config -------------------------------------------------------------------

DATA_PATH = os.getenv("DATA_PATH", "server/data/dataset_final.csv")
SEED      = int(os.getenv("SEED", "42"))

random.seed(SEED)
np.random.seed(SEED)

# --- App ----------------------------------------------------------------------

app = FastAPI(
    title="CascadeGuard Content Moderation Environment",
    description="OpenEnv-compliant RL environment for content moderation research.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading -------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Ensure correct types
    int_cols = [
        "id", "true_toxicity", "correct_action", "follower_bucket",
        "is_adversarial", "user_id", "day", "violation_history",
        "last_action", "escalation_level", "original_action_taken",
        "should_reverse", "label", "appeal_id",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    float_cols = ["noisy_toxicity_score", "confidence_level"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).astype(float)
    str_cols = ["text", "modified_text", "content_type", "group"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df

DF: pd.DataFrame = load_dataset()

# --- Pydantic Models ----------------------------------------------------------

class ActionRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]

class ResetResponse(BaseModel):
    observation: dict[str, Any]
    done: bool
    info: dict[str, Any]

# --- Episode State ------------------------------------------------------------
# One global state dict per task. Single-worker server (HF Spaces).

_state: dict[str, Any] = {
    "task1-single-post":      {},
    "task2-user-trajectory":  {},
    "task3-platform-policy":  {},
    "task4-appeals":          {},
}

# --- Valid Actions ------------------------------------------------------------

VALID_ACTIONS = {
    "task1-single-post":      {"allow", "warn", "remove"},
    "task2-user-trajectory":  {"allow", "warn", "restrict", "remove", "ban"},
    "task3-platform-policy":  {"increase_strictness", "decrease_strictness", "keep_policy_same"},
    "task4-appeals":          {"uphold", "reverse", "escalate"},
}

TASK_NAMES = list(VALID_ACTIONS.keys())

# --- Helpers ------------------------------------------------------------------

def _row_to_post_obs(row: pd.Series) -> dict[str, Any]:
    """Convert a dataset row to a post-level observation dict (Task 1 fields)."""
    return {
        "post_id":             int(row["id"]),
        "text":                str(row["text"]),
        "noisy_toxicity_score":round(float(row["noisy_toxicity_score"]), 4),
        "confidence_level":    round(float(row["confidence_level"]), 4),
        "follower_bucket":     int(row["follower_bucket"]),
        "content_type":        str(row["content_type"]),
    }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ===============================================================================
# TASK 1 -- Single Post Moderation (POMDP)
# ===============================================================================

def _task1_reset() -> dict[str, Any]:
    row = DF.sample(1, random_state=random.randint(0, 99999)).iloc[0]
    _state["task1-single-post"] = {
        "row":  row,
        "step": 0,
        "done": False,
    }
    obs = _row_to_post_obs(row)
    obs["step"] = 0
    return obs


def _task1_step(action: str) -> StepResponse:
    st  = _state["task1-single-post"]
    row = st["row"]

    if st["done"]:
        raise HTTPException(400, "Episode already done. Call /reset first.")

    true_tox      = int(row["true_toxicity"])
    correct       = int(row["correct_action"])   # 0=allow,1=warn,2=remove
    severity      = float(row["noisy_toxicity_score"])
    follower      = int(row["follower_bucket"])

    # Map action string -> int
    action_map = {"allow": 0, "warn": 1, "remove": 2}
    action_int = action_map[action]

    # -- Reward --------------------------------------------------------------
    if action_int == correct:
        # Correct decision -- bonus for catching high-severity content
        reward = 1.0 + (0.2 * true_tox * follower)
    elif action_int == 0 and true_tox == 1:
        # False negative -- allowed harmful content (worst)
        reward = -3.0 - (0.5 * follower)          # worse for large accounts
    elif action_int == 2 and true_tox == 0:
        # False positive -- removed safe content
        reward = -2.0
    else:
        # Off by one (warn instead of remove, etc.)
        reward = -0.5

    reward = _clamp(reward, -4.0, 1.5)

    # Normalise to [0, 1] for grader
    score = _clamp((reward + 4.0) / 5.5)

    st["done"] = True
    st["step"] = 1

    obs = _row_to_post_obs(row)
    obs["step"] = 1

    return StepResponse(
        observation=obs,
        reward=round(score, 4),
        done=True,
        info={
            "true_toxicity":  true_tox,
            "correct_action": correct,
            "action_taken":   action_int,
            "raw_reward":     round(reward, 4),
            "score":          round(score, 4),
        },
    )


def _task1_state() -> dict[str, Any]:
    st = _state["task1-single-post"]
    if not st:
        return {"error": "No active episode. Call /reset first."}
    obs = _row_to_post_obs(st["row"])
    obs["step"] = st["step"]
    obs["done"] = st["done"]
    return obs


# ===============================================================================
# TASK 2 -- User Trajectory (Sequential RL)
# ===============================================================================

def _task2_build_obs(st: dict) -> dict[str, Any]:
    """
    Full observation for Task 2 including COMPLETE post history.
    Agent sees every previous post this user made -- not just a count.
    """
    current_row = st["timeline"][st["current_day_idx"]]
    obs = _row_to_post_obs(current_row)

    # Current user state
    obs["user_id"]           = int(current_row["user_id"])
    obs["day"]               = int(st["current_day_idx"] + 1)
    obs["violation_history"] = int(st["violation_history"])
    obs["last_action"]       = int(st["last_action"])
    obs["escalation_level"]  = int(st["escalation_level"])
    obs["step"]              = int(st["step"])

    # Full post history -- every previous post in this episode
    obs["post_history"] = [
        {
            "day":                  int(i + 1),
            "text":                 str(prev["text"]),
            "noisy_toxicity_score": round(float(prev["noisy_toxicity_score"]), 4),
            "content_type":         str(prev["content_type"]),
            "action_taken":         int(st["actions_taken"][i]),
            "action_label":         ["allow","warn","restrict","remove","ban"][st["actions_taken"][i]],
        }
        for i, prev in enumerate(st["history_rows"])
    ]

    return obs


def _task2_reset() -> dict[str, Any]:
    """Pick a random user with >2 posts and build their timeline."""
    # Users with at least 3 posts
    user_counts = DF[DF["user_id"] > 0]["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 3].index.tolist()
    if not valid_users:
        raise HTTPException(500, "No valid Task 2 users found in dataset.")

    uid  = random.choice(valid_users)
    user_df = DF[DF["user_id"] == uid].sort_values("day").reset_index(drop=True)
    timeline = [user_df.iloc[i] for i in range(len(user_df))]

    _state["task2-user-trajectory"] = {
        "uid":              uid,
        "timeline":         timeline,
        "current_day_idx":  0,
        "step":             0,
        "done":             False,
        "violation_history":0,
        "last_action":      0,
        "escalation_level": 0,
        "history_rows":     [],
        "actions_taken":    [],
        "rewards":          [],
    }

    return _task2_build_obs(_state["task2-user-trajectory"])


def _task2_step(action: str) -> StepResponse:
    st = _state["task2-user-trajectory"]

    if st["done"]:
        raise HTTPException(400, "Episode done. Call /reset first.")

    action_map = {"allow": 0, "warn": 1, "restrict": 2, "remove": 3, "ban": 4}
    action_int = action_map[action]

    current_row  = st["timeline"][st["current_day_idx"]]
    true_tox     = int(current_row["true_toxicity"])
    correct      = int(current_row["correct_action"])   # 0/1/2
    esc          = st["escalation_level"]
    viol_hist    = st["violation_history"]

    # -- Reward --------------------------------------------------------------
    reward = 0.0

    # Correct moderation
    if action_int in {0,1,2} and action_int == correct:
        reward += 1.0
    elif action_int in {3,4} and correct == 2:   # remove/ban when should remove
        reward += 0.8

    # False negative -- allowed harmful content
    if action_int == 0 and true_tox == 1:
        reward -= 2.0

    # False positive -- punished safe user
    if action_int in {3,4} and true_tox == 0:
        reward -= 2.0

    # Over-escalation -- ban too early
    if action_int == 4 and esc < 2:
        reward -= 1.5

    # Correct ban -- persistent offender
    if action_int == 4 and esc >= 3 and viol_hist >= 3:
        reward += 1.0

    # Rehabilitation -- correct allow after user improved
    if action_int == 0 and true_tox == 0 and esc == 0:
        reward += 0.5

    score = _clamp((reward + 3.0) / 5.0)

    # -- Update user state ----------------------------------------------------
    st["history_rows"].append(current_row)
    st["actions_taken"].append(action_int)
    st["rewards"].append(score)

    if true_tox == 1:
        st["violation_history"] += 1
        st["escalation_level"]   = min(3, esc + 1)
    else:
        st["escalation_level"]   = max(0, esc - 1)

    st["last_action"]     = action_int
    st["current_day_idx"] += 1
    st["step"]            += 1

    # -- Done? ----------------------------------------------------------------
    done = (
        action_int == 4 or                               # ban is terminal
        st["current_day_idx"] >= len(st["timeline"])     # no more posts
    )
    st["done"] = done

    if done:
        # Episode-end bonus: reward reduction in overall violation rate
        total_days = st["current_day_idx"]
        total_viols = st["violation_history"]
        viol_rate   = total_viols / max(1, total_days)
        final_bonus = _clamp(1.0 - viol_rate) * 0.3
        score      += final_bonus
        score       = _clamp(score)

    obs = {} if done else _task2_build_obs(st)

    return StepResponse(
        observation=obs,
        reward=round(score, 4),
        done=done,
        info={
            "user_id":          int(st["uid"]),
            "day":              int(st["current_day_idx"]),
            "true_toxicity":    true_tox,
            "escalation_level": int(st["escalation_level"]),
            "violation_history":int(st["violation_history"]),
            "action_taken":     action_int,
            "raw_reward":       round(reward, 4),
            "score":            round(score, 4),
        },
    )


def _task2_state() -> dict[str, Any]:
    st = _state["task2-user-trajectory"]
    if not st:
        return {"error": "No active episode. Call /reset first."}
    obs = _task2_build_obs(st)
    obs["done"] = st["done"]
    return obs


# ===============================================================================
# TASK 3 -- Platform Policy (Multi-objective RL)
# ===============================================================================

MAX_DAYS_T3  = 20
BATCH_SIZE   = 25    # posts processed per day

def _compute_platform_metrics(
    batch: pd.DataFrame,
    strictness: float,
) -> dict[str, float]:
    """
    Simulate one day of platform moderation given current strictness.
    Returns metrics computed from actual dataset labels.
    """
    # Moderation decision: remove if noisy_score > (1 - strictness) threshold
    threshold = 1.0 - strictness
    predicted_remove = batch["noisy_toxicity_score"] >= threshold
    actually_toxic   = batch["true_toxicity"] == 1
    actually_safe    = batch["true_toxicity"] == 0

    # True positives, false positives, false negatives
    tp = (predicted_remove &  actually_toxic).sum()
    fp = (predicted_remove &  actually_safe).sum()
    fn = (~predicted_remove & actually_toxic).sum()
    tn = (~predicted_remove & actually_safe).sum()

    total          = len(batch)
    toxic_total    = actually_toxic.sum()
    safe_total     = actually_safe.sum()

    # Global toxicity rate -- fraction of toxic posts that got through
    global_toxicity_rate = fn / max(1, toxic_total)

    # Engagement -- fraction of safe posts correctly allowed
    engagement_score     = tn / max(1, safe_total)

    # Retention -- penalised by false-positive rate (users leaving after unfair removal)
    fpr_overall          = fp / max(1, safe_total)
    user_retention_rate  = _clamp(1.0 - fpr_overall * 1.5)

    # Fairness -- |group_A_FPR - group_B_FPR|
    group_a = batch[batch["group"] == "A"]
    group_b = batch[batch["group"] == "B"]

    def group_fpr(g: pd.DataFrame) -> float:
        safe_g = g[g["true_toxicity"] == 0]
        if len(safe_g) == 0:
            return 0.0
        rem_g  = safe_g["noisy_toxicity_score"] >= threshold
        return float(rem_g.sum() / len(safe_g))

    fpr_a      = group_fpr(group_a)
    fpr_b      = group_fpr(group_b)
    fairness_gap = abs(fpr_a - fpr_b)

    return {
        "global_toxicity_rate":        round(global_toxicity_rate, 4),
        "engagement_score":            round(engagement_score, 4),
        "user_retention_rate":         round(user_retention_rate, 4),
        "fairness_gap":                round(fairness_gap, 4),
        "group_a_false_positive_rate": round(fpr_a, 4),
        "group_b_false_positive_rate": round(fpr_b, 4),
    }


def _task3_build_obs(st: dict, metrics: dict, day: int) -> dict[str, Any]:
    """
    Full platform observation including TOP 5 active posts needing attention.
    Agent sees the full picture -- not just one sample post.
    """
    batch      = st["current_batch"]
    strictness = st["strictness"]

    # Top 5 posts sorted by urgency (high toxicity x high follower)
    urgency = (
        batch["noisy_toxicity_score"] * (batch["follower_bucket"] + 1)
    )
    top5 = batch.assign(_urgency=urgency).nlargest(5, "_urgency")

    active_posts = [
        {
            "post_id":             int(r["id"]),
            "text":                str(r["text"])[:120],
            "noisy_toxicity_score":round(float(r["noisy_toxicity_score"]), 4),
            "content_type":        str(r["content_type"]),
            "follower_bucket":     int(r["follower_bucket"]),
            "group":               str(r["group"]),
        }
        for _, r in top5.iterrows()
    ]

    # One sample post for backward compat with inference.py
    sample = top5.iloc[0] if len(top5) > 0 else batch.iloc[0]

    # Platform user summary from full dataset user timelines
    user_data = DF[DF["user_id"] > 0]
    escalating_users = int((user_data["escalation_level"] >= 2).sum())
    improving_users  = int((user_data["escalation_level"] == 0).sum())

    obs = {
        "current_day":                 day,
        "global_toxicity_rate":        metrics["global_toxicity_rate"],
        "engagement_score":            metrics["engagement_score"],
        "user_retention_rate":         metrics["user_retention_rate"],
        "moderation_strictness_level": round(strictness, 4),
        "fairness_gap":                metrics["fairness_gap"],
        "group_a_false_positive_rate": metrics["group_a_false_positive_rate"],
        "group_b_false_positive_rate": metrics["group_b_false_positive_rate"],
        # Backward-compatible single sample (for existing inference.py)
        "sample_post_text":            str(sample["text"])[:120],
        "sample_toxicity_score":       round(float(sample["noisy_toxicity_score"]), 4),
        "sample_content_type":         str(sample["content_type"]),
        # NEW -- full active posts list + user summary
        "active_posts":                active_posts,
        "platform_user_summary": {
            "total_active_users":  int(len(DF[DF["user_id"] > 0]["user_id"].unique())),
            "escalating_users":    escalating_users,
            "improving_users":     improving_users,
            "banned_today":        int(st.get("banned_today", 0)),
        },
        "step": day,
    }
    return obs


def _task3_reset() -> dict[str, Any]:
    batch = DF.sample(min(BATCH_SIZE, len(DF)), random_state=random.randint(0, 99999))
    strictness = 0.5   # start at mid strictness
    metrics    = _compute_platform_metrics(batch, strictness)

    _state["task3-platform-policy"] = {
        "day":           1,
        "strictness":    strictness,
        "current_batch": batch,
        "done":          False,
        "rewards":       [],
        "banned_today":  0,
    }

    return _task3_build_obs(_state["task3-platform-policy"], metrics, 1)


def _task3_step(action: str) -> StepResponse:
    st = _state["task3-platform-policy"]

    if st["done"]:
        raise HTTPException(400, "Episode done. Call /reset first.")

    # Adjust strictness
    delta_map = {
        "increase_strictness": +0.1,
        "decrease_strictness": -0.1,
        "keep_policy_same":     0.0,
    }
    st["strictness"] = _clamp(st["strictness"] + delta_map[action])

    # New batch for this day
    batch = DF.sample(min(BATCH_SIZE, len(DF)), random_state=st["day"] * 7)
    st["current_batch"] = batch

    metrics = _compute_platform_metrics(batch, st["strictness"])

    # -- Reward ---------------------------------------------------------------
    # Weighted multi-objective score
    score = (
        0.35 * (1.0 - metrics["global_toxicity_rate"]) +   # safety
        0.30 * metrics["engagement_score"]               +   # engagement
        0.20 * metrics["user_retention_rate"]            +   # retention
        0.15 * (1.0 - metrics["fairness_gap"])               # fairness
    )
    score = _clamp(score)
    st["rewards"].append(score)

    st["day"]        += 1
    st["banned_today"] = 0   # reset daily counter

    done = st["day"] > MAX_DAYS_T3
    st["done"] = done

    obs = {} if done else _task3_build_obs(st, metrics, st["day"])

    return StepResponse(
        observation=obs,
        reward=round(score, 4),
        done=done,
        info={
            "day":                         int(st["day"] - 1),
            "strictness":                  round(st["strictness"], 4),
            "global_toxicity_rate":        metrics["global_toxicity_rate"],
            "engagement_score":            metrics["engagement_score"],
            "user_retention_rate":         metrics["user_retention_rate"],
            "fairness_gap":                metrics["fairness_gap"],
            "group_a_false_positive_rate": metrics["group_a_false_positive_rate"],
            "group_b_false_positive_rate": metrics["group_b_false_positive_rate"],
            "score":                       round(score, 4),
        },
    )


def _task3_state() -> dict[str, Any]:
    st = _state["task3-platform-policy"]
    if not st:
        return {"error": "No active episode. Call /reset first."}
    batch   = st["current_batch"]
    metrics = _compute_platform_metrics(batch, st["strictness"])
    obs     = _task3_build_obs(st, metrics, st["day"])
    obs["done"] = st["done"]
    return obs


# ===============================================================================
# TASK 4 -- Appeals & Fairness (Meta-Decision RL)
# ===============================================================================

MAX_APPEALS = 10

def _task4_build_obs(st: dict) -> dict[str, Any]:
    row = st["current_appeal"]

    obs = {
        "appeal_id":              int(row["appeal_id"]),
        "post_id":                int(row["id"]),
        "text":                   str(row["text"]),
        "noisy_toxicity_score":   round(float(row["noisy_toxicity_score"]), 4),
        "confidence_level":       round(float(row["confidence_level"]), 4),
        "content_type":           str(row["content_type"]),
        "original_action_taken":  int(row["original_action_taken"]),
        "original_action_label":  ["allow","warn","remove"][int(row["original_action_taken"])],
        "follower_bucket":        int(row["follower_bucket"]),
        "step":                   int(st["step"]),
        # Full precedent history -- all past appeals and decisions
        "precedent_history": [
            {
                "appeal_number":    i + 1,
                "text":             str(p["text"])[:80],
                "content_type":     str(p["content_type"]),
                "original_action":  int(p["original_action"]),
                "agent_decision":   str(p["agent_decision"]),
                "was_correct":      bool(p["was_correct"]),
            }
            for i, p in enumerate(st["precedent_history"])
        ],
    }
    return obs


def _task4_reset() -> dict[str, Any]:
    # Sample appeals -- prefer rows where should_reverse == 1 (interesting cases)
    appeal_pool = DF[DF["should_reverse"] == 1]
    if len(appeal_pool) < 5:
        appeal_pool = DF   # fallback to all rows

    appeal = appeal_pool.sample(1, random_state=random.randint(0, 99999)).iloc[0]

    _state["task4-appeals"] = {
        "current_appeal":    appeal,
        "step":              0,
        "done":              False,
        "appeals_done":      0,
        "correct_count":     0,
        "precedent_history": [],   # list of past decisions
        "rewards":           [],
    }

    return _task4_build_obs(_state["task4-appeals"])


def _task4_step(action: str) -> StepResponse:
    st = _state["task4-appeals"]

    if st["done"]:
        raise HTTPException(400, "Episode done. Call /reset first.")

    row         = st["current_appeal"]
    should_rev  = int(row["should_reverse"])
    orig_action = int(row["original_action_taken"])
    true_tox    = int(row["true_toxicity"])

    # Correct decision logic:
    #   should_reverse==1 -> correct is "reverse"
    #   should_reverse==0 -> correct is "uphold"
    #   escalate always gives partial credit (safe choice)
    correct_action = "reverse" if should_rev == 1 else "uphold"

    # -- Reward --------------------------------------------------------------
    reward = 0.0

    if action == correct_action:
        reward += 1.0
        was_correct = True
    elif action == "escalate":
        reward += 0.3    # partial credit -- safe but indecisive
        was_correct = False
    else:
        reward -= 1.0
        was_correct = False

    # -- Consistency penalty --------------------------------------------------
    # Check if similar past cases were decided differently
    similar_past = [
        p for p in st["precedent_history"]
        if p["content_type"] == str(row["content_type"])
    ]
    if similar_past:
        past_decisions = [p["agent_decision"] for p in similar_past]
        inconsistent   = any(d != action and d != "escalate" for d in past_decisions)
        if inconsistent and action != "escalate":
            reward -= 0.5   # consistency penalty

    score = _clamp((reward + 1.0) / 2.0)
    st["rewards"].append(score)

    if was_correct:
        st["correct_count"] += 1

    # Record precedent
    st["precedent_history"].append({
        "text":           str(row["text"])[:80],
        "content_type":   str(row["content_type"]),
        "original_action":orig_action,
        "agent_decision": action,
        "was_correct":    was_correct,
    })

    st["appeals_done"] += 1
    st["step"]         += 1

    done = st["appeals_done"] >= MAX_APPEALS
    st["done"] = done

    # Load next appeal if not done
    if not done:
        appeal_pool = DF[DF["should_reverse"] == 1]
        if len(appeal_pool) < 5:
            appeal_pool = DF
        next_appeal = appeal_pool.sample(
            1, random_state=st["step"] * 13
        ).iloc[0]
        st["current_appeal"] = next_appeal

    if done:
        # Episode-end consistency score
        consistency_score = _clamp(st["correct_count"] / max(1, st["appeals_done"]))
        # Final score = 0.7 * accuracy + 0.3 * consistency
        mean_reward    = sum(st["rewards"]) / max(1, len(st["rewards"]))
        final_score    = 0.7 * mean_reward + 0.3 * consistency_score
        score          = _clamp(final_score)

    obs = {} if done else _task4_build_obs(st)

    return StepResponse(
        observation=obs,
        reward=round(score, 4),
        done=done,
        info={
            "should_reverse":   should_rev,
            "correct_action":   correct_action,
            "action_taken":     action,
            "was_correct":      was_correct,
            "appeals_done":     st["appeals_done"],
            "correct_count":    st["correct_count"],
            "score":            round(score, 4),
        },
    )


def _task4_state() -> dict[str, Any]:
    st = _state["task4-appeals"]
    if not st:
        return {"error": "No active episode. Call /reset first."}
    obs = _task4_build_obs(st)
    obs["done"] = st["done"]
    return obs


# ===============================================================================
# Routes
# ===============================================================================

@app.get("/")
@app.get("/web")
def health():
    return {
        "status":      "ok",
        "environment": "CascadeGuard Content Moderation Env",
        "version":     "0.2.0",
        "tasks":       TASK_NAMES,
        "dataset_rows":len(DF),
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
                "observation_fields": [
                    "post_id", "text", "noisy_toxicity_score",
                    "confidence_level", "follower_bucket", "content_type", "step"
                ],
            },
            {
                "name":        "task2-user-trajectory",
                "description": "Sequential user moderation over time (True RL)",
                "difficulty":  "medium",
                "max_steps":   10,
                "actions":     ["allow", "warn", "restrict", "remove", "ban"],
                "observation_fields": [
                    "post_id", "text", "noisy_toxicity_score",
                    "confidence_level", "follower_bucket", "content_type",
                    "user_id", "day", "violation_history",
                    "last_action", "escalation_level", "post_history", "step"
                ],
            },
            {
                "name":        "task3-platform-policy",
                "description": "Platform-wide policy optimisation (Multi-objective RL)",
                "difficulty":  "hard",
                "max_steps":   20,
                "actions":     ["increase_strictness", "decrease_strictness", "keep_policy_same"],
                "observation_fields": [
                    "current_day", "global_toxicity_rate", "engagement_score",
                    "user_retention_rate", "moderation_strictness_level",
                    "fairness_gap", "group_a_false_positive_rate",
                    "group_b_false_positive_rate", "active_posts",
                    "platform_user_summary", "step"
                ],
            },
            {
                "name":        "task4-appeals",
                "description": "Appeals & consistency system (Meta-Decision RL)",
                "difficulty":  "hard",
                "max_steps":   10,
                "actions":     ["uphold", "reverse", "escalate"],
                "observation_fields": [
                    "appeal_id", "post_id", "text", "noisy_toxicity_score",
                    "confidence_level", "content_type", "original_action_taken",
                    "original_action_label", "follower_bucket",
                    "precedent_history", "step"
                ],
            },
        ]
    }


@app.post("/reset")
def reset(task: str = Query(..., description="Task name")) -> ResetResponse:
    if task not in TASK_NAMES:
        raise HTTPException(400, f"Unknown task '{task}'. Valid: {TASK_NAMES}")

    dispatch = {
        "task1-single-post":     _task1_reset,
        "task2-user-trajectory": _task2_reset,
        "task3-platform-policy": _task3_reset,
        "task4-appeals":         _task4_reset,
    }
    obs = dispatch[task]()
    return ResetResponse(observation=obs, done=False, info={"task": task})


@app.post("/step")
def step(
    body: ActionRequest,
    task: str = Query(..., description="Task name"),
) -> StepResponse:
    if task not in TASK_NAMES:
        raise HTTPException(400, f"Unknown task '{task}'. Valid: {TASK_NAMES}")

    action = body.action.strip().lower()
    if action not in VALID_ACTIONS[task]:
        raise HTTPException(
            400,
            f"Invalid action '{action}' for {task}. "
            f"Valid: {VALID_ACTIONS[task]}"
        )

    dispatch = {
        "task1-single-post":     _task1_step,
        "task2-user-trajectory": _task2_step,
        "task3-platform-policy": _task3_step,
        "task4-appeals":         _task4_step,
    }
    return dispatch[task](action)


@app.get("/state")
def state(task: str = Query(..., description="Task name")) -> dict:
    if task not in TASK_NAMES:
        raise HTTPException(400, f"Unknown task '{task}'. Valid: {TASK_NAMES}")

    dispatch = {
        "task1-single-post":     _task1_state,
        "task2-user-trajectory": _task2_state,
        "task3-platform-policy": _task3_state,
        "task4-appeals":         _task4_state,
    }
    return dispatch[task]()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()