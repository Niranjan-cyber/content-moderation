#!/usr/bin/env python3
"""
inference.py — Content Moderation OpenEnv Baseline
====================================================

Runs all three tasks against the live environment server and emits the
mandatory structured log format required by the hackathon evaluator.

STDOUT FORMAT (mandatory — do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Each task produces exactly one [START] … [END] block.
Score per task is in [0, 1] (enforced by clamp).

Environment variables:
  API_BASE_URL   LLM endpoint      (default: HF router)
  MODEL_NAME     Model id          (default: Qwen2.5-72B-Instruct)
  HF_TOKEN       API key           (required)
  ENV_BASE_URL   Env server        (default: http://localhost:7860)

Runtime: ≤ 20 min on 2 vCPU / 8 GB RAM — designed to stay well under.
"""

import os
import sys
import textwrap
from typing import List, Optional, Dict, Any

import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "content-moderation-env"

TEMPERATURE  = 0.2     # Low temperature → reproducible decisions
MAX_TOKENS   = 60      # Actions are short strings

# Task-level episode budget
TASK1_EPISODES  = 10   # 10 independent single-step episodes → mean score
TASK2_EPISODES  = 3    # 3 full user trajectories → mean score across all steps
TASK3_EPISODES  = 1    # 1 full 20-day simulation → mean reward across days
MAX_STEPS_T2    = 15   # Safety ceiling per trajectory (in case ban never fires)
SUCCESS_THRESHOLD = 0.5

# ─── Valid actions per task ───────────────────────────────────────────────────
VALID_ACTIONS = {
    "task1-single-post":    {"allow", "warn", "remove"},
    "task2-user-trajectory":{"allow", "warn", "restrict", "remove", "ban"},
    "task3-platform-policy":{"increase_strictness", "decrease_strictness", "keep_policy_same"},
}
FALLBACK_ACTIONS = {
    "task1-single-post":    "warn",
    "task2-user-trajectory":"warn",
    "task3-platform-policy":"keep_policy_same",
}

# ─── System prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {

"task1-single-post": textwrap.dedent("""
    You are a content moderation AI for a social media platform.

    Your job: review a post and choose the best action.

    ACTIONS:
      allow  — post is clearly safe, no action needed
      warn   — borderline or uncertain content; notify user and add a warning label
      remove — clearly harmful content; take the post down immediately

    SIGNALS:
      noisy_toxicity_score: AI model estimate of harmfulness (0–1).
        This score is IMPERFECT — it can be wrong.
      confidence_level: how confident the model is in its own score.
        Low confidence (< 0.6) means you should be cautious.
      follower_bucket: poster's reach (0=small, 1=medium, 2=large).
        High-reach posts amplify harm — err on the side of caution.

    DECISION HEURISTICS:
      score > 0.7 AND confidence > 0.7  → remove
      score > 0.6 AND confidence < 0.7  → warn (uncertain — hedge)
      score < 0.3 AND confidence > 0.7  → allow
      score < 0.3 AND confidence < 0.7  → allow or warn conservatively
      follower_bucket == 2              → be more conservative

    Reply with EXACTLY one word: allow, warn, or remove.
    No explanation. No punctuation. Just the single word.
""").strip(),

"task2-user-trajectory": textwrap.dedent("""
    You are moderating a SINGLE USER across multiple posts over time.

    ACTIONS:
      allow    — no intervention; post stays visible
      warn     — notify user; post visible with warning
      restrict — limit how often this user can post
      remove   — take down this specific post
      ban      — remove user from platform permanently (use sparingly)

    USER CONTEXT (use this to shape your strategy):
      violation_history: how many times this user has posted toxic content before
      last_action:       what action was taken on their previous post
                         (0=allow, 1=warn, 2=restrict, 3=remove, 4=ban)
      escalation_level:  0=clean, 1=flagged, 2=escalated, 3=repeat offender

    STRATEGY:
      - First offence (violation_history=0): prefer warn over remove
      - Escalation level 2+ with new toxic post: restrict or remove
      - Escalation level 3 with new toxic post: ban is justified
      - Safe post after a warning: allow (rehabilitation)
      - Never ban at escalation_level 0 or 1 — it will hurt your score

    Reply with EXACTLY one word: allow, warn, restrict, remove, or ban.
    No explanation. No punctuation.
""").strip(),

"task3-platform-policy": textwrap.dedent("""
    You are the Head of Trust & Safety setting PLATFORM-WIDE moderation policy.

    ACTIONS:
      increase_strictness — auto-mod removes more content (safer, less engaging)
      decrease_strictness — auto-mod allows more content (more engaging, riskier)
      keep_policy_same    — no change

    METRICS (all 0–1):
      global_toxicity_rate:  fraction of toxic content leaking through.
        HIGH is bad. If > 0.3, increase strictness.
      engagement_score:      fraction of safe posts correctly allowed.
        LOW is bad. If < 0.6, consider decreasing strictness.
      user_retention_rate:   proxy for whether users are churning.
        LOW means over-moderation — decrease strictness.
      fairness_gap:          |group_A_FPR - group_B_FPR|.
        HIGH (> 0.2) means your policy discriminates — adjust carefully.

    STRATEGY:
      - Ideal state: toxicity < 0.15, engagement > 0.8, retention > 0.8, fairness_gap < 0.05
      - If toxicity is high AND engagement is high → increase strictness (room to tighten)
      - If toxicity is low AND engagement is low  → decrease strictness (over-moderation)
      - If already balanced                        → keep_policy_same
      - Watch fairness_gap; if it widens, try decreasing strictness

    Reply with EXACTLY one of:
      increase_strictness
      decrease_strictness
      keep_policy_same
    No explanation. No extra text.
""").strip(),
}

# ─── Logging (mandatory format) ───────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={done_val} error={err_val}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)

# ─── Env HTTP helpers ─────────────────────────────────────────────────────────

def env_reset(task: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        params={"task": task},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def env_step(task: str, action: str) -> dict:
    r = requests.post(
        f"{ENV_BASE_URL}/step",
        params={"task": task},
        json={"action": action},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# ─── LLM decision ─────────────────────────────────────────────────────────────

def build_user_prompt(task: str, obs: Dict[str, Any],
                      step: int, history: List[str]) -> str:
    """Build a concise, information-dense prompt from the current observation."""

    if task == "task1-single-post":
        return textwrap.dedent(f"""
            Post: "{obs.get('text', '')}"

            noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
            confidence_level     : {obs.get('confidence_level', 0):.3f}
            follower_bucket      : {obs.get('follower_bucket', 0)}  (0=low, 1=mid, 2=high)

            Step {step}. Choose: allow | warn | remove
        """).strip()

    elif task == "task2-user-trajectory":
        history_block = "\n".join(history[-4:]) or "None"
        return textwrap.dedent(f"""
            Post: "{obs.get('text', '')}"

            noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
            confidence_level     : {obs.get('confidence_level', 0):.3f}
            violation_history    : {obs.get('violation_history', 0)}
            last_action          : {obs.get('last_action', 0)}  (0=allow,1=warn,2=restrict,3=remove,4=ban)
            escalation_level     : {obs.get('escalation_level', 0)}  (0-3)

            Recent history:
            {history_block}

            Step {step}. Choose: allow | warn | restrict | remove | ban
        """).strip()

    elif task == "task3-platform-policy":
        return textwrap.dedent(f"""
            Day {obs.get('current_day', step)} platform metrics:

            global_toxicity_rate        : {obs.get('global_toxicity_rate', 0):.3f}
            engagement_score            : {obs.get('engagement_score', 0):.3f}
            user_retention_rate         : {obs.get('user_retention_rate', 0):.3f}
            moderation_strictness_level : {obs.get('moderation_strictness_level', 0):.3f}
            fairness_gap                : {obs.get('fairness_gap', 0):.4f}

            Sample post: "{obs.get('sample_post_text', '')[:100]}"
              score={obs.get('sample_toxicity_score', 0):.3f}  type={obs.get('sample_content_type', '')}

            Step {step}. Choose: increase_strictness | decrease_strictness | keep_policy_same
        """).strip()

    return f"Step {step}. Observation: {obs}"


def get_action(client: OpenAI, task: str, obs: Dict[str, Any],
               step: int, history: List[str]) -> str:
    """Query the LLM for a moderation decision. Returns a safe fallback on error."""
    fallback = FALLBACK_ACTIONS[task]
    valid    = VALID_ACTIONS[task]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": build_user_prompt(
                    task, obs, step, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        # Accept multi-word actions (e.g. "increase_strictness")
        for action in sorted(valid, key=len, reverse=True):
            if action in raw:
                return action
        return fallback
    except Exception as exc:
        print(f"[DEBUG] LLM error (task={task}, step={step}): {exc}", flush=True)
        return fallback

# ─── Per-task runners ─────────────────────────────────────────────────────────

def run_task1(client: OpenAI) -> None:
    """
    Task 1 — 10 independent single-step episodes.
    Score = mean reward across all episodes.
    """
    task        = "task1-single-post"
    all_rewards : List[float] = []
    total_steps = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for ep in range(TASK1_EPISODES):
            error_msg: Optional[str] = None
            try:
                reset_r = env_reset(task)
                obs  = reset_r["observation"]
                done = reset_r.get("done", False)
            except Exception as exc:
                total_steps += 1
                log_step(step=total_steps, action="null", reward=0.0,
                         done=True, error=str(exc))
                all_rewards.append(0.0)
                continue

            if done:
                continue

            action = get_action(client, task, obs, total_steps + 1, [])

            try:
                step_r = env_step(task, action)
                reward = float(step_r.get("reward", 0.0))
                done   = step_r.get("done", True)
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_msg = str(exc)
                print(f"[DEBUG] step failed ep={ep+1}: {exc}", flush=True)

            total_steps  += 1
            all_rewards.append(reward)
            log_step(step=total_steps, action=action, reward=reward,
                     done=done, error=error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task1 unexpected error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=total_steps, score=score,
                rewards=all_rewards)


def run_task2(client: OpenAI) -> None:
    """
    Task 2 — 3 full user trajectory episodes.
    Score = mean reward across all steps across all episodes.
    """
    task        = "task2-user-trajectory"
    all_rewards : List[float] = []
    total_steps = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for ep in range(TASK2_EPISODES):
            history: List[str] = []
            error_msg: Optional[str] = None

            try:
                reset_r = env_reset(task)
                obs  = reset_r["observation"]
                done = reset_r.get("done", False)
            except Exception as exc:
                total_steps += 1
                log_step(step=total_steps, action="null", reward=0.0,
                         done=True, error=str(exc))
                all_rewards.append(0.0)
                continue

            for _s in range(MAX_STEPS_T2):
                if done:
                    break

                action = get_action(client, task, obs, total_steps + 1, history)

                try:
                    step_r  = env_step(task, action)
                    reward  = float(step_r.get("reward", 0.0))
                    done    = step_r.get("done", False)
                    obs     = step_r.get("observation", obs)
                    error_msg = None
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)
                    print(f"[DEBUG] step failed ep={ep+1} s={_s+1}: {exc}",
                          flush=True)

                total_steps  += 1
                all_rewards.append(reward)
                history.append(
                    f"Step {total_steps}: {action!r} → reward {reward:+.2f}"
                )
                log_step(step=total_steps, action=action, reward=reward,
                         done=done, error=error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task2 unexpected error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=total_steps, score=score,
                rewards=all_rewards)


def run_task3(client: OpenAI) -> None:
    """
    Task 3 — 1 full 20-day platform simulation.
    Score = mean reward across all 20 steps.
    """
    task        = "task3-platform-policy"
    all_rewards : List[float] = []
    total_steps = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for ep in range(TASK3_EPISODES):
            try:
                reset_r = env_reset(task)
                obs  = reset_r["observation"]
                done = reset_r.get("done", False)
            except Exception as exc:
                log_step(step=1, action="null", reward=0.0, done=True, error=str(exc))
                log_end(success=False, steps=1, score=0.0, rewards=[0.0])
                return

            for _s in range(25):   # 20 days + small buffer
                if done:
                    break

                action = get_action(client, task, obs, total_steps + 1, [])

                error_msg: Optional[str] = None
                try:
                    step_r  = env_step(task, action)
                    reward  = float(step_r.get("reward", 0.0))
                    done    = step_r.get("done", False)
                    obs     = step_r.get("observation", obs)
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)
                    print(f"[DEBUG] Task3 step failed: {exc}", flush=True)

                total_steps  += 1
                all_rewards.append(reward)
                log_step(step=total_steps, action=action, reward=reward,
                         done=done, error=error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task3 unexpected error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=total_steps, score=score,
                rewards=all_rewards)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[DEBUG] WARNING: No API key found. "
              "Set HF_TOKEN or API_KEY env var.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    run_task1(client)
    run_task2(client)
    run_task3(client)


if __name__ == "__main__":
    main()