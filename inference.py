#!/usr/bin/env python3
"""
inference.py -- CascadeGuard OpenEnv Baseline
=============================================

Runs all four tasks against the live environment server and emits the
mandatory structured log format required by the hackathon evaluator.

STDOUT FORMAT (mandatory -- do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  API_BASE_URL   LLM endpoint      (default: HF router)
  MODEL_NAME     Model id          (default: Qwen2.5-72B-Instruct)
  HF_TOKEN       API key           (required)
  ENV_BASE_URL   Env server        (default: http://localhost:7860)
"""

import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# --- Config -------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN", "") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "content-moderation-env"

TEMPERATURE  = 0.2
MAX_TOKENS   = 60

# Episode budgets
TASK1_EPISODES = 10
TASK2_EPISODES = 3
TASK3_EPISODES = 1
TASK4_EPISODES = 1

MAX_STEPS_T2   = 15    # safety ceiling per trajectory
SUCCESS_THRESHOLD = 0.5

# --- Valid actions ------------------------------------------------------------

VALID_ACTIONS = {
    "task1-single-post":     {"allow", "warn", "remove"},
    "task2-user-trajectory": {"allow", "warn", "restrict", "remove", "ban"},
    "task3-platform-policy": {"increase_strictness", "decrease_strictness", "keep_policy_same"},
    "task4-appeals":         {"uphold", "reverse", "escalate"},
}
FALLBACK_ACTIONS = {
    "task1-single-post":     "warn",
    "task2-user-trajectory": "warn",
    "task3-platform-policy": "keep_policy_same",
    "task4-appeals":         "escalate",
}

# --- System prompts -----------------------------------------------------------

SYSTEM_PROMPTS = {

"task1-single-post": textwrap.dedent("""
    You are a content moderation AI for a social media platform.
    Your job: review a post and choose the best action.

    ACTIONS:
      allow  -- post is clearly safe, no action needed
      warn   -- borderline or uncertain content; notify user and add a warning label
      remove -- clearly harmful content; take the post down immediately

    SIGNALS:
      noisy_toxicity_score: AI model estimate of harmfulness (0-1). IMPERFECT -- can be wrong.
      confidence_level: how confident the model is in its score (0-1).
        Low confidence (< 0.6) -> be more cautious.
      follower_bucket: poster's reach (0=small, 1=medium, 2=large).
        Large reach amplifies harm -- err on the side of caution.

    DECISION HEURISTICS:
      score > 0.7 AND confidence > 0.7  -> remove
      score > 0.6 AND confidence < 0.7  -> warn (uncertain, hedge)
      score < 0.3 AND confidence > 0.7  -> allow
      score < 0.3 AND confidence < 0.7  -> allow or warn conservatively
      follower_bucket == 2              -> be more conservative

    Reply with EXACTLY one word: allow, warn, or remove.
    No explanation. No punctuation.
""").strip(),

"task2-user-trajectory": textwrap.dedent("""
    You are moderating a SINGLE USER across multiple posts over time.
    You have full visibility into everything this user has posted before.

    ACTIONS:
      allow    -- no intervention; post stays visible
      warn     -- notify user; post visible with warning label
      restrict -- limit how often this user can post
      remove   -- take down this specific post
      ban      -- remove user from platform permanently (use sparingly)

    USER STATE SIGNALS:
      violation_history: total toxic posts from this user so far
      last_action:       your previous action (0=allow,1=warn,2=restrict,3=remove,4=ban)
      escalation_level:  0=clean, 1=flagged, 2=escalated, 3=repeat offender

    STRATEGY:
      - First offence (violation_history=0): prefer warn over remove
      - Escalation level 2+ with new toxic post: restrict or remove
      - Escalation level 3 with new toxic post: ban is justified
      - Safe post after a warning: allow (rehabilitation signal)
      - Never ban at escalation_level 0 or 1 -- it will hurt your score
      - Use the FULL POST HISTORY to understand the user's trajectory

    Reply with EXACTLY one word: allow, warn, restrict, remove, or ban.
    No explanation. No punctuation.
""").strip(),

"task3-platform-policy": textwrap.dedent("""
    You are the Head of Trust & Safety setting PLATFORM-WIDE moderation policy.

    ACTIONS:
      increase_strictness -- auto-mod removes more content (safer, less engaging)
      decrease_strictness -- auto-mod allows more content (more engaging, riskier)
      keep_policy_same    -- no change this day

    PLATFORM METRICS (all 0-1):
      global_toxicity_rate:  fraction of toxic content leaking through. HIGH is bad.
      engagement_score:      fraction of safe posts correctly allowed. LOW is bad.
      user_retention_rate:   proxy for user retention. LOW = over-moderation.
      fairness_gap:          |group_A_FPR - group_B_FPR|. HIGH = discriminatory policy.

    IDEAL STATE: toxicity < 0.15, engagement > 0.8, retention > 0.8, fairness_gap < 0.05

    STRATEGY:
      - toxicity high AND engagement high -> increase strictness (room to tighten)
      - toxicity low AND engagement low  -> decrease strictness (over-moderation)
      - already balanced                 -> keep_policy_same
      - fairness_gap > 0.2              -> try decreasing strictness

    Use the ACTIVE POSTS list to understand what content is circulating today.
    Use the PLATFORM USER SUMMARY to understand the health of user behavior.

    Reply with EXACTLY one of:
      increase_strictness
      decrease_strictness
      keep_policy_same
    No explanation. No extra text.
""").strip(),

"task4-appeals": textwrap.dedent("""
    You are an Appeals Officer reviewing moderation decisions that users have contested.

    ACTIONS:
      uphold   -- original decision was correct; stand by it
      reverse  -- original decision was wrong; restore/correct the post
      escalate -- uncertain; send to senior human reviewer

    KEY SIGNALS:
      original_action_taken: what the system decided (0=allow, 1=warn, 2=remove)
      noisy_toxicity_score:  AI estimate of harmfulness (imperfect)
      confidence_level:      how reliable is that score

    CONSISTENCY RULE -- CRITICAL:
      You must be CONSISTENT across appeals of the same content_type.
      If you reversed a decision for a similar post before, you must be
      consistent now. Check the PRECEDENT HISTORY carefully.

    STRATEGY:
      - If noisy_toxicity_score < 0.3 but action was "remove" -> likely false positive -> reverse
      - If noisy_toxicity_score > 0.7 but action was "allow"  -> likely false negative -> reverse
      - If confident the original was right -> uphold
      - If genuinely uncertain -> escalate (safe but only partial credit)
      - ALWAYS check precedent_history for similar past cases first

    Reply with EXACTLY one word: uphold, reverse, or escalate.
    No explanation. No punctuation.
""").strip(),

}

# --- Logging ------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={done_val} error={err_val}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)

# --- Env HTTP helpers ---------------------------------------------------------

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

# --- Prompt builder -----------------------------------------------------------

def build_user_prompt(task: str, obs: Dict[str, Any],
                      step: int) -> str:

    if task == "task1-single-post":
        return textwrap.dedent(f"""
            Post: "{obs.get('text', '')}"

            noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
            confidence_level     : {obs.get('confidence_level', 0):.3f}
            follower_bucket      : {obs.get('follower_bucket', 0)}  (0=low, 1=mid, 2=high)
            content_type         : {obs.get('content_type', '')}

            Step {step}. Choose: allow | warn | remove
        """).strip()

    elif task == "task2-user-trajectory":
        # Build full post history block -- agent sees every previous post
        post_history = obs.get("post_history", [])
        if post_history:
            history_block = "\n".join([
                f"  Day {h['day']}: \"{h['text'][:80]}\"\n"
                f"    toxicity={h['noisy_toxicity_score']:.2f}  "
                f"type={h['content_type']}  "
                f"-> action={h['action_label']}"
                for h in post_history
            ])
        else:
            history_block = "  No previous posts -- this is Day 1."

        last_action_labels = ["allow","warn","restrict","remove","ban"]
        last_label = last_action_labels[int(obs.get('last_action', 0))]

        return textwrap.dedent(f"""
            USER {obs.get('user_id')} -- Day {obs.get('day')}

            CURRENT POST:
              "{obs.get('text', '')}"
              noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
              confidence_level     : {obs.get('confidence_level', 0):.3f}
              content_type         : {obs.get('content_type', '')}
              follower_bucket      : {obs.get('follower_bucket', 0)}

            USER STATE:
              escalation_level  : {obs.get('escalation_level', 0)}  (0=clean -> 3=repeat offender)
              violation_history : {obs.get('violation_history', 0)} past violations
              last_action       : {last_label}

            FULL POST HISTORY (everything this user posted before today):
            {history_block}

            Step {step}. Choose: allow | warn | restrict | remove | ban
        """).strip()

    elif task == "task3-platform-policy":
        # Build active posts block -- top 5 urgent posts
        active_posts = obs.get("active_posts", [])
        if active_posts:
            posts_block = "\n".join([
                f"  [{p['content_type']}] \"{p['text'][:80]}\"\n"
                f"    toxicity={p['noisy_toxicity_score']:.2f}  "
                f"reach={p['follower_bucket']}  "
                f"group={p['group']}"
                for p in active_posts
            ])
        else:
            posts_block = f"  \"{obs.get('sample_post_text', '')}\" " \
                          f"score={obs.get('sample_toxicity_score', 0):.2f}"

        summary = obs.get("platform_user_summary", {})

        return textwrap.dedent(f"""
            Day {obs.get('current_day', step)} -- Platform Dashboard

            PLATFORM METRICS:
              global_toxicity_rate  : {obs.get('global_toxicity_rate', 0):.3f}  (lower=safer)
              engagement_score      : {obs.get('engagement_score', 0):.3f}      (higher=better)
              user_retention_rate   : {obs.get('user_retention_rate', 0):.3f}   (higher=better)
              strictness_level      : {obs.get('moderation_strictness_level', 0):.3f}
              fairness_gap          : {obs.get('fairness_gap', 0):.4f}           (lower=fairer)
              group_A_FPR           : {obs.get('group_a_false_positive_rate', 0):.4f}
              group_B_FPR           : {obs.get('group_b_false_positive_rate', 0):.4f}

            PLATFORM USER BEHAVIOR:
              total active users : {summary.get('total_active_users', 0)}
              escalating users   : {summary.get('escalating_users', 0)}
              improving users    : {summary.get('improving_users', 0)}
              banned today       : {summary.get('banned_today', 0)}

            TOP ACTIVE POSTS NEEDING ATTENTION TODAY:
            {posts_block}

            Step {step}. Choose: increase_strictness | decrease_strictness | keep_policy_same
        """).strip()

    elif task == "task4-appeals":
        action_labels = ["allow", "warn", "remove"]
        orig_label    = action_labels[int(obs.get('original_action_taken', 0))]

        # Build precedent history block
        precedents = obs.get("precedent_history", [])
        if precedents:
            prec_block = "\n".join([
                f"  Appeal {p['appeal_number']}: [{p['content_type']}] "
                f"\"{p['text'][:60]}\" "
                f"original={action_labels[p['original_action']]} "
                f"-> you decided={p['agent_decision']} "
                f"(correct={p['was_correct']})"
                for p in precedents[-5:]   # last 5 precedents
            ])
        else:
            prec_block = "  No previous appeals in this episode."

        return textwrap.dedent(f"""
            APPEAL #{obs.get('appeal_id')}

            CONTESTED POST:
              "{obs.get('text', '')}"
              content_type         : {obs.get('content_type', '')}
              noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
              confidence_level     : {obs.get('confidence_level', 0):.3f}
              follower_bucket      : {obs.get('follower_bucket', 0)}

            ORIGINAL DECISION:
              System action: {orig_label} ({obs.get('original_action_taken', 0)})
              User is contesting this decision.

            PRECEDENT HISTORY (your past decisions on similar content):
            {prec_block}

            Step {step}. Choose: uphold | reverse | escalate
        """).strip()

    return f"Step {step}. Observation: {obs}"

# --- LLM decision -------------------------------------------------------------

def get_action(client: OpenAI, task: str, obs: Dict[str, Any], step: int) -> str:
    fallback = FALLBACK_ACTIONS[task]
    valid    = VALID_ACTIONS[task]
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user",   "content": build_user_prompt(task, obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        for action in sorted(valid, key=len, reverse=True):
            if action in raw:
                return action
        return fallback
    except Exception as exc:
        print(f"[DEBUG] LLM error (task={task}, step={step}): {exc}", flush=True)
        return fallback

# --- Task runners -------------------------------------------------------------

def run_task1(client: OpenAI) -> None:
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
                log_step(total_steps, "null", 0.0, True, str(exc))
                all_rewards.append(0.0)
                continue

            if done:
                continue

            action = get_action(client, task, obs, total_steps + 1)

            try:
                step_r = env_step(task, action)
                reward = float(step_r.get("reward", 0.0))
                done   = step_r.get("done", True)
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_msg = str(exc)

            total_steps += 1
            all_rewards.append(reward)
            log_step(total_steps, action, reward, done, error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task1 error: {exc}", flush=True)
    finally:
        log_end(success, total_steps, score, all_rewards)


def run_task2(client: OpenAI) -> None:
    """
    Task 2 -- 3 full user trajectory episodes.
    Agent receives FULL post history of the user each step.
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
            error_msg: Optional[str] = None

            try:
                reset_r = env_reset(task)
                obs  = reset_r["observation"]
                done = reset_r.get("done", False)
            except Exception as exc:
                total_steps += 1
                log_step(total_steps, "null", 0.0, True, str(exc))
                all_rewards.append(0.0)
                continue

            for _ in range(MAX_STEPS_T2):
                if done:
                    break

                # obs already contains full post_history -- pass directly
                action = get_action(client, task, obs, total_steps + 1)

                try:
                    step_r    = env_step(task, action)
                    reward    = float(step_r.get("reward", 0.0))
                    done      = step_r.get("done", False)
                    obs       = step_r.get("observation", obs) or obs
                    error_msg = None
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)

                total_steps += 1
                all_rewards.append(reward)
                log_step(total_steps, action, reward, done, error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task2 error: {exc}", flush=True)
    finally:
        log_end(success, total_steps, score, all_rewards)


def run_task3(client: OpenAI) -> None:
    """
    Task 3 -- 1 full 20-day platform simulation.
    Agent receives active_posts (top 5 urgent) + platform_user_summary each step.
    Score = mean reward across all 20 days.
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
                log_step(1, "null", 0.0, True, str(exc))
                log_end(False, 1, 0.0, [0.0])
                return

            for _ in range(25):   # 20 days + buffer
                if done:
                    break

                # obs contains active_posts and platform_user_summary
                action = get_action(client, task, obs, total_steps + 1)

                error_msg: Optional[str] = None
                try:
                    step_r    = env_step(task, action)
                    reward    = float(step_r.get("reward", 0.0))
                    done      = step_r.get("done", False)
                    obs       = step_r.get("observation", obs) or obs
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)

                total_steps += 1
                all_rewards.append(reward)
                log_step(total_steps, action, reward, done, error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task3 error: {exc}", flush=True)
    finally:
        log_end(success, total_steps, score, all_rewards)


def run_task4(client: OpenAI) -> None:
    """
    Task 4 -- 1 full appeals episode (10 appeals).
    Agent receives full precedent_history each step for consistency checking.
    Score = 0.7 * accuracy + 0.3 * consistency.
    """
    task        = "task4-appeals"
    all_rewards : List[float] = []
    total_steps = 0
    score       = 0.0
    success     = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for ep in range(TASK4_EPISODES):
            try:
                reset_r = env_reset(task)
                obs  = reset_r["observation"]
                done = reset_r.get("done", False)
            except Exception as exc:
                log_step(1, "null", 0.0, True, str(exc))
                all_rewards.append(0.0)
                continue

            for _ in range(12):   # 10 appeals + buffer
                if done:
                    break

                action = get_action(client, task, obs, total_steps + 1)

                error_msg: Optional[str] = None
                try:
                    step_r    = env_step(task, action)
                    reward    = float(step_r.get("reward", 0.0))
                    done      = step_r.get("done", False)
                    obs       = step_r.get("observation", obs) or obs
                except Exception as exc:
                    reward    = 0.0
                    done      = True
                    error_msg = str(exc)

                total_steps += 1
                all_rewards.append(reward)
                log_step(total_steps, action, reward, done, error_msg)

        score   = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        score   = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task4 error: {exc}", flush=True)
    finally:
        log_end(success, total_steps, score, all_rewards)


# --- Main ---------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[DEBUG] WARNING: No API key. Set HF_TOKEN or API_KEY.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    run_task1(client)
    run_task2(client)
    run_task3(client)
    run_task4(client)


if __name__ == "__main__":
    main()