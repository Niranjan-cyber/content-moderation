#!/usr/bin/env python3
"""
inference.py — CascadeGuard OpenEnv Baseline
=============================================
Runs all four tasks against the live environment server.

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  HF_TOKEN       HuggingFace API key  (required)
  API_BASE_URL   LLM endpoint         (default: HF router)
  MODEL_NAME     Model id             (default: Qwen2.5-72B-Instruct)
  ENV_BASE_URL   Env server           (default: http://localhost:7860)
"""

import os
import textwrap
from collections import deque
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
BENCHMARK    = "content-moderation-env"

TEMPERATURE       = 0.2
MAX_TOKENS        = 80
SUCCESS_THRESHOLD = 0.65   

# ─── Agent Memory ─────────────────────────────────────────────────────────────

AGENT_MEMORY: Dict[str, Any] = {
    "last_bad_action":  None,
    "strategy_notes":   [],
    "recent_actions":   deque(maxlen=5),   # sliding window for loop detection
    "step_rewards":     [],
}

# ─── Valid Actions ────────────────────────────────────────────────────────────

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

# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {

"task1-single-post": textwrap.dedent("""
    You are a content moderation AI for a social media platform.
    Review a post and choose the best action.

    ACTIONS:
      allow  — post is clearly safe; no action needed
      warn   — genuinely borderline; add warning label
      remove — clearly harmful; take down immediately

    SIGNALS:
      noisy_toxicity_score: imperfect AI estimate (0-1); higher = more likely toxic
      confidence_level: reliability of that score (0-1); high conf = trust the score
      follower_bucket: reach (0=small, 1=mid, 2=large) — larger reach = be more cautious
      content_type: hate_speech/threat/sexual require stricter standards

    DECISION RULES:
      score > 0.65 AND confidence > 0.75  → remove
      score > 0.40 AND confidence < 0.70  → warn (uncertain signal)
      score < 0.20 AND confidence > 0.75  → allow (clearly safe)
      score 0.20-0.65 AND confidence high → warn (borderline)
      follower_bucket == 2 and score > 0.4 → remove (high-reach risk)
      content_type in [hate_speech, threat, sexual] → lower threshold to act

    IMPORTANT: Don't default to "warn" on everything — it's evasive.
    Make a decisive call when the signal is clear.

    Reply with EXACTLY one word: allow, warn, or remove.
""").strip(),

"task2-user-trajectory": textwrap.dedent("""
    You are moderating a SINGLE USER across multiple posts over time.
    You have full visibility into everything this user has posted before.

    ACTIONS:
      allow    — no intervention needed
      warn     — notify user; post stays with warning label
      restrict — limit posting frequency (good middle-ground for escalating users)
      remove   — take down this specific post
      ban      — permanent removal (use ONLY for repeat offenders)

    STRATEGY:
      - First offence (violation_history=0): prefer warn
      - Second offence (violation_history=1): warn → restrict
      - escalation_level ≥ 2 with toxic post: restrict or remove
      - escalation_level = 3 and toxic: remove or ban justified
      - Safe post after a period of toxicity: allow (rehabilitation)
      - NEVER ban at escalation_level 0 or 1 — massive penalty
      - Use FULL POST HISTORY to understand user's trajectory

    IMPORTANT: Read the actual post content. The user's pattern over time
    matters more than any single post. Look for improvement or worsening.

    Reply with EXACTLY one word: allow, warn, restrict, remove, or ban.
""").strip(),

"task3-platform-policy": textwrap.dedent("""
    You are Head of Trust & Safety setting PLATFORM-WIDE moderation policy.

    ACTIONS:
      increase_strictness — raise threshold; removes more content (safer, less engaging)
      decrease_strictness — lower threshold; allows more content (more engaging, riskier)
      keep_policy_same    — no change today (use sparingly — passive stance is penalised)

    METRICS (higher is better, except toxicity_rate and fairness_gap):
      global_toxicity_rate : toxic content leaking through — keep BELOW 0.15
      engagement_score     : safe posts correctly allowed — keep ABOVE 0.75
      user_retention_rate  : users staying on platform — keep ABOVE 0.75
      fairness_gap         : |GroupA_FPR - GroupB_FPR| — keep BELOW 0.05

    IDEAL TARGET: toxicity<0.15, engagement>0.75, retention>0.75, fairness_gap<0.05

    DECISION LOGIC:
      toxicity_rate > 0.25                → increase_strictness
      engagement_score < 0.60             → decrease_strictness
      retention < 0.60                    → decrease_strictness (too many FPs)
      fairness_gap > 0.20                 → decrease_strictness (disparity too high)
      all metrics near ideal AND balanced → keep_policy_same (only valid no-op)
      uncertain / mixed signals           → small adjustment, don't stay idle

    Use ACTIVE POSTS and PLATFORM USER SUMMARY to read the situation.

    Reply with EXACTLY one of: increase_strictness / decrease_strictness / keep_policy_same
""").strip(),

"task4-appeals": textwrap.dedent("""
    You are an Appeals Officer reviewing contested moderation decisions.

    ACTIONS:
      uphold   — original decision was CORRECT; stand by it
      reverse  — original decision was WRONG; correct it
      escalate — genuinely uncertain; send to senior reviewer (partial credit only)

    KEY INSIGHT FROM DATA:
      should REVERSE if original decision was wrong (regardless of whether it was allow or remove).
      score HIGH + original was ALLOW → reverse (missed toxic content)
      score LOW + original was REMOVE → reverse (wrongly removed safe content)
          if the original allow seems justified by low score → uphold
          if genuinely ambiguous → escalate

    SIGNALS:
      original_action_label: what was originally decided
      noisy_toxicity_score: AI estimate of harmfulness
      confidence_level: reliability of that estimate

    CONSISTENCY RULE (critical for scoring):
      Check PRECEDENT HISTORY. Be consistent across same content_type.
      If you upheld a similar low-score case before → uphold this one too.
      If you reversed a similar high-score-but-allowed case → reverse this one.

    STRATEGY:
      score > 0.50 and original was "allow" → very likely should REVERSE
      score < 0.25 and original was "allow" → was probably correct → UPHOLD
      score 0.25-0.50 and original was "allow" → check content_type and precedent
      confident in original decision → uphold
      genuinely uncertain despite all signals → escalate (last resort)

    Reply with EXACTLY one word: uphold, reverse, or escalate.
""").strip(),

}

# ─── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)

# ─── Env HTTP helpers ─────────────────────────────────────────────────────────

def env_reset(task: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(task: str, action: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", params={"task": task},
                      json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

# ─── Observation Formatter ────────────────────────────────────────────────────

def format_obs(task: str, obs: Dict[str, Any], step: int) -> str:
    if task == "task1-single-post":
        return textwrap.dedent(f"""
            Post: "{obs.get('text', '')}"
            noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
            confidence_level     : {obs.get('confidence_level', 0):.3f}
            follower_bucket      : {obs.get('follower_bucket', 0)} (0=small,1=mid,2=large)
            content_type         : {obs.get('content_type', '')}
            Step {step}. Choose: allow | warn | remove
        """).strip()

    elif task == "task2-user-trajectory":
        history = obs.get("post_history", [])
        if history:
            hist_lines = "\n".join([
                f"  Day {h['day']}: [{h['content_type']}] "
                f"score={h['noisy_toxicity_score']:.2f} "
                f"→ you={h['action_label']}"
                for h in history
            ])
        else:
            hist_lines = "  (no prior posts)"
        return textwrap.dedent(f"""
            USER #{obs.get('user_id')} — Day {obs.get('day')}
            Post: "{obs.get('text', '')}"
            noisy_toxicity_score : {obs.get('noisy_toxicity_score', 0):.3f}
            confidence_level     : {obs.get('confidence_level', 0):.3f}
            content_type         : {obs.get('content_type', '')}
            violation_history    : {obs.get('violation_history', 0)}
            escalation_level     : {obs.get('escalation_level', 0)} (0=clean, 3=repeat offender)
            last_action          : {obs.get('last_action', 0)}
            HISTORY:
            {hist_lines}
            Step {step}. Choose: allow | warn | restrict | remove | ban
        """).strip()

    elif task == "task3-platform-policy":
        posts  = obs.get("active_posts", [])
        posts_block = "\n".join([
            f"  [{p.get('content_type','')}] \"{str(p.get('text',''))[:80]}\"\n"
            f"    score={p.get('noisy_toxicity_score',0):.2f} "
            f"reach={p.get('follower_bucket',0)} group={p.get('group','')}"
            for p in posts
        ]) if posts else f"  sample: \"{obs.get('sample_post_text','')}\" score={obs.get('sample_toxicity_score',0):.2f}"
        summary = obs.get("platform_user_summary", {})
        return textwrap.dedent(f"""
            Day {obs.get('current_day', step)} — Platform Dashboard
            METRICS:
              toxicity_rate    : {obs.get('global_toxicity_rate',0):.3f}  (ideal < 0.15)
              engagement_score : {obs.get('engagement_score',0):.3f}  (ideal > 0.75)
              retention_rate   : {obs.get('user_retention_rate',0):.3f}  (ideal > 0.75)
              strictness       : {obs.get('moderation_strictness_level',0):.3f}
              fairness_gap     : {obs.get('fairness_gap',0):.4f}  (ideal < 0.05)
              group_A_FPR      : {obs.get('group_a_false_positive_rate',0):.4f}
              group_B_FPR      : {obs.get('group_b_false_positive_rate',0):.4f}
            USERS:  escalating={summary.get('escalating_users',0)}  improving={summary.get('improving_users',0)}  banned={summary.get('banned_today',0)}
            TOP POSTS:
            {posts_block}
            Step {step}. Choose: increase_strictness | decrease_strictness | keep_policy_same
        """).strip()

    elif task == "task4-appeals":
        labels     = ["allow", "warn", "remove"]
        orig_int   = int(obs.get("original_action_taken", 0))
        orig_label = labels[orig_int] if orig_int < len(labels) else "unknown"
        precedents = obs.get("precedent_history", [])
        prec_block = "\n".join([
            f"  Appeal {p['appeal_number']}: [{p['content_type']}] "
            f"original={labels[p['original_action']]} "
            f"→ you={p['agent_decision']} correct={p['was_correct']}"
            for p in precedents[-5:]
        ]) if precedents else "  No previous appeals."
        return textwrap.dedent(f"""
            APPEAL #{obs.get('appeal_id')} (step {step})
            POST: "{obs.get('text', '')}"
              content_type         : {obs.get('content_type','')}
              noisy_toxicity_score : {obs.get('noisy_toxicity_score',0):.3f}
              confidence_level     : {obs.get('confidence_level',0):.3f}
            ORIGINAL DECISION: {orig_label}
            PRECEDENT HISTORY:
            {prec_block}
            Step {step}. Choose: uphold | reverse | escalate
        """).strip()

    return f"Step {step}. Observation: {obs}"

# ─── Reflection (learn from bad decisions) ────────────────────────────────────

def reflect(client: OpenAI, task: str, obs: dict,
            action: str, reward: float) -> None:
    """If reward is low, ask model to briefly diagnose and store the insight."""
    if reward >= 0.55:
        return
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Analyse this bad content moderation decision. One sentence."},
                {"role": "user",   "content": f"Task: {task}\nAction: {action}\nReward: {reward:.2f}\nWhat was wrong and what should I do differently?"},
            ],
            temperature=0.3,
            max_tokens=60,
        )
        insight = resp.choices[0].message.content.strip()
        notes   = AGENT_MEMORY["strategy_notes"]
        notes.append(insight)
        if len(notes) > 6:
            notes.pop(0)
    except Exception:
        pass

# ─── Loop Detection ───────────────────────────────────────────────────────────

def _is_looping(action: str, task: str) -> bool:
    """
    Detect if the agent is stuck repeating the same action.
    Triggers if the last 3 actions are identical AND that action had low reward.
    """
    recent = list(AGENT_MEMORY["recent_actions"])
    if len(recent) < 3:
        return False
    # Last 3 actions all the same
    if len(set(recent[-3:])) == 1 and recent[-1] == action:
        # And it's been performing poorly
        recent_rewards = AGENT_MEMORY["step_rewards"][-3:]
        if recent_rewards and (sum(recent_rewards) / len(recent_rewards)) < 0.5:
            return True
    return False


def _break_loop(task: str, current_action: str) -> str:
    """Pick an action different from the current one to break the loop."""
    valid = list(VALID_ACTIONS[task] - {current_action})
    # Prefer non-fallback actions to break meaningfully
    fallback = FALLBACK_ACTIONS[task]
    non_fallback = [a for a in valid if a != fallback]
    return non_fallback[0] if non_fallback else (valid[0] if valid else current_action)

# ─── Action Selection ─────────────────────────────────────────────────────────

def get_action(client: OpenAI, task: str, obs: Dict[str, Any], step: int) -> str:
    valid    = VALID_ACTIONS[task]
    fallback = FALLBACK_ACTIONS[task]

    # Inject lessons from past mistakes into system prompt
    notes   = AGENT_MEMORY["strategy_notes"][-3:]
    system  = SYSTEM_PROMPTS[task]
    if notes:
        system += "\n\nLESSONS FROM RECENT MISTAKES:\n" + "\n".join(f"- {n}" for n in notes)

    user_prompt = format_obs(task, obs, step)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()

        # Parse: match longest valid action string first (avoid "remove" matching "warn")
        action = fallback
        for a in sorted(valid, key=len, reverse=True):
            if a in raw:
                action = a
                break

        # Avoid repeating the single last bad action
        if AGENT_MEMORY["last_bad_action"] == action:
            valid_others = list(valid - {action})
            if valid_others:
                action = valid_others[0]

        # Detect and break sustained loops
        if _is_looping(action, task):
            action = _break_loop(task, action)

        return action

    except Exception as exc:
        print(f"[DEBUG] LLM error step={step}: {exc}", flush=True)
        return fallback

# ─── Core Runner ──────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task: str, max_steps: int = 10) -> None:
    log_start(task, BENCHMARK, MODEL_NAME)

    rewards: List[float] = []
    steps   = 0
    error   = None

    # Reset agent memory per task
    AGENT_MEMORY["last_bad_action"] = None
    AGENT_MEMORY["strategy_notes"]  = []
    AGENT_MEMORY["recent_actions"]  = deque(maxlen=5)
    AGENT_MEMORY["step_rewards"]    = []

    try:
        reset_r = env_reset(task)
        obs     = reset_r["observation"]
        done    = reset_r.get("done", False)

        for _ in range(max_steps):
            if done:
                break

            action = get_action(client, task, obs, steps + 1)

            try:
                step_r = env_step(task, action)
                reward = float(step_r.get("reward", 0.0))
                done   = step_r.get("done", False)
                obs    = step_r.get("observation", obs) or obs
                error  = None
            except Exception as exc:
                reward = 0.0
                done   = True
                error  = str(exc)

            steps += 1
            rewards.append(reward)

            # Update memory for loop detection and learning
            AGENT_MEMORY["recent_actions"].append(action)
            AGENT_MEMORY["step_rewards"].append(reward)

            reflect(client, task, obs, action, reward)

            if reward < 0.5:
                AGENT_MEMORY["last_bad_action"] = action
            else:
                AGENT_MEMORY["last_bad_action"] = None

            log_step(steps, action, reward, done, error)

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)

    score   = sum(rewards) / len(rewards) if rewards else 0.0
    score   = round(min(max(score, 0.0), 1.0), 3)
    success = score >= SUCCESS_THRESHOLD

    log_end(success, steps, score, rewards)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[DEBUG] WARNING: No API key. Set HF_TOKEN env var.", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "no-key")

    run_task(client, "task1-single-post",     max_steps=3)   # 1-step task, 3 for safety
    run_task(client, "task2-user-trajectory", max_steps=12)
    run_task(client, "task3-platform-policy", max_steps=22)  # 20 days + buffer
    run_task(client, "task4-appeals",         max_steps=10)  # 10 appeals


if __name__ == "__main__":
    main()
