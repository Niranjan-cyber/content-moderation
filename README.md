---
title: CascadeGuard Content Moderation Env
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# CascadeGuard — Content Moderation RL Environment

> An OpenEnv-compliant reinforcement learning environment simulating real-world content moderation decisions at a billion-user social platform.

---

## Overview

Content moderation is one of the hardest operational problems at scale. Platforms like Meta process billions of posts daily, making sequential decisions under uncertainty with limited resources — where one wrong call can cascade into a viral crisis.

**CascadeGuard** turns this into a rigorous RL benchmark with four tasks of increasing complexity, grounded in real toxicity data (Jigsaw Toxic Comment Classification dataset).

---

## Tasks

| Task | Name | Type | Difficulty | Max Steps |
|------|------|------|------------|-----------|
| 1 | Single Post Moderation | POMDP | Easy | 1 |
| 2 | User Trajectory Control | Sequential RL | Medium | 10 |
| 3 | Platform Policy Optimisation | Multi-objective RL | Hard | 20 |
| 4 | Appeals & Consistency | Meta-Decision RL | Hard | 10 |

### Task 1 — Single Post (POMDP)
Agent receives ONE post with a noisy toxicity score (imperfect signal). Must decide `allow / warn / remove`. True toxicity is hidden — agent must learn risk-aware decision making using `confidence_level` as calibration.

### Task 2 — User Trajectory (Sequential RL)
Agent moderates the SAME user across multiple posts. Actions have delayed consequences — too lenient causes escalation, too strict causes unfair churn. Agent receives the **full post history** of the user at every step.

### Task 3 — Platform Policy (Multi-objective RL)
Agent controls platform-wide moderation strictness over 20 days. Must balance four objectives simultaneously: safety, engagement, retention, and demographic fairness. Agent sees the **top 5 most urgent posts** and a **platform user behaviour summary** each day.

### Task 4 — Appeals & Consistency (Meta-Decision RL)
Agent reviews contested moderation decisions across 10 appeals per episode. Must `uphold / reverse / escalate`. A **precedent mechanic** enforces consistency — wrong decisions on early appeals create pressure on future similar cases. Scored 70% accuracy + 30% consistency.

---

## Action Spaces

```
Task 1:  allow | warn | remove
Task 2:  allow | warn | restrict | remove | ban
Task 3:  increase_strictness | decrease_strictness | keep_policy_same
Task 4:  uphold | reverse | escalate
```

---

## Observation Spaces

### Task 1
```json
{
  "post_id": 42,
  "text": "vaccines cause autism",
  "noisy_toxicity_score": 0.71,
  "confidence_level": 0.82,
  "follower_bucket": 2,
  "content_type": "health_misinfo",
  "step": 0
}
```

### Task 2 (extends Task 1)
```json
{
  "...task1_fields": "...",
  "user_id": 12,
  "day": 3,
  "violation_history": 2,
  "last_action": 1,
  "escalation_level": 2,
  "post_history": [
    {"day": 1, "text": "...", "noisy_toxicity_score": 0.45, "action_taken": 1, "action_label": "warn"},
    {"day": 2, "text": "...", "noisy_toxicity_score": 0.61, "action_taken": 2, "action_label": "restrict"}
  ]
}
```

### Task 3
```json
{
  "current_day": 5,
  "global_toxicity_rate": 0.18,
  "engagement_score": 0.76,
  "user_retention_rate": 0.84,
  "moderation_strictness_level": 0.6,
  "fairness_gap": 0.03,
  "group_a_false_positive_rate": 0.04,
  "group_b_false_positive_rate": 0.07,
  "active_posts": [
    {"post_id": 42, "text": "...", "noisy_toxicity_score": 0.91, "follower_bucket": 2, "content_type": "hate_speech", "group": "A"}
  ],
  "platform_user_summary": {
    "total_active_users": 30,
    "escalating_users": 8,
    "improving_users": 12,
    "banned_today": 0
  }
}
```

### Task 4
```json
{
  "appeal_id": 7,
  "text": "...",
  "noisy_toxicity_score": 0.28,
  "confidence_level": 0.91,
  "original_action_taken": 2,
  "original_action_label": "remove",
  "precedent_history": [
    {"appeal_number": 1, "content_type": "political", "original_action": 2, "agent_decision": "reverse", "was_correct": true}
  ]
}
```

---

## Reward Functions

**Task 1:** Correct action = +1.0. False negative (allowed harmful) = -3.0. False positive (removed safe) = -2.0. Normalised to [0, 1].

**Task 2:** Step rewards for correct decisions. Bonus for rehabilitation (+0.5) and justified bans (+1.0). Terminal bonus based on overall violation rate reduction.

**Task 3:** `score = 0.35 * safety + 0.30 * engagement + 0.20 * retention + 0.15 * fairness`

**Task 4:** Correct decision = +1.0. Escalate = +0.3 (partial). Wrong = -1.0. Consistency penalty = -0.5. Episode score = 0.7 * accuracy + 0.3 * consistency.

---

## Setup

```bash
# 1. Generate dataset from Jigsaw raw CSV
python prepare_dataset.py \
  --input  data/raw/jigsaw_train.csv \
  --output server/data/dataset_final.csv \
  --n_rows 1000

# 2. Start environment server
uvicorn server.environment:app --host 0.0.0.0 --port 7860

# 3. Run baseline inference
export HF_TOKEN=your_token_here
python inference.py
```

## Docker

```bash
docker build -t cascadeguard .
docker run -p 7860:7860 cascadeguard
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset?task=<name>` | Start new episode |
| POST | `/step?task=<name>` | Take action `{"action": str}` |
| GET | `/state?task=<name>` | Current state |

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| task1-single-post | Qwen2.5-72B | ~0.72 |
| task2-user-trajectory | Qwen2.5-72B | ~0.58 |
| task3-platform-policy | Qwen2.5-72B | ~0.51 |
| task4-appeals | Qwen2.5-72B | ~0.49 |

---

## Dataset

Built on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) with derived columns for RL training.

| Column | Description | Visible To Agent |
|--------|-------------|-----------------|
| true_toxicity | Binary ground truth | ❌ Grader only |
| noisy_toxicity_score | Imperfect signal | ✅ |
| confidence_level | Signal reliability | ✅ |
| correct_action | Ground truth action | ❌ Grader only |
| user_behavior_pattern | User trajectory type | ❌ Grader only |
| should_reverse | Appeal ground truth | ❌ Grader only |

---

## Project Structure

```
cascadeguard-env/
├── server/
│   ├── environment.py       # FastAPI environment server
│   └── data/
│       └── dataset_final.csv
├── prepare_dataset.py       # Dataset generation from Jigsaw CSV
├── inference.py             # Baseline agent (Qwen2.5-72B)
├── openenv.yaml             # OpenEnv spec
├── requirements.txt
├── Dockerfile
└── README.md
```