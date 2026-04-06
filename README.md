---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - content-moderation
  - reinforcement-learning
  - trust-and-safety
---

# Content Moderation OpenEnv

A real-world RL environment where an AI agent learns to moderate social media content — the same challenge faced by trust & safety teams at every major platform.

## Why this domain?

Content moderation is a genuinely hard RL problem:
- The agent never knows the **true** toxicity — only a noisy model score
- Actions have **long-term consequences** (warn a user today → they behave differently tomorrow)  
- There's a fundamental **tradeoff**: too strict kills engagement, too lenient harms users
- **Fairness constraints** must be satisfied across demographic groups

This is not a toy. This is what companies like Meta, Twitter/X, and YouTube actually solve.

---

## Tasks

### Task 1 — Single Post Moderation (Easy)
**Type:** POMDP (partial observability)

The agent sees one post and makes one decision: `allow / warn / remove`.

The challenge: the toxicity score is **imperfect**. The agent must learn when to trust the signal and when to be cautious.

| Observation field | Description |
|---|---|
| `text` | Raw post content |
| `noisy_toxicity_score` | AI model estimate — NOT ground truth (0–1) |
| `confidence_level` | Model's self-reported confidence (0–1) |
| `follower_bucket` | Poster's reach: 0=low, 1=mid, 2=high |

**Actions:** `allow` · `warn` · `remove`

**Reward:** Shaped [0, 1] — penalises false negatives hardest, adds calibration bonus for confident correct decisions.

---

### Task 2 — User Trajectory Control (Medium) *(coming)*
**Type:** Sequential RL

Agent moderates the same user across 10 time steps. User behavior evolves based on actions taken.

---

### Task 3 — Platform Policy Optimization (Hard) *(coming)*
**Type:** Multi-objective RL

Agent controls global moderation strictness across a simulated platform, balancing engagement, retention, toxicity rate, and demographic fairness.

---

## Action & Observation Spaces

```
Action space:      Discrete(3) — {allow, warn, remove}
Observation space: Dict with text (str), noisy_toxicity_score (float), 
                   confidence_level (float), follower_bucket (int 0-2)
Reward range:      [0.0, 1.0]
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference (set your API key first)
export HF_TOKEN=your_key
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t content-mod-env .
docker run -p 7860:7860 content-mod-env
```

---

## API Reference

```
POST /reset?task=task1-single-post    → ResetResult
POST /step?task=task1-single-post     → StepResult  (body: {"action": "allow"})
GET  /state?task=task1-single-post    → state dict
GET  /health                          → {"status": "ok"}
GET  /tasks                           → available tasks list
```

---

## Baseline Scores

| Task | Model | Score | Episodes |
|------|-------|-------|---------|
| task1-single-post | Qwen2.5-72B-Instruct | ~0.72 | 10 |

---

## Dataset

Built on [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) with derived fields:
- `noisy_toxicity_score` — Gaussian-perturbed true label  
- `confidence_level` — simulated model calibration score
- `is_adversarial` + `modified_text` — character-level obfuscation layer
- `group` — demographic fairness split (A/B)
