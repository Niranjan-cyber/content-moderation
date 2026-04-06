from pydantic import BaseModel, Field
from typing import Literal, Any, Dict, Optional


# ─── Task 1: Single Post Moderation ──────────────────────────────────────────

class Task1Observation(BaseModel):
    post_id:              int
    text:                 str
    noisy_toxicity_score: float = Field(ge=0.0, le=1.0)  # Imperfect — NOT ground truth
    confidence_level:     float = Field(ge=0.0, le=1.0)  # Model's self-reported confidence
    follower_bucket:      int   = Field(ge=0,   le=2)    # 0=low, 1=mid, 2=high reach
    step:                 int

class Task1Action(BaseModel):
    action: Literal["allow", "warn", "remove"]

class StepResult(BaseModel):
    observation: Task1Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class ResetResult(BaseModel):
    observation: Task1Observation
    done:        bool
    info:        Dict[str, Any]


# ─── Task 2: User Trajectory Control ─────────────────────────────────────────

class Task2Observation(BaseModel):
    post_id:              int
    text:                 str
    noisy_toxicity_score: float = Field(ge=0.0, le=1.0)
    confidence_level:     float = Field(ge=0.0, le=1.0)
    follower_bucket:      int   = Field(ge=0,   le=2)
    user_id:              int
    day:                  int
    violation_history:    int   # Number of past violations by this user
    last_action:          int   # Agent's action on user's previous post (0=allow,1=warn,2=restrict,3=remove,4=ban)
    escalation_level:     int   = Field(ge=0, le=3)  # 0-3 tier
    step:                 int

class Task2Action(BaseModel):
    action: Literal["allow", "warn", "restrict", "remove", "ban"]

class Task2StepResult(BaseModel):
    observation: Task2Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class Task2ResetResult(BaseModel):
    observation: Task2Observation
    done:        bool
    info:        Dict[str, Any]


# ─── Task 3: Platform Policy Optimization ────────────────────────────────────

class Task3Observation(BaseModel):
    # Episode position
    current_day:                 int

    # Core platform health (all 0.0–1.0)
    global_toxicity_rate:        float = Field(ge=0.0, le=1.0,
        description="Fraction of truly toxic posts that slipped through moderation")
    engagement_score:            float = Field(ge=0.0, le=1.0,
        description="Fraction of safe posts that were correctly allowed")
    user_retention_rate:         float = Field(ge=0.0, le=1.0,
        description="Proxy for user retention — penalised by false-positive rate")

    # Policy control
    moderation_strictness_level: float = Field(ge=0.0, le=1.0,
        description="Current auto-mod threshold; higher = more aggressive removal")

    # Fairness tracking (key differentiator from toy envs)
    fairness_gap:                float = Field(ge=0.0, le=1.0,
        description="|group_A_FPR - group_B_FPR|; 0 = perfectly fair")
    group_a_false_positive_rate: float = Field(ge=0.0, le=1.0,
        description="Fraction of safe group-A posts incorrectly removed")
    group_b_false_positive_rate: float = Field(ge=0.0, le=1.0,
        description="Fraction of safe group-B posts incorrectly removed")

    # Grounding: one sampled post so the agent can reason about real content
    sample_post_text:            str
    sample_toxicity_score:       float = Field(ge=0.0, le=1.0)
    sample_content_type:         str

    step: int

class Task3Action(BaseModel):
    action: Literal["increase_strictness", "decrease_strictness", "keep_policy_same"]

class Task3StepResult(BaseModel):
    observation: Task3Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any]

class Task3ResetResult(BaseModel):
    observation: Task3Observation
    done:        bool
    info:        Dict[str, Any]
    
# ─── Task 4: Appeals & Consistency ─────────────────────────────────────────

class Task4Observation(BaseModel):
    appeal_id:             int
    original_post_text:    str
    original_action_taken: str
    noisy_toxicity_score:  float = Field(ge=0.0, le=1.0)
    confidence_level:      float = Field(ge=0.0, le=1.0)
    user_id:               int
    user_history:          int
    should_reverse:        int   # 1 if wrong decision, 0 if correct

    # Consistency signals
    similar_cases:         int
    similar_agreement:     float = Field(ge=0.0, le=1.0)

    step: int


class Task4Action(BaseModel):
    action: Literal["uphold", "reverse", "escalate"]


class Task4StepResult(BaseModel):
    observation: Task4Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class Task4ResetResult(BaseModel):
    observation: Task4Observation
    done: bool
    info: Dict[str, Any]