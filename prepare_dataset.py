#!/usr/bin/env python3
"""
prepare_dataset.py -- Generate dataset_final.csv from raw Jigsaw CSV
=====================================================================

Run from repo root:
    python prepare_dataset.py

Or with custom paths:
    python prepare_dataset.py --input data/raw/jigsaw_train.csv \
                              --output server/data/dataset_final.csv \
                              --n_rows 1000

What this does
--------------
Reads the raw Jigsaw Toxic Comment Classification CSV and produces a
fully derived dataset with ALL canonical columns needed by Tasks 1-4.

Canonical column order (no additions, no removals):
    id, appeal_id,
    text, modified_text, content_type,
    true_toxicity, correct_action,
    noisy_toxicity_score, confidence_level,
    follower_bucket, group, is_adversarial,
    user_id, day,
    violation_history, last_action, escalation_level,
    original_action_taken, should_reverse,
    label

Derivation rules
----------------
true_toxicity       -> binary (0/1). 1 if any of: toxic, severe_toxic,
                      threat, identity_hate == 1 in Jigsaw labels.

noisy_toxicity_score-> severity_score (weighted float) + Gaussian noise.
                      severity_score = weighted sum of all 6 Jigsaw labels.

confidence_level    -> 1 - |severity_score - noisy_toxicity_score|

correct_action      -> 0=allow / 1=warn / 2=remove, thresholded on severity_score.
                      0 if severity < 0.35
                      1 if 0.35 <= severity < 0.65
                      2 if severity >= 0.65

follower_bucket     -> 0/1/2 (small/mid/large), seeded random per post.

group               -> "A" or "B" (fairness split), seeded random per post.

is_adversarial      -> 0 for all base rows (adversarial rows added separately).

user_id             -> Task 2. Posts grouped into 30 fake users,
                      each user gets a timeline of posts ordered by day.

day                 -> Position within user's timeline (1-indexed).

violation_history   -> Count of true_toxicity==1 posts BEFORE current day
                      within the same user's timeline.

last_action         -> correct_action of the PREVIOUS day's post for this user.
                      0 (allow) for day 1.

escalation_level    -> 0-3 scale. Starts at 0. Increases with each violation,
                      decreases on clean posts. Capped at [0, 3].

appeal_id           -> same as id (every post is also an appeal candidate).

original_action_taken -> same as correct_action (what the system decided).

should_reverse      -> 1 if the post was incorrectly moderated:
                        - removed (correct_action==2) but true_toxicity==0 -> false positive
                        - allowed (correct_action==0) but true_toxicity==1 -> false negative
                      0 otherwise (decision was correct).

label               -> same as true_toxicity (binary ground truth).

modified_text       -> same as text for all base rows.
"""

import argparse
import re
import numpy as np
import pandas as pd

# --- Reproducibility ---------------------------------------------------------
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# --- Weighted Scoring Classifier ---------------------------------------------
CATEGORY_PATTERNS: dict[str, list[tuple[int, str]]] = {
    "hate_speech": [
        (6, r"\b(nigger|niglet|faggot|retard|tranny|spic|chink|kike|coon|gook|sandnigger)\b"),
        (6, r"\b(hate|scum|inferior|subhuman|vermin|disgusting|despise|loathe)\b.{0,30}\b(jews?|muslims?|blacks?|whites?|gays?|trans\b|mexicans?|arabs?|refugees?)\b"),
        (5, r"\b(racist|racism|white.suprem|nazi|neo.nazi|kkk|antisemit|islamophob|bigot)\b"),
        (4, r"\b(jews?|muslims?|blacks?|gays?|trans\b|mexicans?|arabs?|refugees?)\b.{0,40}\b(kill|die|scum|inferior|criminal|terrorist|rapist|evil|disgusting|hate)\b"),
        (4, r"\b(holocaust|auschwitz|genocide|gas.the|gas the|ethnic.cleansing|slavery)\b"),
        (3, r"\b(not.racist|no.racism)\b.{0,30}\b(but|because|they|those)\b"),
        (4, r"\b(instead of|ban|deport|remove|exclude)\b.{0,20}\b(muslims?|blacks?|mexicans?|arabs?|refugees?|immigrants?|jews?)\b"),
        (3, r"\b(great.replacement|race.war|white.genocide|replacement.theory|race.mixing)\b"),
        (3, r"\b(crime|welfare|violent|lazy|dirty|smelly|iq)\b.{0,40}\b(black|mexican|arab|muslim|jew|asian|gay|trans)\b"),
    ],
    "threat": [
        (5, r"\b(kill you|murder you|i.ll.kill|gonna.kill|want.to.kill|shoot.you|stab.you|death.threat)\b"),
        (4, r"\b(kill|murder|shoot|stab|bomb|blow.?up|execute|assassinate|slaughter)\b"),
        (3, r"\b(gun|knife|weapon|explosive|grenade|attack|threaten|harm.you|hurt.you)\b"),
        (2, r"\b(violence|blood|die|dead|end.you|destroy.you|wipe.out|come.for.you)\b"),
        (1, r"\b(beat.up|punch|smash|crush|torture)\b"),
    ],
    "sexual": [
        (5, r"\b(porn|pornography|nude|naked|masturbat|orgasm|cum|blowjob|handjob|dildo)\b"),
        (4, r"\b(fuck|fucking|fucked|dick|cock|pussy|ass.?hole|tits|boobs|penis|vagina)\b"),
        (3, r"\b(rape|sexual.assault|molest|grope|harass.sexually|sex.with|have.sex)\b"),
        (2, r"\b(horny|aroused|erect|strip|panties|underwear|seduce|lust|kinky|fetish)\b"),
        (1, r"\b(sex|sexy|sexual|sensual|intimate|hook.?up|one.night.stand)\b"),
    ],
    "political": [
        (4, r"\b(election|president|congress|senate|parliament|democrat|republican|socialist)\b"),
        (4, r"\b(isis|isil|al.qaeda|jihad|terrorism|radical.islam|extremist|insurgent)\b"),
        (3, r"\b(trump|obama|biden|hillary|pelosi|bernie|putin|boris|macron)\b"),
        (3, r"\b(immigration|border.wall|deportation|refugee|asylum.seeker|illegal.alien)\b"),
        (2, r"\b(liberal|conservative|left.wing|right.wing|fascist|communist|antifa|proud.boy)\b"),
        (2, r"\b(vote|voting|ballot|rigged|stolen.election|conspiracy|deep.state)\b"),
        (1, r"\b(policy|law|bill|government|politics|protest|activist|revolution)\b"),
    ],
    "humor": [
        (4, r"\b(lol|lmao|lmfao|rofl|haha|hehe|xd)\b"),
        (3, r"\b(joke|meme|funny|hilarious|comedy|punchline|prank|troll)\b"),
        (3, r"\b(when you|that moment when|me irl|nobody:|no one:|them:|boss:|teacher:)\b"),
        (2, r"\b(sarcas|irony|ironic|satire|deadpan|dark.humor|dry.humor)\b"),
        (2, r"\b(imagine if|plot twist|unpopular opinion|fun fact|shower thought)\b"),
        (1, r"\b(wait|imagine|meanwhile|basically|literally|actually.though)\b"),
    ],
    "personal": [
        (4, r"\b(i am|i'm|i was|i feel|i love|i hate|i need|i want|i think|i believe)\b"),
        (3, r"\b(my (life|story|friend|family|mom|dad|sister|brother|boyfriend|girlfriend))\b"),
        (3, r"\b(confession|personal|vent|diary|opened up|honest(ly)?|true story)\b"),
        (2, r"\b(relationship|breakup|dating|marriage|divorce|heartbreak|toxic.person)\b"),
        (2, r"\b(anxiety|depression|mental.health|therapy|trauma|abuse|struggle)\b"),
        (1, r"\b(today i|yesterday i|last (week|night|year) i|when i was)\b"),
    ],
    "news_info": [
        (4, r"\b(study|research|report|scientists?|university|published|findings|data.shows)\b"),
        (4, r"\b(breaking.news|headline|journalist|press|media|according.to|sources.say)\b"),
        (3, r"\b(percent|statistics?|million|billion|survey|poll|results.show)\b"),
        (2, r"\b(expert|official|government.says|health.officials?|cdc|who|fbi|cia)\b"),
        (1, r"\b(discovered|revealed|confirmed|announced|reports?|claims?|alleges?)\b"),
    ],
}

PRIORITY_ORDER = [
    "hate_speech", "threat", "sexual", "political",
    "humor", "personal", "news_info", "general",
]


def classify_content_type(text: str) -> str:
    t = str(text).lower()
    scores: dict[str, float] = {cat: 0.0 for cat in PRIORITY_ORDER}
    for category, patterns in CATEGORY_PATTERNS.items():
        for weight, pat in patterns:
            if re.search(pat, t):
                scores[category] += weight
    best_score = max(scores.values())
    if best_score == 0.0:
        return "general"
    for cat in PRIORITY_ORDER:
        if scores[cat] == best_score:
            return cat
    return "general"


# --- Column Derivation --------------------------------------------------------

def derive_severity_score(row: pd.Series) -> float:
    """
    Weighted float severity score from Jigsaw binary labels.
    Used as the base for noisy_toxicity_score derivation.
    NOT exposed to the agent -- only used internally.
    """
    return float(np.clip(
        row.get("toxic", 0)        * 0.20 +
        row.get("severe_toxic", 0) * 0.35 +
        row.get("threat", 0)       * 0.25 +
        row.get("obscene", 0)      * 0.05 +
        row.get("insult", 0)       * 0.05 +
        row.get("identity_hate", 0)* 0.10,
        0.0, 1.0
    ))


def derive_true_toxicity(row: pd.Series) -> int:
    """
    Binary ground truth. 1 if the post contains any high-severity signal.
    Grader uses this -- agent never sees it.
    """
    return int(
        row.get("toxic", 0) == 1 or
        row.get("severe_toxic", 0) == 1 or
        row.get("threat", 0) == 1 or
        row.get("identity_hate", 0) == 1
    )


def derive_correct_action(severity: float) -> int:
    """
    0 = allow, 1 = warn, 2 = remove
    Thresholded on severity score.
    """
    if severity >= 0.65:
        return 2   # remove
    elif severity >= 0.35:
        return 1   # warn
    else:
        return 0   # allow


def derive_noisy_score(severity: float, noise_std: float = 0.15) -> float:
    """Add Gaussian noise to severity -- this is what the agent sees."""
    noisy = severity + rng.normal(0, noise_std)
    return float(np.clip(noisy, 0.0, 1.0))


def derive_confidence(severity: float, noisy: float) -> float:
    """How trustworthy is the noisy signal? Inverse of noise gap."""
    return float(np.clip(1.0 - abs(severity - noisy), 0.0, 1.0))


def derive_follower_bucket(n: int) -> np.ndarray:
    """
    0 = small (<1k), 1 = mid (1k-100k), 2 = large (>100k)
    Realistic distribution: most accounts are small.
    """
    return rng.choice([0, 1, 2], size=n, p=[0.65, 0.25, 0.10])


def derive_group(n: int) -> np.ndarray:
    """A/B fairness split. ~50/50."""
    return rng.choice(["A", "B"], size=n, p=[0.50, 0.50])


# --- Task 2: User Timeline Columns -------------------------------------------

def build_user_timelines(df: pd.DataFrame, n_users: int = 30) -> pd.DataFrame:
    """
    Assigns posts to fake users and builds sequential timeline columns:
        user_id, day, violation_history, last_action, escalation_level

    Strategy:
    - n_users users, each gets a timeline of posts sampled from df.
    - Each user is assigned a hidden behavior pattern that determines
      how posts are ordered (escalating / improving / stable / relapsing).
    - violation_history, last_action, escalation_level are derived
      deterministically from the timeline order.

    Posts NOT assigned to any user (Task 1 / Task 3 rows) get user_id=0, day=0,
    and all other timeline columns = 0.
    """
    df = df.copy()

    # Initialise with Task-1 defaults (user_id=0 means "no user context")
    df["user_id"]          = 0
    df["day"]              = 0
    df["violation_history"]= 0
    df["last_action"]      = 0
    df["escalation_level"] = 0

    # Separate toxic and clean pools for realistic timelines
    toxic_idx = df.index[df["true_toxicity"] == 1].tolist()
    clean_idx = df.index[df["true_toxicity"] == 0].tolist()

    rng.shuffle(toxic_idx)
    rng.shuffle(clean_idx)

    BEHAVIOR_PATTERNS = ["escalating", "improving", "stable", "relapsing"]
    # Posts per user: between 5 and 10
    posts_per_user = rng.integers(5, 11, size=n_users)

    toxic_ptr = 0
    clean_ptr = 0

    for uid in range(1, n_users + 1):
        n_posts  = int(posts_per_user[uid - 1])
        pattern  = BEHAVIOR_PATTERNS[(uid - 1) % len(BEHAVIOR_PATTERNS)]

        # Build a timeline of indices
        # escalating  -> more toxic posts toward the end
        # improving   -> more toxic posts at the start
        # stable      -> mixed evenly
        # relapsing   -> clean middle, toxic bookends

        n_toxic = max(1, n_posts // 2)
        n_clean = n_posts - n_toxic

        # Guard: don't exceed available indices
        n_toxic = min(n_toxic, len(toxic_idx) - toxic_ptr)
        n_clean = min(n_clean, len(clean_idx) - clean_ptr)
        if n_toxic + n_clean < 2:
            break   # ran out of posts

        t_slice = toxic_idx[toxic_ptr: toxic_ptr + n_toxic]
        c_slice = clean_idx[clean_ptr: clean_ptr + n_clean]
        toxic_ptr += n_toxic
        clean_ptr += n_clean

        # Order posts by pattern
        if pattern == "escalating":
            timeline = c_slice + t_slice          # clean first, toxic later
        elif pattern == "improving":
            timeline = t_slice + c_slice          # toxic first, clean later
        elif pattern == "stable":
            # interleave safely -- zip stops at shortest, append remainder
            merged = []
            for t, c in zip(t_slice, c_slice):
                merged.extend([t, c])
            # append any leftover elements from the longer slice
            longer = t_slice if len(t_slice) > len(c_slice) else c_slice
            merged.extend(longer[len(min(t_slice, c_slice, key=len)):])
            timeline = merged
        else:   # relapsing
            mid     = len(c_slice) // 2
            half_t  = len(t_slice) // 2
            timeline = t_slice[:half_t] + c_slice + t_slice[half_t:]

        # Assign user timeline columns
        violation_count = 0
        prev_action     = 0   # 0 = allow (default for day 1)
        esc_level       = 0

        for day, idx in enumerate(timeline, start=1):
            df.at[idx, "user_id"]           = uid
            df.at[idx, "day"]               = day
            df.at[idx, "violation_history"] = violation_count
            df.at[idx, "last_action"]       = prev_action
            df.at[idx, "escalation_level"]  = esc_level

            # Update trackers for next day
            is_violation = int(df.at[idx, "true_toxicity"] == 1)
            if is_violation:
                violation_count += 1
                esc_level = min(3, esc_level + 1)
            else:
                esc_level = max(0, esc_level - 1)

            prev_action = int(df.at[idx, "correct_action"])

    return df


# --- Task 4: Appeals Columns --------------------------------------------------

def build_appeals_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    appeal_id           -> same as id (every post is an appeal candidate)
    original_action_taken -> same as correct_action
    should_reverse      -> 1 if the moderation decision was wrong:
                            false positive: correct_action==2 but true_toxicity==0
                            false negative: correct_action==0 but true_toxicity==1
                          0 if the decision was correct
    """
    df = df.copy()
    df["appeal_id"]             = df["id"]
    df["original_action_taken"] = df["correct_action"]

    false_positive = (df["correct_action"] == 2) & (df["true_toxicity"] == 0)
    false_negative = (df["correct_action"] == 0) & (df["true_toxicity"] == 1)
    df["should_reverse"] = ((false_positive) | (false_negative)).astype(int)

    return df


# --- Canonical Column Order ---------------------------------------------------

CANONICAL_COLUMN_ORDER = [
    "id", "appeal_id",
    "text", "modified_text", "content_type",
    "true_toxicity", "correct_action",
    "noisy_toxicity_score", "confidence_level",
    "follower_bucket", "group", "is_adversarial",
    "user_id", "day",
    "violation_history", "last_action", "escalation_level",
    "original_action_taken", "should_reverse",
    "label",
]


# --- Main ---------------------------------------------------------------------

def main(input_path: str, output_path: str, n_rows: int) -> None:

    print(f"[1/7] Loading {input_path} ...")
    raw = pd.read_csv(input_path)
    print(f"      {len(raw)} rows loaded")

    # Sample if requested
    if n_rows and n_rows < len(raw):
        # Stratified: keep ratio of toxic to clean
        toxic = raw[raw["toxic"] == 1]
        clean = raw[raw["toxic"] == 0]
        n_toxic = min(len(toxic), n_rows // 2)
        n_clean = n_rows - n_toxic
        raw = pd.concat([
            toxic.sample(n_toxic, random_state=RNG_SEED),
            clean.sample(n_clean, random_state=RNG_SEED),
        ]).sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)
        print(f"      Sampled {len(raw)} rows ({n_toxic} toxic, {n_clean} clean)")

    # -- Step 1: Core text + id ------------------------------------------------
    print("[2/7] Building core columns ...")
    df = pd.DataFrame()
    df["id"]   = raw.index + 1
    df["text"] = raw["comment_text"].fillna("").astype(str)

    # -- Step 2: Derive toxicity signals --------------------------------------
    print("[3/7] Deriving toxicity signals ...")

    severity = raw.apply(derive_severity_score, axis=1).values
    df["true_toxicity"]        = raw.apply(derive_true_toxicity, axis=1).values.astype(int)
    df["correct_action"]       = [derive_correct_action(s) for s in severity]
    noisy                      = np.array([derive_noisy_score(s) for s in severity])
    df["noisy_toxicity_score"] = np.round(noisy, 4)
    df["confidence_level"]     = np.round(
        [derive_confidence(s, n) for s, n in zip(severity, noisy)], 4
    )

    # -- Step 3: Post metadata -------------------------------------------------
    print("[4/7] Assigning post metadata ...")
    df["content_type"]   = df["text"].apply(classify_content_type)
    df["follower_bucket"]= derive_follower_bucket(len(df)).astype(int)
    df["group"]          = derive_group(len(df))
    df["is_adversarial"] = 0        # base rows are not adversarial
    df["modified_text"]  = df["text"]   # same as text for base rows
    df["label"]          = df["true_toxicity"]   # binary ground truth alias

    # -- Step 4: Task 2 -- user timeline columns --------------------------------
    print("[5/7] Building user timelines (Task 2) ...")
    df = build_user_timelines(df, n_users=30)

    # -- Step 5: Task 4 -- appeals columns -------------------------------------
    print("[6/7] Building appeals columns (Task 4) ...")
    df = build_appeals_columns(df)

    # -- Step 6: Reorder to canonical schema -----------------------------------
    existing = [c for c in CANONICAL_COLUMN_ORDER if c in df.columns]
    df = df[existing]

    # -- Step 7: Save ----------------------------------------------------------
    print(f"[7/7] Saving to {output_path} ...")
    df.to_csv(output_path, index=False)

    # -- Sanity report ---------------------------------------------------------
    print()
    print("-- Sanity Checks ----------------------------------------------------")
    print(f"  Total rows              : {len(df)}")
    print(f"  Total NaNs              : {int(df.isnull().sum().sum())}  (must be 0)")
    print(f"  content_type classes    : {sorted(df['content_type'].unique())}")
    print(f"  true_toxicity classes   : {sorted(df['true_toxicity'].unique())}  (should be [0, 1])")
    print(f"  correct_action classes  : {sorted(df['correct_action'].unique())}  (should be [0, 1, 2])")
    print(f"  follower_bucket classes : {sorted(df['follower_bucket'].unique())}  (should be [0, 1, 2])")
    print(f"  group classes           : {sorted(df['group'].unique())}  (should be ['A', 'B'])")
    print(f"  is_adversarial classes  : {sorted(df['is_adversarial'].unique())}  (should be [0])")
    print(f"  unique user_ids         : {df['user_id'].nunique()}  (should be 31: 0 + 30 users)")
    print(f"  users with >2 posts     : {(df['user_id'][df['user_id']>0].value_counts() > 2).sum()}  (Task 2 needs > 0)")
    print(f"  should_reverse==1 rows  : {(df['should_reverse']==1).sum()}  (Task 4 appeals)")
    print(f"  max escalation_level    : {df['escalation_level'].max()}  (should be <= 3)")
    print()

    print("-- Column Distribution ----------------------------------------------")
    print(f"  correct_action breakdown:")
    for v, name in [(0, "allow"), (1, "warn"), (2, "remove")]:
        count = (df["correct_action"] == v).sum()
        print(f"    {v} ({name:6}) : {count:5}  ({count/len(df)*100:.1f}%)")

    print(f"\n  content_type breakdown:")
    for cat, cnt in df["content_type"].value_counts().items():
        print(f"    {cat:15} : {cnt:5}  ({cnt/len(df)*100:.1f}%)")

    print()
    print(f"[ok] Dataset saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate dataset_final.csv from raw Jigsaw CSV"
    )
    parser.add_argument(
        "--input",  default="server/data/jigsaw_train.csv",
        help="Path to raw Jigsaw train.csv"
    )
    parser.add_argument(
        "--output", default="server/data/dataset_final.csv",
        help="Path to write final dataset"
    )
    parser.add_argument(
        "--n_rows", type=int, default=1000,
        help="Number of rows to sample (0 = use all). Default: 1000"
    )
    args = parser.parse_args()
    main(args.input, args.output, args.n_rows)