#!/usr/bin/env python3
"""
prepare_dataset.py — Patch dataset_final.csv with correct content_type labels
==========================================================================

Run from repo root:
    python prepare_dataset.py

Or with custom paths:
    python prepare_dataset.py --input server/data/dataset_final.csv --output server/data/dataset_final.csv

What this fixes
───────────────
1. content_type — replaces the naive first-match keyword approach with a
   WEIGHTED SCORING classifier.  Every category has multiple semantic anchor
   groups; the category with the highest total score wins.
2. NaN values — forward-filled / default-filled so the CSV is clean.
3. Re-orders columns to the canonical schema.

Output: overwrites the input file (or writes to --output if specified).
"""

import argparse
import re
import pandas as pd

# ─── Weighted Scoring Classifier ─────────────────────────────────────────────
# Structure: { category: [(weight, regex_pattern), ...] }
# A single post gets score[cat] = sum of weights for every pattern that matches.
# Ties broken by priority order (hate_speech > threat > sexual > political > ...).

CATEGORY_PATTERNS: dict[str, list[tuple[int, str]]] = {

    # ── Hate Speech ────────────────────────────────────────────────────────────
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

    # ── Threat / Violence ──────────────────────────────────────────────────────
    "threat": [
        (5, r"\b(kill you|murder you|i.ll.kill|gonna.kill|want.to.kill|shoot.you|stab.you|death.threat)\b"),
        (4, r"\b(kill|murder|shoot|stab|bomb|blow.?up|execute|assassinate|slaughter)\b"),
        (3, r"\b(gun|knife|weapon|explosive|grenade|attack|threaten|harm.you|hurt.you)\b"),
        (2, r"\b(violence|blood|die|dead|end.you|destroy.you|wipe.out|come.for.you)\b"),
        (1, r"\b(beat.up|punch|smash|crush|torture)\b"),
    ],

    # ── Sexual Content ─────────────────────────────────────────────────────────
    "sexual": [
        (5, r"\b(porn|pornography|nude|naked|masturbat|orgasm|cum|blowjob|handjob|dildo)\b"),
        (4, r"\b(fuck|fucking|fucked|dick|cock|pussy|ass.?hole|tits|boobs|penis|vagina)\b"),
        (3, r"\b(rape|sexual.assault|molest|grope|harass.sexually|sex.with|have.sex)\b"),
        (2, r"\b(horny|aroused|erect|strip|panties|underwear|seduce|lust|kinky|fetish)\b"),
        (1, r"\b(sex|sexy|sexual|sensual|intimate|hook.?up|one.night.stand)\b"),
    ],

    # ── Political / Ideological ────────────────────────────────────────────────
    "political": [
        (4, r"\b(election|president|congress|senate|parliament|democrat|republican|socialist)\b"),
        (4, r"\b(isis|isil|al.qaeda|jihad|terrorism|radical.islam|extremist|insurgent)\b"),
        (3, r"\b(trump|obama|biden|hillary|pelosi|bernie|putin|boris|macron)\b"),
        (3, r"\b(immigration|border.wall|deportation|refugee|asylum.seeker|illegal.alien)\b"),
        (2, r"\b(liberal|conservative|left.wing|right.wing|fascist|communist|antifa|proud.boy)\b"),
        (2, r"\b(vote|voting|ballot|rigged|stolen.election|conspiracy|deep.state)\b"),
        (1, r"\b(policy|law|bill|government|politics|protest|activist|revolution)\b"),
    ],

    # ── Humor / Meme ──────────────────────────────────────────────────────────
    "humor": [
        (4, r"\b(lol|lmao|lmfao|rofl|haha|hehe|xd)\b"),
        (3, r"\b(joke|meme|funny|hilarious|comedy|punchline|prank|troll)\b"),
        (3, r"\b(when you|that moment when|me irl|nobody:|no one:|them:|boss:|teacher:)\b"),
        (2, r"\b(sarcas|irony|ironic|satire|deadpan|dark.humor|dry.humor)\b"),
        (2, r"\b(imagine if|plot twist|unpopular opinion|fun fact|shower thought)\b"),
        (1, r"\b(wait|imagine|meanwhile|basically|literally|actually.though)\b"),
    ],

    # ── Personal / Experiential ────────────────────────────────────────────────
    "personal": [
        (4, r"\b(i am|i'm|i was|i feel|i love|i hate|i need|i want|i think|i believe)\b"),
        (3, r"\b(my (life|story|friend|family|mom|dad|sister|brother|boyfriend|girlfriend))\b"),
        (3, r"\b(confession|personal|vent|diary|opened up|honest(ly)?|true story)\b"),
        (2, r"\b(relationship|breakup|dating|marriage|divorce|heartbreak|toxic.person)\b"),
        (2, r"\b(anxiety|depression|mental.health|therapy|trauma|abuse|struggle)\b"),
        (1, r"\b(today i|yesterday i|last (week|night|year) i|when i was)\b"),
    ],

    # ── News / Information ─────────────────────────────────────────────────────
    "news_info": [
        (4, r"\b(study|research|report|scientists?|university|published|findings|data.shows)\b"),
        (4, r"\b(breaking.news|headline|journalist|press|media|according.to|sources.say)\b"),
        (3, r"\b(percent|statistics?|million|billion|survey|poll|results.show)\b"),
        (2, r"\b(expert|official|government.says|health.officials?|cdc|who|fbi|cia)\b"),
        (1, r"\b(discovered|revealed|confirmed|announced|reports?|claims?|alleges?)\b"),
    ],
}

# Priority order for tie-breaking (leftmost wins ties)
PRIORITY_ORDER = [
    "hate_speech", "threat", "sexual", "political",
    "humor", "personal", "news_info", "general",
]


def classify_content_type(text: str) -> str:
    """
    Weighted multi-pattern scoring classifier.
    Returns the category with the highest total score.
    Falls back to 'general' if no patterns match.
    """
    t = text.lower()
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


# ─── NaN Filler Defaults ──────────────────────────────────────────────────────
# NOTE: user_id and day are included here as safety fallbacks.
# Your CSV already has valid values for these, but this protects
# against any future data pipeline that might introduce NaNs.

NAN_DEFAULTS = {
    "user_id":               0,
    "day":                   0,
    "violation_history":     0,
    "last_action":           0,
    "escalation_level":      0,
    "original_action_taken": 0,
    "should_reverse":        0,
    "is_adversarial":        0,
    "follower_bucket":       0,
    "group":                 "A",
    "label":                 0,
    "modified_text":         "",
}

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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(input_path: str, output_path: str) -> None:
    print(f"[1/5] Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"      {len(df)} rows, {len(df.columns)} columns")

    # ── Step 1: Fix NaN values ────────────────────────────────────────────────
    print("[2/5] Fixing NaN values ...")
    total_nans_before = int(df.isnull().sum().sum())
    print(f"      NaNs before: {total_nans_before}")

    for col, default in NAN_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # modified_text: if empty/NaN, copy from text (no adversarial transform)
    if "modified_text" in df.columns:
        mask = df["modified_text"].isna() | (df["modified_text"] == "")
        df.loc[mask, "modified_text"] = df.loc[mask, "text"]

    print(f"      NaNs after:  {int(df.isnull().sum().sum())}")

    # ── Step 2: Reclassify content_type ──────────────────────────────────────
    print("[3/5] Reclassifying content_type with weighted scoring ...")
    old_dist = df["content_type"].value_counts().to_dict()
    df["content_type"] = df["text"].apply(classify_content_type)
    new_dist = df["content_type"].value_counts().to_dict()

    print("      OLD distribution:")
    for k, v in sorted(old_dist.items(), key=lambda x: -x[1]):
        print(f"        {k:15} {v:4}")
    print("      NEW distribution:")
    for k, v in sorted(new_dist.items(), key=lambda x: -x[1]):
        print(f"        {k:15} {v:4}")

    # ── Step 3: Spot-check toxic → general misclassifications ────────────────
    print("[4/5] Spot-check: toxic posts still labeled 'general' ...")
    toxic_general = df[(df["content_type"] == "general") & (df["true_toxicity"] == 1)]
    print(f"      Toxic posts still in 'general': {len(toxic_general)} "
          f"(was {old_dist.get('general', 0)} total general before)")
    if len(toxic_general) > 0:
        print("      Sample of remaining 'general' toxic posts:")
        for _, r in toxic_general.head(5).iterrows():
            print(f"        [{r['id']}] {r['text'][:70]!r}")

    # ── Step 4: Reorder columns ───────────────────────────────────────────────
    existing_cols = [c for c in CANONICAL_COLUMN_ORDER if c in df.columns]
    extra_cols    = [c for c in df.columns if c not in CANONICAL_COLUMN_ORDER]
    df = df[existing_cols + extra_cols]

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    print(f"[5/5] Saving to {output_path} ...")
    df.to_csv(output_path, index=False)
    print(f"      Done. {len(df)} rows, {len(df.columns)} columns.")

    # ── Final sanity report ───────────────────────────────────────────────────
    print()
    print("── Final sanity checks ─────────────────────────────────────────────")
    print(f"  Total rows:             {len(df)}")
    print(f"  Total NaNs:             {int(df.isnull().sum().sum())} (must be 0)")
    print(f"  content_type classes:   {sorted(df['content_type'].unique())}")
    print(f"  correct_action classes: {sorted(df['correct_action'].unique())} (should be [0,1,2])")
    print(f"  true_toxicity classes:  {sorted(df['true_toxicity'].unique())} (should be [0,1])")
    print(f"  user_id unique users:   {df['user_id'].nunique()}")
    print(f"  users with >2 posts:    {(df['user_id'].value_counts() > 2).sum()} (Task 2 needs > 0)")
    print(f"  appeal_id rows:         {len(df)}")
    print()
    print("Dataset is ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default paths match the actual repo structure: data/ lives inside server/
    parser.add_argument("--input",  default="server/data/dataset_final.csv")
    parser.add_argument("--output", default="server/data/dataset_final_1000.csv")
    args = parser.parse_args()
    main(args.input, args.output)