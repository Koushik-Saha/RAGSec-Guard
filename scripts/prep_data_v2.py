#!/usr/bin/env python3
"""RAGSec Guard - Day 1 data prep v2: merge the agent-guard backbone.

Pool = existing deepset+saluteface rows (from guard_train/guard_test v1)
     + ankitkr225/agent-guard-open-data train+validation parquet.
Labels: benign -> safe, prompt_injection -> injection, jailbreak -> jailbreak.
harmful_request rows are parked in data/harmful_request_parked.jsonl (v2 class).
agent-guard's own test.parquet is left untouched as an external eval set.

Rebalanced to ~2,200 safe / ~2,200 injection / ~1,700 jailbreak, deduped
(exact + prefix), stratified 85/15 split, seed 42, leakage asserted.
Run with the project venv (needs pyarrow).
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "agent-guard"
SEED = 42

SAFE_REASONS = [
    "ordinary request; nothing tries to override the system instructions.",
    "a normal question or task with no hidden instruction for the assistant.",
    "no attempt to change the assistant's rules or reveal its prompt.",
]
INJ_REASONS = [
    "tries to override the system instructions and make the assistant follow new ones.",
    "attempts to make the assistant ignore its original instructions.",
    "embeds an instruction meant to hijack the assistant's behavior.",
]
JB_REASONS = [
    "tries to bypass the assistant's safety rules with a persona or framing trick.",
    "uses role-play or a fictional frame to get around the assistant's restrictions.",
]
REASON_POOL = {"safe": SAFE_REASONS, "injection": INJ_REASONS, "jailbreak": JB_REASONS}
LABEL_MAP = {"benign": "safe", "prompt_injection": "injection", "jailbreak": "jailbreak"}
TARGETS = {"safe": 2200, "injection": 2200, "jailbreak": 1700}
INSTRUCTION = (
    "Classify this prompt for a RAG system. Answer with one label - safe, "
    "injection, or jailbreak - then a one-line reason."
)


def norm(text):
    return " ".join(str(text).split()).lower()


def main():
    rng = random.Random(SEED)
    pool = []

    # existing public-source rows (v1 files)
    for name in ("guard_train.jsonl", "guard_test.jsonl"):
        path = DATA / name
        if path.exists():
            for line in open(path):
                pool.append(json.loads(line))
    print(f"v1 rows carried in: {len(pool)}  {Counter(r['label'] for r in pool)}")

    # agent-guard backbone
    parked = []
    n_ag = 0
    for split in ("train", "validation"):
        table = pq.read_table(RAW / f"{split}.parquet")
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        for text, label in zip(cols["text"], cols["label"]):
            raw = str(label).strip().lower()
            if raw == "harmful_request":
                parked.append({"input": " ".join(str(text).split())[:4000],
                               "label": "harmful_request", "source": "agent-guard"})
                continue
            mapped = LABEL_MAP.get(raw)
            if mapped is None or not str(text).strip():
                continue
            reason = REASON_POOL[mapped][rng.randrange(len(REASON_POOL[mapped]))]
            pool.append({
                "instruction": INSTRUCTION,
                "input": " ".join(str(text).split())[:4000],
                "output": f"{mapped}\nReason: {reason}",
                "label": mapped,
                "source": "agent-guard",
            })
            n_ag += 1
    print(f"agent-guard rows added: {n_ag}; harmful_request parked: {len(parked)}")
    with open(DATA / "harmful_request_parked.jsonl", "w") as f:
        for r in parked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # dedupe: exact normalized text + prefix key
    seen, seen_prefix, deduped = set(), set(), []
    for r in pool:
        key = norm(r["input"])
        if key in seen or key[:80] in seen_prefix:
            continue
        seen.add(key)
        seen_prefix.add(key[:80])
        deduped.append(r)
    print(f"after dedupe: {len(deduped)}  {Counter(r['label'] for r in deduped)}")

    # balance to targets (keep all rows of any class below target)
    by_label = defaultdict(list)
    for r in deduped:
        by_label[r["label"]].append(r)
    balanced = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        take = rows[:TARGETS.get(label, len(rows))]
        balanced.extend(take)
        print(f"class {label}: kept {len(take)} of {len(rows)} (target {TARGETS.get(label)})")

    # stratified 85/15 split
    by_label = defaultdict(list)
    for r in balanced:
        by_label[r["label"]].append(r)
    train, test = [], []
    for rows in by_label.values():
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * 0.15))
        test.extend(rows[:cut])
        train.extend(rows[cut:])
    rng.shuffle(train)
    rng.shuffle(test)

    overlap = {norm(r["input"]) for r in train} & {norm(r["input"]) for r in test}
    assert not overlap, f"LEAKAGE: {len(overlap)}"

    for name, rows in (("guard_train.jsonl", train), ("guard_test.jsonl", test)):
        with open(DATA / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows  {Counter(r['label'] for r in rows)}  "
              f"sources: {Counter(r['source'] for r in rows)}")
    print("leakage check: 0 overlapping prompts - PASS")


if __name__ == "__main__":
    main()
