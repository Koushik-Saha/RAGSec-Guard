#!/usr/bin/env python3
"""RAGSec Guard - Day 1 data prep (public sources only).

Builds data/guard_train.jsonl and data/guard_test.jsonl from:
  - deepset/prompt-injections        (0 -> safe, 1 -> injection)
  - SalutefaceAAC/prompt-injection-classification (benign -> safe, injection -> injection;
    rows whose attack_type is explicitly a jailbreak -> jailbreak)

The gated backbone (ankitkr225/agent-guard-open-data) and the RAGSec probe
export are merged by a later step; every row carries a `source` field.

Split: dedupe on normalized text (plus a prefix key to keep near-duplicates
on one side), stratified 85/15, seed 42. Leakage check asserts zero overlap.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

INSTRUCTION = (
    "Classify this prompt for a RAG system. Answer with one label - safe, "
    "injection, or jailbreak - then a one-line reason."
)
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


def fetch_rows(dataset, split):
    rows, offset = [], 0
    while True:
        r = requests.get(
            "https://datasets-server.huggingface.co/rows",
            params={"dataset": dataset, "config": "default", "split": split,
                    "offset": offset, "length": 100},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        batch = [item["row"] for item in payload["rows"]]
        rows.extend(batch)
        offset += len(batch)
        if offset >= payload["num_rows_total"] or not batch:
            return rows


def norm(text):
    return " ".join(str(text).split()).lower()


def one_line(text, cap=220):
    text = " ".join(str(text).split())
    return text[:cap].rstrip()


def make_row(text, label, reason, source, rng):
    if not reason:
        pool = {"safe": SAFE_REASONS, "injection": INJ_REASONS,
                "jailbreak": JB_REASONS}[label]
        reason = pool[rng.randrange(len(pool))]
    return {
        "instruction": INSTRUCTION,
        "input": one_line(text, cap=4000),
        "output": f"{label}\nReason: {one_line(reason)}",
        "label": label,
        "source": source,
    }


def main():
    rng = random.Random(SEED)
    pool = []

    # --- deepset/prompt-injections (train + test combined; we re-split) ---
    n_deepset = 0
    for split in ("train", "test"):
        for row in fetch_rows("deepset/prompt-injections", split):
            label = "injection" if int(row["label"]) == 1 else "safe"
            pool.append(make_row(row["text"], label, None, "deepset", rng))
            n_deepset += 1
    print(f"deepset rows: {n_deepset}")

    # --- SalutefaceAAC ---
    splits = requests.get(
        "https://datasets-server.huggingface.co/info",
        params={"dataset": "SalutefaceAAC/prompt-injection-classification"},
        timeout=60,
    ).json()["dataset_info"]["default"]["splits"].keys()
    n_sal = 0
    attack_types = Counter()
    for split in splits:
        for row in fetch_rows("SalutefaceAAC/prompt-injection-classification", split):
            raw = str(row["label"]).strip().lower()
            atype = str(row.get("attack_type") or "").strip().lower()
            attack_types[atype] += 1
            if raw == "benign":
                label = "safe"
            elif "jailbreak" in atype:
                label = "jailbreak"
            else:
                label = "injection"
            pool.append(make_row(row["text"], label,
                                 row.get("explanation"), "saluteface", rng))
            n_sal += 1
    print(f"saluteface rows: {n_sal}; attack_types: {dict(attack_types)}")

    print(f"pool before dedupe: {len(pool)}  {Counter(r['label'] for r in pool)}")

    # --- dedupe: exact normalized text, then prefix key (near-dup guard) ---
    seen, seen_prefix, deduped = set(), set(), []
    for r in pool:
        key = norm(r["input"])
        prefix = key[:80]
        if key in seen or prefix in seen_prefix:
            continue
        seen.add(key)
        seen_prefix.add(prefix)
        deduped.append(r)
    print(f"after dedupe: {len(deduped)}  {Counter(r['label'] for r in deduped)}")

    # --- balance: cap the majority classes at ~1.4x the smallest attack class ---
    by_label = defaultdict(list)
    for r in deduped:
        by_label[r["label"]].append(r)
    for rows in by_label.values():
        rng.shuffle(rows)
    counts = {k: len(v) for k, v in by_label.items()}
    attack_min = min(counts.get("injection", 0), counts.get("jailbreak", 0)) or counts.get("injection", 0)
    cap = max(int(attack_min * 1.4), 50) if attack_min else 10**9
    balanced = []
    for label, rows in by_label.items():
        take = rows if label == "jailbreak" else rows[:cap]
        balanced.extend(take)
        print(f"class {label}: kept {len(take)} of {len(rows)} (cap {cap})")

    # --- stratified 85/15 split ---
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

    # --- leakage check ---
    train_keys = {norm(r["input"]) for r in train}
    test_keys = {norm(r["input"]) for r in test}
    overlap = train_keys & test_keys
    assert not overlap, f"LEAKAGE: {len(overlap)} overlapping prompts"

    for name, rows in (("guard_train.jsonl", train), ("guard_test.jsonl", test)):
        with open(DATA / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows  {Counter(r['label'] for r in rows)}  "
              f"sources: {Counter(r['source'] for r in rows)}")
    print("leakage check: 0 overlapping prompts - PASS")


if __name__ == "__main__":
    sys.exit(main())
