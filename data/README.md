# Data

The exact files used for the reported run.

- `guard_train.jsonl` — 5,200 rows (1,879 safe / 1,876 injection / 1,445 jailbreak)
- `guard_test.jsonl` — 915 held-out rows, never trained on

Schema per row: `instruction`, `input`, `output` (label + one-line reason),
`label`, `source`.

Sources: `ankitkr225/agent-guard-open-data` (benign / prompt_injection /
jailbreak classes only), `deepset/prompt-injections`, a SalutefaceAAC
injection set, and 15 RAGSec Scanner probes (train only). Merged, balanced,
deduplicated (exact + 80-char prefix), stratified 85/15 split, seed 42,
leakage-checked. Build pipeline: `../scripts/`.
