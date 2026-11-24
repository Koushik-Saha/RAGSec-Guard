# RAGSec Guard — writeup

**Problem.** A RAG system takes untrusted text twice: the user's prompt and
the retrieved documents. Either can carry a prompt injection — an instruction
smuggled in to override the system prompt, leak it, or steer the answer. My
earlier project, RAGSec Scanner, tests RAG systems for exactly these attacks
from the outside. RAGSec Guard is the defensive counterpart: a small model
that sits in front of the pipeline and labels each incoming prompt
**safe**, **injection**, or **jailbreak**, with a one-line reason a human or
a downstream policy can act on.

**Why fine-tune?** The baseline experiment is the argument. Asked to do this
classification zero-shot, Qwen2.5-7B-Instruct scores 53.7% accuracy on the
held-out set and recalls 1.5% of injections — it labels almost everything
"safe," because nothing in its general instruction tuning makes it treat
manipulation detection as the job. A guard that misses 98.5% of attacks is
not a guard. The classification behavior has to be trained in.

## Data

Three public sources, merged and rebuilt rather than used as-is:

- `ankitkr225/agent-guard-open-data` — benign, prompt_injection, jailbreak
  (its `harmful_request` class, 10,473 rows, was parked unused: refusing
  harmful requests is the base model's alignment job; this guard targets
  manipulation of the system, a different failure mode).
- `deepset/prompt-injections` — 662 rows.
- A SalutefaceAAC prompt-injection set — 2,564 rows.

Plus 15 rows exported from RAGSec Scanner's probe set (train split only) —
realistic RAG-specific attacks such as poisoned FAQ passages and
cross-document leakage questions, which generic injection datasets lack.

Pipeline (`scripts/`): normalize to one schema
(`instruction / input / output / label / source`), balance the three
classes, deduplicate exactly and by 80-character prefix, stratified 85/15
split with seed 42, then a leakage check that no test row's text appears in
train. Final: **5,200 train** (1,879 safe / 1,876 injection / 1,445
jailbreak) and **915 test** rows.

## Training

QLoRA on the 4-bit base (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`): r=16,
alpha=16, dropout 0, target modules q/k/v/o/gate/up/down — 40.4M trainable
parameters (0.53% of the model). Two epochs, effective batch 8
(per-device 2 × grad-accumulation 4), max sequence 1024 with packing
(5,200 rows → 912 packed sequences), 228 steps, final training loss ≈ 1.165.
Trained on Kaggle's free tier (Tesla T4), about 1.5 hours of GPU time; the
whole project cost $0.

## Evaluation

One harness for both models, so the comparison is clean: greedy decoding,
batch 16, 48 new tokens, the label parsed from the first output line, the
rest of the line kept as the reason. Same 915 held-out rows for base and
tuned.

| | Base | Tuned |
|---|---|---|
| Accuracy | 0.537 | **0.924** |
| Injection — precision / recall / F1 | 0.333 / 0.015 / 0.029 | 0.934 / 0.948 / **0.941** |
| Jailbreak — precision / recall / F1 | 0.575 / 0.690 / 0.627 | 0.905 / 0.933 / **0.919** |
| Safe — precision / recall / F1 | 0.524 / 0.939 / 0.672 | 0.927 / 0.891 / **0.909** |
| False alarms on safe | 6.1% | 10.9% |

Confusion detail for the tuned model: of 330 injections, 313 caught, 13
passed as safe, 4 called jailbreak; of 255 jailbreaks, 238 caught; of 330
safe prompts, 294 passed, 36 flagged (15 as injection, 21 as jailbreak).

**Reading the trade-off honestly.** The tuned model's false-alarm rate
(10.9%) is *higher* than the base model's (6.1%) — and that comparison is
nearly meaningless, because the base model achieves it by flagging almost
nothing (1.5% injection recall). The tuned model flags more because it
flags, full stop. Still, 36 false alarms per 330 safe prompts is the v1's
real weakness: in production that is user friction, and it is where a v2
should spend its effort (a borderline-safe training slice and probability
calibration before thresholding, rather than a hard three-way label).

## What went wrong during the runs (kept in on purpose)

Two bugs, both mine, both in the notebook rather than the model:

1. The first committed run died in 16 seconds: a `%%capture` cell magic sat
   below a comment line, and cell magics must be the first line of a cell.
   No GPU time was meaningfully burned; the fix was deleting the line.
2. The successful training run then "failed" in its *final* cell: the
   results dict carried a confusion matrix keyed by `(gold, predicted)`
   tuples, which JSON cannot serialize, so `eval_results.json` was never
   written and the tuned per-class scores were lost. The adapter and both
   accuracy prints survived. Recovery was an eval-only notebook that
   re-scored the tuned model against the saved adapter (~9 minutes) and
   regenerated the complete results file — which is why the tuned accuracy
   appears as 0.9224 in the original run's log and 0.9235 in the final
   results file (a one-row difference on an identical model and test set).

Both are mundane, and both are the kind of thing that only surfaces when a
pipeline is actually run end to end — the reason the eval harness, not the
training loop, got the recovery notebook.

## Limitations

- One seed, one run: no variance estimate on any number above.
- English only; no multilingual or obfuscated-encoding (base64, homoglyph)
  slices in the test set yet.
- The test set comes from the same public sources as train (disjoint rows,
  decontaminated) — an external benchmark pass is the next credibility step.
- A prompt classifier is one layer of a defense, not the defense: pair with
  output-side leak checks and least-privilege tool design.

## Next

1. Cut false alarms: borderline-safe data slice + calibration.
2. External eval on an independent injection benchmark, reported alongside.
3. Wire the guard into the RAGSec Scanner demo path — scan, then guard,
   then re-scan — so the two projects compose into one story: measure the
   attacks, then block them.
