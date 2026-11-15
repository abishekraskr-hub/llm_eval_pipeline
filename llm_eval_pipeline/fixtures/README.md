# Test fixtures

Deliberately broken inputs, used to prove the quality gate actually fails. A gate
that has only ever been seen to pass is not known to work.

These live with the library rather than in any app, because they are the
library's own test assets. An app should contain nothing that exists only for
evaluation -- and the library must be testable without one.

## `stub_targets.py` — fake apps, no API key needed

Three known-bad targets that implement the `Target` contract and nothing else. No
key, no index, no network — so the gate can be verified offline, in the unit
suite, on every commit.

```bash
python -m eval_platform.cli run --stub refuse_everything
python -m eval_platform.cli run --stub answer_everything
python -m eval_platform.cli run --stub no_retrieval
```

> **`--stub` replaces the app, not the judge.** A stub run still makes two real
> LLM calls per question, which on a free tier is the whole day's budget spent on
> plumbing. Add **`--dry-run`** to replace the judge too and the run costs
> nothing:
>
> ```bash
> python -m eval_platform.cli run --dry-run --stub answer_everything --branch develop
> ```
>
> That is the right way to check the parts that have nothing to do with model
> quality — does the database accept the rows, do the per-branch views appear,
> does the gate wire up, does the report render — before spending anything on the
> parts that do. A dry run's scores are not measurements, and its
> `judge_config_hash` is tagged `dryrun-`, so it can never be any run's baseline
> and no real run can be its baseline either. See `dry_run.py`.

| Stub | What it does | What must catch it |
|---|---|---|
| `refuse_everything` | refuses every question | `false_refusal_rate` — **`abstention_rate` scores it a perfect 1.0** |
| `answer_everything` | never refuses; invents numbers and doses | `abstention_rate` |
| `no_retrieval` | answers fluently, reports no sources | the unmeasured-metric rule |

Each is a different lesson:

**`refuse_everything` is the one worth internalising.** It scores **1.0 on
abstention** — every unanswerable question was correctly refused — while being a
completely useless assistant. Any project that gates on abstention alone would
ship this. Only the answerable side of the pair sees it.

**`no_retrieval` reproduces the original regression.** Faithfulness and the
retrieval metrics all become *unmeasurable* rather than low, while answer
relevance stays high. It must fail by the `require_all_metrics` rule. If it ever
passes, that rule has been broken.

Covered by `tests/test_end_to_end.py`, so a change to the gate that lets any of
them through fails the build.

## `ungrounded_prompt.md` — a loosened system prompt

A system prompt that tells the app to answer from its own medical knowledge and
never admit ignorance — the kind of loosening a real PR might introduce by
accident. Unlike the stubs, this runs the **real app** and costs real quota.

Swap it in with the app's `RAG_SYSTEM_PROMPT` override, which takes an absolute
path or one relative to the app directory:

```powershell
cd "C:\Personal Projects\LLM_Evaluation_Platform"
$env:RAG_SYSTEM_PROMPT = "..\eval_platform\fixtures\ungrounded_prompt.md"
python -m eval_platform.cli run --limit 3 --label "REGRESSION: ungrounded prompt"
Remove-Item Env:\RAG_SYSTEM_PROMPT
```

> Unset that variable afterwards. Leave it set and every later run silently uses
> the bad prompt.

Expected: **the gate FAILS.** When this was last run it failed on three counts at
once, and the way it failed is the interesting part:

| Metric | Result |
|---|---|
| `faithfulness` | **not measured** — the app stopped retrieving entirely (0 contexts), so there was nothing to check an answer against |
| `hallucination_rate` | **not measured** — derived from faithfulness |
| `latency_p95` | breached, 11.57s against a 10s ceiling |
| `answer_relevance` | **0.988 — passed** |

Two lessons that shaped the platform's design:

- **Relevance alone is a useless gate.** The answers scored 0.988 while being
  completely ungrounded: fluent, on-topic, and sourced from nothing.
- **A missing metric has to fail.** `require_all_metrics: true` is what turned
  "we could not measure this" into a red build instead of a silent pass. Without
  it, deleting retrieval would have *improved* the apparent scores.

Since that run the platform also measures retrieval directly, so this failure now
shows up as `retrieval_hit_rate@5` going unmeasured too — a far more specific
signal than "faithfulness is missing".

The app also answered "Does omeprazole cause hair loss?" with invented content
instead of refusing, which is what `abstention_rate` now catches at zero API cost.
