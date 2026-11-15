# llm_eval_pipeline

**CI/CD for LLM applications.** A pull request that makes answer quality worse
gets blocked, the same way a failing unit test blocks one. Point it at any LLM
app — RAG, agent or plain prompt, in any language, over any storage — with one
config file and either a ~25-line adapter class or nothing at all if the app
already speaks HTTP. You never edit library code.

```
pip install "git+https://github.com/abishekraskr-hub/llm_eval_pipeline@v0.1.0"
```

## Quickstart

The bundled example needs no API key, no network and no database. It is also the
library's integration test, so if it breaks, CI goes red.

```bash
git clone https://github.com/abishekraskr-hub/llm_eval_pipeline
cd llm_eval_pipeline && pip install -e ".[dev]"

llmeval validate --config examples/echo_target/llmeval.yaml
llmeval run      --config examples/echo_target/llmeval.yaml --branch develop
```

```
  [PASS] Answered                         >= 1        actual 1.000
  [PASS] Faithfulness                     >= 0.75     actual 1.000
  [PASS] Abstention on unanswerable       >= 0.7      actual 1.000
  [PASS] False refusal on answerable      <= 0.2      actual 0.000
  [PASS] Retrieval hit rate@5             >= 0.7      actual 1.000
  GATE: PASSED
```

`examples/http_stub/` does the same thing over HTTP, against a standard-library
server whose field names are deliberately *not* the ones this library uses.

## What it measures

| Metric | What it catches |
|---|---|
`answered_rate` | the app returned nothing. Needs no judge, so it cannot itself go missing |
`faithfulness` | claims not supported by the retrieved passages |
`hallucination_rate` | derived from faithfulness, as a rate |
`answer_relevance` | fluent answers to a different question |
`abstention_rate` | inventing an answer instead of declining, on unanswerable questions |
`false_refusal_rate` | declining a question it *should* have answered |
`retrieval_hit_rate@k` | the search step missing the right source entirely |
`retrieval_recall@k` | finding some of the needed sources but not all |
`retrieval_mrr` | the right source ranked too low to be used |
`latency_p50` / `latency_p95` | the app getting slower |
`cost_per_query` | the app getting more expensive |

Two pairs exist because each half is gameable alone:

- **abstention vs. false refusal.** An app that refuses *everything* scores a
  perfect 1.000 abstention rate while being useless. Only the answerable side
  sees that.
- **relevance vs. faithfulness.** When retrieval broke completely in this
  project's own history, `answer_relevance` still scored **0.988** — the answers
  stayed fluent and on-topic, grounded in nothing. Only faithfulness going
  *unmeasurable* caught it.

## The target contract

Your app answers one question at a time and says what it retrieved. Only
`answer` is required; everything else degrades a specific metric rather than
breaking the run.

```python
{
  "answer": str,
  "retrieved": [            # None/absent = "this app does not expose retrieval"
    {
      "source_id": str,     # ANY stable id: filename, URL, doc id, DB key
      "page": int | None,   # optional
      "content": str,       # the text the app used -- faithfulness needs it
      "rank": int,          # 1 = best
    }
  ],
  "refused": bool | None,   # the app saying it declined. Beats any pattern
  "usage": {"input_tokens": int, "output_tokens": int} | None,
  "trace": [...] | None,    # agent steps, for debugging
  "latency_s": float,
  "model": str,
}
```

**`source_id` is the only thing that must be stable.** It is whatever your app
can name the thing it retrieved, and the golden dataset's gold labels are written
against it. A volatile chunk id would make every label rot on the next re-index.

**`page` is optional, deliberately.** A page is compared only when *both* the
gold label and the retrieved hit declare one. An app querying OpenSearch,
Postgres or the web omits it and matches on `source_id` alone.

### Option A — a Python adapter

```yaml
target:
  type: python
  entry: eval.my_target:MyTarget     # module:Class, relative to this config
```

A complete adapter for an app backed by **OpenSearch** — no documents, no pages:

```python
# eval/my_target.py
from myapp import answer_question, search          # your existing code

class MyTarget:
    def ask(self, question: str) -> dict:
        hits = search(question, size=5)            # whatever your app already does
        reply = answer_question(question, hits)
        return {
            "answer": reply.text,
            "retrieved": [
                {
                    "source_id": hit["_id"],       # the OpenSearch document id
                    "content": hit["_source"]["body"],
                    "rank": i + 1,
                }                                  # no `page` -- there are none
                for i, hit in enumerate(hits)
            ],
            "usage": {
                "input_tokens": reply.in_tokens,
                "output_tokens": reply.out_tokens,
            },
        }

    def describe(self) -> dict:
        return {
            "name": "support-bot",
            "model": "gpt-4o-mini",
            "supports": {"retrieval": True},
        }
```

Gold labels are then `{ source_id: "a1b2c3" }` with no page.

### Option B — no adapter at all

If the app answers an HTTP request, map its field names and you are done. It does
not have to be Python, and it needs no knowledge that an evaluator exists:

```yaml
target:
  type: http
  url: http://localhost:8000/chat
  headers: { Authorization: "Bearer ${MYAPP_TOKEN}" }   # ${ENV} only; a literal key is refused
  request:
    question_field: query          # sends {"query": "..."}
  response:                        # dot-paths into the JSON reply
    answer: reply
    retrieved: sources
    source_id: url
    content: text
    refused: meta.refused
  start:                           # optional: launch it, wait for health, stop it after
    command: "python app_server.py"
    health_url: http://localhost:8000/health
```

Streaming responses are not supported: the evaluator needs the whole answer and
its passages before it can score anything. Point it at the non-streaming
endpoint.

## Config reference

Everything app-specific lives in `llmeval.yaml`. **Every relative path inside it
resolves from its own folder**, never from the current working directory — so a
config that works locally works identically in CI.

```yaml
version: 1

target:          # see above
dataset: eval/golden.yaml

refusal:
  # How a refusal is recognised. This is yours, not the library's: it cannot know
  # how your app phrases "I don't know", or what language it answers in.
  use_target_flag: true          # trust the app's own `refused` flag
  patterns:                      # otherwise, case-insensitive regexes
    - "the documents? (do not|don't) (cover|contain)"

judge:
  provider: groq                 # groq | gemini | fake
  model: openai/gpt-oss-120b
  api_key_env: GROQ_API_KEY      # the NAME of an env var, never the key
  requests_per_minute: 12
  metrics: { faithfulness: true, answer_relevancy: true }

storage:
  backend: postgres              # postgres | sqlite
  sqlite_path: .llmeval/history.sqlite
  reports: .llmeval/reports
  postgres:
    host: localhost
    dbname: llm_eval_pipeline
    user: postgres
    password_env: EVAL_PG_PASSWORD   # a password in this file is REFUSED
    create_database: true

retrieval:
  k: 5                           # part of the metric NAME: retrieval_hit_rate@5
  page_tolerance: 0

paired:
  enabled: true
  base_branch: auto              # auto = the PR's target branch in CI
  alpha: 0.05
  min_flips: 2                   # binary: net flips needed
  min_delta: 0.03                # continuous: mean drop needed
  bootstrap_resamples: 10000
  seed: 42                       # fixed, so the gate cannot change its mind
  min_pair_coverage: 0.8

thresholds:
  faithfulness:         { floor: 0.75, type: continuous, direction: higher_is_better }
  retrieval_hit_rate@k: { floor: 0.70, type: binary,     direction: higher_is_better }
  false_refusal_rate:   { floor: 0.20, type: binary,     direction: lower_is_better }
  latency_p95: 20.0              # a bare number is the floor

require_all_metrics: true        # a declared metric that cannot be measured FAILS
min_metric_coverage: 0.9         # ...and so does one measured on only part of the data
```

Each metric's **direction is declared as data**, never inferred from its name.
Getting a direction wrong silently inverts the gate — it would pass exactly the
runs it exists to block — so it is stated, not guessed.

`llmeval validate` checks all of this, loads the dataset *and* constructs the
target, and calls no model. Every error names the exact field.

### Why those last two lines matter

`require_all_metrics` turns "we could not measure this" into a red build. It is
what caught a real regression where retrieval was deleted and relevance stayed at
0.988.

`min_metric_coverage` is the sharper version, added after it turned out the first
rule was too coarse. It asks "was this measured *at all*", so one item measuring
satisfied it — and because unmeasured items are *excluded from the mean*, a run
that returned an empty answer for one question in five reported faithfulness
**1.000** and passed every quality check. Partial coverage now fails, because an
average over the items that happened to work is not a measurement of the dataset.

## The paired gate

A fixed threshold on a 20–100 question benchmark either flaps or sits so low it
catches nothing: judges are stochastic, and on 20 questions one item flipping is
worth 5%.

So the real work is done by comparing **the same questions** against the last
passing run on the base branch:

- **binary** metrics → exact McNemar test on the flips
- **continuous** metrics → seeded paired bootstrap of the mean difference

Both also need an effect-size floor (`min_flips`, `min_delta`), so a
technically-significant two-item wobble does not block a merge.

**A baseline must be comparable, or there is no baseline.** It has to be a
passing run, on the base branch, with the same dataset hash *and* the same
comparability hash. When nothing comparable exists — a first run, an edited
dataset, a new judge, a library upgrade — the paired test is **skipped and the
report says so**. That is not a failure; the floors carry the gate for that run.

**One hard limit to know before trusting it.** With a two-sided exact test at
p = 0.5, **six** questions must flip the same way before a binary regression is
significant at all — five gives p = 0.0625. On a 20-question run, a binary
regression smaller than that is undetectable at any setting. Only a bigger
dataset changes that. The continuous metrics are far more sensitive, because they
use the size of each change rather than only its direction.

## GitHub Action

```yaml
name: eval
on:
  pull_request: { branches: [develop] }
  push:        { branches: [develop] }

permissions:
  contents: read
  actions: read            # to download the baseline
  pull-requests: write     # to post the report comment

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - uses: abishekraskr-hub/llm_eval_pipeline@v0.1.0
        with:
          config: llmeval.yaml
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

It downloads the baseline from the last successful run on the base branch, posts
**one** PR comment and updates it in place on every push, fills the job summary,
and uploads the new history as the next baseline — but only from a push to the
base branch, never from a PR. A PR's own numbers must not become the baseline
other PRs are judged against.

If the judge's key is missing (a fork PR cannot see secrets) it emits a warning
and exits 3 rather than passing silently. A green check on an evaluation that
never ran is worse than no check at all.

## Exit codes

| Code | Meaning | Blocks a merge? |
|---|---|---|
`0` | passed | no |
`1` | the quality gate failed | **yes** |
`2` | invalid config or usage | **yes** — not about quality, but still broken |
`3` | could not complete: quota, outage, missing key | no, warns |

`3` never blocks, on purpose. Treating a quota wall as a quality failure is how a
team learns to ignore a red pipeline.

## Versioning

Releases are **git tags** (`v0.1.0`, `v0.2.0`, …). There is no `main`; `develop`
is the default branch.

Always pin a tag, never a branch:

```
llm_eval_pipeline @ git+https://github.com/abishekraskr-hub/llm_eval_pipeline@v0.1.0
```

**The library version is part of the comparability hash.** An upgrade can change
how a metric is computed, so after a bump there is no comparable baseline and the
paired test is skipped for one run while a new one is established. That is
intentional: comparing numbers produced by different code is worse than not
comparing at all. Bump the tag in `requirements.txt` **and** in the workflow
together.

## Judge providers

| Provider | Notes |
|---|---|
`groq` | the default. Keep the judge on a *different* provider from the app, so grading never competes with the app for quota and no model grades its own output |
`gemini` | via the OpenAI-compatible endpoint |
`fake` | no network at all, deterministic. What CI uses |

Adding one is a new file in `judges/` plus a line in `REGISTRY`.

## Development

```bash
pip install -e ".[dev]"
pytest -q                     # fully offline
```

The live-Postgres tests skip themselves when no server answers, so the suite runs
on a machine that has never had Postgres on it. `EVAL_SKIP_PG_TESTS=1` forces
that, which is what CI sets.

One architectural rule is asserted rather than trusted:
`tests/test_architecture.py` fails the build if any library module imports an
application. That is the property which keeps this package reusable.

## Licence

MIT.
