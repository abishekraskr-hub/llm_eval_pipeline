"""A judge that costs nothing, for exercising the pipeline without spending quota.

`--stub` replaces the APP under test. It does not replace the judge, so a stub
run still makes two real LLM calls per question. On a free tier that is the
entire day's budget spent on plumbing.

This replaces the judge instead. Combined with `--stub` the whole run is free and
offline, which is what makes it possible to check the parts that have nothing to
do with model quality -- does Postgres accept the rows, do the per-branch views
appear, does the gate wire up, does the report render -- before spending anything
on the parts that do.

Two safeguards, because a fabricated number that reaches a dashboard is worse
than no number at all:

* every score is visibly synthetic and derived from the text length, not from
  any judgement of quality;
* the run's judge identity is tagged `dryrun`, and baseline selection requires a
  matching `judge_config_hash`. So a dry run can never be the baseline for a
  real run, and a real run can never be the baseline for a dry run. The paired
  test will say "no comparable baseline", which is the truth.
"""
from ..metrics import ItemScores

DRY_RUN_TAG = "dryrun"


class DryRunJudge:
    """Satisfies the two things the runner asks of a judge, and makes no calls."""

    def __init__(self, name: str = f"{DRY_RUN_TAG}:no-model"):
        self._name = name
        self.calls = 0

    def get_model_name(self) -> str:
        return self._name


class DryRunScorer:
    """Deterministic pseudo-scores. Never a measurement of anything.

    The values vary a little between questions so the reports and the per-item
    tables are not uniformly identical -- uniform columns hide bugs in the
    pairing code, which joins on item_id and would look fine if every value
    matched. They are derived from the answer's length, which is not a quality
    signal and is not meant to be read as one.
    """

    def __init__(self, judge: DryRunJudge | None = None):
        self.judge = judge or DryRunJudge()

    def score(self, question: str, answer: str, contexts: list[str]) -> ItemScores:
        if not contexts:
            # What a real Scorer returns when the target retrieved nothing:
            # faithfulness is UNMEASURABLE, not zero. The gate must fail on this
            # under require_all_metrics, and a dry run has to reproduce it or it
            # would not be exercising the interesting path.
            return ItemScores(
                faithfulness=None,
                answer_relevancy=None,
                errors={"faithfulness": "dry run: no contexts to check against"},
            )
        spread = (len(answer or "") % 7) / 100.0  # 0.00 .. 0.06
        return ItemScores(
            faithfulness=round(0.90 + spread, 4),
            answer_relevancy=round(0.92 + spread, 4),
            reasons={
                "faithfulness": "dry run: synthetic score, no judge was called",
                "answer_relevancy": "dry run: synthetic score, no judge was called",
            },
        )
