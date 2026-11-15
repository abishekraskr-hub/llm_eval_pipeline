"""Run a golden dataset against a target, score it, gate it, store it.

The order matters: every question is asked and scored before the gate runs, so a
report always covers the whole dataset. A run that cannot finish (quota, app
down) is recorded as a hard failure, never as a quality failure -- conflating the
two is how teams learn to ignore a red pipeline.
"""
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import gate as gate_mod
from .aggregate import RefusalPolicy, item_metric_values, summarize
from .dataset import GoldenItem
from .judge import JudgeQuotaExceeded, LLMJudge
from .metrics import Scorer
from .paired import PairedComparison, PairedConfig, compare
from .retrieval import score_retrieval
from .targets.base import Target, TargetUnavailable


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@dataclass
class Baseline:
    """A previous run this one is compared against, already loaded."""

    run_id: int
    label: str | None
    values: dict[str, dict] = field(default_factory=dict)


@dataclass
class RunOutcome:
    results: list[dict]
    summary: dict
    gate: gate_mod.GateResult
    target_info: dict
    judge_model: str
    # {metric: {item_id: value_or_None}} -- persisted, and the baseline for the
    # next run's paired comparison.
    item_values: dict[str, dict] = field(default_factory=dict)
    paired: PairedComparison | None = None


def run_evaluation(
    items: list[GoldenItem],
    target: Target,
    judge: LLMJudge,
    scorer: Scorer,
    thresholds: dict,
    require_all_metrics: bool = True,
    min_metric_coverage: float = 0.9,
    refusal: RefusalPolicy | None = None,
    retrieval_k: int | None = None,
    page_tolerance: int = 0,
    paired_cfg: PairedConfig | None = None,
    baseline: Baseline | None = None,
    on_progress=None,
) -> RunOutcome:
    results: list[dict] = []
    hard_failure: str | None = None

    for index, item in enumerate(items, start=1):
        record: dict = {
            "id": item.id,
            "question": item.question,
            "kind": item.kind,
            "expected_answer": item.expected_answer,
            "tags": item.tags,
            "gold_sources": [list(g) for g in item.gold_sources],
        }
        try:
            response = target.ask(item.question)
        except TargetUnavailable as exc:
            # Stop immediately: every later question would fail the same way and
            # burn whatever quota is left for no information.
            hard_failure = f"target unavailable after {index - 1}/{len(items)} questions: {exc}"
            record["error"] = str(exc)
            results.append(record)
            if on_progress:
                on_progress(index, len(items), record)
            break

        record.update(
            {
                "answer": response.answer,
                # Whichever the target gave us: explicit `contexts`, else the
                # content carried on each retrieved passage. One of the two must
                # exist or faithfulness is unmeasurable, which FAILS the gate.
                "contexts": response.faithfulness_contexts(),
                "refused": response.refused,
                "trace": (
                    None if response.trace is None
                    else [step.to_dict() for step in response.trace]
                ),
                "sources": response.sources,
                "retrieved_sources": (
                    None
                    if response.retrieved is None
                    else [r.to_dict() for r in response.retrieved]
                ),
                "latency_s": response.latency_s,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "model": response.model,
            }
        )

        # Retrieval is scored before the judge: it costs nothing, so it is still
        # recorded for this item even if the judge then dies on quota.
        if retrieval_k and item.is_answerable and item.gold_sources:
            scores = score_retrieval(
                item.gold_sources,
                response.retrieved,
                k=retrieval_k,
                page_tolerance=page_tolerance,
            )
            record["retrieval"] = {
                "hit": scores.hit,
                "recall": scores.recall,
                "rr": scores.rr,
            }

        try:
            scores = scorer.score(
                item.question, response.answer, response.faithfulness_contexts()
            )
        except JudgeQuotaExceeded as exc:
            hard_failure = f"judge out of quota after {index - 1}/{len(items)} questions: {exc}"
            record["error"] = str(exc)
            results.append(record)
            if on_progress:
                on_progress(index, len(items), record)
            break

        record.update(
            {
                "faithfulness": scores.faithfulness,
                "answer_relevancy": scores.answer_relevancy,
                "hallucinated": scores.hallucinated,
                "reasons": scores.reasons,
                "metric_errors": scores.errors,
            }
        )
        results.append(record)
        if on_progress:
            on_progress(index, len(items), record)

    summary = summarize(
        results, judge_calls=judge.calls, retrieval_k=retrieval_k, refusal=refusal
    )
    summary_dict = summary.to_dict()
    item_values = item_metric_values(results, retrieval_k, refusal)

    # Paired comparison, when there is something comparable to compare against.
    paired_cfg = paired_cfg or PairedConfig()
    specs = gate_mod.parse_specs(thresholds, retrieval_k)
    paired: PairedComparison | None = None
    if not paired_cfg.enabled:
        paired = PairedComparison(skipped_reason="paired comparison disabled in config")
    elif baseline is None:
        paired = PairedComparison(
            skipped_reason="no comparable baseline: needs a PASSING run on the base "
            "branch with the same dataset and the same judge"
        )
    elif hard_failure:
        # Half a dataset is not a fair comparison against a full baseline: the
        # items that never ran would read as missing pairs, not as a regression.
        paired = PairedComparison(
            baseline_run_id=baseline.run_id,
            skipped_reason="run did not complete, so it is not comparable to the baseline",
        )
    else:
        paired = compare(
            current_values=item_values,
            baseline_values=baseline.values,
            specs=specs,
            cfg=paired_cfg,
            baseline_run_id=baseline.run_id,
            baseline_label=baseline.label,
        )

    gate_result = gate_mod.evaluate(
        summary_dict,
        thresholds,
        require_all=require_all_metrics,
        min_metric_coverage=min_metric_coverage,
        paired=paired.verdicts if paired else None,
        retrieval_k=retrieval_k,
    )
    if hard_failure:
        # Preserve the threshold table for context, but the verdict is "incomplete".
        gate_result.hard_failure = hard_failure
        gate_result.passed = False

    return RunOutcome(
        results=results,
        summary=summary_dict,
        gate=gate_result,
        target_info=target.describe(),
        judge_model=judge.get_model_name(),
        item_values=item_values,
        paired=paired,
    )


def git_context() -> dict:
    """Which commit and branch this run measured.

    The branch decides whether a run can later serve as a baseline, so getting it
    right matters. CI's own variables win over git: `actions/checkout` leaves a
    detached HEAD, where `rev-parse --abbrev-ref HEAD` answers "HEAD" -- so a push
    to main would never be recorded as a main run and no baseline would ever exist.

    On a pull_request event GITHUB_REF_NAME is like "123/merge", which is correctly
    *not* the base branch: a PR's own run must not become anyone's baseline.
    """
    sha = os.getenv("GITHUB_SHA")
    return {
        "git_sha": sha[:7] if sha else _git("rev-parse", "--short", "HEAD"),
        "git_branch": os.getenv("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
