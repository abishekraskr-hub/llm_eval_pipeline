"""Human-readable run reports.

The markdown version is written for a PR reader: the verdict first, then the
numbers, then only the failing questions. Nobody scrolls 100 passing rows.
"""
from .gate import GateResult


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_sem(value: float | None, sem: float | None, digits: int = 3) -> str:
    """`0.910 (±0.021)` -- the number plus how much it is allowed to wobble.

    Without the spread, a reader has no way to tell a 0.01 move that means
    something from a 0.01 move that is noise.
    """
    if value is None:
        return "n/a"
    base = f"{value:.{digits}f}"
    return base if sem is None else f"{base} (±{sem:.{digits}f})"


def _fmt_delta(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:+.{digits}f}"


def _paired_section(paired) -> list[str]:
    """The 'did these questions get worse' table."""
    lines: list[str] = []
    lines.append("### Paired comparison vs baseline")
    lines.append("")

    if paired is None:
        lines.append("_Not run._")
        lines.append("")
        return lines

    if paired.skipped_reason:
        # Explicitly NOT a failure. A first run, a dataset edit or a judge change
        # all land here, and the absolute floors carry the gate on their own.
        lines.append(f"**No comparable baseline — paired test skipped.** {paired.skipped_reason}")
        lines.append("")
        lines.append("_Gated on absolute floors only for this run._")
        lines.append("")
        return lines

    base_bits = f"run #{paired.baseline_run_id}"
    if paired.baseline_label:
        base_bits += f" ({paired.baseline_label})"
    lines.append(
        f"Baseline: **{base_bits}** &nbsp;|&nbsp; {paired.paired_items} items paired"
        + (
            f", {paired.unpaired_current} new / {paired.unpaired_baseline} dropped"
            if (paired.unpaired_current or paired.unpaired_baseline)
            else ""
        )
    )
    lines.append("")
    lines.append("| Metric | Baseline | Current | Δ | Worse / Better | p or 95% CI | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    for verdict in paired.verdicts.values():
        if verdict.paired_n == 0:
            lines.append(
                f"| {verdict.label} | n/a | n/a | - | - | - | ⚠️ {verdict.note or 'no pairs'} |"
            )
            continue
        if verdict.kind == "binary":
            flips = f"{verdict.mcnemar.worse} / {verdict.mcnemar.better}" if verdict.mcnemar else "-"
            detail = f"p = {verdict.mcnemar.p_value:.3f}" if verdict.mcnemar else "-"
            baseline_cell = _fmt(verdict.baseline)
            current_cell = _fmt(verdict.current)
        else:
            flips = "—"
            detail = verdict.detail
            baseline_cell = _fmt_sem(verdict.baseline, verdict.sem_baseline)
            current_cell = _fmt_sem(verdict.current, verdict.sem_current)

        if verdict.regression:
            mark = "❌ regression"
        elif verdict.note:
            mark = f"✅ {verdict.note}"
        else:
            mark = "✅ noise"

        lines.append(
            f"| {verdict.label} | {baseline_cell} | {current_cell} | "
            f"{_fmt_delta(verdict.delta)} | {flips} | {detail} | {mark} |"
        )
    lines.append("")

    regressions = paired.regressions
    if regressions:
        lines.append("**Items that got worse:**")
        lines.append("")
        for verdict in regressions:
            shown = verdict.worse_items[:15]
            suffix = f" … +{len(verdict.worse_items) - len(shown)} more" if len(
                verdict.worse_items
            ) > len(shown) else ""
            ids = ", ".join(f"`{i}`" for i in shown) or "—"
            lines.append(f"- **{verdict.label}**: {ids}{suffix}")
        lines.append("")

    return lines


def markdown_report(
    summary: dict,
    gate: GateResult,
    results: list[dict],
    target: dict,
    judge_model: str,
    max_failures: int = 10,
    paired=None,
    hash_parts: dict | None = None,
) -> str:
    lines: list[str] = []

    if gate.hard_failure:
        lines.append("## LLM evaluation: DID NOT COMPLETE")
        lines.append("")
        lines.append(f"**{gate.hard_failure}**")
        lines.append("")
        lines.append("_No quality verdict: the run could not finish, so this is not a quality failure._")
        return "\n".join(lines)

    verdict = "PASSED" if gate.passed else "FAILED"
    lines.append(f"## LLM evaluation: {verdict}")
    lines.append("")

    target_bits = ", ".join(
        f"{k}=`{v}`" for k, v in target.items() if k not in ("kind", "app_dir") and v is not None
    )
    lines.append(f"**Target:** {target_bits or target.get('kind', 'unknown')}")
    lines.append(f"**Judge:** `{judge_model}` &nbsp;|&nbsp; **Questions:** {summary.get('total')} "
                 f"({summary.get('errors')} errored)")
    lines.append("")
    # The comparability inputs, so "no comparable baseline" is never a mystery:
    # the reader can see which part changed instead of guessing.
    if hash_parts:
        bits = " &nbsp;|&nbsp; ".join(f"{k} `{v}`" for k, v in hash_parts.items())
        lines.append(f"<sub>Baseline comparability: {bits}</sub>")
        lines.append("")

    lines.append("| Metric | Threshold | Actual | Measured on | Status | Why |")
    lines.append("|---|---|---|---|---|---|")
    for check in gate.checks:
        rule = f"{'≤' if check.direction == 'max' else '≥'} {check.threshold:g}"
        actual = _fmt(check.actual)
        note = f" <sub>{check.note}</sub>" if check.note else ""
        reason = check.failure_reason or ""
        # Coverage is shown for every metric, not just the failing ones. A
        # reader who can see "4/4" next to a score knows it covers the dataset;
        # without it, a flattering average over a subset looks identical to an
        # honest one.
        if check.applicable:
            seen = f"{check.measured}/{check.applicable}"
            if check.partial_coverage:
                seen = f"**{seen}**"
        else:
            seen = "—"
        lines.append(
            f"| {check.label} | {rule} | {actual}{note} | {seen} | {check.status} | {reason} |"
        )
    lines.append("")
    lines.append(
        "<sub>Thresholds are **safety floors** for disasters; the paired comparison "
        "below is what detects day-to-day regressions. **Measured on** is how many "
        "of the items a metric applies to actually got a value — a metric scored on "
        "only part of the dataset fails, because unscored items are dropped from the "
        "mean rather than counted against it.</sub>"
    )
    lines.append("")

    lines.extend(_paired_section(paired))

    # Both sides of the refusal coin, together. Reading either alone is misleading:
    # a bot that refuses everything scores a perfect abstention rate.
    if summary.get("unanswerable_total") or summary.get("answerable_total"):
        lines.append("### Refusal behaviour")
        lines.append("")
        lines.append("| | Value | Items | Good direction | Meaning |")
        lines.append("|---|---|---|---|---|")
        lines.append(
            f"| Abstention (unanswerable) | {_fmt(summary.get('abstention_rate'))} | "
            f"{summary.get('unanswerable_total')} | high | refuses what it should |"
        )
        lines.append(
            f"| False refusal (answerable) | {_fmt(summary.get('false_refusal_rate'))} | "
            f"{summary.get('answerable_total')} | low | doesn't refuse what it shouldn't |"
        )
        lines.append("")

    # `retrieval_k` and `retrieval_scored` share the prefix but are metadata, not
    # scores -- rendering them as "5.000" alongside real metrics reads as a result.
    retrieval_meta = {"retrieval_k", "retrieval_scored"}
    retrieval_keys = sorted(
        k for k in summary if k.startswith("retrieval_") and k not in retrieval_meta
    )
    if retrieval_keys:
        lines.append("<details><summary>Retrieval</summary>")
        lines.append("")
        lines.append("| | |")
        lines.append("|---|---|")
        for key in retrieval_keys:
            lines.append(f"| {key} | {_fmt(summary.get(key))} |")
        lines.append(
            f"| items scored | {summary.get('retrieval_scored')} "
            f"(answerable items with gold labels) |"
        )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("<details><summary>Run metrics</summary>")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Faithfulness (mean) | {_fmt(summary.get('faithfulness'))} |")
    lines.append(f"| Answer relevance (mean) | {_fmt(summary.get('answer_relevance'))} |")
    lines.append(f"| Hallucination rate | {_fmt(summary.get('hallucination_rate'))} |")
    lines.append(f"| Latency p50 / p95 | {_fmt(summary.get('latency_p50'), 2)}s / "
                 f"{_fmt(summary.get('latency_p95'), 2)}s |")
    lines.append(f"| Cost per query | ${summary.get('cost_per_query') or 0:.8f} |")
    lines.append(f"| Tokens in / out | {summary.get('input_tokens')} / {summary.get('output_tokens')} |")
    lines.append(f"| Judge requests | {summary.get('judge_calls')} |")
    lines.append("")
    lines.append(
        "<sub>p95 latency is noisy on a small dataset — one slow question moves it a "
        "lot. It is gated on its absolute ceiling only, never on the paired test.</sub>"
    )
    lines.append("")
    lines.append("</details>")
    lines.append("")

    for note in summary.get("notes") or []:
        lines.append(f"> {note}")
    if summary.get("notes"):
        lines.append("")

    worst = sorted(
        (r for r in results if r.get("hallucinated") or r.get("error")),
        key=lambda r: (r.get("faithfulness") is None, r.get("faithfulness") or 0.0),
    )[:max_failures]
    if worst:
        lines.append(f"### Lowest-scoring items ({len(worst)} shown)")
        lines.append("")
        for r in worst:
            lines.append(f"**{r['id']}** — {r['question']}")
            if r.get("error"):
                lines.append(f"- error: `{r['error']}`")
            else:
                lines.append(
                    f"- faithfulness `{_fmt(r.get('faithfulness'))}`, "
                    f"relevance `{_fmt(r.get('answer_relevancy'))}`"
                )
                retrieval = r.get("retrieval") or {}
                if retrieval:
                    lines.append(
                        f"- retrieval hit `{_fmt(retrieval.get('hit'), 0)}`, "
                        f"recall `{_fmt(retrieval.get('recall'))}`, "
                        f"RR `{_fmt(retrieval.get('rr'))}`"
                    )
                answer = (r.get("answer") or "").replace("\n", " ")
                lines.append(f"- answer: {answer[:240]}{'...' if len(answer) > 240 else ''}")
                reason = (r.get("reasons") or {}).get("faithfulness")
                if reason:
                    lines.append(f"- judge: {reason[:300]}")
            lines.append("")

    return "\n".join(lines)


def console_report(summary: dict, gate: GateResult, paired=None) -> str:
    lines = []
    if gate.hard_failure:
        return f"RUN INCOMPLETE: {gate.hard_failure}"
    for check in gate.checks:
        rule = "<=" if check.direction == "max" else ">="
        flag = "  <-- regression vs baseline" if check.paired_regression else ""
        lines.append(
            f"  [{check.status}] {check.label:32s} {rule} {check.threshold:<8g} "
            f"actual {_fmt(check.actual)}{flag}"
        )
    lines.append("")
    if paired is not None:
        if paired.skipped_reason:
            lines.append(f"  paired: skipped ({paired.skipped_reason})")
        else:
            regressions = paired.regressions
            lines.append(
                f"  paired: vs run #{paired.baseline_run_id}, {paired.paired_items} items, "
                f"{len(regressions)} regression(s)"
            )
    lines.append(f"  p50 {_fmt(summary.get('latency_p50'), 2)}s / p95 {_fmt(summary.get('latency_p95'), 2)}s"
                 f" | judge requests {summary.get('judge_calls')}")
    lines.append(f"  GATE: {'PASSED' if gate.passed else 'FAILED'}")
    return "\n".join(lines)
