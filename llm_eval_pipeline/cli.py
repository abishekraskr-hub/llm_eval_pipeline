"""`llmeval` -- the command line entry point.

    llmeval run      --config llmeval.yaml [--branch X] [--label "..."]
    llmeval validate --config llmeval.yaml
    llmeval history  --config llmeval.yaml
    llmeval db-check --config llmeval.yaml

`python -m llm_eval_pipeline.cli` is the same thing, for when the console script
is not on PATH.

Exit codes are the contract with CI, and the distinction between them is the
point:

    0  pass
    1  the quality gate failed            -> block the merge
    2  invalid config or usage            -> block, but it is not about quality
    3  the run could not complete         -> WARN only; quota, outage, no key

Treating 3 as a failure is how a team learns to ignore a red pipeline, so it
never blocks. Treating 2 as a pass would let a broken config look like a green
build, so it does.
"""
import argparse
import json
import os
import sys

from . import __version__, report as report_mod
from .aggregate import RefusalPolicy, values_by_item
from .config import (
    ConfigError,
    hash_parts,
    judge_config_hash,
    load_config,
    paired_settings,
    resolve,
    validate as validate_config,
)
from .dataset import DatasetError, dataset_fingerprint, load_dataset
from .judges import UnknownJudge, build as build_judge
from .paired import PairedConfig
from .runner import Baseline, git_context, run_evaluation, utc_now
from .storage import BACKENDS, StoreUnavailable, describe_params, open_store, postgres_params
from .targets.base import TargetUnavailable

EXIT_OK, EXIT_GATE_FAILED, EXIT_USAGE, EXIT_INCOMPLETE = 0, 1, 2, 3


def _check_to_dict(check) -> dict:
    """Serialise a gate Check, including its computed status and reason.

    The stored detail is what a dashboard reads months later, so the derived
    fields are written out rather than left to be recomputed from flags.
    """
    data = dict(vars(check))
    data["status"] = check.status
    data["failure_reason"] = check.failure_reason
    return data


def _paired_to_dict(paired) -> dict | None:
    if paired is None:
        return None
    return {
        "baseline_run_id": paired.baseline_run_id,
        "baseline_label": paired.baseline_label,
        "skipped_reason": paired.skipped_reason,
        "paired_items": paired.paired_items,
        "unpaired_current": paired.unpaired_current,
        "unpaired_baseline": paired.unpaired_baseline,
        "verdicts": {
            name: {
                "metric": v.metric,
                "kind": v.kind,
                "baseline": v.baseline,
                "current": v.current,
                "delta": v.delta,
                "paired_n": v.paired_n,
                "regression": v.regression,
                "detail": v.detail,
                "note": v.note,
                "worse_items": v.worse_items,
                "sem_current": v.sem_current,
                "sem_baseline": v.sem_baseline,
            }
            for name, v in paired.verdicts.items()
        },
    }


def build_target(cfg: dict):
    """Construct the app under test from `target:` in the config.

    Three types, and only the first two are for real apps:

    python -- import an adapter class the app's own repo provides. The usual
              choice for a Python app you can import.
    http   -- call a running app over HTTP. The language-agnostic path: the app
              needs no Python and no knowledge of this library at all.
    stub   -- a built-in known-bad fake app. Exists so the GATE can be tested:
              a gate that has only ever been seen to pass is not known to work.
    """
    tcfg = cfg.get("target") or {}
    kind = str(tcfg.get("type") or "python").lower()
    config_dir = cfg.get("_dir") or "."

    if kind == "python":
        from .targets.python_target import PythonTarget

        return PythonTarget(str(tcfg.get("entry")), config_dir)
    if kind == "http":
        from .targets.http_target import HttpTarget

        return HttpTarget(tcfg, config_dir)
    if kind == "stub":
        from .fixtures.stub_targets import build_stub

        return build_stub(tcfg.get("stub") or "refuse_everything")
    raise ConfigError(f"unknown target.type {kind!r} (expected python|http|stub)")


def exit_code_for(gate) -> int:
    """The CI contract, in one place so it can be tested without a live run.

    A run that could not finish is never a quality verdict: it must not be able
    to block a merge, or a quota wall teaches the team to ignore red builds.
    """
    if gate.hard_failure:
        return EXIT_INCOMPLETE
    return EXIT_OK if gate.passed else EXIT_GATE_FAILED


def _load(args) -> dict:
    """Load the config, applying the flags that override it."""
    cfg = load_config(args.config)
    if getattr(args, "dataset", None):
        cfg["dataset"] = args.dataset
    if getattr(args, "db", None):
        cfg["storage"]["sqlite_path"] = args.db
    if getattr(args, "report_dir", None):
        cfg["storage"]["reports"] = args.report_dir
    if getattr(args, "storage", None):
        cfg["storage"]["backend"] = args.storage
    if getattr(args, "judge_provider", None):
        cfg["judge"]["provider"] = args.judge_provider
        # A model and key chosen for the old provider mean nothing on the new
        # one, so they are cleared and that provider's defaults apply.
        cfg["judge"]["model"] = None
        cfg["judge"]["api_key_env"] = None
    if getattr(args, "judge_model", None):
        cfg["judge"]["model"] = args.judge_model
    if getattr(args, "base_branch", None):
        cfg["paired"]["base_branch"] = args.base_branch
    if getattr(args, "no_paired", False):
        cfg["paired"]["enabled"] = False
    if getattr(args, "stub", None):
        cfg["target"]["type"] = "stub"
        cfg["target"]["stub"] = args.stub
    return cfg


def cmd_validate(args) -> int:
    """Check everything a run needs, without calling a single model.

    Loads the config, the dataset AND the target -- constructing the target is
    what catches a broken adapter, a missing index or an unimportable module,
    which are the failures that otherwise surface several minutes into a paid
    run.
    """
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    problems = validate_config(cfg)
    if problems:
        print(f"{len(problems)} problem(s) in {cfg['_path']}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_USAGE

    print(f"config   : {cfg['_path']}")

    try:
        items = load_dataset(
            resolve(cfg["dataset"], cfg),
            require_gold_sources=not args.allow_missing_gold,
        )
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    answerable = sum(1 for i in items if i.is_answerable)
    print(
        f"dataset  : {cfg['dataset']} -- {len(items)} questions "
        f"({answerable} answerable, {len(items) - answerable} unanswerable)"
    )

    try:
        target = build_target(cfg)
    except TargetUnavailable as exc:
        print(f"target unavailable: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except (ConfigError, ValueError) as exc:
        print(f"target error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"target   : {json.dumps(target.describe())}")

    try:
        judge, _ = build_judge(cfg["judge"])
    except UnknownJudge as exc:
        print(f"judge error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:
        # A missing API key. Usage, not a quality verdict.
        print(f"judge error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"judge    : {judge.get_model_name()}")

    refusal = RefusalPolicy.from_config(cfg)
    if not refusal.can_measure:
        print(
            "warning  : no refusal.patterns and use_target_flag is off, so "
            "abstention_rate and false_refusal_rate cannot be measured"
        )

    print("\nvalid.")
    return EXIT_OK


def cmd_run(args) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    problems = validate_config(cfg)
    if problems:
        print(f"config error in {cfg['_path']}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_USAGE

    try:
        items = load_dataset(
            resolve(cfg["dataset"], cfg),
            limit=args.limit,
            tag=args.tag,
            require_gold_sources=not args.allow_missing_gold,
        )
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        target = build_target(cfg)
    except TargetUnavailable as exc:
        print(f"target unavailable: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except (ConfigError, ValueError) as exc:
        print(f"target error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        judge, scorer = build_judge(cfg["judge"])
    except UnknownJudge as exc:
        print(f"judge error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:
        print(f"judge error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    retrieval_cfg = cfg.get("retrieval") or {}
    retrieval_k = retrieval_cfg.get("k")
    page_tolerance = int(retrieval_cfg.get("page_tolerance", 0) or 0)
    refusal = RefusalPolicy.from_config(cfg)

    try:
        paired_cfg = PairedConfig.from_dict(paired_settings(cfg))
    except (TypeError, ValueError) as exc:
        print(f"paired config error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # Fingerprint the whole dataset FILE, not the --limit slice, so a subset run
    # is still comparable to a full one on the base branch.
    ds_hash = dataset_fingerprint(
        resolve(cfg["dataset"], cfg), require_gold_sources=not args.allow_missing_gold
    )
    jc_hash = judge_config_hash(cfg)
    git = git_context()
    if args.branch:
        # Needed outside a git checkout. The branch decides which runs this one
        # can ever be compared against, so guessing it is not an option.
        git["git_branch"] = args.branch

    target_info = target.describe()
    print(f"llmeval  : {__version__}")
    print(f"target   : {json.dumps(target_info)}")
    print(f"judge    : {judge.get_model_name()}")
    print(f"dataset  : {cfg['dataset']} ({len(items)} questions, hash {ds_hash})")
    # A model grading its own output is a known source of self-preference bias.
    if target_info.get("model") and target_info["model"] == cfg["judge"].get("model"):
        print(
            f"WARNING  : the judge and the app under test are both "
            f"{target_info['model']!r}. A model grading its own output is biased "
            f"towards it."
        )

    try:
        store = open_store(
            cfg,
            backend=args.storage,
            create_db=_create_db_choice(args),
            notify=lambda message: print(f"  {message}"),
        )
    except StoreUnavailable as exc:
        print(f"storage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(f"store    : {store.describe()}")
    print(
        f"branch   : {git.get('git_branch') or '(unknown)'} "
        f"-> baseline from '{paired_cfg.base_branch}'"
    )

    run_id = store.start_run(
        started_at=utc_now(),
        label=args.label,
        dataset=str(cfg["dataset"]),
        dataset_size=len(items),
        dataset_hash=ds_hash,
        judge_config_hash=jc_hash,
        target=target_info,
        judge_model=judge.get_model_name(),
        **git,
    )

    # The baseline must be a PASSING run, on the base branch, that asked the same
    # questions and was graded the same way. Anything else would compare two
    # different measurements and call the difference a regression.
    baseline = None
    if paired_cfg.enabled:
        row = store.find_baseline(
            branch=paired_cfg.base_branch,
            dataset_hash=ds_hash,
            judge_config_hash=jc_hash,
            exclude_run_id=run_id,
        )
        if row:
            baseline = Baseline(
                run_id=row["id"],
                label=row.get("label"),
                values=store.item_values(row["id"]),
            )
            print(
                f"baseline : run #{baseline.run_id} "
                f"({row.get('git_sha') or '?'}, {(row.get('finished_at') or '')[:19]})"
            )
        else:
            print(
                f"baseline : none comparable on '{paired_cfg.base_branch}' "
                f"-- paired test skipped, floors still apply"
            )
    print()

    def progress(index: int, total: int, record: dict) -> None:
        if record.get("error"):
            flag = "ERR "
        elif record.get("hallucinated"):
            flag = "HALL"
        else:
            flag = "ok  "
        faith = record.get("faithfulness")
        rel = record.get("answer_relevancy")
        print(
            f"[{index:>3}/{total}] {flag} {record['id']:<10} "
            f"faith={'n/a' if faith is None else f'{faith:.2f}'} "
            f"rel={'n/a' if rel is None else f'{rel:.2f}'} "
            f"{record.get('latency_s', 0):.1f}s  {record['question'][:48]}",
            flush=True,
        )

    # An http target may need starting first, and must be stopped even on error.
    starter = getattr(target, "start", None)
    if callable(starter):
        try:
            starter()
        except TargetUnavailable as exc:
            print(f"target unavailable: {exc}", file=sys.stderr)
            store.close()
            return EXIT_INCOMPLETE
    try:
        outcome = run_evaluation(
            items=items,
            target=target,
            judge=judge,
            scorer=scorer,
            thresholds=cfg["thresholds"],
            require_all_metrics=cfg.get("require_all_metrics", True),
            min_metric_coverage=float(cfg.get("min_metric_coverage", 0.9)),
            retrieval_k=retrieval_k,
            page_tolerance=page_tolerance,
            paired_cfg=paired_cfg,
            baseline=baseline,
            refusal=refusal,
            on_progress=progress,
        )
    finally:
        stopper = getattr(target, "stop", None)
        if callable(stopper):
            stopper()

    by_item = values_by_item(outcome.item_values)
    for record in outcome.results:
        store.add_result(run_id, record, by_item.get(record["id"]))
    store.finish_run(
        run_id,
        finished_at=utc_now(),
        summary=outcome.summary,
        gate_passed=outcome.gate.passed,
        gate_detail=[_check_to_dict(c) for c in outcome.gate.checks],
        hard_failure=outcome.gate.hard_failure,
        baseline_run_id=(outcome.paired.baseline_run_id if outcome.paired else None),
    )
    views = store.refresh_branch_views()
    view_schema = getattr(store, "branch_schema", None)
    store.close()

    print()
    print(report_mod.console_report(outcome.summary, outcome.gate, outcome.paired))
    slug = views.get(git.get("git_branch") or "")
    if slug and view_schema:
        print(f"  rows: {view_schema}.{slug} and {view_schema}.{slug}__items")

    md = report_mod.markdown_report(
        outcome.summary,
        outcome.gate,
        outcome.results,
        outcome.target_info,
        outcome.judge_model,
        paired=outcome.paired,
        hash_parts=hash_parts(cfg),
    )
    reports_dir = resolve((cfg.get("storage") or {}).get("reports") or ".llmeval/reports", cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"run_{run_id:04d}.md"
    md_path.write_text(md, encoding="utf-8")
    json_path = reports_dir / f"run_{run_id:04d}.json"
    json_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "llmeval_version": __version__,
                "summary": outcome.summary,
                "gate": {
                    "passed": outcome.gate.passed,
                    "hard_failure": outcome.gate.hard_failure,
                    "checks": [_check_to_dict(c) for c in outcome.gate.checks],
                },
                "paired": _paired_to_dict(outcome.paired),
                "target": outcome.target_info,
                "comparability": hash_parts(cfg),
                "results": outcome.results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n  run #{run_id} -> {md_path.name}, {json_path.name}")

    summary_out = args.summary_out or os.getenv("GITHUB_STEP_SUMMARY")
    if summary_out:
        with open(summary_out, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")

    return exit_code_for(outcome.gate)


def cmd_history(args) -> int:
    cfg = _load(args)
    try:
        store = open_store(
            cfg,
            backend=args.storage,
            create_db=_create_db_choice(args),
            notify=lambda message: print(f"  {message}"),
        )
    except StoreUnavailable as exc:
        print(f"storage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    runs = store.history(limit=args.limit)
    if args.branch:
        runs = [r for r in runs if r.get("git_branch") == args.branch]
    store.close()
    if not runs:
        print("no runs recorded yet")
        return EXIT_OK
    header = (
        f"{'run':>4}  {'when':<20} {'branch':<14} {'gate':<6} {'halluc':>7} "
        f"{'relev':>7} {'faith':>7} {'p95':>7}  label"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        s = run.get("summary") or {}

        def fmt(key, digits=3):
            v = s.get(key)
            return "n/a" if v is None else f"{v:.{digits}f}"

        gate = "INCOMPL" if run.get("hard_failure") else ("PASS" if run.get("gate_passed") else "FAIL")
        print(
            f"{run['id']:>4}  {(run.get('started_at') or '')[:19]:<20} "
            f"{(run.get('git_branch') or '-')[:14]:<14} {gate:<6} "
            f"{fmt('hallucination_rate'):>7} {fmt('answer_relevance'):>7} "
            f"{fmt('faithfulness'):>7} {fmt('latency_p95', 2):>7}  {run.get('label') or ''}"
        )
    return EXIT_OK


def _create_db_choice(args) -> bool | None:
    """--create-database / --no-create-database, or None to use the config."""
    if getattr(args, "no_create_database", False):
        return False
    if getattr(args, "create_database", False):
        return True
    return None


def cmd_db_check(args) -> int:
    """Prove the store works before spending a single judge call on it.

    Creates the schema if it is missing, so this doubles as the setup step: an
    empty database needs no hand-written DDL.
    """
    cfg = _load(args)
    backend = (args.storage or (cfg.get("storage") or {}).get("backend") or "sqlite").lower()
    if backend == "postgres":
        print(f"Target : {describe_params(postgres_params(cfg))}")
        pw_env = ((cfg.get("storage") or {}).get("postgres") or {}).get("password_env")
        if pw_env:
            state = "set" if os.getenv(pw_env) else "NOT SET"
            print(f"Password: ${pw_env} is {state}")

    try:
        store = open_store(
            cfg,
            backend=args.storage,
            create_db=_create_db_choice(args),
            notify=lambda message: print(f"  {message}"),
        )
    except StoreUnavailable as exc:
        print(f"\nstorage error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(f"Store  : {store.describe()}  OK")
    runs = store.history(limit=1000)
    print(f"Runs   : {len(runs)}")
    if runs:
        by_branch: dict[str, list[dict]] = {}
        for run in runs:
            by_branch.setdefault(run.get("git_branch") or "(none)", []).append(run)
        views = store.refresh_branch_views()
        schema = getattr(store, "branch_schema", None)
        print()
        print(f"  {'branch':<24} {'runs':>5} {'passed':>7}  latest   view")
        for branch, rows in sorted(by_branch.items()):
            passed = sum(1 for r in rows if r.get("gate_status") == "pass")
            slug = views.get(branch)
            where = f"{schema}.{slug}" if slug and schema else "-"
            print(
                f"  {branch[:24]:<24} {len(rows):>5} {passed:>7}  "
                f"#{max(r['id'] for r in rows):<6}  {where}"
            )
        base = cfg["paired"]["base_branch"]
        passing = [
            r for r in by_branch.get(base, [])
            if r.get("gate_status") == "pass" and r.get("finished_at")
        ]
        print()
        if passing:
            newest = max(passing, key=lambda r: r["id"])
            print(
                f"  baseline branch '{base}': last passing run #{newest['id']} "
                f"(dataset {newest.get('dataset_hash') or '?'}, "
                f"judge {newest.get('judge_config_hash') or '?'})"
            )
            print("  a run only pairs against it if BOTH of those hashes match.")
        else:
            print(
                f"  baseline branch '{base}': no passing run yet -- the paired test "
                f"will be skipped and the absolute floors are the whole gate"
            )
    else:
        print("  (empty -- schema created, nothing recorded yet)")
    store.close()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmeval",
        description="CI/CD for LLM applications: block a PR that makes answers worse.",
    )
    parser.add_argument("--version", action="version", version=f"llmeval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the golden dataset and apply the gate")
    val_p = sub.add_parser(
        "validate",
        help="check the config, dataset and target without calling any model",
    )
    hist_p = sub.add_parser("history", help="list past runs")
    check_p = sub.add_parser(
        "db-check", help="connect to the store, create the schema, and show what is there"
    )

    # Shared by every subcommand: which config, and where to put things.
    for sub_parser in (run_p, val_p, hist_p, check_p):
        sub_parser.add_argument(
            "--config",
            default="llmeval.yaml",
            help="path to llmeval.yaml (default: ./llmeval.yaml). Every relative "
            "path INSIDE it resolves from its own folder, not from the cwd.",
        )
        sub_parser.add_argument(
            "--storage",
            choices=sorted(BACKENDS),
            help="override storage.backend from the config",
        )
        sub_parser.add_argument("--db", help="override storage.sqlite_path")
        sub_parser.add_argument("--report-dir", help="override storage.reports")
        group = sub_parser.add_mutually_exclusive_group()
        group.add_argument(
            "--create-database",
            action="store_true",
            help="create the Postgres database if it does not exist (default: "
            "storage.postgres.create_database)",
        )
        group.add_argument(
            "--no-create-database",
            action="store_true",
            help="fail instead of creating a missing database. Use this when a "
            "typo in the name must not quietly produce a second empty database.",
        )

    for sub_parser in (run_p, val_p):
        sub_parser.add_argument("--dataset", help="override the dataset path")
        sub_parser.add_argument("--judge-model", help="override judge.model")
        sub_parser.add_argument(
            "--judge-provider", help="override judge.provider (resets the model to its default)"
        )
        sub_parser.add_argument(
            "--stub",
            help="run a built-in known-bad fake app instead of the real target: "
            "refuse_everything | answer_everything | no_retrieval",
        )
        sub_parser.add_argument(
            "--allow-missing-gold",
            action="store_true",
            help="load answerable items that have no gold_sources (the retrieval "
            "metrics are then unmeasured, which FAILS the gate if declared)",
        )

    run_p.add_argument("--limit", type=int, help="only the first N questions (cheap smoke run)")
    run_p.add_argument("--tag", help="only items carrying this tag")
    run_p.add_argument("--label", help="name this run in the history table")
    run_p.add_argument(
        "--branch",
        help="record this run under this branch name instead of asking git. "
        "Needed outside a git checkout -- the branch decides which runs this one "
        "can ever be compared against.",
    )
    run_p.add_argument(
        "--base-branch",
        help="the integration branch whose last PASSING run is the baseline "
        "(default: paired.base_branch)",
    )
    run_p.add_argument(
        "--no-paired",
        action="store_true",
        help="skip the paired comparison and gate on absolute floors only",
    )
    run_p.add_argument(
        "--summary-out",
        help="append the markdown report to this file (defaults to "
        "$GITHUB_STEP_SUMMARY when that is set)",
    )

    hist_p.add_argument("--limit", type=int, default=20)
    hist_p.add_argument("--branch", help="only runs recorded on this branch")

    run_p.set_defaults(func=cmd_run)
    val_p.set_defaults(func=cmd_validate)
    hist_p.set_defaults(func=cmd_history)
    check_p.set_defaults(func=cmd_db_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_INCOMPLETE
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:
        # CI reads exit codes, not tracebacks. An unexpected crash is
        # "incomplete", never a quality verdict -- a stack trace must not look
        # like a failed gate and block a merge.
        print(f"\nevaluation crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
