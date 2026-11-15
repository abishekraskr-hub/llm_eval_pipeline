"""Load and validate `llmeval.yaml`, the one file an app writes.

Everything app-specific lives in that file: which app to call, where the golden
dataset is, which judge to use, what the floors are, and what counts as a
refusal. The library holds no app knowledge at all -- that is the property which
lets the same package gate a medical RAG bot and a SQL-backed support agent
without either of them editing library code.

PATH RESOLUTION. Every relative path inside the config resolves from the CONFIG
FILE's folder, never from the current working directory. So

    llmeval run --config ../other-app/llmeval.yaml

behaves exactly as if you had run it from `../other-app/`. Resolving from the cwd
would make the same config work or fail depending on where you happened to be
standing -- which in practice means working locally and breaking in CI.
"""
import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from . import __version__

# Defaults for everything an app does not state. A config can be as short as a
# target, a dataset and a judge.
DEFAULTS: dict = {
    "version": 1,
    "dataset": "eval/golden.yaml",
    "target": {
        # python -> import an adapter class from the app's own repo
        # http   -> call a running app over HTTP, in any language
        # stub   -> a built-in known-bad fake app, for proving the gate fails
        "type": "python",
        "entry": None,
        "url": None,
        "method": "POST",
        "timeout_s": 120,
        "retries": 0,
        "headers": {},
        "request": {},
        "response": {},
        "start": {},
        "stub": "refuse_everything",
    },
    # What counts as the app declining to answer. This USED to be a hard-coded
    # list of English phrases inside the library, which silently made the library
    # wrong for any app that phrased it differently or answered in another
    # language. It belongs to the app.
    "refusal": {
        # If the target reports `refused: true/false` itself, believe it: the app
        # knows whether it took its refusal branch, and no pattern can be as
        # accurate as that.
        "use_target_flag": True,
        # Case-insensitive regular expressions, applied to the answer when the
        # target reports no flag. Empty patterns AND no flag means
        # abstention_rate and false_refusal_rate are UNMEASURABLE -- which fails
        # the gate if they are declared, rather than quietly scoring every answer
        # as a non-refusal.
        "patterns": [],
    },
    "judge": {
        # groq | gemini | fake. Groq by default so the judge never shares a quota
        # bucket with the app under test, and so no model grades its own output.
        # `fake` makes no network calls -- see judges/fake.py.
        "provider": "groq",
        "model": None,
        "api_key_env": None,
        "requests_per_minute": 12,
        "temperature": 0.0,
        # Which judge-scored metrics to compute at all.
        "metrics": {"faithfulness": True, "answer_relevancy": True},
    },
    "storage": {
        # postgres is the real store; sqlite is a single file needing no server,
        # used by CI and by anyone without a database. There is deliberately NO
        # automatic fallback between them.
        "backend": "postgres",
        "sqlite_path": ".llmeval/history.sqlite",
        "reports": ".llmeval/reports",
        "postgres": {
            "host": "localhost",
            "port": 5432,
            "dbname": "llm_eval_pipeline",
            "user": "postgres",
            # The NAME of the env var holding the password, never the password.
            "password_env": "EVAL_PG_PASSWORD",
            "connect_timeout": 10,
            "create_database": True,
            "schema": "eval",
            "branch_schema": "branches",
        },
    },
    "retrieval": {
        # k is baked into the retrieval metric NAMES, so changing it starts a new
        # series instead of silently redefining the old one.
        "k": 5,
        # Pages of slack when matching a retrieved page to a gold page. Every
        # page of slack makes the metric easier to pass, so this stays 0 until a
        # real boundary mismatch is observed.
        "page_tolerance": 0,
    },
    "paired": {
        "enabled": True,
        # The integration branch whose last PASSING run is the baseline.
        # "auto" = the PR's target branch in CI, else the --branch given.
        "base_branch": "develop",
        "alpha": 0.05,
        "min_flips": 2,
        "min_delta": 0.03,
        "bootstrap_resamples": 10000,
        "seed": 42,
        "min_pair_coverage": 0.8,
    },
    "pricing": {"input_per_1m_tokens": 0.0, "output_per_1m_tokens": 0.0},
    "thresholds": {},
    "require_all_metrics": True,
    "min_metric_coverage": 0.9,
}

# Keys allowed at the top level. An unknown key is REJECTED rather than ignored:
# a typo in a gate setting would otherwise disable the very thing it was meant to
# configure, and the run would pass for the wrong reason.
TOP_LEVEL = set(DEFAULTS) | {"_dir", "_path"}

TARGET_TYPES = ("python", "http", "stub")


class ConfigError(ValueError):
    """The config is invalid. Always exit 2 -- usage, never a quality verdict."""


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Read `llmeval.yaml` and return the merged config.

    Adds two private keys: `_path` (the config file) and `_dir` (its folder,
    which every relative path resolves against).
    """
    cfg_path = Path(path or "llmeval.yaml").expanduser()
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    cfg_path = cfg_path.resolve()

    try:
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{cfg_path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"{cfg_path} must hold a mapping at the top level")

    unknown = set(loaded) - TOP_LEVEL
    if unknown:
        raise ConfigError(
            f"{cfg_path.name}: unknown top-level key(s) {sorted(unknown)}; "
            f"known: {sorted(DEFAULTS)}"
        )

    cfg = _merge(DEFAULTS, loaded)
    cfg["_path"] = str(cfg_path)
    cfg["_dir"] = str(cfg_path.parent)

    # The app's own .env, loaded from beside its config. `override=False` so a
    # variable already in the environment wins -- CI secrets must beat a stale
    # local file.
    load_dotenv(cfg_path.parent / ".env", override=False)

    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict) -> None:
    """Environment overrides, for CI and for one-off local runs."""
    if os.getenv("LLMEVAL_STORAGE_BACKEND"):
        cfg["storage"]["backend"] = os.environ["LLMEVAL_STORAGE_BACKEND"]
    if os.getenv("LLMEVAL_BASE_BRANCH"):
        cfg["paired"]["base_branch"] = os.environ["LLMEVAL_BASE_BRANCH"]
    # Provider first: switching it invalidates a model name chosen for the old
    # one, so the model is cleared unless it is also given explicitly.
    if os.getenv("LLMEVAL_JUDGE_PROVIDER"):
        cfg["judge"]["provider"] = os.environ["LLMEVAL_JUDGE_PROVIDER"]
        cfg["judge"]["model"] = None
        cfg["judge"]["api_key_env"] = None
    if os.getenv("LLMEVAL_JUDGE_MODEL"):
        cfg["judge"]["model"] = os.environ["LLMEVAL_JUDGE_MODEL"]
    if os.getenv("LLMEVAL_DATASET"):
        cfg["dataset"] = os.environ["LLMEVAL_DATASET"]


def resolve(path: str | Path, cfg: dict) -> Path:
    """Resolve one config path against the config file's folder."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(cfg.get("_dir") or ".") / candidate).resolve()


def paired_settings(cfg: dict) -> dict:
    """The paired block with `base_branch: auto` resolved.

    `auto` means "whatever this PR is targeting", which only CI knows. Outside
    CI it falls back to the configured default rather than guessing.
    """
    paired = dict(cfg.get("paired") or {})
    if str(paired.get("base_branch") or "").lower() == "auto":
        paired["base_branch"] = (
            os.getenv("GITHUB_BASE_REF")
            or os.getenv("LLMEVAL_BASE_BRANCH")
            or DEFAULTS["paired"]["base_branch"]
        )
    return paired


def validate(cfg: dict) -> list[str]:
    """Return a list of problems. Empty means valid.

    Every message names the exact field, because the fastest way to make someone
    stop reading error output is to tell them something is wrong without saying
    where.
    """
    import re

    problems: list[str] = []

    if int(cfg.get("version") or 1) != 1:
        problems.append(f"version: only 1 is supported, got {cfg.get('version')!r}")

    target = cfg.get("target") or {}
    kind = str(target.get("type") or "").lower()
    if kind not in TARGET_TYPES:
        problems.append(
            f"target.type: must be one of {', '.join(TARGET_TYPES)}, "
            f"got {kind or '(missing)'!r}"
        )
    elif kind == "python":
        entry = target.get("entry")
        if not entry:
            problems.append(
                "target.entry: required for target.type: python "
                "(e.g. eval.my_target:MyTarget)"
            )
        elif ":" not in str(entry):
            problems.append(f"target.entry: must be 'module:Class', got {entry!r}")
    elif kind == "http" and not (target.get("url") or target.get("base_url")):
        problems.append("target.url: required for target.type: http")

    if not cfg.get("dataset"):
        problems.append("dataset: required (path to the golden dataset, relative to this file)")
    else:
        dataset = resolve(cfg["dataset"], cfg)
        if not dataset.exists():
            problems.append(f"dataset: file not found: {dataset}")

    judge = cfg.get("judge") or {}
    from .judges import REGISTRY

    provider = str(judge.get("provider") or "").lower()
    if provider not in REGISTRY:
        problems.append(
            f"judge.provider: must be one of {', '.join(sorted(REGISTRY))}, got {provider!r}"
        )
    for key in ("model", "api_key_env"):
        value = judge.get(key)
        if value is not None and not isinstance(value, str):
            problems.append(
                f"judge.{key}: must be a string or omitted, got {type(value).__name__}"
            )
    api_key_env = judge.get("api_key_env")
    if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(api_key_env)):
        problems.append(
            f"judge.api_key_env: must be the NAME of an environment variable, "
            f"not a value, got {api_key_env!r}"
        )

    storage = cfg.get("storage") or {}
    if str(storage.get("backend") or "").lower() not in ("postgres", "sqlite"):
        problems.append(
            f"storage.backend: must be postgres or sqlite, got {storage.get('backend')!r}"
        )
    pg = storage.get("postgres") or {}
    if "password" in pg:
        problems.append(
            "storage.postgres.password: not allowed. Use password_env to name an "
            "environment variable -- a password in a config file ends up in git."
        )

    retrieval = cfg.get("retrieval") or {}
    try:
        if int(retrieval.get("k", 5)) < 1:
            problems.append("retrieval.k: must be 1 or more")
    except (TypeError, ValueError):
        problems.append(f"retrieval.k: must be an integer, got {retrieval.get('k')!r}")

    refusal = cfg.get("refusal") or {}
    patterns = refusal.get("patterns")
    if patterns is not None and not isinstance(patterns, list):
        problems.append("refusal.patterns: must be a list of regular expressions")
    else:
        for i, pattern in enumerate(patterns or []):
            try:
                re.compile(str(pattern), re.IGNORECASE)
            except re.error as exc:
                problems.append(
                    f"refusal.patterns[{i}]: not a valid regex ({exc}): {pattern!r}"
                )

    thresholds = cfg.get("thresholds")
    if thresholds is not None and not isinstance(thresholds, dict):
        problems.append("thresholds: must be a mapping of metric -> floor")
    else:
        from .gate import UnknownMetric, parse_specs

        try:
            parse_specs(thresholds or {}, int(retrieval.get("k", 5) or 5))
        except UnknownMetric as exc:
            problems.append(f"thresholds: {exc}")

    try:
        from .paired import PairedConfig

        PairedConfig.from_dict(paired_settings(cfg))
    except (TypeError, ValueError) as exc:
        problems.append(f"paired: {exc}")

    return problems


def judge_config_hash(cfg: dict) -> str:
    """Identity of the grader AND of the code that graded. Baseline comparability.

    A paired comparison is only meaningful if both runs were graded the same way.
    Faithfulness 0.91 from one judge is not the same measurement as 0.91 from
    another, so changing any of these must start a fresh baseline rather than
    produce a phantom regression.

    What is included, and why each one earns its place:

    * provider, model, temperature -- a different grader gives different numbers
    * which judge metrics are enabled
    * retrieval k and page tolerance -- they change what counts as a hit
    * the installed DeepEval version -- the judge's actual PROMPTS live in that
      package, so an upgrade can change scoring with no config edit at all
    * this library's own version -- the same argument, for our code
    * the target type and entry -- pointing the gate at a different app is not a
      regression in the old one
    """
    import hashlib

    judge = cfg.get("judge") or {}
    retrieval = cfg.get("retrieval") or {}
    target = cfg.get("target") or {}

    try:
        from importlib.metadata import version

        deepeval_version = version("deepeval")
    except Exception:
        deepeval_version = "unknown"

    parts = [
        f"provider={judge.get('provider')}",
        f"model={judge.get('model')}",
        f"temperature={judge.get('temperature')}",
        f"metrics={sorted((k, bool(v)) for k, v in (judge.get('metrics') or {}).items())}",
        f"retrieval_k={retrieval.get('k')}",
        f"page_tolerance={retrieval.get('page_tolerance')}",
        f"deepeval={deepeval_version}",
        f"llm_eval_pipeline={__version__}",
        f"target={target.get('type')}:"
        f"{target.get('entry') or target.get('url') or target.get('stub')}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def hash_parts(cfg: dict) -> dict:
    """The comparability inputs, for the report header.

    Shown so that "no comparable baseline" is never a mystery: the reader can see
    which part moved instead of guessing.
    """
    judge = cfg.get("judge") or {}
    retrieval = cfg.get("retrieval") or {}
    target = cfg.get("target") or {}
    try:
        from importlib.metadata import version

        deepeval_version = version("deepeval")
    except Exception:
        deepeval_version = "unknown"
    return {
        "judge": f"{judge.get('provider')}:{judge.get('model') or 'provider default'}",
        "retrieval": f"k={retrieval.get('k')}, page_tolerance={retrieval.get('page_tolerance')}",
        "library": __version__,
        "deepeval": deepeval_version,
        "target": f"{target.get('type')}:"
        f"{target.get('entry') or target.get('url') or target.get('stub')}",
    }
