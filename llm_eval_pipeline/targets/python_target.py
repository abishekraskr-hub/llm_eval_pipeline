"""Load an adapter class the app's own repo provides, and call it directly.

This is the path for an app written in Python that you can import. The app repo
writes a small class -- typically 20-30 lines -- that wraps whatever it already
has and returns the contract from `base.py`. The library imports it by name:

    target:
      type: python
      entry: eval.rag_target:RagTarget

`entry` is `module:Class`, resolved with the CONFIG FILE's folder on `sys.path`.
Resolving from the config rather than the current working directory is what makes
`llmeval run --config ../other/llmeval.yaml` behave the same as running it from
that folder -- otherwise the same config would work or fail depending on where
you happened to be standing.

The class is instantiated with no arguments. Anything it needs comes from its own
repo: its own config file, its own environment. The library passes nothing,
because the moment it passes something app-shaped it stops being generic.
"""
import importlib
import sys
from pathlib import Path

from .base import TargetResponse, TargetUnavailable


class TargetConfigError(ValueError):
    """The `entry` could not be resolved. A config problem, not a run failure."""


def load_entry(entry: str, config_dir: Path):
    """Import `module:Class` with `config_dir` on sys.path, and return the class."""
    if ":" not in entry:
        raise TargetConfigError(
            f"target.entry must be 'module:Class', got {entry!r}. "
            f"Example: entry: eval.rag_target:RagTarget"
        )
    module_name, _, class_name = entry.partition(":")
    if not module_name or not class_name:
        raise TargetConfigError(f"target.entry must be 'module:Class', got {entry!r}")

    root = str(Path(config_dir).resolve())
    if root not in sys.path:
        # Prepended, not appended: the app's own modules must win over anything
        # of the same name that happens to be installed.
        sys.path.insert(0, root)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TargetConfigError(
            f"cannot import {module_name!r} for target.entry {entry!r}: {exc}\n"
            f"  looked in: {root}\n"
            f"  the module path is relative to the config file's folder"
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError:
        available = [n for n in vars(module) if not n.startswith("_")]
        raise TargetConfigError(
            f"{module_name!r} has no {class_name!r}. Found: {', '.join(sorted(available)[:12])}"
        ) from None


class PythonTarget:
    """Wraps the app's adapter class and normalises whatever it returns.

    An adapter may return a `TargetResponse` or a plain dict; both are accepted,
    because requiring an import from this library in the app's adapter would make
    the app depend on the evaluator's internals just to be evaluated.
    """

    def __init__(self, entry: str, config_dir: Path):
        self.entry = entry
        cls = load_entry(entry, config_dir)
        try:
            self._app = cls()
        except Exception as exc:
            # Construction usually means loading an index or a model, so a
            # failure here is "the app is not usable", not "the config is wrong".
            raise TargetUnavailable(
                f"{entry}: adapter raised on construction: {type(exc).__name__}: {exc}"
            ) from exc
        if not hasattr(self._app, "ask"):
            raise TargetConfigError(f"{entry}: the class has no ask(question) method")

    def ask(self, question: str) -> TargetResponse:
        try:
            raw = self._app.ask(question)
        except TargetUnavailable:
            raise
        except Exception as exc:
            raise TargetUnavailable(
                f"{self.entry}: ask() raised: {type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(raw, TargetResponse):
            return raw
        if isinstance(raw, dict):
            return TargetResponse.from_dict(raw)
        raise TargetConfigError(
            f"{self.entry}: ask() must return a dict or TargetResponse, got {type(raw).__name__}"
        )

    def describe(self) -> dict:
        info = {"type": "python", "entry": self.entry}
        describe = getattr(self._app, "describe", None)
        if callable(describe):
            try:
                extra = describe()
                if isinstance(extra, dict):
                    info.update(extra)
            except Exception as exc:  # describe() is optional metadata, never fatal
                info["describe_error"] = f"{type(exc).__name__}: {exc}"
        return info
