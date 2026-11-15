"""The platform must not depend on the app it tests.

This is the property that makes the platform reusable, and it is the easiest one
to break by accident -- one convenient import and the evaluator only works for
this RAG app forever. So it is asserted rather than trusted.

The single sanctioned bridge is targets/inprocess_target.py, which exists to adapt
this app. Even there the import is LAZY (inside a function) so that importing the
platform does not require the app to be present or installable.
"""
import ast
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parent.parent / "llm_eval_pipeline"
ADAPTER = "targets/inprocess_target.py"

# config.py may name the app as DEFAULT CONFIG DATA -- the default `app_dir`, and
# where to look for a .env. That is a pointer the user can repoint at any app, not
# a dependency: nothing is imported and no behaviour assumes this app. The test
# below proves the name appears there only inside string literals.
NAME_ALLOWED_AS_DATA = {"config.py"}


def python_files() -> list[Path]:
    return sorted(p for p in PLATFORM.rglob("*.py") if "__pycache__" not in p.parts)


def imports_in(tree: ast.AST, *, top_level_only: bool) -> list[str]:
    """Module names imported, optionally restricted to module scope."""
    names: list[str] = []
    nodes = tree.body if top_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_there_are_files_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(python_files()) >= 10


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_module_names_rag_application(path):
    if path.name in NAME_ALLOWED_AS_DATA:
        pytest.skip("named only as default config data; see test_..._only_as_a_string")
    assert "RAG_Application" not in path.read_text(encoding="utf-8"), (
        f"{path.name} refers to RAG_Application by name. The app under test must "
        f"reach the platform through the Target contract, not by path."
    )


@pytest.mark.parametrize("name", sorted(NAME_ALLOWED_AS_DATA))
def test_allowlisted_files_name_the_app_only_as_a_string(name):
    """A default path is data. An import, attribute or call would not be."""
    source = (PLATFORM / name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    in_strings = sum(
        node.value.count("RAG_Application")
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    # Comments are not in the AST, so count them separately and allow them.
    in_comments = sum(
        line.count("RAG_Application") for line in source.splitlines() if line.lstrip().startswith("#")
    )
    assert source.count("RAG_Application") == in_strings + in_comments


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_top_level_app_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        name for name in imports_in(tree, top_level_only=True) if name == "app" or name.startswith("app.")
    ]
    assert not offenders, (
        f"{path.name} imports {offenders} at module scope. Importing the platform "
        f"would then require the app to be importable."
    )


def test_nothing_touches_the_app_at_all():
    """The rule is now absolute.

    There used to be ONE sanctioned bridge -- an in-process adapter that
    imported the medical RAG app directly. That file moved into the app's own
    repo, which is where an app-specific adapter belongs. So the library now has
    no exception: no module may import an application, at any depth, lazily or
    otherwise.
    """
    touching = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(n == "app" or n.startswith("app.") for n in imports_in(tree, top_level_only=False)):
            touching.append(path.relative_to(PLATFORM).as_posix())
    assert touching == [], f"the library must import no application, got {touching}"


def test_core_modules_import_without_the_app_present():
    """Everything but the adapter must import with no app on sys.path."""
    import importlib

    for module in (
        "llm_eval_pipeline.aggregate",
        "llm_eval_pipeline.dataset",
        "llm_eval_pipeline.gate",
        "llm_eval_pipeline.paired",
        "llm_eval_pipeline.report",
        "llm_eval_pipeline.retrieval",
        "llm_eval_pipeline.stats",
        "llm_eval_pipeline.store",
        # Both must import with no psycopg installed: the driver is pulled in
        # lazily so that an offline sqlite run never needs it.
        "llm_eval_pipeline.store_pg",
        "llm_eval_pipeline.storage",
        "llm_eval_pipeline.targets.base",
        "llm_eval_pipeline.fixtures.stub_targets",
    ):
        assert importlib.import_module(module) is not None
