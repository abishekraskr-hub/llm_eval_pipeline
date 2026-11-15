"""Static checks on `action.yml` and the workflows.

These exist because of a shipped bug. v0.1.0's action had TWO `env:` blocks in
one step, and GitHub rejected the whole manifest before running anything:

    action.yml (Line: 104, Col: 7): 'env' is already defined

Nothing caught it. The library's own CI drives `llmeval` directly rather than
through the action, so the manifest was published without ever being parsed by
anything strict. `yaml.safe_load` does not help: it accepts duplicate keys
silently and keeps the last one, so the file looked fine locally and was
unusable in the one place it mattered.

A GitHub runner is the only thing that can truly execute the action, but most of
what went wrong is visible without one.
"""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
ACTION = REPO / "action.yml"
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))


class StrictLoader(yaml.SafeLoader):
    """A loader that REFUSES duplicate mapping keys.

    GitHub's parser rejects them; PyYAML's silently takes the last. That gap is
    the entire reason v0.1.0's action was broken on arrival.
    """


def _no_duplicates(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.YAMLError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
)


def strict_load(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)


@pytest.mark.parametrize(
    "path", [ACTION, *WORKFLOWS], ids=lambda p: p.name
)
def test_no_duplicate_keys(path):
    """The exact bug that made v0.1.0's action unusable."""
    strict_load(path)


def test_there_are_workflows_to_check():
    """Guard against the glob silently matching nothing."""
    assert WORKFLOWS, "no workflows found -- this test file would pass vacuously"


class TestActionManifest:
    @pytest.fixture(scope="class")
    def action(self):
        return strict_load(ACTION)

    def test_it_is_a_composite_action(self, action):
        assert action["runs"]["using"] == "composite"

    def test_every_run_step_declares_a_shell(self, action):
        """A composite action's `run` step without `shell` is a hard error on
        the runner, and nothing local tells you."""
        for step in action["runs"]["steps"]:
            if "run" in step:
                assert step.get("shell"), f"step {step.get('name')!r} has no shell"

    def test_every_step_is_named(self, action):
        """Unnamed steps make a failing job summary unreadable."""
        for step in action["runs"]["steps"]:
            assert step.get("name") or step.get("uses"), step

    def test_every_referenced_input_exists(self, action):
        """A typo in `inputs.base_branch` vs `inputs.base-branch` expands to an
        empty string and the step quietly does the wrong thing."""
        import re

        declared = set(action.get("inputs") or {})
        referenced = set(re.findall(r"inputs\.([A-Za-z0-9_-]+)", ACTION.read_text(encoding="utf-8")))
        assert referenced <= declared, f"undeclared input(s): {sorted(referenced - declared)}"

    def test_the_outputs_point_at_a_real_step(self, action):
        import re

        step_ids = {s.get("id") for s in action["runs"]["steps"] if s.get("id")}
        for name, spec in (action.get("outputs") or {}).items():
            for step_id in re.findall(r"steps\.([A-Za-z0-9_-]+)\.", str(spec.get("value", ""))):
                assert step_id in step_ids, f"output {name!r} references unknown step {step_id!r}"

    def test_it_installs_the_library_from_its_own_checkout(self, action):
        """`github.action_path` is what keeps the library version equal to the
        action tag. Installing from PyPI or a branch would let the two drift,
        and a mismatched pair is very hard to diagnose from a report."""
        text = ACTION.read_text(encoding="utf-8")
        assert "github.action_path" in text

    def test_the_exit_codes_are_all_handled(self, action):
        """0/1/2/3 each mean something different, and 3 must not block."""
        gate = next(
            s for s in action["runs"]["steps"] if s.get("name") == "Apply the gate"
        )
        body = gate["run"]
        for code in ("0)", "1)", "2)", "3)"):
            assert code in body, f"exit code {code} is not handled"
        assert "*)" in body, "an unexpected exit code must not pass silently"


class TestLibraryCi:
    @pytest.fixture(scope="class")
    def ci(self):
        return strict_load(REPO / ".github" / "workflows" / "ci.yml")

    def test_it_proves_the_gate_can_fail(self, ci):
        """A gate that has only ever been seen to pass is not known to work."""
        names = [s.get("name", "") for s in ci["jobs"]["test"]["steps"]]
        assert any("gate fails" in n for n in names)

    def test_it_needs_no_secrets(self, ci):
        """The suite and both examples use the `fake` judge, so CI must run on a
        fork PR with no secrets available."""
        text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "secrets." not in text
