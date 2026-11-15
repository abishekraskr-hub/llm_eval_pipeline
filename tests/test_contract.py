"""The v1 target contract, and the config that drives it.

These are the tests for the properties that make the library reusable. Each one
guards a specific way the library could quietly become usable for only one kind
of app again.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_eval_pipeline.aggregate import RefusalPolicy, summarize
from llm_eval_pipeline.config import ConfigError, load_config, resolve, validate
from llm_eval_pipeline.retrieval import matches, score_retrieval
from llm_eval_pipeline.targets.base import (
    RetrievedSource,
    TargetResponse,
    rank_sources,
    supports,
)
from llm_eval_pipeline.targets.http_target import TargetConfigError, dig, expand_env
from llm_eval_pipeline.targets.python_target import PythonTarget, load_entry

REPO = Path(__file__).resolve().parent.parent
ECHO = REPO / "examples" / "echo_target" / "llmeval.yaml"
HTTP_STUB = REPO / "examples" / "http_stub" / "llmeval.yaml"


class TestSourceIdIsAnyStableId:
    """The library must not assume documents with pages."""

    @pytest.mark.parametrize(
        "hit,expected",
        [
            ({"source_id": "doc-7"}, "doc-7"),
            ({"file": "paper.pdf"}, "paper.pdf"),
            ({"source": "paper.pdf"}, "paper.pdf"),
            ({"url": "https://example.com/a"}, "https://example.com/a"),
            ({"id": "42"}, "42"),
            ({"uri": "s3://bucket/key"}, "s3://bucket/key"),
        ],
    )
    def test_every_accepted_spelling(self, hit, expected):
        assert rank_sources([hit])[0].source_id == expected

    def test_a_hit_with_no_identifier_is_dropped(self):
        """It cannot be matched against a gold label, so counting it would only
        dilute the metric."""
        assert rank_sources([{"score": 0.9, "page": 3}]) == []

    def test_file_is_an_alias_not_a_second_concept(self):
        source = RetrievedSource(source_id="doc-7", page=None, rank=1)
        assert source.file == "doc-7"

    def test_constructing_with_file_still_works(self):
        """An adapter written against the first version of the contract."""
        source = RetrievedSource(file="paper.pdf", page=4, rank=1)
        assert (source.source_id, source.file, source.page) == ("paper.pdf", "paper.pdf", 4)

    def test_to_dict_carries_both_names(self):
        """Run history and reports written before the rename are read by this
        same code, so both keys are emitted."""
        data = RetrievedSource(source_id="doc-7", page=1, rank=1).to_dict()
        assert data["source_id"] == "doc-7" and data["file"] == "doc-7"

    def test_ranks_are_renumbered_from_the_apps_order(self):
        """A target cannot hand us duplicate or sparse ranks."""
        ranked = rank_sources(
            [
                {"source_id": "a", "rank": 99},
                {"source_id": "a"},  # duplicate of the first, dropped
                {"source_id": "b", "rank": 99},
            ]
        )
        assert [(r.source_id, r.rank) for r in ranked] == [("a", 1), ("b", 2)]


class TestOptionalPages:
    """The single change that makes non-document retrieval work."""

    def test_pages_compare_only_when_both_sides_have_one(self):
        assert matches(("doc", 4), ("doc", 4)) is True
        assert matches(("doc", 4), ("doc", 9)) is False
        # Either side missing -> the id match stands alone.
        assert matches(("doc", None), ("doc", 4)) is True
        assert matches(("doc", 4), ("doc", None)) is True
        assert matches(("doc", None), ("doc", None)) is True

    def test_a_different_source_never_matches(self):
        assert matches(("a", None), ("b", None)) is False

    def test_a_page_less_app_scores_normally(self):
        """If a missing page were a mismatch, every app without page numbers
        would score 0 on every retrieval metric."""
        retrieved = rank_sources([{"source_id": "doc-7"}, {"source_id": "doc-9"}])
        scores = score_retrieval([("doc-7", None)], retrieved, k=5)
        assert (scores.hit, scores.recall, scores.rr) == (1.0, 1.0, 1.0)

    def test_page_tolerance_still_applies_when_both_have_pages(self):
        assert matches(("doc", 11), ("doc", 12), page_tolerance=1) is True
        assert matches(("doc", 10), ("doc", 12), page_tolerance=1) is False


class TestFaithfulnessNeedsContext:
    def test_contexts_are_preferred_when_given(self):
        response = TargetResponse(answer="a", contexts=["explicit"])
        assert response.faithfulness_contexts() == ["explicit"]

    def test_it_falls_back_to_the_retrieved_content(self):
        """An app can report WHAT it retrieved and the text in one go."""
        response = TargetResponse(
            answer="a",
            retrieved=[RetrievedSource(source_id="d", page=1, rank=1, content="passage")],
        )
        assert response.faithfulness_contexts() == ["passage"]

    def test_retrieval_without_content_leaves_faithfulness_unmeasurable(self):
        """Reporting WHERE a passage came from is not the same as reporting what
        it said. Retrieval can still be scored; grounding cannot."""
        response = TargetResponse(
            answer="a", retrieved=[RetrievedSource(source_id="d", page=1, rank=1)]
        )
        assert response.faithfulness_contexts() == []

        summary = summarize(
            [{"id": "a1", "kind": "answerable", "answer": "a", "contexts": []}]
        ).to_dict()
        assert summary["faithfulness"] is None, "unmeasured, which FAILS the gate"

    def test_retrieved_none_differs_from_retrieved_empty(self):
        """None = "this target does not expose retrieval"; [] = "it found
        nothing". The first is unmeasurable, the second is a measured zero."""
        assert TargetResponse.from_dict({"answer": "a"}).retrieved is None
        assert TargetResponse.from_dict({"answer": "a", "retrieved": []}).retrieved == []


class TestRefusalPrecedence:
    FLAGGED = {"answer": "I cannot help with that.", "refused": True}
    PATTERNED = {"answer": "The documents do not cover that."}

    def test_the_target_flag_wins(self):
        """The app knows which branch it took; no pattern can beat that."""
        policy = RefusalPolicy.from_config(
            {"refusal": {"use_target_flag": True, "patterns": [r"do not cover"]}}
        )
        # Pattern says refused, app says it is not.
        assert policy.verdict({"answer": "do not cover", "refused": False}) is False

    def test_patterns_decide_when_the_app_reports_nothing(self):
        policy = RefusalPolicy.from_config({"refusal": {"patterns": [r"do not cover"]}})
        assert policy.verdict(self.PATTERNED) is True
        assert policy.verdict({"answer": "Invoices are monthly."}) is False

    def test_the_flag_can_be_switched_off(self):
        policy = RefusalPolicy.from_config(
            {"refusal": {"use_target_flag": False, "patterns": [r"do not cover"]}}
        )
        assert policy.verdict({"answer": "do not cover", "refused": False}) is True

    def test_patterns_are_case_insensitive_regexes_not_substrings(self):
        policy = RefusalPolicy.from_config(
            {"refusal": {"patterns": [r"documents? (do not|don't) (cover|contain)"]}}
        )
        assert policy.verdict({"answer": "The DOCUMENT DOESN'T CONTAIN it."}) is False
        assert policy.verdict({"answer": "The DOCUMENTS DO NOT CONTAIN it."}) is True

    def test_no_flag_and_no_patterns_is_unmeasurable_not_a_pass(self):
        """The hole this closes: with the old hard-coded English list, an app
        that phrased refusals differently -- or answered in another language --
        scored a perfect 0.000 false-refusal rate while refusing everything."""
        policy = RefusalPolicy.from_config({"refusal": {"use_target_flag": False}})
        assert policy.can_measure is False
        assert policy.verdict(self.PATTERNED) is None

        summary = summarize(
            [
                {"id": "u1", "kind": "unanswerable", "answer": "whatever", "contexts": ["c"]},
                {"id": "a1", "kind": "answerable", "answer": "whatever", "contexts": ["c"]},
            ],
            refusal=policy,
        ).to_dict()
        assert summary["abstention_rate"] is None
        assert summary["false_refusal_rate"] is None


class TestHeaderSecrets:
    def test_env_references_are_expanded(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "s3cret")
        assert expand_env("Bearer ${MY_TOKEN}", "target.headers.Authorization") == "Bearer s3cret"

    def test_a_missing_variable_is_an_error_not_an_empty_string(self):
        """Sending `Bearer ` would surface as a confusing 401 from the app
        instead of as the config problem it is."""
        with pytest.raises(TargetConfigError, match="NOT_SET"):
            expand_env("Bearer ${NOT_SET}", "target.headers.Authorization")

    @pytest.mark.parametrize(
        "literal",
        [
            "AIzaSy" + "B" * 33,
            "sk-" + "a" * 25,
            "gsk_" + "b" * 40,
            "ghp_" + "c" * 36,
            "xoxb-" + "1" * 20,
        ],
    )
    def test_a_literal_looking_key_is_refused(self, literal):
        """A key written into a config file ends up in git history."""
        with pytest.raises(TargetConfigError, match="literal API key"):
            expand_env(literal, "target.headers.Authorization")

    def test_an_ordinary_value_passes_through(self):
        assert expand_env("application/json", "target.headers.Accept") == "application/json"


class TestResponseMapping:
    def test_dot_paths_reach_into_nested_replies(self):
        assert dig({"data": {"meta": {"refused": True}}}, "data.meta.refused") is True

    def test_a_missing_path_is_none_not_an_error(self):
        assert dig({"a": 1}, "b.c.d") is None

    def test_a_non_dict_on_the_way_down_is_none(self):
        assert dig({"a": [1, 2]}, "a.b") is None


class TestConfigPathsResolveFromTheConfig:
    """A config that worked locally and broke in CI is the failure this prevents."""

    def test_relative_paths_use_the_config_folder_not_the_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "eval").mkdir()
        (tmp_path / "eval" / "golden.yaml").write_text("items: []", encoding="utf-8")
        (tmp_path / "llmeval.yaml").write_text(
            "version: 1\ntarget:\n  type: stub\ndataset: eval/golden.yaml\n"
            "judge:\n  provider: fake\n",
            encoding="utf-8",
        )
        # Run from somewhere else entirely.
        monkeypatch.chdir(tmp_path.parent)
        cfg = load_config(tmp_path / "llmeval.yaml")
        assert resolve(cfg["dataset"], cfg) == tmp_path / "eval" / "golden.yaml"

    def test_absolute_paths_are_left_alone(self, tmp_path):
        cfg = {"_dir": str(tmp_path)}
        absolute = tmp_path / "elsewhere" / "x.yaml"
        assert resolve(absolute, cfg) == absolute

    def test_a_missing_config_says_so(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_an_unknown_top_level_key_is_rejected(self, tmp_path):
        """A typo in a gate setting would otherwise disable the thing it was
        meant to configure, and the run would pass for the wrong reason."""
        path = tmp_path / "llmeval.yaml"
        path.write_text("version: 1\nthreshholds:\n  faithfulness: 0.8\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="threshholds"):
            load_config(path)


class TestValidateNamesTheField:
    def _cfg(self, text, tmp_path):
        path = tmp_path / "llmeval.yaml"
        path.write_text(text, encoding="utf-8")
        return load_config(path)

    def test_a_python_target_needs_an_entry(self, tmp_path):
        cfg = self._cfg("version: 1\ntarget:\n  type: python\n", tmp_path)
        assert any(p.startswith("target.entry:") for p in validate(cfg))

    def test_an_entry_must_be_module_colon_class(self, tmp_path):
        cfg = self._cfg("version: 1\ntarget:\n  type: python\n  entry: just_a_module\n", tmp_path)
        assert any("module:Class" in p for p in validate(cfg))

    def test_an_http_target_needs_a_url(self, tmp_path):
        cfg = self._cfg("version: 1\ntarget:\n  type: http\n", tmp_path)
        assert any(p.startswith("target.url:") for p in validate(cfg))

    def test_an_unknown_target_type(self, tmp_path):
        cfg = self._cfg("version: 1\ntarget:\n  type: grpc\n", tmp_path)
        assert any(p.startswith("target.type:") for p in validate(cfg))

    def test_a_missing_dataset_file(self, tmp_path):
        cfg = self._cfg("version: 1\ntarget:\n  type: stub\ndataset: nope.yaml\n", tmp_path)
        assert any("file not found" in p for p in validate(cfg))

    def test_an_unknown_judge_provider(self, tmp_path):
        cfg = self._cfg("version: 1\njudge:\n  provider: anthropic\n", tmp_path)
        assert any(p.startswith("judge.provider:") for p in validate(cfg))

    def test_an_api_key_pasted_where_its_name_belongs(self, tmp_path):
        """`api_key_env` is the NAME of a variable. Pasting the key there would
        put it in git."""
        cfg = self._cfg(
            'version: 1\njudge:\n  provider: fake\n  api_key_env: "AIzaSyB-real-key"\n',
            tmp_path,
        )
        assert any("api_key_env" in p for p in validate(cfg))

    def test_a_password_in_the_config_is_refused(self, tmp_path):
        cfg = self._cfg(
            "version: 1\nstorage:\n  postgres:\n    password: hunter2\n", tmp_path
        )
        assert any("password_env" in p for p in validate(cfg))

    def test_a_broken_refusal_regex_names_its_index(self, tmp_path):
        cfg = self._cfg('version: 1\nrefusal:\n  patterns: ["[unclosed"]\n', tmp_path)
        assert any(p.startswith("refusal.patterns[0]:") for p in validate(cfg))

    def test_an_unknown_metric_in_thresholds(self, tmp_path):
        cfg = self._cfg("version: 1\nthresholds:\n  vibes: 0.9\n", tmp_path)
        assert any("vibes" in p for p in validate(cfg))

    def test_the_shipped_examples_are_valid(self):
        """If the quickstart drifts, this fails -- which is the point of using
        the examples as fixtures."""
        for path in (ECHO, HTTP_STUB):
            assert validate(load_config(path)) == [], path.name


class TestPythonEntry:
    def test_it_imports_relative_to_the_config_folder(self):
        cls = load_entry("eval.echo_target:EchoTarget", ECHO.parent)
        assert cls.__name__ == "EchoTarget"

    def test_a_bad_format_is_reported_before_importing(self):
        with pytest.raises(ValueError, match="module:Class"):
            load_entry("no_colon_here", ECHO.parent)

    def test_a_missing_module_says_where_it_looked(self, tmp_path):
        with pytest.raises(ValueError, match="looked in"):
            load_entry("nope:Thing", tmp_path)

    def test_a_missing_class_lists_what_is_there(self):
        with pytest.raises(ValueError, match="EchoTarget"):
            load_entry("eval.echo_target:Wrong", ECHO.parent)

    def test_a_dict_from_ask_is_normalised(self):
        target = PythonTarget("eval.echo_target:EchoTarget", ECHO.parent)
        response = target.ask("How long do expenses take to be reimbursed?")
        assert isinstance(response, TargetResponse)
        assert response.retrieved and response.retrieved[0].rank == 1
        assert response.refused is False

    def test_describe_is_merged_into_the_run_record(self):
        target = PythonTarget("eval.echo_target:EchoTarget", ECHO.parent)
        info = target.describe()
        assert info["type"] == "python"
        assert supports(info, "retrieval") is True
        assert supports(info, "nothing_declared") is None


class TestExamplesRunEndToEnd:
    """The quickstart IS the integration test. No key, no network, no database."""

    def _run(self, config: Path, *extra: str):
        return subprocess.run(
            [
                sys.executable, "-m", "llm_eval_pipeline.cli", "run",
                "--config", str(config), "--branch", "develop",
                "--db", str(config.parent / ".llmeval" / "test.sqlite"),
                "--report-dir", str(config.parent / ".llmeval" / "test-reports"),
                *extra,
            ],
            capture_output=True, text=True, cwd=str(REPO), timeout=300,
        )

    def test_echo_target_passes(self, tmp_path):
        result = self._run(ECHO)
        assert "GATE: PASSED" in result.stdout, result.stdout + result.stderr
        assert result.returncode == 0

    def test_echo_target_makes_no_judge_calls(self):
        result = self._run(ECHO)
        assert "judge requests 0" in result.stdout
        assert "fake:no-model" in result.stdout

    def test_validate_passes_on_both_examples(self):
        for config in (ECHO, HTTP_STUB):
            result = subprocess.run(
                [sys.executable, "-m", "llm_eval_pipeline.cli", "validate",
                 "--config", str(config)],
                capture_output=True, text=True, cwd=str(REPO), timeout=120,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "valid." in result.stdout

    def test_http_stub_starts_maps_and_stops(self):
        """Exercises `start`, the health poll, and every remapped field name --
        the app's API is `query`/`reply`/`sources`/`url`/`text`, none of which
        are what this library calls them."""
        result = self._run(HTTP_STUB)
        assert "GATE: PASSED" in result.stdout, result.stdout + result.stderr
        # URL-based gold labels matched, including one with no page at all.
        assert "Retrieval hit rate@5" in result.stdout

    def test_a_stub_target_fails_the_gate(self):
        """A gate that has only ever been seen to pass is not known to work."""
        result = self._run(ECHO, "--stub", "refuse_everything")
        assert "GATE: FAILED" in result.stdout, result.stdout + result.stderr
        assert result.returncode == 1, "a quality failure must exit 1"

    def test_the_report_records_the_comparability_hash(self):
        self._run(ECHO)
        reports = sorted((ECHO.parent / ".llmeval" / "test-reports").glob("run_*.json"))
        assert reports
        data = json.loads(reports[-1].read_text(encoding="utf-8"))
        # The library version is in there, so a version bump resets the baseline
        # rather than comparing numbers produced by different code.
        assert "library" in data["comparability"]
        assert data["llmeval_version"] == data["comparability"]["library"]
