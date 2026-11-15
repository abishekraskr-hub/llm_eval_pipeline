"""Dataset loading and validation -- the cheap failures, before any API call."""
import json

import pytest

from llm_eval_pipeline.dataset import (
    DatasetError,
    dataset_fingerprint,
    dataset_hash,
    load_dataset,
)


def write(tmp_path, items, name="golden.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"items": items}), encoding="utf-8")
    return path


ANSWERABLE = {
    "id": "a1",
    "question": "What are the side effects?",
    "kind": "answerable",
    "gold_sources": [{"file": "doc.pdf", "page": 12}],
}
UNANSWERABLE = {"id": "u1", "question": "Who won in 2022?", "kind": "unanswerable"}


class TestGoldSources:
    def test_answerable_without_gold_sources_is_rejected(self, tmp_path):
        path = write(tmp_path, [{"id": "a1", "question": "q", "kind": "answerable"}])
        with pytest.raises(DatasetError) as exc:
            load_dataset(path)
        assert "no gold_sources" in str(exc.value)
        assert "source_id" in str(exc.value)
        # The message has to tell you how to fix it.
        assert "suggest_gold_sources" in str(exc.value)

    def test_can_be_relaxed_for_bootstrapping(self, tmp_path):
        path = write(tmp_path, [{"id": "a1", "question": "q", "kind": "answerable"}])
        items = load_dataset(path, require_gold_sources=False)
        assert items[0].gold_sources == []

    def test_unanswerable_needs_none(self, tmp_path):
        items = load_dataset(write(tmp_path, [UNANSWERABLE]))
        assert items[0].gold_sources == []
        assert items[0].is_unanswerable

    def test_unanswerable_with_gold_sources_is_rejected(self, tmp_path):
        bad = dict(UNANSWERABLE, gold_sources=[{"file": "d.pdf", "page": 1}])
        with pytest.raises(DatasetError, match="unanswerable but has gold_sources"):
            load_dataset(write(tmp_path, [bad]))

    def test_parsed_as_tuples(self, tmp_path):
        items = load_dataset(write(tmp_path, [ANSWERABLE]))
        assert items[0].gold_sources == [("doc.pdf", 12)]

    def test_duplicates_collapse(self, tmp_path):
        rec = dict(
            ANSWERABLE,
            gold_sources=[
                {"file": "doc.pdf", "page": 12},
                {"file": "doc.pdf", "page": 12},
            ],
        )
        assert load_dataset(write(tmp_path, [rec]))[0].gold_sources == [("doc.pdf", 12)]

    def test_missing_page_is_accepted(self, tmp_path):
        """A page is OPTIONAL, and that is what lets this work beyond documents.

        An app retrieving from OpenSearch, a SQL table or the web has a stable
        source_id and no page at all. Requiring one would have scored every such
        app 0 on every retrieval metric.
        """
        rec = dict(ANSWERABLE, gold_sources=[{"source_id": "doc-7"}])
        items = load_dataset(write(tmp_path, [rec]))
        assert items[0].gold_sources == [("doc-7", None)]

    def test_a_bare_string_is_shorthand_for_a_page_less_source(self, tmp_path):
        rec = dict(ANSWERABLE, gold_sources=["https://example.com/a"])
        items = load_dataset(write(tmp_path, [rec]))
        assert items[0].gold_sources == [("https://example.com/a", None)]

    def test_file_is_still_accepted_as_an_alias_for_source_id(self, tmp_path):
        """Datasets written against the first version of the format still load."""
        rec = dict(ANSWERABLE, gold_sources=[{"file": "doc.pdf", "page": 3}])
        items = load_dataset(write(tmp_path, [rec]))
        assert items[0].gold_sources == [("doc.pdf", 3)]

    def test_zero_page_is_rejected_because_pages_are_1_based(self, tmp_path):
        rec = dict(ANSWERABLE, gold_sources=[{"file": "doc.pdf", "page": 0}])
        with pytest.raises(DatasetError, match="1-based"):
            load_dataset(write(tmp_path, [rec]))

    def test_non_integer_page_is_rejected(self, tmp_path):
        rec = dict(ANSWERABLE, gold_sources=[{"file": "doc.pdf", "page": "twelve"}])
        with pytest.raises(DatasetError, match="must be an integer"):
            load_dataset(write(tmp_path, [rec]))

    def test_legacy_source_plus_page_is_accepted(self, tmp_path):
        """A hand-written item that already names one file and page still loads."""
        rec = {"id": "a1", "question": "q", "kind": "answerable", "source": "d.pdf", "page": 3}
        assert load_dataset(write(tmp_path, [rec]))[0].gold_sources == [("d.pdf", 3)]

    def test_legacy_source_without_page_is_now_a_complete_label(self, tmp_path):
        """It used to be rejected, because a page was mandatory. Now the
        source_id alone locates the thing, and the page is optional."""
        rec = {"id": "a1", "question": "q", "kind": "answerable", "source": "d.pdf"}
        items = load_dataset(write(tmp_path, [rec]))
        assert items[0].gold_sources == [("d.pdf", None)]


class TestExistingValidation:
    def test_duplicate_ids_rejected(self, tmp_path):
        with pytest.raises(DatasetError, match="duplicate item id"):
            load_dataset(write(tmp_path, [ANSWERABLE, dict(ANSWERABLE)]))

    def test_bad_kind_rejected(self, tmp_path):
        with pytest.raises(DatasetError, match="answerable\\|unanswerable"):
            load_dataset(write(tmp_path, [dict(ANSWERABLE, kind="maybe")]))

    def test_missing_question_rejected(self, tmp_path):
        with pytest.raises(DatasetError, match="no question"):
            load_dataset(write(tmp_path, [{"id": "a1", "kind": "unanswerable"}]))

    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope.json")

    def test_limit_takes_the_first_n(self, tmp_path):
        items = [dict(UNANSWERABLE, id=f"u{i}", question=f"q{i}") for i in range(5)]
        assert [i.id for i in load_dataset(write(tmp_path, items), limit=2)] == ["u0", "u1"]

    def test_tag_filter(self, tmp_path):
        items = [
            dict(UNANSWERABLE, id="u1", tags=["safety"]),
            dict(UNANSWERABLE, id="u2", tags=["dosing"]),
        ]
        got = load_dataset(write(tmp_path, items), tag="dosing")
        assert [i.id for i in got] == ["u2"]


class TestDatasetHash:
    def test_stable_and_order_independent(self, tmp_path):
        a = load_dataset(write(tmp_path, [ANSWERABLE, UNANSWERABLE], "a.json"))
        b = load_dataset(write(tmp_path, [UNANSWERABLE, ANSWERABLE], "b.json"))
        assert dataset_hash(a) == dataset_hash(b)

    def test_changes_when_a_question_changes(self, tmp_path):
        a = load_dataset(write(tmp_path, [ANSWERABLE], "a.json"))
        b = load_dataset(write(tmp_path, [dict(ANSWERABLE, question="different?")], "b.json"))
        assert dataset_hash(a) != dataset_hash(b)

    def test_changes_when_gold_labels_change(self, tmp_path):
        """A relabelled page changes what the retrieval metrics mean."""
        a = load_dataset(write(tmp_path, [ANSWERABLE], "a.json"))
        moved = dict(ANSWERABLE, gold_sources=[{"file": "doc.pdf", "page": 13}])
        b = load_dataset(write(tmp_path, [moved], "b.json"))
        assert dataset_hash(a) != dataset_hash(b)

    def test_ignores_tags(self, tmp_path):
        a = load_dataset(write(tmp_path, [ANSWERABLE], "a.json"))
        b = load_dataset(write(tmp_path, [dict(ANSWERABLE, tags=["new"])], "b.json"))
        assert dataset_hash(a) == dataset_hash(b)


class TestDatasetFingerprint:
    def test_ignores_limit_so_a_pr_subset_matches_a_full_baseline(self, tmp_path):
        """Without this, the paired gate would never run on a --limit PR."""
        items = [
            dict(ANSWERABLE, id=f"a{i}", question=f"q{i}", gold_sources=[{"file": "d.pdf", "page": i + 1}])
            for i in range(6)
        ]
        path = write(tmp_path, items)
        full = dataset_fingerprint(path)
        # The slice has a different hash; the file does not.
        assert dataset_hash(load_dataset(path, limit=2)) != full
        assert dataset_fingerprint(path) == full

    def test_still_changes_when_the_file_changes(self, tmp_path):
        a = dataset_fingerprint(write(tmp_path, [ANSWERABLE], "a.json"))
        b = dataset_fingerprint(write(tmp_path, [ANSWERABLE, UNANSWERABLE], "b.json"))
        assert a != b
