"""The golden dataset: the fixed benchmark every run is scored against.

Kept deliberately dumb -- JSON or YAML in, validated records out. The value is in
the questions, not in this loader.

GOLD LOCATIONS. Each entry of `gold_sources` is a `source_id` plus an optional
`page`:

    gold_sources:
      - { source_id: "handbook", page: 12 }    # a document, located to the page
      - { source_id: "doc-7" }                 # a record with no pages at all
      - { source_id: "https://example.com/a" } # a web page

`source_id` must be whatever the app reports for the same thing, and must be
STABLE across runs -- a volatile chunk id would make every label rot on the next
re-index. `file` is accepted as an alias, because the first version of this
format used that name.

PAGE NUMBERING, when pages are used: `page` is **1-based**, which is what a
document reader and a human citation both mean by "page 4". A gold label and a
retrieved hit are then directly comparable with no offset. A page of 0 is
rejected rather than silently shifted, since a 0-based index would make every
label off by one. Pages are compared only when both the label and the hit have
one -- see `retrieval.matches`.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

# (source_id, 1-based page or None)
Location = tuple[str, int | None]


@dataclass
class GoldenItem:
    id: str
    question: str
    # The known-good answer. Optional: an item can be useful for faithfulness and
    # relevance scoring even without one.
    expected_answer: str | None = None
    # "answerable" | "unanswerable" -- unanswerable items are how you measure
    # whether the app refuses rather than invents, which is the whole point of a
    # hallucination gate. Answerable items are also where false_refusal_rate is
    # measured, so both kinds have to be present for the gate to be complete.
    kind: str = "answerable"
    # The page(s) that actually contain the answer, human-verified. Required on
    # answerable items: without them retrieval cannot be scored, and a silently
    # unlabelled item would quietly shrink the denominator of every retrieval
    # metric instead of failing.
    gold_sources: list[Location] = field(default_factory=list)
    # Free-form grouping (drug name, document, difficulty) for per-slice reporting.
    tags: list[str] = field(default_factory=list)
    source: str | None = None
    page: int | None = None

    @property
    def is_unanswerable(self) -> bool:
        return self.kind == "unanswerable"

    @property
    def is_answerable(self) -> bool:
        return self.kind == "answerable"


class DatasetError(ValueError):
    pass


# Every spelling accepted for a gold entry's identifier. `source_id` is the
# documented one; the rest exist so a dataset written for an earlier version, or
# for an app that naturally names things differently, still loads.
_ID_KEYS = ("source_id", "file", "source", "id", "uri", "url", "path")


def _parse_gold_sources(rec: dict, item_id: str) -> list[Location]:
    """Read `gold_sources`, falling back to a legacy single `source` + `page`.

    `page` is OPTIONAL. An entry with an id and no page is a complete label for
    anything that has no pages -- a database row, an API record, a web page --
    and `retrieval.matches` compares pages only when both sides have one.
    """
    raw = rec.get("gold_sources")
    if raw is None:
        if rec.get("source"):
            raw = [{"source_id": rec["source"], "page": rec.get("page")}]
        else:
            return []
    if not isinstance(raw, list):
        raise DatasetError(f"item {item_id}: gold_sources must be a list")

    out: list[Location] = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            # Shorthand for a source with no page.
            entry = {"source_id": entry}
        if not isinstance(entry, dict):
            raise DatasetError(
                f"item {item_id}: gold_sources[{i}] must be an object or a string"
            )
        name = next((entry[k] for k in _ID_KEYS if entry.get(k)), None)
        if not name:
            raise DatasetError(
                f"item {item_id}: gold_sources[{i}] has no 'source_id' "
                f"(accepted: {', '.join(_ID_KEYS)})"
            )
        page = entry.get("page")
        page_no: int | None = None
        if page is not None and page != "":
            try:
                page_no = int(page)
            except (TypeError, ValueError):
                raise DatasetError(
                    f"item {item_id}: gold_sources[{i}].page must be an integer "
                    f"or omitted, got {page!r}"
                ) from None
            if page_no < 1:
                raise DatasetError(
                    f"item {item_id}: gold_sources[{i}].page is {page_no}; "
                    f"pages are 1-based"
                )
        location = (str(name), page_no)
        if location not in out:
            out.append(location)
    return out


def load_dataset(
    path: str | Path,
    limit: int | None = None,
    tag: str | None = None,
    require_gold_sources: bool = True,
) -> list[GoldenItem]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"golden dataset not found: {path}")

    text = path.read_text(encoding="utf-8")
    # YAML is a superset of JSON, so one parser could read both -- but JSON gets
    # its own branch because its error messages point at the offending line far
    # more precisely, and a malformed dataset is a thing people have to fix by
    # hand.
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DatasetError(f"{path} is not valid YAML: {exc}") from exc
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path} is not valid JSON: {exc}") from exc

    records = raw.get("items", raw) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise DatasetError(f"{path} must hold a list of items (or {{'items': [...]}})")

    items: list[GoldenItem] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DatasetError(f"item {i} is not an object")
        question = (rec.get("question") or "").strip()
        if not question:
            raise DatasetError(f"item {i} has no question")
        item_id = str(rec.get("id") or f"q{i + 1:03d}")
        if item_id in seen:
            # Duplicate ids would silently overwrite each other in the results store,
            # and would break pairing against a baseline run.
            raise DatasetError(f"duplicate item id {item_id!r}")
        seen.add(item_id)
        kind = rec.get("kind", "answerable")
        if kind not in ("answerable", "unanswerable"):
            raise DatasetError(f"item {item_id}: kind must be answerable|unanswerable, got {kind!r}")

        gold_sources = _parse_gold_sources(rec, item_id)
        if kind == "answerable" and require_gold_sources and not gold_sources:
            raise DatasetError(
                f"item {item_id} is answerable but has no gold_sources. Add the "
                f"source(s) that contain the answer, e.g.\n"
                f'  gold_sources: [{{source_id: "handbook", page: 12}}]\n'
                f'  gold_sources: [{{source_id: "doc-7"}}]   # no pages\n'
                f"`python -m llm_eval_pipeline.scripts.suggest_gold_sources` shows what "
                f"your app currently retrieves. Then verify each label by reading "
                f"it: a wrong gold label makes a healthy retriever look broken."
            )
        if kind == "unanswerable" and gold_sources:
            raise DatasetError(
                f"item {item_id} is unanswerable but has gold_sources. An unanswerable "
                f"question has no correct page by definition -- drop one or the other."
            )

        items.append(
            GoldenItem(
                id=item_id,
                question=question,
                expected_answer=(rec.get("expected_answer") or None),
                kind=kind,
                gold_sources=gold_sources,
                tags=list(rec.get("tags") or []),
                source=rec.get("source"),
                page=rec.get("page"),
            )
        )

    if tag:
        items = [it for it in items if tag in it.tags]
    if limit is not None:
        items = items[:limit]
    if not items:
        raise DatasetError("dataset is empty after filtering")
    return items


def dataset_fingerprint(path: str | Path, require_gold_sources: bool = True) -> str:
    """Hash of the WHOLE dataset file, ignoring --limit and --tag.

    This must NOT be the hash of the slice that was run. A PR runs a 20-question
    subset while `main` runs all 100; if the identity of the benchmark changed with
    the slice, those two runs would never be considered comparable and the paired
    gate would be permanently skipped on every PR -- which is exactly where it is
    most needed. Pairing then happens over the items the two runs share, which the
    coverage check already handles.
    """
    return dataset_hash(load_dataset(path, require_gold_sources=require_gold_sources))


def dataset_hash(items: list[GoldenItem]) -> str:
    """Identity of the benchmark itself, for baseline comparability.

    Two runs are only comparable question-by-question if they asked the same
    questions. This covers id, question text, kind and gold labels -- everything
    that changes what a score means. It deliberately ignores `tags` and
    `expected_answer`, which no metric currently reads.
    """
    import hashlib

    digest = hashlib.sha256()
    for item in sorted(items, key=lambda it: it.id):
        digest.update(item.id.encode("utf-8"))
        digest.update(item.question.encode("utf-8"))
        digest.update(item.kind.encode("utf-8"))
        for file_name, page in sorted(item.gold_sources):
            digest.update(f"{file_name}:{page}".encode("utf-8"))
    return digest.hexdigest()[:16]
