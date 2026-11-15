"""The common API every evaluated application must satisfy.

This is the contract that makes the library reusable: any LLM app -- RAG, agent,
plain prompt, any language, any storage -- becomes evaluable by exposing a single
`ask()` that returns an answer plus whatever it retrieved. Nothing else in the
library knows what the app is or how it works.

Identifying a retrieved passage
-------------------------------

A passage is located by `source_id` plus an optional `page`.

`source_id` is ANY stable identifier the app can produce for the thing it
retrieved: a filename, a URL, a database primary key, a document id. It must be
stable across runs, because the golden dataset's gold labels are written against
it -- a volatile chunk id would make every label rot on the next re-index.

`page` is optional, and that is what lets this work outside document retrieval.
A match needs the pages to agree only when BOTH sides declare one, so a gold
label with no page matches on `source_id` alone. An app retrieving from
OpenSearch or a SQL table simply omits it.

`file=` is still accepted everywhere `source_id=` is, because the first version
of this contract used that name. It is an alias, not a second concept.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedSource:
    """One passage location the app retrieved, in rank order.

    Separate from the free-form `sources` list (which carries whatever per-hit
    detail an app wants to report) because the retrieval metrics need exactly
    three things and need them in a fixed shape: what was retrieved, where in
    it, and how highly it ranked.
    """

    source_id: str = ""
    page: int | None = None
    rank: int = 0  # 1 = best
    # The passage text the app actually gave the model. Faithfulness is scored
    # against this. Optional: a target may report WHAT it retrieved without
    # reporting the contents, in which case faithfulness falls back to the
    # response's `contexts`.
    content: str | None = None

    def __init__(
        self,
        source_id: str | None = None,
        page: int | None = None,
        rank: int = 0,
        content: str | None = None,
        *,
        file: str | None = None,
    ):
        """Written out by hand so that `file=` works as a keyword too.

        An adapter written against the first version of this contract passes
        `file=`; the dataclass-generated `__init__` would reject it. `object.
        __setattr__` is how a frozen dataclass assigns its own fields.
        """
        if source_id is None and file is None:
            raise TypeError("RetrievedSource needs source_id (or its alias, file)")
        object.__setattr__(self, "source_id", str(source_id if source_id is not None else file))
        object.__setattr__(self, "page", None if page is None else int(page))
        object.__setattr__(self, "rank", int(rank))
        object.__setattr__(self, "content", content)

    @property
    def file(self) -> str:
        """The original name for `source_id`, kept so existing callers work."""
        return self.source_id

    def to_dict(self) -> dict:
        data = asdict(self)
        # `file` is written out as well: run history and reports produced before
        # the rename are read back by this same code.
        data["file"] = self.source_id
        return data


def _source_id_of(hit: dict) -> str | None:
    """Accept any of the spellings an app or an older config might use."""
    for key in ("source_id", "file", "source", "id", "uri", "url", "path"):
        value = hit.get(key)
        if value:
            return str(value)
    return None


def _page_of(hit: dict) -> int | None:
    page = hit.get("page")
    if page is None or page == "":
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def rank_sources(sources: list[dict] | None) -> list[RetrievedSource]:
    """Turn an app's per-hit dicts into a ranked, de-duplicated list.

    An agentic app may search several times for one question, so the same
    passage can come back more than once. We keep **first appearance** order --
    the first search's hits rank above the second's -- de-duplicate on
    (source_id, page), and then number from 1.

    Ranking by score instead would be wrong: scores from different queries are
    not comparable, because each query is a different point in the embedding
    space.

    A hit with no usable `source_id` is dropped: it cannot be matched against a
    gold label, so counting it would only dilute the metric.
    """
    ranked: list[RetrievedSource] = []
    seen: set[tuple[str, int | None]] = set()
    for hit in sources or []:
        source_id = _source_id_of(hit)
        if not source_id:
            continue
        key = (source_id, _page_of(hit))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            RetrievedSource(
                source_id=key[0],
                page=key[1],
                rank=len(ranked) + 1,
                content=hit.get("content") or hit.get("text") or None,
            )
        )
    return ranked


@dataclass
class TraceStep:
    """One step an agentic app took, for debugging and step metrics."""

    type: str  # "retrieval" | "tool" | "llm"
    name: str = ""
    output_summary: str = ""
    latency_ms: float | None = None
    tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TargetResponse:
    """One answer from the application under test. Only `answer` is required."""

    answer: str
    # The passages faithfulness is judged against. Kept as its own field rather
    # than derived only from `retrieved`, because an app can report the text it
    # used without being able to say which document each piece came from.
    contexts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    # Ranked retrieval locations. None means "this target does not expose its
    # retrieval", which is NOT the same as an empty list ("it retrieved
    # nothing"). The gate must fail an unmeasurable metric, so the distinction
    # has to survive all the way to the summary.
    retrieved: list[RetrievedSource] | None = None
    # Whether the app itself says it declined to answer. None means it does not
    # report that, and the configured refusal patterns decide instead.
    refused: bool | None = None
    trace: list[TraceStep] | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = "unknown"

    def faithfulness_contexts(self) -> list[str]:
        """The text faithfulness is scored against.

        Prefers explicit `contexts`, then falls back to the content carried on
        the retrieved passages. One of the two must be present or faithfulness
        is unmeasurable -- which FAILS the gate rather than scoring zero.
        """
        if self.contexts:
            return list(self.contexts)
        return [r.content for r in (self.retrieved or []) if r.content]

    @classmethod
    def from_dict(cls, data: dict) -> "TargetResponse":
        retrieved = None
        if data.get("retrieved") is not None:
            # Ranks are renumbered from the app's order, so a target cannot hand
            # us duplicate or sparse ranks.
            retrieved = rank_sources(data["retrieved"])

        usage = data.get("usage") or {}
        trace = None
        if data.get("trace") is not None:
            trace = [
                TraceStep(
                    type=str(step.get("type") or "unknown"),
                    name=str(step.get("name") or ""),
                    output_summary=str(step.get("output_summary") or ""),
                    latency_ms=step.get("latency_ms"),
                    tokens=step.get("tokens"),
                )
                for step in data["trace"]
            ]

        refused = data.get("refused")
        return cls(
            answer=data.get("answer") or "",
            contexts=list(data.get("contexts") or []),
            sources=list(data.get("sources") or []),
            retrieved=retrieved,
            refused=None if refused is None else bool(refused),
            trace=trace,
            latency_s=float(data.get("latency_s") or 0.0),
            input_tokens=int(usage.get("input_tokens") or data.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or data.get("output_tokens") or 0),
            cost_usd=float(data.get("cost_usd") or 0.0),
            model=str(data.get("model") or "unknown"),
        )


@runtime_checkable
class Target(Protocol):
    """An application under evaluation."""

    def ask(self, question: str) -> TargetResponse: ...

    def describe(self) -> dict:
        """Config worth recording with the run: name, model, prompt version, index id.

        Runs are grouped by this, so a metric change can be traced to whichever
        knob actually moved. Should carry `supports`, so the library can say
        whether the retrieval and trace metrics are applicable at all:

            {"name": "my-rag", "model": "...",
             "supports": {"retrieval": True, "trace": False}}
        """
        ...


class TargetUnavailable(RuntimeError):
    """The app could not be reached or is out of quota -- not a bad answer.

    Kept separate so a run that dies on an outage or a quota wall is never
    reported as a quality failure. It maps to exit code 3, which warns rather
    than blocking a merge.
    """


def supports(info: dict[str, Any] | None, capability: str) -> bool | None:
    """Read one capability flag out of `describe()`. None = the app did not say."""
    block = (info or {}).get("supports")
    if isinstance(block, dict) and capability in block:
        return bool(block[capability])
    # The first version of describe() used a flat key.
    legacy = (info or {}).get(f"supports_{capability}")
    return None if legacy is None else bool(legacy)
