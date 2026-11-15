"""Fake targets with known-bad behaviour, for proving the gate actually fails.

A gate that has only ever been seen to pass is not known to work. These need no
API key, no index and no network, so the gate can be verified offline and in a
unit test -- which means a change to the gate itself gets caught.

Select one from eval_config.yaml:

    target:
      kind: stub
      stub: refuse_everything

or from the command line:

    python -m llm_eval_pipeline.cli run --target-kind stub --stub refuse_everything
"""
from ..targets.base import RetrievedSource, TargetResponse

# A plausible-looking passage, so faithfulness has something to score against.
_PASSAGE = (
    "Reported adverse events for this agent include headache and abdominal pain, "
    "with diarrhoea observed in short-term use."
)


class _StubTarget:
    """Common plumbing. Subclasses define `answer_for`."""

    name = "stub"
    description = "a stub target"
    # Where this stub claims to have retrieved from. Matches the gold labels in
    # the shipped dataset closely enough to exercise the retrieval metrics.
    retrieves = True

    def answer_for(self, question: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def ask(self, question: str) -> TargetResponse:
        retrieved = (
            [
                RetrievedSource(source_id="doc-a", page=1, rank=1),
                RetrievedSource(source_id="doc-b", page=2, rank=2),
            ]
            if self.retrieves
            else None
        )
        return TargetResponse(
            answer=self.answer_for(question),
            contexts=[_PASSAGE] if self.retrieves else [],
            sources=[],
            retrieved=retrieved,
            latency_s=0.01,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.0,
            model=f"stub:{self.name}",
        )

    def describe(self) -> dict:
        return {
            "kind": "stub",
            "stub": self.name,
            "description": self.description,
            "model": f"stub:{self.name}",
            "supports_retrieval": self.retrieves,
        }


class RefuseEverythingTarget(_StubTarget):
    """Refuses every question, answerable or not.

    This is the failure `abstention_rate` cannot see on its own: it scores a
    perfect 1.0 here, because every unanswerable question was correctly refused.
    The app is useless, and only `false_refusal_rate` says so.
    """

    name = "refuse_everything"
    description = "refuses every question"

    def answer_for(self, question: str) -> str:
        return "This is not mentioned in the available documents."


class AnswerEverythingTarget(_StubTarget):
    """Never refuses -- invents an answer for questions with no source.

    The mirror image: `false_refusal_rate` is a perfect 0.0 while the app
    confidently makes things up about questions the documents do not cover.
    """

    name = "answer_everything"
    description = "never refuses; invents answers"

    def answer_for(self, question: str) -> str:
        return (
            "Yes. This agent causes headache, abdominal pain and kidney failure in "
            "roughly 12% of patients, and the recommended dose is 40 mg twice daily."
        )


class NoRetrievalTarget(_StubTarget):
    """Answers fluently but reports no retrieval at all.

    Reproduces the regression that started all of this: the answers stay relevant,
    faithfulness becomes unmeasurable, and the retrieval metrics have nothing to
    score. Everything here must FAIL by the unmeasured rule, not pass quietly.
    """

    name = "no_retrieval"
    description = "answers without retrieving; reports no sources"
    retrieves = False

    def answer_for(self, question: str) -> str:
        return "Headache and abdominal pain are the commonly reported adverse events."


STUBS: dict[str, type[_StubTarget]] = {
    RefuseEverythingTarget.name: RefuseEverythingTarget,
    AnswerEverythingTarget.name: AnswerEverythingTarget,
    NoRetrievalTarget.name: NoRetrievalTarget,
}


def build_stub(name: str) -> _StubTarget:
    if name not in STUBS:
        raise ValueError(f"unknown stub target {name!r}; known: {sorted(STUBS)}")
    return STUBS[name]()
