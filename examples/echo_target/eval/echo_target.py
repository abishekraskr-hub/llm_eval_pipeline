"""A fake app, so the quickstart and CI need no API key, no index and no network.

This is also the shortest honest answer to "what do I have to write to evaluate
my app?" -- an adapter is a class with `ask()` and `describe()`. Everything else
is the app you already have.

The behaviour is rigged to be *interesting* rather than good: it answers the
questions whose keywords it recognises and declines the rest, so one run
exercises faithfulness, relevance, retrieval, abstention AND false refusal. A
stub that answered everything perfectly would prove only that the pipeline runs.
"""

# A pretend index: source_id -> (page, text). `source_id` is any stable id; here
# they are filenames, but a URL or a database key works identically.
CORPUS = {
    "handbook.md": [
        (1, "Expenses are reimbursed within 30 days of approval."),
        (2, "Travel over 500 km requires manager approval in advance."),
    ],
    "policy.md": [
        (1, "Annual leave is 25 days plus public holidays."),
        (3, "Unused leave may be carried over, to a maximum of 5 days."),
    ],
}

# Which passages answer which question, by keyword. A real app would search.
KEYWORDS = {
    "expense": [("handbook.md", 1)],
    "reimburse": [("handbook.md", 1)],
    "travel": [("handbook.md", 2)],
    "leave": [("policy.md", 1), ("policy.md", 3)],
    "holiday": [("policy.md", 1)],
    "carry": [("policy.md", 3)],
}

REFUSAL = "That is not covered by the available documents."


def _lookup(source_id: str, page: int) -> str:
    for number, text in CORPUS[source_id]:
        if number == page:
            return text
    return ""


class EchoTarget:
    """An adapter. The whole contract is these two methods."""

    def ask(self, question: str) -> dict:
        lowered = question.lower()
        hits: list[tuple[str, int]] = []
        for keyword, locations in KEYWORDS.items():
            if keyword in lowered:
                for location in locations:
                    if location not in hits:
                        hits.append(location)

        if not hits:
            # Declines, and says so explicitly rather than only in prose. The
            # flag is what `refusal.use_target_flag` trusts, and it beats any
            # pattern: the app knows which branch it took.
            return {
                "answer": REFUSAL,
                "retrieved": [],
                "refused": True,
                "usage": {"input_tokens": 12, "output_tokens": 9},
                "latency_s": 0.01,
                "model": "echo-v1",
            }

        retrieved = [
            {
                "source_id": source_id,
                "page": page,
                "content": _lookup(source_id, page),
                "rank": i + 1,
            }
            for i, (source_id, page) in enumerate(hits)
        ]
        # Every sentence is a quote from a retrieved passage, so a faithfulness
        # judge has something real to agree with.
        answer = " ".join(hit["content"] for hit in retrieved)
        return {
            "answer": answer,
            "retrieved": retrieved,
            "refused": False,
            "usage": {"input_tokens": 40 + 10 * len(retrieved), "output_tokens": 20},
            "trace": [
                {
                    "type": "retrieval",
                    "name": "keyword_lookup",
                    "output_summary": f"{len(retrieved)} passage(s)",
                    "latency_ms": 1.0,
                }
            ],
            "latency_s": 0.01 * len(retrieved),
            "model": "echo-v1",
        }

    def describe(self) -> dict:
        return {
            "name": "echo_target",
            "version": "1",
            "model": "echo-v1",
            "supports": {"retrieval": True, "trace": True},
        }
