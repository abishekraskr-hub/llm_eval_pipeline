"""A tiny HTTP app with DELIBERATELY NON-STANDARD field names.

Standard library only: no framework, no dependency, so the integration test can
start it anywhere.

The field names are awkward on purpose. It takes `{"query": ...}` and replies
with `reply`, `sources`, `url`, `text`, `meta.refused` -- none of which are what
the library calls them. If this example used the library's own names it would
prove nothing: the point is that an app which already serves real users does not
have to change its API to be evaluated. `llmeval.yaml` maps the names instead.

It also demonstrates the thing people assume is impossible: the app is not
Python as far as the evaluator is concerned. It speaks HTTP, and that is all the
contract requires. A Go or TypeScript service is evaluated exactly this way.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# source_id -> list of (page, text). A page is OPTIONAL in the contract; this
# example has them so that page matching is exercised too.
DOCS = {
    "https://docs.example.com/billing": [
        (1, "Invoices are issued on the first working day of each month."),
        (2, "Payment is due within 14 days of the invoice date."),
    ],
    "https://docs.example.com/accounts": [
        (1, "An account can be closed at any time from the settings page."),
    ],
}

KEYWORDS = {
    "invoice": [("https://docs.example.com/billing", 1)],
    "issued": [("https://docs.example.com/billing", 1)],
    "payment": [("https://docs.example.com/billing", 2)],
    "due": [("https://docs.example.com/billing", 2)],
    "close": [("https://docs.example.com/accounts", 1)],
    "account": [("https://docs.example.com/accounts", 1)],
}

DECLINE = "The documentation does not cover that."


def _text(source_id: str, page: int) -> str:
    for number, body in DOCS[source_id]:
        if number == page:
            return body
    return ""


def answer(query: str) -> dict:
    lowered = (query or "").lower()
    hits: list[tuple[str, int]] = []
    for keyword, locations in KEYWORDS.items():
        if keyword in lowered:
            for location in locations:
                if location not in hits:
                    hits.append(location)

    if not hits:
        return {"reply": DECLINE, "sources": [], "meta": {"refused": True}}

    return {
        # Note every key: `reply` not `answer`, `sources` not `retrieved`,
        # `url` not `source_id`, `text` not `content`.
        "reply": " ".join(_text(s, p) for s, p in hits),
        "sources": [
            {"url": source_id, "page": page, "text": _text(source_id, page)}
            for source_id, page in hits
        ],
        "meta": {"refused": False},
        "usage": {"input_tokens": 30, "output_tokens": 15},
        "model": "http-stub-v1",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        # The health endpoint `start.health_url` polls, so the evaluator waits
        # for readiness instead of sleeping an arbitrary number of seconds.
        if self.path == "/health":
            self._send(200, {"ok": True})
        elif self.path == "/config":
            # Optional: lets the app declare itself, which lands in the report.
            self._send(
                200,
                {
                    "name": "http_stub",
                    "model": "http-stub-v1",
                    "supports": {"retrieval": True, "trace": False},
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/chat":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return
        self._send(200, answer(payload.get("query", "")))

    def log_message(self, *args) -> None:
        """Silence the per-request log: the evaluator's output is the signal."""


def main(port: int = 8077) -> None:
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8077)
