"""llm_eval_pipeline -- CI/CD for LLM applications.

A pull request that makes answer quality worse gets blocked, like a failing unit
test. Point it at any LLM app -- RAG, agent or plain prompt, in any language,
over any storage -- with one config file, plus either a small adapter class or
nothing at all if the app already speaks HTTP.
"""

# The single source of truth for the version. `pyproject.toml` reads it from
# here, and it feeds the baseline comparability hash: a library upgrade can
# change how a metric is computed, so it must start a fresh baseline rather than
# quietly compare numbers produced by different code.
__version__ = "0.1.0"
