"""Retrieval evaluation: precision / recall / MRR of RAG table retrieval vs ground truth.

For each labeled question we compare the tables the retriever surfaced against `expected_tables`:
  * recall     — did we surface every table needed to answer? (the one that matters for RAG:
                 a missing table means the LLM can't ground its SQL)
  * MRR        — mean reciprocal rank of the first relevant table (is it ranked high?)
  * precision@k — fraction of surfaced tables that were relevant (low noise)

Deterministic, no API key.  Run:  .venv/bin/python -m agent.eval_retrieval
"""
from __future__ import annotations

from . import retrieval
from .eval_dataset import ANSWERABLE


def _retrieved(question: str) -> list[str]:
    return [t["name"] for t in retrieval.retrieve(question)["tables"]]


def main() -> int:
    P = R = M = 0.0
    n = 0
    for g in ANSWERABLE:
        if not g.expected_tables:
            continue
        got = _retrieved(g.question)
        exp = set(g.expected_tables)
        inter = set(got) & exp
        precision = len(inter) / len(got) if got else 0.0
        recall = len(inter) / len(exp)
        rr = next((1.0 / i for i, t in enumerate(got, 1) if t in exp), 0.0)
        P += precision; R += recall; M += rr; n += 1
        print(f"  {'✅' if recall == 1.0 else '⚠️'} {g.id:18} "
              f"recall={recall:.2f} mrr={rr:.2f} prec={precision:.2f} → {got[:4]}")

    print(f"\n  Mean over {n} questions:  recall={R/n:.0%}  MRR={M/n:.2f}  precision@k={P/n:.0%}")
    return 0 if R / n >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
