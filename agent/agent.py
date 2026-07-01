"""The agent loop.

    question
      → RETRIEVE   (RAG over the semantic catalog)
      → PLAN       (hypothesis + approach)                     [LLM]
      → SQL        (generate a read-only DuckDB query)         [LLM]
      → EXECUTE    (run; on error feed the message back)       [self-heal, up to N tries]
      → GUARDRAIL  (deterministic statistical checks)          [no LLM]
      → INTERPRET  (findings + recommendation, honoring caveats)[LLM]

Returns an AgentResult carrying the full trace so the UI can show its work.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd

from . import guardrails, llm, retrieval
from .warehouse import run_query, QueryError

MAX_SQL_TRIES = 4

_SYSTEM = (
    "You are a meticulous healthcare data analyst working over a dbt-modeled DuckDB warehouse "
    "(schema `main`). You write DuckDB SQL. Rules: (1) use ONLY tables and columns that appear in "
    "the provided catalog — never invent names; (2) read-only single SELECT statements only; "
    "(3) prefer the analytics marts (mart_*) when they directly answer the question; "
    "(4) all costs/data are synthetic. Be precise about grain and denominators. "
    "(5) To filter by a clinical NAME (a condition, medication, or procedure), match the "
    "corresponding *_description column with ILIKE '%name%' — the *_code columns hold coded "
    "identifiers (SNOMED/RxNorm/LOINC), not names. Use the example values in the catalog to ground "
    "your filters. "
    "(6) Only add GROUP BY when the question explicitly asks for a per-category breakdown "
    "(e.g. 'by age group', 'by condition', 'for each class'). A single overall figure — including "
    "phrasings like 'average cost per encounter/patient' — means one aggregate row over all units, "
    "not a grouped result. "
    "(7) When you compute a rate/proportion/prevalence, also SELECT its numerator (the count) and "
    "denominator (the group size), not only the percentage — the downstream statistical guardrail "
    "needs them to compute confidence intervals and group contrasts."
)


@dataclass
class AgentResult:
    question: str
    hypothesis: str = ""
    plan: str = ""
    sql: str = ""
    attempts: list[dict] = field(default_factory=list)   # [{sql, error|None}]
    dataframe: pd.DataFrame | None = None
    findings: list[guardrails.Finding] = field(default_factory=list)
    interpretation: str = ""
    error: str | None = None

    @property
    def n_rows(self) -> int:
        return 0 if self.dataframe is None else len(self.dataframe)


def _clean_sql(text: str) -> str:
    """Strip markdown fences / stray prose the model might add around the SQL."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lower().startswith("sql"):
            t = t[3:]
    return t.strip().strip("`").strip()


def _plan(question: str, context: str) -> tuple[str, str]:
    out = llm.complete_json(
        _SYSTEM,
        f"{context}\n\nQUESTION: {question}\n\n"
        "Return JSON: {\"hypothesis\": one-sentence testable hypothesis, "
        "\"analysis_plan\": 1-3 sentences on how you'll answer it with the tables above}.",
    )
    return out.get("hypothesis", ""), out.get("analysis_plan", "")


def _gen_sql(question: str, context: str, plan: str) -> str:
    return _clean_sql(llm.complete(
        _SYSTEM,
        f"{context}\n\nQUESTION: {question}\nPLAN: {plan}\n\n"
        "Write ONE read-only DuckDB SELECT that answers the question using only the catalog above. "
        "Return ONLY the SQL — no markdown, no commentary.",
    ))


def _fix_sql(question: str, context: str, bad_sql: str, error: str) -> str:
    return _clean_sql(llm.complete(
        _SYSTEM,
        f"{context}\n\nQUESTION: {question}\n\nThis query failed:\n{bad_sql}\n\n"
        f"DuckDB error:\n{error}\n\nReturn a corrected single read-only SELECT. Only the SQL.",
    ))


def _interpret(question: str, sql: str, df: pd.DataFrame, findings: list[guardrails.Finding]) -> str:
    preview = df.head(30).to_csv(index=False)
    caveats = guardrails.render(findings)
    return llm.complete(
        _SYSTEM,
        f"QUESTION: {question}\n\nSQL:\n{sql}\n\nRESULT (up to 30 rows, CSV):\n{preview}\n\n"
        f"STATISTICAL CAVEATS (computed deterministically — you must respect these, do not overstate):\n"
        f"{caveats}\n\n"
        "Write the answer in three short sections using markdown headers exactly:\n"
        "**Findings** — 3-5 sentences on what the data shows.\n"
        "**Recommendation** — one concrete, actionable recommendation.\n"
        "**Statistical caveats** — restate the caveats above in plain language; never claim "
        "significance or causation the data can't support.",
        temperature=0.2,
    )


def run_analysis(question: str, max_tries: int = MAX_SQL_TRIES) -> AgentResult:
    result = AgentResult(question=question)
    try:
        retrieved = retrieval.retrieve(question)
        context = retrieval.render_context(retrieved)

        result.hypothesis, result.plan = _plan(question, context)
        sql = _gen_sql(question, context, result.plan)

        df = None
        for attempt in range(1, max_tries + 1):
            try:
                candidate = run_query(sql)
            except QueryError as e:                              # self-heal on SQL error
                result.attempts.append({"sql": sql, "error": str(e)})
                if attempt == max_tries:
                    result.sql = sql
                    result.error = f"SQL failed after {max_tries} attempts: {e}"
                    return result
                sql = _fix_sql(question, context, sql, str(e))
                continue
            # executed OK — but an empty result usually means a wrong filter; retry with a hint
            if len(candidate) == 0 and attempt < max_tries:
                hint = ("Query executed but returned 0 rows. Reconsider the filters — e.g. to match "
                        "a clinical name use ILIKE on a *_description column, not equality on a "
                        "*_code column; check the example values in the catalog.")
                result.attempts.append({"sql": sql, "error": hint})
                sql = _fix_sql(question, context, sql, hint)
                continue
            result.attempts.append({"sql": sql, "error": None})
            df = candidate
            break

        result.sql = sql
        result.dataframe = df
        result.findings = guardrails.analyze(df, question, sql)
        result.interpretation = _interpret(question, sql, df, result.findings)
    except llm.LLMError as e:
        result.error = str(e)
    return result


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the 30-day readmission rate, and does it vary by age group?"
    r = run_analysis(q)
    if r.error:
        print("ERROR:", r.error)
    else:
        print(f"HYPOTHESIS: {r.hypothesis}\n\nSQL ({len(r.attempts)} attempt/s):\n{r.sql}\n")
        print(r.dataframe.head(15).to_string(index=False), "\n")
        print("GUARDRAIL:\n" + guardrails.render(r.findings), "\n")
        print(r.interpretation)
