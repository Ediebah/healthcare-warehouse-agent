"""The agent loop.

    question
      → TRIAGE     (answerable, or ask a clarifying question?)      [LLM]
      → RETRIEVE   (RAG over the semantic catalog)
      → PLAN       (hypothesis + approach)                          [LLM]
      → SQL        (generate a read-only DuckDB query)              [LLM]
      → EXECUTE    (run; on error/empty feed it back)               [self-heal, up to N tries]
      → CITE       (which catalog tables the SQL used)              [deterministic]
      → GUARDRAIL  (statistical checks)                             [deterministic]
      → VERIFY     (does the SQL answer THIS question? confidence)  [LLM critic]
      → INTERPRET  (findings + recommendation, honoring caveats)    [LLM]

Returns an AgentResult carrying the full trace so the UI can show its work.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import guardrails, llm, retrieval
from .warehouse import QueryError, run_query

MAX_SQL_TRIES = 4
MAX_QUESTION_LEN = 2000
_TRACE_LOG = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"

# Defense-in-depth: block obvious prompt-injection before spending any tokens. (The SQL layer is
# already read-only + validated, so the blast radius is small; this stops instruction-override too.)
_INJECTION = re.compile(
    r"(ignore (all |the )?(previous|above|prior)|disregard (the |your |all )?(previous|instruction|rule)|"
    r"system prompt|prompt injection|you are now|you must now|new (system )?instructions|"
    r"reveal (your|the) (instructions|prompt|system)|print (your |the )?(instructions|prompt|system)|"
    r"dump (your|the) (instructions|prompt|system)|jailbreak|developer mode|"
    r"override (the |your )?(rules|instructions)|bypass (the |your )?(rules|guard|filter|instructions)|"
    r"forget (your|the|all)|pretend (to be|you are))", re.I)


def _looks_like_injection(q: str) -> bool:
    q = q or ""
    if len(q) > MAX_QUESTION_LEN:        # absurdly long input — likely stuffing/exfiltration
        return True
    return bool(_INJECTION.search(q))

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
    "needs them to compute confidence intervals and group contrasts. "
    "(8) To count DISTINCT patients who have a condition or medication, use COUNT(DISTINCT patient_id) "
    "on fct_conditions / fct_medications (join the dim_ by code to filter by name) — never SUM a "
    "per-group numerator from an analytics mart, which double-counts patients across strata."
)


@dataclass
class AgentResult:
    question: str
    clarification: str = ""                                  # set if the agent needs to ask back
    hypothesis: str = ""
    plan: str = ""
    sql: str = ""
    attempts: list[dict] = field(default_factory=list)       # [{sql, error|None}]
    dataframe: pd.DataFrame | None = None
    citations: list[str] = field(default_factory=list)        # catalog tables the SQL used
    findings: list[guardrails.Finding] = field(default_factory=list)
    verification: dict | None = None                          # {answers_question, confidence, issues}
    interpretation: str = ""
    trace: dict | None = None                                 # {calls, tokens, latency_ms, est_cost_usd}
    error: str | None = None

    @property
    def n_rows(self) -> int:
        return 0 if self.dataframe is None else len(self.dataframe)


def _clean_sql(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lower().startswith("sql"):
            t = t[3:]
    return t.strip().strip("`").strip()


def _triage(question: str, context: str) -> dict:
    """Decide ONLY whether the question is too vague to attempt. Deliberately conservative:
    it must not second-guess whether a specific column exists (that's the SQL step's job)."""
    return llm.complete_json(
        _SYSTEM,
        f"QUESTION: {question}\n\n"
        "You are ONLY deciding whether this question is too vague to attempt at all. "
        "Default answerable=true. Set answerable=false ONLY when the question names no concrete "
        "metric or entity — e.g. 'show me the trends', 'what's interesting', 'tell me about the data', "
        "'which treatment is best' — or asks for something clearly outside a healthcare EHR warehouse. "
        "A question that names any count, rate, cost, condition, medication, demographic "
        "(age/sex/gender/race), payer, provider, or time period is ANSWERABLE. Do NOT judge whether a "
        "specific column exists; assume standard EHR fields are present and let the SQL step handle it. "
        'Return JSON: {"answerable": bool, "clarification": "one specific question if not answerable, else empty"}.',
    )


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


def _verify(question: str, sql: str, df: pd.DataFrame) -> dict:
    """Critic pass: does the SQL actually answer THIS question? Confidence + issues."""
    preview = df.head(15).to_csv(index=False)
    out = llm.complete_json(
        _SYSTEM,
        f"QUESTION: {question}\n\nSQL:\n{sql}\n\nRESULT (up to 15 rows):\n{preview}\n\n"
        "Critically review whether this SQL answers the EXACT question asked (right grain, filters, "
        "metric, denominators) and whether the result is plausible. Be skeptical. "
        'Return JSON: {"answers_question": bool, "confidence": "high"|"medium"|"low", '
        '"issues": [short strings, empty if none]}.',
    )
    out.setdefault("answers_question", True)
    out.setdefault("confidence", "medium")
    out.setdefault("issues", [])
    return out


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


def _citations(sql: str, table_names: list[str]) -> list[str]:
    return [t for t in table_names if re.search(rf"\b{re.escape(t)}\b", sql)]


def _degenerate(df) -> bool:
    """A single-row aggregate that is all zero/NULL — usually a filter that matched nothing."""
    if df is None or len(df) != 1:
        return False
    nums = [df.iloc[0][c] for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return bool(nums) and all((pd.isna(v) or v == 0) for v in nums)


def _persist_trace(question: str, trace: dict | None, error: str | None) -> None:
    """Append each run's trace (tokens/latency/cost) to logs/traces.jsonl for observability."""
    try:
        _TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"),
               "question": (question or "")[:200], "model": llm.MODEL, "error": bool(error),
               **(trace or {})}
        with _TRACE_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def run_analysis(question: str, max_tries: int = MAX_SQL_TRIES) -> AgentResult:
    result = AgentResult(question=question)
    llm.reset_trace()
    if _looks_like_injection(question):
        result.error = ("This request looks like an attempt to override the agent's instructions and "
                        "was blocked. Please ask a data question about the healthcare warehouse.")
        return result
    try:
        retrieved = retrieval.retrieve(question)
        context = retrieval.render_context(retrieved)

        triage = _triage(question, context)
        if not triage.get("answerable", True) and triage.get("clarification"):
            result.clarification = triage["clarification"]
            return result

        result.hypothesis, result.plan = _plan(question, context)
        sql = _gen_sql(question, context, result.plan)

        df = None
        degen_used = False
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
            empty = len(candidate) == 0
            degen = (not empty) and (not degen_used) and _degenerate(candidate)
            if (empty or degen) and attempt < max_tries:         # self-heal on empty / null-aggregate
                if empty:
                    hint = ("Query executed but returned 0 rows. Reconsider the filters — e.g. match a "
                            "clinical name with ILIKE on a *_description column, not equality on a "
                            "*_code column; check the example values in the catalog.")
                else:
                    degen_used = True
                    hint = ("The query returned a single all-zero/NULL aggregate — the filter likely "
                            "matched no rows. Check exact category values (e.g. gender is 'M'/'F', not "
                            "'female'; match clinical names with ILIKE on *_description). If 0 is "
                            "genuinely correct, return the same query.")
                result.attempts.append({"sql": sql, "error": hint})
                sql = _fix_sql(question, context, sql, hint)
                continue
            result.attempts.append({"sql": sql, "error": None})
            df = candidate
            break

        result.sql = sql
        result.dataframe = df
        result.citations = _citations(sql, retrieved["all_table_names"])
        result.findings = guardrails.analyze(df, question, sql)
        result.verification = _verify(question, sql, df)
        result.interpretation = _interpret(question, sql, df, result.findings)
    except llm.LLMError as e:
        result.error = str(e)
    finally:
        result.trace = llm.trace_summary()
        _persist_trace(question, result.trace, result.error)
    return result


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the 30-day readmission rate, and does it vary by age group?"
    r = run_analysis(q)
    if r.clarification:
        print("NEEDS CLARIFICATION:", r.clarification)
    elif r.error:
        print("ERROR:", r.error)
    else:
        print(f"HYPOTHESIS: {r.hypothesis}\n\nSQL ({len(r.attempts)} attempt/s):\n{r.sql}\n")
        print(r.dataframe.head(15).to_string(index=False), "\n")
        print("CITATIONS:", ", ".join(r.citations))
        print("VERIFY:", r.verification)
        print("\nGUARDRAIL:\n" + guardrails.render(r.findings), "\n")
        print(r.interpretation)
