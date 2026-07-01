"""Safe, read-only query execution against the DuckDB warehouse.

Guardrails (the spec's "agent can't run destructive or runaway SQL"):
  1. Engine-level: the connection is opened read_only=True — DuckDB itself rejects any write.
  2. Statement-level: we validate the SQL is a single SELECT/WITH with no write keywords.
  3. Runaway-level: results are capped by wrapping the query in an outer LIMIT.

On any validation or execution error we raise QueryError with a clear message — the agent
feeds that message back to the model to self-heal.
"""
from __future__ import annotations
import os
import re
from pathlib import Path

import duckdb
import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _resolve_db_path() -> Path:
    """Full warehouse locally; committed slim demo DB on deploy (where the full one is absent).
    Override with the WAREHOUSE_DB env var."""
    if os.getenv("WAREHOUSE_DB"):
        return Path(os.environ["WAREHOUSE_DB"])
    full = _DATA_DIR / "healthcare.duckdb"
    return full if full.exists() else _DATA_DIR / "healthcare_demo.duckdb"


DB_PATH = _resolve_db_path()
MAX_ROWS = 1000

# Whole-word write/DDL keywords that must never appear in an analytics query.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|replace|truncate|attach|detach|copy|"
    r"install|load|pragma|export|import|call|set|vacuum|checkpoint)\b",
    re.IGNORECASE,
)


class QueryError(Exception):
    """Raised on validation failure or a DuckDB execution error."""


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)          # line comments
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)  # block comments
    return sql


def validate(sql: str) -> str:
    """Return a cleaned, validated single-statement SELECT/WITH, or raise QueryError."""
    if not sql or not sql.strip():
        raise QueryError("Empty query.")
    cleaned = sql.strip().rstrip(";").strip()
    body = _strip_sql_comments(cleaned)
    if ";" in body:
        raise QueryError("Only a single statement is allowed (found ';').")
    if not re.match(r"^\s*(select|with)\b", body, re.IGNORECASE):
        raise QueryError("Only read-only SELECT/WITH queries are allowed.")
    if _FORBIDDEN.search(body):
        bad = _FORBIDDEN.search(body).group(0)
        raise QueryError(f"Write/DDL keyword '{bad}' is not permitted.")
    return cleaned


def run_query(sql: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Validate + execute read-only, capped at max_rows. Returns a DataFrame or raises QueryError."""
    cleaned = validate(sql)
    if not DB_PATH.exists():
        raise QueryError(f"Warehouse not found at {DB_PATH}. Run the loader + `dbt build` first.")
    wrapped = f"select * from (\n{cleaned}\n) as _agent_q limit {max_rows}"
    try:
        con = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            return con.execute(wrapped).df()
        finally:
            con.close()
    except QueryError:
        raise
    except Exception as e:  # noqa: BLE001 — surface the DB message to the self-heal loop
        raise QueryError(str(e)) from e


if __name__ == "__main__":
    df = run_query("select encounter_class, count(*) n from fct_encounters group by 1 order by n desc")
    print(df.to_string(index=False))
