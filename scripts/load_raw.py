#!/usr/bin/env python3
"""Load Synthea CSVs into DuckDB as a faithful RAW copy.

Design choices (see CONCEPTS.md):
  * Every column is loaded as VARCHAR (`all_varchar=true`). Raw is a byte-faithful copy
    of the source CSVs; DuckDB never guesses a type. All casting happens in dbt staging.
  * CREATE OR REPLACE makes this idempotent — re-run any time to rebuild the raw schema.
  * We load only the tables the warehouse needs (claims_transactions.csv, 482MB, is unused).

Usage:  .venv/bin/python scripts/load_raw.py
"""
from pathlib import Path
import sys
import duckdb

ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "synthea" / "output" / "csv"
DB_PATH = ROOT / "data" / "healthcare.duckdb"

# Tables the star schema is built from (order is cosmetic).
TABLES = [
    "patients", "encounters", "conditions", "medications",
    "observations", "procedures", "claims",
    "organizations", "providers", "payers",
]


def main() -> int:
    if not CSV_DIR.exists():
        sys.exit(f"CSV dir not found: {CSV_DIR}\nRun the Synthea generate step first.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")

    print(f"Loading into {DB_PATH.relative_to(ROOT)}  (schema: raw)\n")
    for name in TABLES:
        csv = CSV_DIR / f"{name}.csv"
        if not csv.exists():
            print(f"  ! missing {csv.name} — skipping")
            continue
        # Inline the local path (trusted); all_varchar keeps raw faithful.
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{name} AS "
            f"SELECT * FROM read_csv('{csv.as_posix()}', "
            f"header=true, all_varchar=true, sample_size=-1)"
        )
        rows = con.execute(f"SELECT count(*) FROM raw.{name}").fetchone()[0]
        cols = len(con.execute(f"DESCRIBE raw.{name}").fetchall())
        print(f"  raw.{name:<14} {rows:>9,} rows  {cols:>3} cols")

    con.close()
    print("\nDone. Inspect with:  duckdb data/healthcare.duckdb  (or via Python)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
