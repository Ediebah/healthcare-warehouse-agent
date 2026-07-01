#!/usr/bin/env python3
"""Generate an LLM-readable semantic catalog from dbt artifacts.

Reads warehouse/target/{manifest,catalog}.json (produced by `dbt docs generate`) and emits:
  * agent/semantic_catalog.json  — machine-readable, one entry per mart table + named metrics.
  * agent/semantic_catalog.md    — the same content as browsable Markdown.

Why this exists (CONCEPTS §9): the agent retrieves over this catalog so it knows the exact
tables, grains, columns, types, primary keys, join keys, and metric SQL — instead of
hallucinating column names. Every description here was written once in the dbt .yml files.

What we include: the modeled query surface = marts/ (core dims+facts + analytics). Staging is
intermediate and intentionally omitted so the agent is steered to the clean star schema.

Usage:  .venv/bin/python agent/build_catalog.py   (after `dbt docs generate`)
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "warehouse" / "target"
DB_PATH = ROOT / "data" / "healthcare.duckdb"
OUT_JSON = ROOT / "agent" / "semantic_catalog.json"
OUT_MD = ROOT / "agent" / "semantic_catalog.md"

REF_RE = re.compile(r"ref\(\s*'([^']+)'\s*\)")


def ref_name(s: str | None) -> str | None:
    """Pull the model name out of a dbt ref() string, e.g. "ref('dim_patient')" -> dim_patient."""
    if not s:
        return None
    m = REF_RE.search(s)
    return m.group(1) if m else None


def load_artifacts() -> tuple[dict, dict]:
    manifest = json.loads((TARGET / "manifest.json").read_text())
    catalog = json.loads((TARGET / "catalog.json").read_text())
    return manifest, catalog


def collect_keys(manifest: dict) -> tuple[dict, dict]:
    """Scan test nodes → {model: [pk cols]} and {model: [(col, to_model, to_field)]}."""
    primary_keys: dict[str, list[str]] = {}
    foreign_keys: dict[str, list[dict]] = {}
    for node in manifest["nodes"].values():
        if node["resource_type"] != "test":
            continue
        meta = node.get("test_metadata") or {}
        name = meta.get("name")
        kw = meta.get("kwargs", {})
        model = ref_name(kw.get("model"))
        if not model:
            continue
        if name == "unique":
            # a `unique` test on a column marks it as (part of) the primary key
            col = kw.get("column_name")
            if col:
                primary_keys.setdefault(model, [])
                if col not in primary_keys[model]:
                    primary_keys[model].append(col)
        elif name == "unique_combination_of_columns":
            # a composite primary key (dbt_utils model-level test)
            primary_keys.setdefault(model, [])
            for col in kw.get("combination_of_columns", []):
                if col not in primary_keys[model]:
                    primary_keys[model].append(col)
        elif name == "relationships":
            foreign_keys.setdefault(model, []).append({
                "column": kw.get("column_name"),
                "references_table": ref_name(kw.get("to")),
                "references_column": kw.get("field"),
            })
    return primary_keys, foreign_keys


def column_types(catalog: dict, unique_id: str) -> dict[str, str]:
    """Real SQL types per column from catalog.json (keyed lower-case for matching)."""
    cnode = catalog["nodes"].get(unique_id, {})
    return {c["name"].lower(): c["type"] for c in cnode.get("columns", {}).values()}


def sample_values(con, table_name: str, col: str, limit: int = 4) -> list[str]:
    """A few distinct example values for a column — grounds the LLM (e.g. names live in
    *_description, codes in *_code). Returns [] on any error."""
    try:
        rows = con.execute(
            f'select distinct "{col}" as v from main."{table_name}" '
            f'where "{col}" is not null limit {limit}'
        ).fetchall()
        return [str(r[0]) for r in rows]
    except Exception:
        return []


def build_tables(manifest: dict, catalog: dict, pks: dict, fks: dict, con=None) -> list[dict]:
    tables = []
    for uid, node in manifest["nodes"].items():
        if node["resource_type"] != "model":
            continue
        if not node["path"].startswith("marts/"):     # marts only (core + analytics)
            continue
        name = node["name"]
        layer = node["path"].split("/")[1]             # 'core' or 'analytics'
        types = column_types(catalog, uid)
        described = node.get("columns", {})            # only columns documented in .yml
        # Full column list comes from the catalog (every physical column); enrich w/ descriptions.
        cols = []
        for col_lower, sql_type in types.items():
            # find the documented entry case-insensitively
            doc = next((d for cn, d in described.items() if cn.lower() == col_lower), {})
            # sample example values for text columns (skip UUID id columns — noise)
            examples = []
            if con is not None and sql_type == "VARCHAR" and not col_lower.endswith("_id"):
                examples = sample_values(con, name, col_lower)
            cols.append({
                "name": col_lower,
                "type": sql_type,
                "description": (doc or {}).get("description", ""),
                "example_values": examples,
            })
        tables.append({
            "name": name,
            "layer": layer,
            "relation": node["relation_name"].replace('"', ""),
            "description": node.get("description", "").strip(),
            "primary_key": pks.get(name, []),
            "foreign_keys": fks.get(name, []),
            "columns": cols,
        })
    tables.sort(key=lambda t: (t["layer"] != "core", t["name"]))   # core first, then analytics
    return tables


def named_metrics() -> list[dict]:
    """Hand-authored metric definitions: plain-English + the exact SQL logic + source model."""
    return [
        {
            "name": "readmission_rate_30d",
            "definition": "Share of index inpatient stays followed by another inpatient admission within 0-30 days of discharge.",
            "model": "mart_readmissions",
            "sql": "avg(is_30d_readmission::int)  -- optionally *100 for a percentage",
            "caveats": "Inpatient-only. Denominator is index inpatient stays, not patients.",
        },
        {
            "name": "avg_encounter_cost",
            "definition": "Average total billed cost across encounters.",
            "model": "fct_encounters",
            "sql": "avg(total_claim_cost)",
            "caveats": "Billed (claim) cost, synthetic. Segment by encounter_class for fair comparisons.",
        },
        {
            "name": "avg_patient_out_of_pocket",
            "definition": "Average amount the patient pays after insurance, per encounter.",
            "model": "fct_encounters",
            "sql": "avg(patient_out_of_pocket)  -- total_claim_cost - payer_coverage",
            "caveats": "Synthetic payer logic; not real benefit design.",
        },
        {
            "name": "condition_prevalence_by_age",
            "definition": "Percent of patients in an age band who have a given condition.",
            "model": "mart_condition_prevalence",
            "sql": "prevalence_pct  -- 100 * patients_with_condition / total_patients_in_age_group",
            "caveats": "Small age bands (e.g. 75+, n≈91) give noisy estimates — check total_patients_in_age_group.",
        },
        {
            "name": "avg_diagnosing_encounter_cost_by_condition",
            "definition": "Average cost of the encounter at which a condition was diagnosed, per condition.",
            "model": "mart_cost_by_condition",
            "sql": "avg_diagnosing_encounter_cost",
            "caveats": "NOT lifetime cost of treating the condition — no causal attribution of downstream care.",
        },
        {
            "name": "total_medication_cost",
            "definition": "Total cost of medication orders.",
            "model": "fct_medications",
            "sql": "sum(total_cost)",
            "caveats": "Order-level synthetic cost; a patient may have many orders of one drug.",
        },
    ]


def to_markdown(catalog: dict) -> str:
    lines = ["# Semantic Catalog", "",
             f"Generated from dbt artifacts. {len(catalog['tables'])} tables, "
             f"{len(catalog['metrics'])} named metrics.", "",
             "## Named metrics", ""]
    for m in catalog["metrics"]:
        lines += [f"### `{m['name']}`",
                  f"- **Definition:** {m['definition']}",
                  f"- **Source model:** `{m['model']}`",
                  f"- **SQL:** `{m['sql']}`",
                  f"- **Caveats:** {m['caveats']}", ""]
    lines += ["## Tables", ""]
    for t in catalog["tables"]:
        lines += [f"### `{t['name']}`  ({t['layer']})",
                  f"{t['description']}", "",
                  f"- **Relation:** `{t['relation']}`",
                  f"- **Primary key:** {', '.join(t['primary_key']) or '(none)'}"]
        if t["foreign_keys"]:
            fk = "; ".join(f"{f['column']} → {f['references_table']}.{f['references_column']}"
                           for f in t["foreign_keys"])
            lines.append(f"- **Foreign keys:** {fk}")
        lines += ["", "| column | type | description | examples |", "|---|---|---|---|"]
        for c in t["columns"]:
            ex = ", ".join(str(e) for e in c.get("example_values", [])[:3])
            lines.append(f"| `{c['name']}` | {c['type']} | {c['description']} | {ex} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    manifest, catalog_art = load_artifacts()
    pks, fks = collect_keys(manifest)
    con = duckdb.connect(str(DB_PATH), read_only=True) if DB_PATH.exists() else None
    try:
        tables = build_tables(manifest, catalog_art, pks, fks, con=con)
    finally:
        if con is not None:
            con.close()
    catalog = {
        "warehouse": "healthcare (DuckDB)",
        "generated_from": "dbt manifest.json + catalog.json",
        "tables": tables,
        "metrics": named_metrics(),
    }
    OUT_JSON.write_text(json.dumps(catalog, indent=2))
    OUT_MD.write_text(to_markdown(catalog))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}  ({len(tables)} tables, {len(catalog['metrics'])} metrics)")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    for t in tables:
        print(f"  {t['layer']:9} {t['name']:24} pk={t['primary_key'] or '-'}  cols={len(t['columns'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
