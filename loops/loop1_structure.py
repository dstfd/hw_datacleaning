"""Loop 1 — Structure.

CODE  ⚙ : profile every sheet (schema, nulls, distincts, top-K, FK candidates)
LLM   🧠: one call to pick fuzzy threshold + decide which FK candidates are valid.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from ..lib.llm import json_call
from ..lib.strings import norm


def summarize_sheet(name: str, df: pd.DataFrame, top_k: int = 5) -> dict:
    cols = []
    for c in df.columns:
        col = df[c]
        non_null = col.dropna()
        try:
            top = non_null.astype(str).value_counts().head(top_k).index.tolist()
        except Exception:
            top = []
        cols.append({
            "name": str(c),
            "dtype": str(col.dtype),
            "null_count": int(col.isna().sum()),
            "distinct_count": int(non_null.nunique()),
            "top_values": top,
        })
    return {"sheet": name, "row_count": int(len(df)), "columns": cols}


def fk_candidates(sheets: dict[str, pd.DataFrame]) -> list[dict]:
    """For every pair of (id-ish) columns across sheets, compute match rate."""
    out = []
    id_cols = []
    for sname, df in sheets.items():
        for c in df.columns:
            if c.lower().endswith("_id"):
                id_cols.append((sname, c))
    for src_sheet, src_col in id_cols:
        src_vals = set(sheets[src_sheet][src_col].dropna().astype(str))
        if not src_vals:
            continue
        for tgt_sheet, tgt_col in id_cols:
            if (src_sheet, src_col) == (tgt_sheet, tgt_col):
                continue
            if src_col != tgt_col and not tgt_col.lower() == "id":
                # Same column name = obvious FK candidate (e.g. restaurant_id ↔ restaurant_id)
                if src_col != tgt_col:
                    continue
            tgt_vals = set(sheets[tgt_sheet][tgt_col].dropna().astype(str))
            if not tgt_vals:
                continue
            overlap = len(src_vals & tgt_vals) / len(src_vals)
            if overlap > 0.05:
                out.append({
                    "from": f"{src_sheet}.{src_col}",
                    "to": f"{tgt_sheet}.{tgt_col}",
                    "match_rate": round(overlap, 3),
                    "src_distinct": len(src_vals),
                    "tgt_distinct": len(tgt_vals),
                })
    return out


def name_pairs(sheets: dict[str, pd.DataFrame]) -> list[dict]:
    """Find columns named '*_name' that might fuzzy-match a sibling sheet's 'name'."""
    out = []
    for src_sheet, df in sheets.items():
        for c in df.columns:
            if c.lower().endswith("_name") and c.lower() != "name":
                target_sheet = c.lower()[:-5] + "s"  # restaurant_name → restaurants
                if target_sheet in sheets and "name" in sheets[target_sheet].columns:
                    out.append({
                        "from": f"{src_sheet}.{c}",
                        "to": f"{target_sheet}.name",
                        "matchable_rows": int(df[c].notna().sum()),
                    })
    return out


def detect_quality_issues(sheets: dict[str, pd.DataFrame]) -> list[dict]:
    issues = []
    for sname, df in sheets.items():
        for c in df.columns:
            null_pct = df[c].isna().mean()
            if null_pct > 0.3:
                issues.append({
                    "table": sname, "column": c,
                    "issue": "high_null_rate",
                    "rate": round(float(null_pct), 3),
                    "affected_count": int(df[c].isna().sum()),
                })
            if df[c].dtype == object:
                non_null = df[c].dropna().astype(str)
                if not non_null.empty:
                    norm_vals = non_null.map(norm)
                    distinct_raw = non_null.nunique()
                    distinct_norm = norm_vals.nunique()
                    if distinct_raw > distinct_norm * 1.05:
                        issues.append({
                            "table": sname, "column": c,
                            "issue": "case_or_whitespace_dupes",
                            "distinct_raw": int(distinct_raw),
                            "distinct_norm": int(distinct_norm),
                            "affected_count": int(distinct_raw - distinct_norm),
                        })
    return issues


SYSTEM = """You are a data reconciliation strategist.
Given a deterministic profile of a multi-sheet dataset, decide:
  1. which FK / name relationships should be enforced for linking
  2. one global fuzzy threshold to use for name matching (between 0.70 and 0.90)
  3. which tables need cleaning before linking
  4. which detected quality issues genuinely matter

Respond ONLY with a JSON object of the exact shape requested. Do not invent fields."""


PROMPT_TMPL = """## Dataset profile

{profile_json}

## Pre-computed FK candidates (by id-column intersection)

{fk_json}

## Pre-computed name-match candidates

{name_json}

## Pre-computed quality issues

{issues_json}

## Task

Return a JSON object with this shape:

{{
  "relationships": [
    {{
      "source": "table.column",
      "target": "table.column",
      "method": "exact" | "fuzzy",
      "when": "ALWAYS" | "IF_SOURCE_NOT_NULL" | "IF_SOURCE_NULL_AND_NAME_AVAILABLE",
      "fuzzy_threshold": null | 0.70..0.90
    }}
  ],
  "global_fuzzy_threshold": 0.70..0.90,
  "cleaning_targets": ["table_name", ...],
  "quality_issues": [
    {{"table": "...", "column": "...", "issue": "...", "matters": true|false, "reason": "..."}}
  ],
  "strategy_notes": "one paragraph explaining the chosen approach"
}}
"""


def run(xlsx_path: Path, out_dir: Path) -> dict:
    print("→ Loop 1: profiling sheets...")
    xl = pd.ExcelFile(xlsx_path)
    sheets = {name: pd.read_excel(xl, name) for name in xl.sheet_names}
    sheet_keys = {k.lower(): v for k, v in sheets.items()}
    profiles = [summarize_sheet(n, df) for n, df in sheets.items()]
    fks = fk_candidates(sheet_keys)
    names = name_pairs(sheet_keys)
    issues = detect_quality_issues(sheet_keys)

    print(f"  sheets: {[s['sheet'] for s in profiles]}")
    print(f"  fk candidates: {len(fks)}, name pairs: {len(names)}, quality issues: {len(issues)}")

    prompt = PROMPT_TMPL.format(
        profile_json=json.dumps(profiles, indent=2),
        fk_json=json.dumps(fks, indent=2),
        name_json=json.dumps(names, indent=2),
        issues_json=json.dumps(issues, indent=2),
    )

    print("→ Loop 1: LLM strategy call...")
    strategy = json_call(prompt, system=SYSTEM, max_tokens=4096)

    out = {
        "profile": profiles,
        "fk_candidates": fks,
        "name_pairs": names,
        "detected_issues": issues,
        "strategy": strategy,
    }
    out_path = out_dir / "loop1_structure.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    print(f"  fuzzy threshold: {strategy.get('global_fuzzy_threshold')}")
    print(f"  cleaning targets: {strategy.get('cleaning_targets')}")
    return out
