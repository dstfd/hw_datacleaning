"""Loop 2b — Recipe-token bridge.

Recipe ingredients are pipe-delimited free-text tokens; most never appear in
the Ingredients sheet exactly. Bridge them to canonical ids:

  CODE  ⚙ : exact lookup → fuzzy>=0.92 → candidate list (0.75-0.92) → unresolved
  LLM   🧠: for every ambiguous token, pick from top-5 candidates by INDEX.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process

from ..lib.llm import json_call, batched
from ..lib.strings import norm

HIGH_AUTO = 92        # ≥ this → accept deterministically as fuzzy match
LOW_CUTOFF = 75       # < this → unresolved (don't even ask LLM)
LLM_BATCH = 25


def collect_tokens(recipes_df: pd.DataFrame) -> list[str]:
    seen = set()
    tokens = []
    for raw in recipes_df["ingredients"].dropna():
        for t in str(raw).split("|"):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                tokens.append(t)
    return tokens


SYSTEM = """You are a culinary ingredient matcher.
For each free-text recipe token, choose the best canonical ingredient from the
provided numbered candidate list — or return null if no candidate is a good
match.

Match semantically: "bee broth" → "beef broth"; "chicken thighs" → "chicken"
if no specific "chicken thighs" canonical exists. Reject obvious garbage
("and sometimes mushrooms" → null).

Respond ONLY with the JSON array shape requested."""


PROMPT_TMPL = """## Tokens needing adjudication

Each entry has:
  - token: the free-text recipe ingredient
  - candidates: top-5 canonical names by fuzzy score, indexed 0-4

For each, pick the best candidate by INDEX (0..4) or null if none fits.
Give a confidence 0.60-0.95.

{batch_json}

## Return

```json
[
  {{"token": "...", "pick": 0|1|2|3|4|null, "confidence": 0.0..1.0}},
  ...
]
```

Return only the JSON array."""


def run(cleaning: dict, xlsx_path: Path, out_dir: Path) -> dict:
    print("→ Loop 2b: collecting recipe ingredient tokens...")
    recipes = pd.read_excel(xlsx_path, sheet_name="Recipes")
    tokens = collect_tokens(recipes)
    merge_map = dict(cleaning["merge_map"])
    canon_by_id = {c["ingredient_id"]: c for c in cleaning["canonical_ingredients"]}
    canon_names = [(c["ingredient_id"], c["name"]) for c in cleaning["canonical_ingredients"]]
    canon_name_list = [n for _, n in canon_names]
    canon_id_list = [i for i, _ in canon_names]
    print(f"  {len(tokens)} unique tokens against {len(canon_name_list)} canonicals")

    mapping_meta: dict[str, dict] = {}  # raw_token → {ingredient_id, method, confidence}
    ambiguous: list[dict] = []
    exact = 0
    auto_fuzzy = 0
    rejected = 0

    for token in tokens:
        nn = norm(token)
        if not nn:
            mapping_meta[token] = {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}
            rejected += 1
            continue

        # 1. Exact match via merge_map (variant name → canonical_id)
        if nn in merge_map:
            cid = merge_map[nn]
            mapping_meta[token] = {"ingredient_id": cid, "method": "exact", "confidence": 1.0}
            exact += 1
            continue

        # 2. Fuzzy against canonical names
        top5 = process.extract(nn, canon_name_list, scorer=fuzz.token_sort_ratio, limit=5)
        # top5 entries: (matched_name, score, original_index)
        if not top5:
            mapping_meta[token] = {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}
            rejected += 1
            continue
        top_score = top5[0][1]
        if top_score >= HIGH_AUTO:
            idx = top5[0][2]
            mapping_meta[token] = {
                "ingredient_id": canon_id_list[idx],
                "method": "fuzzy_high",
                "confidence": round(top_score / 100.0, 3),
            }
            auto_fuzzy += 1
        elif top_score >= LOW_CUTOFF:
            ambiguous.append({
                "token": token,
                "candidates": [
                    {"id": canon_id_list[m[2]], "name": m[0], "score": m[1]} for m in top5
                ],
            })
        else:
            mapping_meta[token] = {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}
            rejected += 1

    print(f"  exact={exact} auto_fuzzy={auto_fuzzy} ambiguous={len(ambiguous)} unresolved={rejected}")

    # LLM adjudication on ambiguous
    llm_resolved = 0
    llm_rejected = 0
    if ambiguous:
        print(f"→ Loop 2b: LLM resolves {len(ambiguous)} ambiguous tokens in batches of {LLM_BATCH}...")
        for i, chunk in enumerate(batched(ambiguous, LLM_BATCH), start=1):
            # Strip candidate IDs from the prompt — LLM only sees indexed names.
            payload = [
                {
                    "token": a["token"],
                    "candidates": [
                        {"index": j, "name": c["name"], "score": c["score"]}
                        for j, c in enumerate(a["candidates"])
                    ],
                }
                for a in chunk
            ]
            prompt = PROMPT_TMPL.format(batch_json=json.dumps(payload, indent=2))
            result = json_call(prompt, system=SYSTEM, max_tokens=4096)
            if not isinstance(result, list):
                raise RuntimeError(f"Bridge batch {i}: non-list response")
            # Apply
            tok_to_cands = {a["token"]: a["candidates"] for a in chunk}
            for r in result:
                tok = r.get("token")
                pick = r.get("pick")
                conf = r.get("confidence")
                if tok not in tok_to_cands:
                    continue
                if pick is None or not isinstance(pick, int) or pick < 0 or pick >= len(tok_to_cands[tok]):
                    mapping_meta[tok] = {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}
                    llm_rejected += 1
                else:
                    chosen = tok_to_cands[tok][pick]
                    mapping_meta[tok] = {
                        "ingredient_id": chosen["id"],
                        "method": "llm_disambiguated",
                        "confidence": round(float(conf or 0.7), 3),
                    }
                    llm_resolved += 1
            print(f"  batch {i}/{(len(ambiguous)+LLM_BATCH-1)//LLM_BATCH} ✓")

    # Ensure every token has an entry
    for token in tokens:
        if token not in mapping_meta:
            mapping_meta[token] = {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}

    out = {
        "tokens_total": len(tokens),
        "exact": exact,
        "auto_fuzzy": auto_fuzzy,
        "ambiguous": len(ambiguous),
        "llm_resolved": llm_resolved,
        "llm_rejected": llm_rejected,
        "still_unresolved": sum(1 for v in mapping_meta.values() if v["ingredient_id"] is None),
        "mapping_meta": mapping_meta,
    }
    out_path = out_dir / "loop2b_bridge.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    return out
