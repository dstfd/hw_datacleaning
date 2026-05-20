"""Loop 2 — Cleaning.

CODE  ⚙ : cluster ingredient rows with rapidfuzz; assign stable canonical IDs.
LLM   🧠: per cluster, confirm canonical name and assign a semantic category.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz

from ..lib.llm import json_call, batched
from ..lib.strings import norm

# Cluster threshold: token_sort_ratio. 86 was sweet-spot in earlier experiments.
CLUSTER_THRESHOLD = 86
LLM_BATCH = 30  # clusters per LLM call


def cluster_ingredients(df: pd.DataFrame) -> list[dict]:
    """Greedy clustering: each ingredient joins the highest-scoring cluster ≥ threshold."""
    clusters: list[dict] = []
    for _, row in df.iterrows():
        name = str(row["name"])
        cat = str(row.get("category") or "")
        iid = str(row["ingredient_id"]).strip()
        nn = norm(name)
        if not nn:
            continue
        best_i, best_s = -1, 0.0
        for i, cl in enumerate(clusters):
            s = fuzz.token_sort_ratio(nn, cl["centroid"])
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0 and best_s >= CLUSTER_THRESHOLD:
            clusters[best_i]["members"].append({"id": iid, "name": name, "src_category": cat})
        else:
            clusters.append({
                "centroid": nn,
                "members": [{"id": iid, "name": name, "src_category": cat}],
            })
    # Assign stable canonical ids
    for i, cl in enumerate(clusters):
        cl["canonical_id"] = f"C{i+1:04d}"
        # Provisional canonical: longest member name (typos drop letters)
        members = cl["members"]
        names = [m["name"] for m in members]
        scores = [sum(fuzz.token_sort_ratio(n, o) for o in names if o != n) for n in names]
        if len(members) == 1:
            cl["provisional_canonical"] = members[0]["name"]
        else:
            best = max(range(len(members)), key=lambda j: (scores[j], len(names[j]), names[j]))
            cl["provisional_canonical"] = names[best]
    return clusters


SYSTEM = """You are a culinary data canonicalizer.
Given a cluster of ingredient name variants (most are typos of the same thing),
produce:
  1. the cleanest canonical name (proper spelling, lowercase unless a proper noun)
  2. the correct semantic category from this fixed set:
     Spice, Herb, Oil, Sweetener, Grain, Dairy, Protein, Sauce, Vegetable,
     Fruit, Nut, Seed, Legume, Pasta & Noodles, Seafood, Meat, Condiment,
     Beverage, Alcohol, Baking, Broth, Other

Respond ONLY with the JSON array shape requested. The cluster IDs are stable —
return one entry per input cluster, preserving order is fine but match by id."""


PROMPT_TMPL = """## Clusters needing canonical name + category

For each cluster, pick the cleanest canonical_name from members (or write a
corrected spelling if all members are corrupted), and assign the correct
category.

{clusters_json}

## Return

```json
[
  {{"canonical_id": "C0001", "canonical_name": "...", "category": "..."}},
  ...
]
```

Return only the JSON array."""


def run(structure: dict, xlsx_path: Path, out_dir: Path) -> dict:
    print("→ Loop 2: clustering ingredients...")
    df = pd.read_excel(xlsx_path, sheet_name="Ingredients")
    clusters = cluster_ingredients(df)
    print(f"  {len(df)} rows → {len(clusters)} clusters")

    # Prepare LLM payloads — small slices per cluster, plus only top src_categories
    llm_input = [
        {
            "canonical_id": cl["canonical_id"],
            "members": [m["name"] for m in cl["members"]],
            "provisional_canonical": cl["provisional_canonical"],
        }
        for cl in clusters
    ]

    print(f"→ Loop 2: LLM labels {len(clusters)} clusters in batches of {LLM_BATCH}...")
    labels: dict[str, dict] = {}
    for i, chunk in enumerate(batched(llm_input, LLM_BATCH), start=1):
        prompt = PROMPT_TMPL.format(clusters_json=json.dumps(chunk, indent=2))
        result = json_call(prompt, system=SYSTEM, max_tokens=8192)
        if not isinstance(result, list):
            raise RuntimeError(f"Batch {i}: LLM returned non-list: {type(result)}")
        for r in result:
            cid = r.get("canonical_id")
            if cid:
                labels[cid] = {
                    "name": r.get("canonical_name") or "",
                    "category": r.get("category") or "Other",
                }
        print(f"  batch {i}/{(len(llm_input)+LLM_BATCH-1)//LLM_BATCH} ✓ labeled {len(result)}")

    # Build merge map (variant name → canonical_id) and canonical list
    canonicals = []
    merge_map: dict[str, str] = {}
    missing_labels = 0
    for cl in clusters:
        lbl = labels.get(cl["canonical_id"])
        if lbl is None:
            missing_labels += 1
            lbl = {"name": cl["provisional_canonical"], "category": "Other"}
        canonicals.append({
            "ingredient_id": cl["canonical_id"],
            "name": lbl["name"],
            "category": lbl["category"],
            "variant_count": len(cl["members"]),
            "source_ids": [m["id"] for m in cl["members"]],
        })
        for m in cl["members"]:
            merge_map[norm(m["name"])] = cl["canonical_id"]

    out = {
        "cluster_count": len(clusters),
        "canonical_ingredients": canonicals,
        "merge_map": merge_map,
        "stats": {
            "input_rows": int(len(df)),
            "clusters": len(clusters),
            "merge_map_size": len(merge_map),
            "missing_labels": missing_labels,
        },
    }
    out_path = out_dir / "loop2_cleaning.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    print(f"  canonicals: {len(canonicals)}, missing_labels: {missing_labels}")
    return out
