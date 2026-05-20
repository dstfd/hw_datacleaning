"""Loop 3 — Linking.

CODE  ⚙ : prepass — exact_id, exact_name, fuzzy>=0.92 resolved deterministically.
LLM   🧠: adjudicate residual fuzzy_name + pick menu by recipe-name semantics.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz, process

from ..lib.llm import json_call, batched
from ..lib.strings import norm

NAME_AUTO_HIGH = 92    # ≥ → accept fuzzy_high
NAME_LOW_CUTOFF = 70   # < → null without asking LLM
LLM_BATCH = 20
MENU_LLM_BATCH = 8  # smaller batches — menu JSON was truncating at 4096 tokens


SYSTEM_RESTAURANT = """You are a restaurant name matcher.
For each query name + numbered candidate list, pick the best candidate by INDEX
or null if no candidate is a plausible match. Account for typos and word-order
differences. Output JSON only."""

PROMPT_RESTAURANT = """## Restaurant fuzzy adjudication

For each entry, pick the candidate index (0..N-1) that best matches the query,
or null if none. Confidence 0.60-0.95.

{batch_json}

```json
[
  {{"query": "...", "pick": 0|1|...|null, "confidence": 0.0..1.0}}
]
```
"""


SYSTEM_MENU = """You are a restaurant menu selector.
Given a recipe name and the candidate menus of its restaurant, pick the menu
that best fits the recipe (e.g. a vegan dish → Vegan Menu; pancakes → Breakfast
Menu; steak → Dinner Menu). If a restaurant has only one menu, pick it. Output
JSON only."""

PROMPT_MENU = """## Menu selection

For each recipe, pick the best-fitting menu BY INDEX from the candidates.
Confidence: 0.85+ if recipe keywords strongly match a menu name; 0.55-0.75 if
weak fit; null if no candidate fits.

{batch_json}

```json
[
  {{"recipe_id": "...", "pick": 0|1|...|null, "confidence": 0.0..1.0}}
]
```
"""


def run(structure: dict, cleaning: dict, bridge: dict, xlsx_path: Path, out_dir: Path) -> dict:
    print("→ Loop 3: loading sheets and indexing...")
    recipes = pd.read_excel(xlsx_path, sheet_name="Recipes")
    restaurants = pd.read_excel(xlsx_path, sheet_name="Restaurants")
    menus_df = pd.read_excel(xlsx_path, sheet_name="Menus")

    threshold = structure["strategy"].get("global_fuzzy_threshold", 0.80) or 0.80
    name_match_low = max(NAME_LOW_CUTOFF, int(threshold * 100) - 5)
    print(f"  using fuzzy threshold {threshold} (LLM range {name_match_low}..{NAME_AUTO_HIGH})")

    rest_by_id = {str(r["restaurant_id"]).strip(): {
        "restaurant_id": str(r["restaurant_id"]).strip(),
        "name": str(r["name"]),
        "cuisine_type": str(r.get("cuisine_type") or ""),
        "city": str(r.get("city") or ""),
    } for _, r in restaurants.iterrows()}
    rest_norm_to_id = {norm(r["name"]): str(r["restaurant_id"]).strip() for _, r in restaurants.iterrows()}
    rest_names = [str(r["name"]) for _, r in restaurants.iterrows()]
    rest_ids = list(rest_by_id.keys())

    menus_by_id = {str(m["menu_id"]).strip(): {
        "menu_id": str(m["menu_id"]).strip(),
        "menu_name": str(m["menu_name"]),
        "restaurant_id": str(m["restaurant_id"]).strip(),
    } for _, m in menus_df.iterrows()}
    menus_by_restaurant: dict[str, list[dict]] = {}
    for m in menus_by_id.values():
        menus_by_restaurant.setdefault(m["restaurant_id"], []).append(m)

    mapping_meta = bridge["mapping_meta"]
    canon_by_id = {c["ingredient_id"]: c for c in cleaning["canonical_ingredients"]}

    # ── Prepass over every recipe ───────────────────────────────────────────
    decisions: list[dict] = []
    rest_ambiguous: list[dict] = []
    menu_ambiguous: list[dict] = []

    stats = {"exact_id_rest": 0, "exact_name_rest": 0, "fuzzy_high_rest": 0,
             "ambiguous_rest": 0, "via_menu_rest": 0, "no_rest_signal": 0,
             "exact_id_menu": 0, "menu_inferred_single": 0,
             "menu_ambiguous": 0, "no_menu_signal": 0}

    for _, rec in recipes.iterrows():
        rid_raw = rec.get("restaurant_id")
        rid = str(rid_raw).strip() if pd.notna(rid_raw) else ""
        rname_raw = rec.get("restaurant_name")
        rname = str(rname_raw).strip() if pd.notna(rname_raw) else ""

        rest_resolution = None  # dict or None
        if rid and rid in rest_by_id:
            rest_resolution = {"id": rid, "method": "exact_id", "confidence": 1.0,
                               "evidence": [rest_by_id[rid]["name"]]}
            stats["exact_id_rest"] += 1
        elif rname:
            nn = norm(rname)
            if nn in rest_norm_to_id:
                target = rest_norm_to_id[nn]
                rest_resolution = {"id": target, "method": "exact_name", "confidence": 1.0,
                                   "evidence": [rest_by_id[target]["name"]]}
                stats["exact_name_rest"] += 1
            else:
                top5 = process.extract(rname, rest_names, scorer=fuzz.token_sort_ratio, limit=5)
                if not top5:
                    stats["no_rest_signal"] += 1
                elif top5[0][1] >= NAME_AUTO_HIGH:
                    idx = top5[0][2]
                    target_id = rest_ids[idx]
                    rest_resolution = {"id": target_id, "method": "fuzzy_high",
                                       "confidence": round(top5[0][1] / 100.0, 3),
                                       "evidence": [rest_by_id[target_id]["name"]]}
                    stats["fuzzy_high_rest"] += 1
                elif top5[0][1] >= name_match_low:
                    rest_ambiguous.append({
                        "recipe_id": str(rec["recipe_id"]),
                        "query": rname,
                        "candidates": [
                            {"index": j, "id": rest_ids[m[2]], "name": m[0], "score": m[1]}
                            for j, m in enumerate(top5)
                        ],
                    })
                    stats["ambiguous_rest"] += 1
                else:
                    stats["no_rest_signal"] += 1
        else:
            stats["no_rest_signal"] += 1

        mid_raw = rec.get("menu_id")
        mid = str(mid_raw).strip() if pd.notna(mid_raw) else ""
        menu_resolution = None
        if mid and mid in menus_by_id:
            m = menus_by_id[mid]
            menu_resolution = {"id": mid, "method": "exact_id", "confidence": 1.0,
                               "evidence": [m["menu_name"]]}
            stats["exact_id_menu"] += 1

        decisions.append({
            "recipe_id": str(rec["recipe_id"]),
            "recipe_name": str(rec.get("recipe_name") or ""),
            "_rest_resolution": rest_resolution,
            "_menu_resolution": menu_resolution,
            "_raw_rest": rid, "_raw_rname": rname, "_raw_mid": mid,
        })

    # ── LLM adjudicates ambiguous restaurants ──────────────────────────────
    rec_by_id = {d["recipe_id"]: d for d in decisions}
    if rest_ambiguous:
        print(f"→ Loop 3: LLM adjudicates {len(rest_ambiguous)} ambiguous restaurants...")
        for i, chunk in enumerate(batched(rest_ambiguous, LLM_BATCH), start=1):
            payload = [
                {"query": e["query"],
                 "candidates": [{"index": c["index"], "name": c["name"], "score": c["score"]}
                                for c in e["candidates"]]}
                for e in chunk
            ]
            result = json_call(PROMPT_RESTAURANT.format(batch_json=json.dumps(payload, indent=2)),
                               system=SYSTEM_RESTAURANT, max_tokens=4096)
            if not isinstance(result, list):
                raise RuntimeError(f"Rest batch {i}: non-list response")
            for r, entry in zip(result, chunk):
                rec = rec_by_id[entry["recipe_id"]]
                pick = r.get("pick")
                if pick is None or not isinstance(pick, int) or pick < 0 or pick >= len(entry["candidates"]):
                    continue
                chosen = entry["candidates"][pick]
                rec["_rest_resolution"] = {
                    "id": chosen["id"], "method": "llm_fuzzy",
                    "confidence": round(float(r.get("confidence", 0.7)), 3),
                    "evidence": [chosen["name"]],
                }
            print(f"  rest batch {i}/{(len(rest_ambiguous)+LLM_BATCH-1)//LLM_BATCH} ✓")

    # ── Recovery: restaurant still null but menu_id valid → use menu's restaurant
    via_menu_count = 0
    for d in decisions:
        if d["_rest_resolution"] is not None:
            continue
        mid = d["_raw_mid"]
        if not mid or mid not in menus_by_id:
            continue
        via_rid = menus_by_id[mid]["restaurant_id"]
        if via_rid not in rest_by_id:
            continue
        d["_rest_resolution"] = {
            "id": via_rid, "method": "via_menu", "confidence": 1.0,
            "evidence": [f"menu {mid} → {rest_by_id[via_rid]['name']}"],
        }
        via_menu_count += 1
    stats["via_menu_rest"] = via_menu_count
    stats["no_rest_signal"] = max(0, stats["no_rest_signal"] - via_menu_count)
    if via_menu_count:
        print(f"  recovered {via_menu_count} restaurants via menu_id")

    # ── Menu inference: for resolved-restaurant + missing-menu, ask LLM ────
    for d in decisions:
        if d["_menu_resolution"] is not None:
            continue
        rr = d["_rest_resolution"]
        if rr is None:
            continue
        rmenus = menus_by_restaurant.get(rr["id"], [])
        if not rmenus:
            continue
        if len(rmenus) == 1:
            m = rmenus[0]
            d["_menu_resolution"] = {"id": m["menu_id"], "method": "only_menu",
                                     "confidence": 0.6, "evidence": [m["menu_name"]]}
            stats["menu_inferred_single"] += 1
        else:
            menu_ambiguous.append({
                "recipe_id": d["recipe_id"],
                "recipe_name": d["recipe_name"],
                "candidates": [{"index": j, "id": m["menu_id"], "name": m["menu_name"]}
                               for j, m in enumerate(rmenus)],
            })
            stats["menu_ambiguous"] += 1

    if menu_ambiguous:
        print(f"→ Loop 3: LLM picks menus for {len(menu_ambiguous)} recipes...")
        for i, chunk in enumerate(batched(menu_ambiguous, MENU_LLM_BATCH), start=1):
            payload = [
                {"recipe_id": e["recipe_id"], "recipe_name": e["recipe_name"],
                 "candidates": [{"index": c["index"], "name": c["name"]} for c in e["candidates"]]}
                for e in chunk
            ]
            result = json_call(PROMPT_MENU.format(batch_json=json.dumps(payload, indent=2)),
                               system=SYSTEM_MENU, max_tokens=8192)
            if not isinstance(result, list):
                raise RuntimeError(f"Menu batch {i}: non-list response")
            by_recipe = {r.get("recipe_id"): r for r in result if isinstance(r, dict)}
            for entry in chunk:
                rec = rec_by_id[entry["recipe_id"]]
                r = by_recipe.get(entry["recipe_id"])
                if r is None:
                    continue
                pick = r.get("pick")
                if pick is None or not isinstance(pick, int) or pick < 0 or pick >= len(entry["candidates"]):
                    continue
                chosen = entry["candidates"][pick]
                rec["_menu_resolution"] = {
                    "id": chosen["id"], "method": "llm_menu",
                    "confidence": round(float(r.get("confidence", 0.65)), 3),
                    "evidence": [chosen["name"]],
                }
            print(f"  menu batch {i}/{(len(menu_ambiguous)+MENU_LLM_BATCH-1)//MENU_LLM_BATCH} ✓")

    # ── Resolve ingredients from bridge mapping_meta ───────────────────────
    raw_recipes = recipes.set_index("recipe_id")
    linked_recipes = []
    for d in decisions:
        recipe_id = d["recipe_id"]
        rr = d["_rest_resolution"]
        mr = d["_menu_resolution"]
        rest_out = None
        if rr is not None:
            r = rest_by_id[rr["id"]]
            rest_out = {**r, "confidence": rr["confidence"], "method": rr["method"]}
        menu_out = None
        if mr is not None:
            m = menus_by_id[mr["id"]]
            menu_out = {"menu_id": m["menu_id"], "menu_name": m["menu_name"],
                        "confidence": mr["confidence"], "method": mr["method"]}

        raw_ing = raw_recipes.loc[recipe_id]["ingredients"] if recipe_id in raw_recipes.index else None
        tokens = [t.strip() for t in str(raw_ing).split("|")] if pd.notna(raw_ing) else []
        tokens = [t for t in tokens if t]
        ings = []
        for t in tokens:
            meta = mapping_meta.get(t) or {"ingredient_id": None, "method": "unresolved", "confidence": 0.0}
            cid = meta["ingredient_id"]
            if cid and cid in canon_by_id:
                c = canon_by_id[cid]
                ings.append({"raw": t, "ingredient_id": cid, "name": c["name"],
                             "category": c["category"], "method": meta["method"],
                             "confidence": meta["confidence"]})
            else:
                ings.append({"raw": t, "ingredient_id": None, "name": t,
                             "category": None, "method": "unresolved", "confidence": 0.0})

        linked_recipes.append({
            "recipe_id": recipe_id,
            "recipe_name": d["recipe_name"],
            "restaurant": rest_out,
            "menu": menu_out,
            "ingredients": ings,
        })

    out = {
        "stats": stats,
        "linked_recipes": linked_recipes,
    }
    out_path = out_dir / "loop3_linking.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    print(f"  stats: {stats}")
    return out
