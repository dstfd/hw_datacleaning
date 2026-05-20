"""Loop 3b — Coherence check (cuisine_type + observed-categories signal).

A resolved restaurant link can be wrong even when the name match is high.
Example: "Brazilian Feijoada Stew" matched to "Sushi Cove Cafe" (cuisine=Japanese).
Name spelled correctly, link is fishy.

CODE  ⚙ : per restaurant, build a histogram of ingredient categories observed
          across its currently-linked recipes. This is a second cuisine signal
          derived from the data itself — independent of the (often noisy)
          source `cuisine_type` column.
LLM   🧠: per recipe with resolved restaurant, judge plausibility using BOTH
          the source cuisine_type AND the observed category profile.
          Incoherent matches get confidence downgraded.
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from ..lib.llm import json_call, batched

LLM_BATCH = 6  # keep responses small — coherence JSON was truncating
SUSPICIOUS_FLOOR = 0.35  # downgrade confidence to this if incoherent
SIGNAL_TOP_K = 4          # how many top categories to surface per restaurant
MIN_RECIPES_FOR_SIGNAL = 3  # ignore signal for restaurants with too few recipes


def build_cuisine_signal(linked_recipes: list[dict]) -> dict[str, dict]:
    """For each restaurant, derive (top categories, recipe_count) from its
    linked recipes' resolved ingredient categories. Pure deterministic."""
    by_rest: dict[str, Counter] = {}
    counts: dict[str, int] = {}
    for r in linked_recipes:
        rest = r.get("restaurant")
        if not rest:
            continue
        rid = rest.get("restaurant_id") or rest.get("id")
        if not rid:
            continue
        counts[rid] = counts.get(rid, 0) + 1
        cat_counter = by_rest.setdefault(rid, Counter())
        for ing in (r.get("ingredients") or []):
            cat = ing.get("category")
            if cat and ing.get("ingredient_id"):
                cat_counter[cat] += 1

    signal = {}
    for rid, counter in by_rest.items():
        total_ings = sum(counter.values())
        top = counter.most_common(SIGNAL_TOP_K)
        signal[rid] = {
            "recipe_count": counts[rid],
            "ingredient_count": total_ings,
            "top_categories": [
                {"category": cat, "share": round(n / total_ings, 3)}
                for cat, n in top
            ] if total_ings else [],
        }
    return signal


SYSTEM = """You are a culinary plausibility judge.
For each (recipe_name, restaurant_cuisine_type, observed_categories) triple,
decide whether the recipe plausibly belongs to that restaurant.

Two cuisine signals are given:
  - cuisine_type   : the restaurant's source label (can be wrong or vague)
  - observed_categories : the most common ingredient categories actually used
                          in this restaurant's other recipes (derived from data)

Be permissive: many restaurants serve dishes outside their headline cuisine.
Only mark INCOHERENT when BOTH signals strongly disagree with the recipe
(e.g. a Mexican Burrito at a Japanese sushi place that mostly uses Seafood
and Rice categories).

If observed_categories is empty or recipe_count is small, weight cuisine_type
more. If they contradict each other, trust observed_categories.

Output JSON only."""


PROMPT_TMPL = """## Plausibility check

For each entry, output:
  coherent: true | false
  confidence: 0.0..1.0  (how sure you are about the judgment)
  reason: max 6 words

{batch_json}

Return a JSON array, one entry per input.

```json
[
  {{"recipe_id": "...", "coherent": true|false, "confidence": 0.0..1.0, "reason": "..."}}
]
```
"""


def run(linking: dict, out_dir: Path) -> dict:
    print("→ Loop 3b: building cuisine signal from linked ingredients...")
    recs = linking["linked_recipes"]
    cuisine_signal = build_cuisine_signal(recs)
    print(f"  signal computed for {len(cuisine_signal)} restaurants")

    # Build LLM payload — include both source cuisine_type and observed categories.
    pairs = []
    for r in recs:
        rest = r.get("restaurant")
        if not rest:
            continue
        cuisine = (rest.get("cuisine_type") or "").strip()
        rid = rest.get("restaurant_id") or rest.get("id")
        sig = cuisine_signal.get(rid) or {}
        observed = sig.get("top_categories") if sig.get("recipe_count", 0) >= MIN_RECIPES_FOR_SIGNAL else []
        # Skip if absolutely no signal in either direction.
        if not cuisine and not observed:
            continue
        pairs.append({
            "recipe_id": r["recipe_id"],
            "recipe_name": r["recipe_name"],
            "cuisine_type": cuisine or None,
            "observed_categories": observed,
            "restaurant_recipe_count": sig.get("recipe_count", 0),
        })
    print(f"  {len(pairs)} (recipe, signal) triples to judge")

    judgments: dict[str, dict] = {}
    for i, chunk in enumerate(batched(pairs, LLM_BATCH), start=1):
        prompt = PROMPT_TMPL.format(batch_json=json.dumps(chunk, separators=(",", ":")))
        result = json_call(prompt, system=SYSTEM, max_tokens=16384)
        if not isinstance(result, list):
            raise RuntimeError(f"Coherence batch {i}: non-list response")
        for r in result:
            rid = r.get("recipe_id")
            if not rid:
                continue
            judgments[rid] = {
                "coherent": bool(r.get("coherent", True)),
                "confidence": round(float(r.get("confidence", 0.7)), 3),
                "reason": str(r.get("reason", ""))[:120],
            }
        print(f"  batch {i}/{(len(pairs)+LLM_BATCH-1)//LLM_BATCH} ✓")

    # Apply judgments to linking output.
    downgraded = 0
    coherent = 0
    skipped_low_conf = 0
    for r in recs:
        j = judgments.get(r["recipe_id"])
        if j is None:
            continue
        rest = r.get("restaurant")
        if not rest:
            continue
        if j["coherent"]:
            coherent += 1
            rest["coherence"] = {"coherent": True, "reason": j["reason"]}
            continue
        if j["confidence"] < 0.6:
            skipped_low_conf += 1
            rest["coherence"] = {"coherent": False, "judged_confidence": j["confidence"],
                                 "reason": j["reason"], "applied": False}
            continue
        prior_method = rest.get("method", "?")
        prior_conf = rest.get("confidence", 1.0)
        rest["confidence"] = min(prior_conf, SUSPICIOUS_FLOOR)
        rest["method"] = f"suspicious:{prior_method}"
        rest["coherence"] = {"coherent": False, "judged_confidence": j["confidence"],
                             "reason": j["reason"], "applied": True}
        downgraded += 1

    stats = {
        "restaurants_with_signal": len(cuisine_signal),
        "pairs_checked": len(pairs),
        "coherent": coherent,
        "incoherent_downgraded": downgraded,
        "incoherent_skipped_low_conf": skipped_low_conf,
    }
    out = {
        "stats": stats,
        "cuisine_signal": cuisine_signal,
        "judgments": judgments,
    }
    out_path = out_dir / "loop3b_coherence.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    print(f"  coherent: {coherent}, downgraded: {downgraded}, skipped (low conf): {skipped_low_conf}")
    return out
