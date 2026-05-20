"""Loop 4 — Confidence rollup + output.json assembly.

Pure deterministic. No LLM.
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

LOW_CONFIDENCE_FLAG = 0.6


def run(linking: dict, run_id: str, out_dir: Path, root_out: Path) -> dict:
    print("→ Loop 4: assembly + confidence rollup...")
    recipes_out = []
    rest_resolved = 0
    menu_resolved = 0
    ing_resolved = 0
    ing_total = 0
    review = 0
    conf_sum = 0.0

    for r in linking["linked_recipes"]:
        rest = r.get("restaurant")
        menu = r.get("menu")
        ings = r.get("ingredients") or []
        resolved = sum(1 for i in ings if i.get("ingredient_id") is not None)

        rest_conf = rest["confidence"] if rest else 0.0
        menu_conf = menu["confidence"] if menu else 0.0
        ing_conf = (sum(i.get("confidence", 0.0) for i in ings) / len(ings)) if ings else None

        parts = [rest_conf, menu_conf]
        if ing_conf is not None:
            parts.append(ing_conf)
        overall = sum(parts) / len(parts) if parts else 0.0

        if rest is not None:
            rest_resolved += 1
        if menu is not None:
            menu_resolved += 1
        ing_resolved += resolved
        ing_total += len(ings)
        conf_sum += overall
        flagged = overall < LOW_CONFIDENCE_FLAG
        if flagged:
            review += 1

        recipes_out.append({
            "recipe_id": r["recipe_id"],
            "recipe_name": r["recipe_name"],
            "restaurant": rest,
            "menu": menu,
            "ingredients": [
                {k: v for k, v in ing.items() if k != "raw"} | {"raw": ing.get("raw", "")}
                for ing in ings
            ],
            "mapping_quality": {
                "overall_confidence": round(overall, 3),
                "restaurant_method": rest["method"] if rest else None,
                "menu_method": menu["method"] if menu else None,
                "ingredients_resolved": resolved,
                "ingredients_total": len(ings),
                "human_review": flagged,
            },
        })

    summary = {
        "total_recipes": len(recipes_out),
        "recipes_with_restaurant": rest_resolved,
        "recipes_with_menu": menu_resolved,
        "ingredients_resolved_total": ing_resolved,
        "ingredients_unresolved_total": ing_total - ing_resolved,
        "average_confidence": round(conf_sum / len(recipes_out), 3) if recipes_out else 0.0,
        "records_in_human_review": review,
        "run_id": run_id,
        "loops_run": 6,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "method": "live-overlay (deterministic prepass + LLM judgment on residual)",
    }

    out = {"recipes": recipes_out, "summary": summary}
    out_path = out_dir / "output.json"
    out_path.write_text(json.dumps(out, indent=2))
    root_out.write_text(json.dumps(out, indent=2))
    print(f"  ✓ {out_path}")
    print(f"  ✓ {root_out}")
    print(f"  restaurants: {rest_resolved}/{len(recipes_out)}, menus: {menu_resolved}/{len(recipes_out)}, ingredients: {ing_resolved}/{ing_total}")
    print(f"  avg confidence: {summary['average_confidence']}, human review: {review}")
    return out
