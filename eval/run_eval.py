"""Eval — compare pipeline output.json against the 50-record gold set.

Usage:
    python3 live/eval/run_eval.py output.json
    python3 live/eval/run_eval.py data/runs/<runId>/output.json

Metrics, per field (restaurant / menu / ingredients):
  - precision: of records where pipeline gave an answer, how many were right
  - recall:    of records where gold has an answer, how many did pipeline get
  - abstention_correct: pipeline returned null and gold is also null
  - hallucination:      pipeline returned an answer but gold is null

Ingredients are matched by NAME (fuzz>=85) so the score is robust to different
ID-namespace conventions across pipelines.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent.parent
GOLD = Path(__file__).resolve().parent / "gold.json"
INGREDIENT_NAME_MATCH = 85  # fuzz token_sort_ratio threshold


def load_pipeline(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {r["recipe_id"]: r for r in data["recipes"]}


def names_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return fuzz.token_sort_ratio(str(a).strip().lower(), str(b).strip().lower()) >= INGREDIENT_NAME_MATCH


def eval_against_gold(pipeline_path: Path) -> dict:
    gold = json.loads(GOLD.read_text())
    pipe = load_pipeline(pipeline_path)

    # Counters
    rest_tp = rest_fp = rest_fn = rest_tn = rest_wrong = 0
    menu_tp = menu_fp = menu_fn = menu_tn = menu_wrong = 0
    ing_tp = ing_fp = ing_fn = 0
    missing_recipes = []
    rest_examples_wrong = []
    menu_examples_wrong = []

    for g in gold["records"]:
        rid = g["recipe_id"]
        p = pipe.get(rid)
        if p is None:
            missing_recipes.append(rid)
            continue

        # ── Restaurant ─────────────────────────────────────────────────────
        gold_rest = g["gold_restaurant_id"]
        pipe_rest = p["restaurant"]["restaurant_id"] if p.get("restaurant") else None
        if gold_rest is None and pipe_rest is None:
            rest_tn += 1
        elif gold_rest is None and pipe_rest is not None:
            rest_fp += 1
            rest_examples_wrong.append({"recipe_id": rid, "type": "hallucination",
                                        "gold": None, "pipe": pipe_rest})
        elif gold_rest is not None and pipe_rest is None:
            rest_fn += 1
        elif gold_rest == pipe_rest:
            rest_tp += 1
        else:
            rest_wrong += 1
            rest_examples_wrong.append({"recipe_id": rid, "type": "wrong",
                                        "gold": gold_rest, "pipe": pipe_rest})

        # ── Menu ───────────────────────────────────────────────────────────
        gold_menu = g["gold_menu_id"]
        pipe_menu = p["menu"]["menu_id"] if p.get("menu") else None
        if gold_menu is None and pipe_menu is None:
            menu_tn += 1
        elif gold_menu is None and pipe_menu is not None:
            menu_fp += 1
            menu_examples_wrong.append({"recipe_id": rid, "type": "hallucination",
                                        "gold": None, "pipe": pipe_menu})
        elif gold_menu is not None and pipe_menu is None:
            menu_fn += 1
        elif gold_menu == pipe_menu:
            menu_tp += 1
        else:
            menu_wrong += 1
            menu_examples_wrong.append({"recipe_id": rid, "type": "wrong",
                                        "gold": gold_menu, "pipe": pipe_menu})

        # ── Ingredients (matched by name, fuzz>=85) ────────────────────────
        # Prefer matching by `raw` if pipeline stores it (live/ does).
        # Fall back to positional match (OODA pipeline drops `raw`).
        gold_ings = g["gold_ingredients"]
        pipe_ings = p.get("ingredients", [])
        has_raw = any(ing.get("raw") for ing in pipe_ings)
        pipe_by_raw = {ing.get("raw", "").strip().lower(): ing for ing in pipe_ings} if has_raw else {}
        for idx, gi in enumerate(gold_ings):
            raw = gi["raw"].strip().lower()
            gold_name = gi["expected_name"]
            if has_raw:
                pi = pipe_by_raw.get(raw)
            else:
                pi = pipe_ings[idx] if idx < len(pipe_ings) else None
            pipe_name = (pi or {}).get("name") if pi and pi.get("ingredient_id") else None
            if gold_name is None and pipe_name is None:
                pass  # both abstain; not counted
            elif gold_name is None and pipe_name is not None:
                ing_fp += 1
            elif gold_name is not None and pipe_name is None:
                ing_fn += 1
            elif names_match(pipe_name, gold_name):
                ing_tp += 1
            else:
                ing_fn += 1  # wrong name = miss
                ing_fp += 1  # wrong name = also a hallucination of a different one

    def prf(tp, fp, fn):
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
        return round(p_, 3), round(r_, 3), round(f1, 3)

    rp, rr, rf = prf(rest_tp, rest_fp + rest_wrong, rest_fn + rest_wrong)
    mp, mr, mf = prf(menu_tp, menu_fp + menu_wrong, menu_fn + menu_wrong)
    ip, ir, if_ = prf(ing_tp, ing_fp, ing_fn)

    return {
        "pipeline_file": str(pipeline_path),
        "gold_records": len(gold["records"]),
        "missing_recipes_in_pipeline": missing_recipes,
        "restaurant": {
            "tp": rest_tp, "wrong": rest_wrong, "fp_hallucination": rest_fp,
            "fn_missed": rest_fn, "tn_correct_abstain": rest_tn,
            "precision": rp, "recall": rr, "f1": rf,
            "examples_wrong": rest_examples_wrong[:5],
        },
        "menu": {
            "tp": menu_tp, "wrong": menu_wrong, "fp_hallucination": menu_fp,
            "fn_missed": menu_fn, "tn_correct_abstain": menu_tn,
            "precision": mp, "recall": mr, "f1": mf,
            "examples_wrong": menu_examples_wrong[:5],
        },
        "ingredients": {
            "tp": ing_tp, "fp_hallucination_or_wrong_name": ing_fp,
            "fn_missed_or_wrong_name": ing_fn,
            "precision": ip, "recall": ir, "f1": if_,
        },
    }


def main():
    if len(sys.argv) < 2:
        path = ROOT / "output.json"
    else:
        path = Path(sys.argv[1])
    if not path.exists():
        print(f"Pipeline output not found: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"Evaluating: {path}")
    print(f"Gold set:   {GOLD}")
    print()
    report = eval_against_gold(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
