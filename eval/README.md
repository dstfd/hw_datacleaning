# Eval — 50-record gold set + scorer

## What this is

A small, hand-audited gold set of 50 recipes and an eval script that scores
any pipeline's `output.json` against it.

Use it to compare pipelines honestly — instead of arguing about average
confidence numbers that two pipelines compute differently.

## Files

- `gold.json` — the 50 gold records, with a `reason` field on every decision
- `run_eval.py` — scorer; runs against any pipeline output that follows the
  shared schema (`recipes[].recipe_id`, `restaurant.restaurant_id`,
  `menu.menu_id`, `ingredients[].raw + .name + .ingredient_id`)

## Methodology (the honest part)

The gold set is constructed from the source data alone — no external knowledge.

**Sampling.** Stratified across data-completeness classes:
- 15× `B_rid_only`     (has restaurant_id only)
- 15× `D_rname_only`   (has restaurant_name only)
- 10× `E_mid_only`     (has menu_id only — need to recover restaurant via menu)
- 10× `F_orphan`       (has nothing — should stay unresolved)
- Total = 50. Deterministic (`random_state=42`).

**Gold values.** Per recipe:
- `gold_restaurant_id`: from source if valid; else exact normalized name in
  `Restaurants.name`; else fuzz≥90 against `Restaurants.name`; else via
  `menus[menu_id].restaurant_id`; else `null`.
- `gold_menu_id`: from source if valid; else `null`.
  (We do **not** invent a menu_id for cases where the source has none — there
  is no ground truth signal for menu inference.)
- `gold_ingredients[]`: per token, the expected canonical **name** (not ID).
  Exact in `Ingredients.name` or fuzz≥92. The gold names are normalized
  (lowercased, whitespace-collapsed) and the eval matches on fuzz≥85 so it
  tolerates different canonical-name spellings across pipelines.

**What this gold set is NOT.**
- Not an external ground truth — it is derived from the same source XLSX.
- Not a benchmark of which recipe-restaurant pairs are "true in the world".
- Not 5000 records or stratified by cuisine. 50 is enough to detect a
  pipeline that confidently outputs wrong answers; not enough to compare two
  good pipelines at fine granularity.

## What the eval reports

Per field (restaurant / menu / ingredients):
- `tp` — pipeline answer matches gold answer
- `wrong` — pipeline returned an answer, gold has an answer, they differ
- `fp_hallucination` — pipeline returned an answer, gold says null
- `fn_missed` — gold has an answer, pipeline returned null
- `tn_correct_abstain` — both null (correctly gave up)
- `precision` = `tp / (tp + fp + wrong)`
- `recall`    = `tp / (tp + fn + wrong)`
- `f1`        = harmonic mean

Hallucinations (returning an answer when there is no signal) are penalised
twice — once in precision, once because they replace a correct abstention.

## Run

```bash
# from project root
python3 live/eval/run_eval.py output.json
# or against a specific run
python3 live/eval/run_eval.py data/runs/<runId>/output.json
```

The eval is read-only — it never modifies the pipeline output or the gold
file.
