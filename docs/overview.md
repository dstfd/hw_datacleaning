# Recipe Data Reconciliation

Reads `data/recipe_data.xlsx`. Writes `output.json`.

---

## 1. Summary

A six-loop pipeline that reconciles 1000 recipes against restaurants,
menus, and a noisy ingredients catalog. Deterministic code owns the
bookkeeping (profiling, clustering, lookup, scoring, ID generation). The
LLM is used only where judgment helps: picking canonical names, assigning
categories, disambiguating fuzzy matches, choosing menus, and judging
recipe-restaurant plausibility. **The LLM never types an ID.**

Result on a 50-record hand-audited gold set:
- Restaurant F1 = **1.000**
- Ingredients F1 = **0.988**
- Menu F1 = 0.556 (eval methodology debate — see §10)

---

## 2. What was found in the data

**Quality issues**
- Ingredient names: many typos. "garlic / gaarlic / galic / garlik" are all
  the same thing. 901 rows collapse to ~280 real ingredients.
- Ingredient `category` column: ~90% wrong. "olive oil" labeled "Spice".
  Not usable.
- Garbage rows in `Ingredients`: `"and i'll be happy to help!"`,
  `"and chili oil)"`. Leftover text from editing.
- 642 / 1000 recipes have no `restaurant_id`. 867 / 1000 have no
  `menu_id`. 613 / 1000 have no `restaurant_name`.
- `recipes.ingredients` is a pipe-delimited string, not a structured list.

**How recipes link to restaurants**
1. Valid `restaurant_id` → direct match.
2. No id, but `restaurant_name` → exact name lookup, then fuzzy.
3. No id, no name, but valid `menu_id` → use the menu's restaurant.
4. Nothing → cannot link, goes to review bucket.

**How recipes link to menus**
1. Valid `menu_id` → direct match.
2. No `menu_id`, restaurant known → pick from that restaurant's menus
   (single menu auto-attached; otherwise LLM picks by recipe name).
3. No signal → null.

**Ingredient completeness**
- 6882 ingredient mentions across 1000 recipes (~6.9 per recipe).
- 110 recipes have no ingredients at all.
- Each mention is free text. Each one is resolved to a canonical row.

**How missing or unclear info is handled**
- Every field carries a confidence (0.0 = unresolved, 1.0 = exact match).
- Overall recipe confidence is the mean of all field confidences. A null
  field counts as 0 and drags the score down.
- Overall confidence < 0.6 → `human_review: true`.
- The pipeline never silently guesses. If no match is found, the field is
  `null`.

---

## 3. Approach

Each loop solves one problem. Each writes a JSON file under
`live/data/runs/{runId}/`.

| Loop | Question | Problem |
|---|---|---|
| **1 Structure** | What sheets, columns, links exist? | Schema and FKs are unknown until runtime. |
| **2 Cleaning** | Which reference rows are duplicates? | Reference rows have typos. Need one canonical per real thing. |
| **2b Bridge** | Which canonical does this free-text token mean? | Records store values as text, not IDs. |
| **3 Linking** | Which related row does this record belong to? | FKs are missing or partial. |
| **3b Coherence** | Is the resolved link plausible? | A clean fuzzy match can still be wrong (Brazilian dish at sushi place). |
| **4 Confidence** | How sure is the result per record, which need a human? | Confidence must reflect ALL fields, including missing ones. |

**Split of work**
- Code: profile, normalize, cluster, exact lookup, ID generation, scoring.
- LLM: pick canonical names, assign semantic categories, disambiguate
  fuzzy candidates by index, pick the right menu, judge plausibility.

---

## 4. Assumptions

- The `Ingredients` sheet is the source of truth for canonical ingredients.
- `restaurant_name` typos are recoverable by fuzzy + LLM disambiguation.
- A recipe's correct menu must belong to its resolved restaurant.
- One cluster of similar names = one real ingredient.
- Categories use a fixed list of 22 (Spice, Herb, Oil, Protein, etc.).

---

## 5. Known limitations

- Cannot recover a restaurant if both `restaurant_id` and `restaurant_name`
  are missing AND there is no valid `menu_id`.
- Cannot recover a menu if `menu_id` is missing AND the restaurant is
  unresolved.
- Some garbage ingredient rows form their own cluster. They never match a
  recipe token but stay in the canonical list.
- Confidence is rule-based, not learned. Cross-method comparison is
  approximate.
- One run = one output. No learning between runs.
- The 50-record gold set is hand-audited from source data, not external
  ground truth.

---

## 6. How to run

```bash
# Use a venv on macOS / recent Linux — system pip is locked down (PEP 668).
python3 -m venv .venv-live && source .venv-live/bin/activate

pip install google-genai rapidfuzz pandas openpyxl
gcloud auth application-default login
echo 'GOOGLE_CLOUD_PROJECT=your-project' > .env.local
echo 'VERTEX_LOCATION=us-central1'      >> .env.local

PYTHONUNBUFFERED=1 python3 -m live.pipeline      # unbuffered = live progress
```

Resume a failed run from Loop 3b:
```bash
python3 -m live.pipeline --resume <runId>
```

Run the gold-set eval:
```bash
python3 live/eval/run_eval.py output.json
```

---

## 7. Using a different LLM

Only `live/lib/llm.py` knows about the LLM. The rest of the pipeline calls
`json_call(prompt, system=...)` and gets parsed JSON back. Swap the inside
of that one function — Gemini direct, OpenAI, Anthropic, Ollama all work.

**Direct Gemini API key**
```python
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
```

**OpenAI**
```python
from openai import OpenAI
_client = OpenAI()

def json_call(prompt, *, system=None, max_tokens=8192, temperature=0.0):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": prompt})
    r = _client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs,
        max_tokens=max_tokens, temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)
```

**Anthropic**
```python
from anthropic import Anthropic
_client = Anthropic()

def json_call(prompt, *, system=None, max_tokens=8192, temperature=0.0):
    r = _client.messages.create(
        model="claude-sonnet-4-5", system=system or "",
        max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(_strip_code_fence(r.content[0].text))
```

**Local (Ollama, vLLM)** — point OpenAI client at the local URL:
```python
_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

Rules across providers: temperature 0.0, ask for JSON in the prompt, keep
the code-fence stripper. OpenAI's JSON mode returns objects, not arrays —
some prompts here expect arrays; wrap them in `{"items": [...]}` if needed.

---

## 8. File layout

```
live/
├── README.md           (this file)
├── pipeline.py         driver — runs all 6 loops
├── lib/
│   ├── llm.py          LLM client + json_call()
│   └── strings.py      normalize + fuzzy helpers
├── loops/
│   ├── loop1_structure.py
│   ├── loop2_cleaning.py
│   ├── loop2b_bridge.py
│   ├── loop3_linking.py     (includes via_menu recovery)
│   ├── loop3b_coherence.py  (cuisine signal + plausibility judge)
│   └── loop4_confidence.py
├── eval/
│   ├── gold.json       50-record hand-audited gold
│   ├── run_eval.py     P/R/F1 scorer
│   └── README.md       methodology
└── data/runs/{runId}/  per-loop artifacts
```

---

## 9. Output schema

```jsonc
{
  "recipes": [
    {
      "recipe_id": "RC0001",
      "recipe_name": "Chicken Shawarma",
      "restaurant": {
        "restaurant_id": "R0086", "name": "...", "cuisine_type": "...",
        "city": "...", "method": "exact_id|exact_name|fuzzy_high|llm_fuzzy|via_menu|suspicious:*",
        "confidence": 0.0,
        "coherence": {                 // present after Loop 3b runs
          "coherent": true,             // false if recipe vs cuisine mismatched
          "reason": "short clause",
          "judged_confidence": 0.0,    // only when coherent=false
          "applied": true               // only when coherent=false; true = confidence was floored
        }
      } | null,
      "menu": {
        "menu_id": "M0001", "menu_name": "...",
        "method": "exact_id|only_menu|llm_menu|default_menu",
        "confidence": 0.0
      } | null,
      "ingredients": [
        { "raw": "olive oil", "ingredient_id": "C0001", "name": "olive oil",
          "category": "Oil",
          "method": "exact|fuzzy_high|llm_disambiguated|unresolved",
          "confidence": 0.0 }
      ],
      "mapping_quality": {
        "overall_confidence": 0.0,
        "restaurant_method": "...", "menu_method": "...",
        "ingredients_resolved": 0, "ingredients_total": 0,
        "human_review": false
      }
    }
  ],
  "summary": {
    "total_recipes": 1000,
    "recipes_with_restaurant": 0, "recipes_with_menu": 0,
    "ingredients_resolved_total": 0, "ingredients_unresolved_total": 0,
    "average_confidence": 0.0,
    "records_in_human_review": 0,
    "run_id": "...", "completed_at": "..."
  }
}
```

---

## 10. Mapping to the assessment criteria

| Criterion | Where |
|---|---|
| Correctness | Gold eval F1 — restaurant 1.000, ingredients 0.988. See `live/eval/`. |
| Completeness | All 1000 recipes processed. Every field has a method tag and a confidence. |
| Confidence scoring | Per-field methods (`exact_id`, `fuzzy_high`, etc.) + per-record `overall_confidence`. Null fields drag the score down. Threshold 0.6 → `human_review`. |
| Code quality | Six small loops, one file each. One LLM client file. Tested via gold-set eval. |
| Approach | Deterministic owns bookkeeping. LLM owns judgment. LLM never types an ID. |

---

## 11. Bonus — an approach that was explored but not shipped

An early experiment let the LLM drive the whole pipeline OODA-style — the
model chose what to read, what to write, and when to stop, using its own
tool-calls to navigate Observe → Orient → Decide → Act.

What it produced:
- Average confidence 0.94 — looked great on paper.
- IDs were invented (`I_CAN_019`, `I0561_canonical`, mixed formats).
- Merge groups got polluted (a "tomato" cluster contained "all-purpose
  flour").
- 350 records reached confidence 1.0 while still missing fields, because
  the confidence formula averaged only over present fields.

Lesson: high LLM autonomy looks confident but hides bookkeeping failures.
The shipped pipeline keeps the LLM only for judgment calls and lets code
own every ID and every score. This bonus code is not part of the delivered
solution.
