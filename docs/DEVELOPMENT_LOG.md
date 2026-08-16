# Smartour Backend — Development Log

Session log: what was built, problems encountered, fixes applied, and the
algorithms behind the chat system.

---

## Session 2026-08-16 — Trainable intent model + CSV datasets

### Goal

The Smartour chat system (`backend/app/ai/engine.py`) was **100% rule-based**
(regex intent detection + fuzzy spot-name matching). The goal was to add a
**real, trainable model** so the bot could be taught new phrasings and new
answers without editing code, and to store all datasets as **CSV files**
instead of Python source.

### What was built

| Piece | File(s) | What it does |
| ----- | ------- | ------------ |
| Trainable intent model | `app/ai/trainable_model.py` | TF-IDF + Logistic Regression classifier, `fit()` / `predict()` / `predict_top()`, joblib persistence, confidence gating |
| Default dataset | `app/ai/seed_training_data.csv` + `seed_training_data.py` | 98 labeled examples across 7 intents, loaded from CSV at startup |
| CSV helpers | `app/ai/csv_data.py` | Read/write labeled-example CSVs (`text,intent[,response]`), header optional, BOM-safe |
| Training API | `app/routers/ai.py` + `app/models/ai.py` | `GET /api/v1/ai/model`, `POST /api/v1/ai/predict`, `POST /api/v1/ai/train`, `POST /api/v1/ai/train/file` (JSON **or** CSV), `POST /api/v1/ai/model/reset` |
| Training CLI | `scripts/train_model.py` | Offline training from CSV/JSON, `--seed` reset, `--help` |
| Engine integration | `app/ai/engine.py` | Trained model takes priority (confidence ≥ 0.5); rule-based detection is the fallback; **clarifying questions** when both are unsure |
| Docs & deps | `README.md`, `requirements.txt` | Full training guide, dependency table; `numpy` declared, `httpx` as dev-only |

**Custom intents:** any intent outside the built-in set must carry a canned
`response` — this lets the bot answer brand-new topics with zero code changes
(e.g. train `kalesa_rental` with a reply and the bot instantly answers kalesa
questions).

**Dataset lifecycle:** training writes the current dataset to
`backend/data/training_data.csv`. Startup priority is
**dataset CSV → saved model → seed CSV**, so datasets can be edited by hand
and reloaded on restart. `POST /ai/model/reset` deletes the custom dataset and
restores the seed model.

### Problems encountered & fixes

| # | Problem | Fix |
| - | ------- | --- |
| 1 | **Timid model** — sklearn's default `C=1.0` L2 regularization on small data made *every* prediction weak (exact training matches scored only ~0.37 confidence), so the model almost never beat the 0.5 threshold and the feature felt broken. | Tuned `C=10.0`. Result: exact matches ~0.8–0.9, fuzzy matches ~0.5, out-of-scope stays low. |
| 2 | **Intent/handler name mismatch** — intent `fuel_cost` mapped to a method named `_handle_fuel`, which broke the new generic dispatch (`getattr(self, "_handle_fuel_cost")` → `None` → wrong fallback). | Renamed the handler to `_handle_fuel_cost` for consistency. |
| 3 | **Arity bug** — `_handle_greeting()` / `_handle_help()` took no `message` argument, but the generic router passes one. | Added `message: str = ""` to both signatures. |
| 4 | **Windows console crash** — CLI printed a checkmark emoji and crashed with `UnicodeEncodeError` (cp1252 console). | `sys.stdout.reconfigure(encoding="utf-8")` at the top of the CLI script. |
| 5 | **Missing runtime dep** — `python-multipart` was declared in `requirements.txt` but not installed in the venv, so the file-upload route failed at import. | `pip install python-multipart`. |
| 6 | **Pydantic dropped a field** — `dataset_csv` was added to the model status dict but not to the `ModelStatusResponse` schema, so it vanished from the API response. | Declared the field in the pydantic model. |
| 7 | **Category false-positive** — `_detect_category()` matched the substring `"eco"` *inside the word "recommend"*, so `"recommend places in vigan"` returned **Nature Spots**. Same class of bug could hit `"sea"` in `"search"` or `"top"` in `"stop"`. | Word-boundary regex matching for single-word keywords (`\b…\b`); multi-word phrases still substring-match. |
| 8 | **Training gap** — `"best to go on vigan"` scored 0.48 (below threshold) and the regex rules missed it → bot said "I don't know". | Added 14 Vigan/Ilocos Sur recommend phrasings (incl. the exact failing phrase, variations, Taglish) to the seed → 89 examples. The phrase now scores 0.86 and returns the must-see list. |
| 9 | **Clarification hint lost** — the new "ask a clarifying question" feature passed a hint (the model's below-threshold best guess) to `_handle_unknown`, but `unknown` is itself a registered handler, so the generic dispatch called it *without* the hint and the targeted nudges never fired. | Special-cased `intent == "unknown"` in `_route()` to forward the hint. |
| 10 | **Mid-word truncation in recommendation lists** — spot descriptions were sliced at 90 chars (`description[:90]`), producing broken cuts like `"relax..."` / `"white tiger..."`. | Added `_truncate()` — cuts at the last word boundary and appends `…` only when actually truncated. |
| 11 | **"How do I get to X?" misrouted** — `"paano pumunta sa university of northern phillipines?"` scored `recommend` at 0.585 (just over threshold) because the word `pumunta` appears in the recommend seed examples and there were zero "how to get to" examples, so the bot returned the must-see list instead of the UNP spot card. | Added 9 `describe` seed examples for `paano pumunta sa X` / `how to get to X` / `directions to X` phrasings → 98 examples. The query now scores `describe` 0.85 and returns the UNP card (which includes its location). |
| 12 | **Spot matching failures on long/typo'd phrases** — `get_close_matches` rejected a literal 14-char match (`"bantay church "` in a long query scored 0.48 < cutoff) and returned the wrong spot (`"paano pumunta sa bantay church"` → Santa Maria Church); `"how to get to unp"` matched nothing because `unp` alone has only 1 overlapping token. | Reworked `_fuzzy_match_spot` fallback order: spot-id word check (e.g. `unp`) → token-overlap ≥ 2 (most reliable for long phrases) → `get_close_matches` demoted to last resort (still catches short typos like `balwarte` → Baluarte). |
| 13 | **No directions-style answer for "how do I get to X?"** — these queries (now correctly routed to `describe`) returned the generic spot card instead of actually helping someone get there. | Added a directions response for describe queries phrased as getting-there questions (`_is_directions_query` detects English + Taglish patterns): replies with the spot's location, distance from Vigan city center (Calle Crisologo — table distances exist for all spots), and the nearest known landmark (via `_nearest_spot`). Plain info queries like "tell me about X" still get the normal spot card. |
| 14 | **Compound "distance + fuel" requests dropped the fuel part** — `"how far santa maria church from vigan calle crisologo and it fuel estimation?"` routed to `distance` (0.69) and the distance handler answered with only the distance; the fuel request was silently ignored. | Distance handler now detects a fuel ask in the same message (`_wants_fuel`: fuel keywords *or* a named vehicle) and appends a fuel-estimate block — a full estimate when a car type is given (`"…using a sedan"` → cost in ₱), otherwise a car-type prompt. Symmetrically, the fuel handler now shows the distance while asking for the vehicle when the route is already known. |
| 15 | **Knowledge base lived in code** — `knowledge_base.py` was a hard-coded Python dict of all tourist spots, distances, car types, and fuel data; editing it required touching code. | Moved the raw data to `app/ai/embedded_data.py` (fallback only) and added a **MySQL data layer**: `app/ai/db.py` (pymysql) loads all tables at startup into the same dict shapes; `knowledge_base.py` is now a facade that uses MySQL when `SMART_DB_*` env vars are set and falls back to embedded otherwise — so all existing `from app.ai.knowledge_base import ...` code is unchanged. `scripts/import_seed_to_db.py` creates the schema and seeds it from the embedded dataset; `/health` reports the active data source (`mysql` / `embedded`). |
| 16 | **No conversation memory** — the engine's `respond(message, history)` accepted the chat history but never used it, so every message was answered in a vacuum: after `"fuel cost from calle crisologo to santa maria using a sedan"`, the follow-up `"and the fuel cost?"` couldn't compute anything (no route/car in the message) and `"and to unp?"` fell to the clarifying prompt. | Added conversation context: `_extract_context()` scans the last few history messages (route, car type, last spot, last topic) and handlers reuse it — `_handle_fuel_cost`/`_handle_distance` fill missing endpoints via `_fill_route()` (full route reuse, `"from X"`/`"to X"` single-spot replacement), `_handle_describe` reuses the last spot, and short follow-ups that name a spot but carry no intent keywords are rerouted to the last topic (`fuel_cost`/`distance`/`describe`) instead of `unknown`. Follow-ups now answer from memory; fresh sessions without history behave exactly as before. Three sub-bugs found while testing: (a) the route regex only handled `from X to Y` order, so `estimated cost to calle from santa` extracted no route — added a reversed `to X from Y` regex in both handlers and in `_route_from_text()`; (b) assistant replies are full of spot lists and example phrases (`Available spots: …`, `…using a sedan`), which leaked into the remembered route/car — context now reads spots/car only from *user* messages, assistant replies contribute only the topic; (c) `_spots_in_order()` matched only exact full names, so partial mentions like `bantay church` (full name: Bantay Church & Bell Tower) weren't found — added word-boundary first-word matching with exact-match priority at equal positions.
| 17 | **Bare vehicle replies were forgotten** — after the bot asked for a vehicle (`"fuel cot calle crisologo to bantay church"` → "Tell me your vehicle type"), answering just `"motorcycle"` got the recommend clarify prompt ("looking for places to visit") instead of the fuel estimate: the model scores bare car words as `recommend` at ~0.26 (below threshold, so it became the clarify hint), and the context reroute required a spot in the message, which a bare car reply has none of. | Added `_is_bare_car_reply()` — true only when the message is essentially just a vehicle word (`motorcycle`, `a sedan`, `suv po`, `i have a pickup`), with word-boundary alias stripping so `motor` inside `motorcycle` or `city` in `vigan city` don't confuse it. `_topic_of_reply()` now also marks replies containing "vehicle type" (the bot's own vehicle prompts) as `fuel_cost`. In `respond()`, an `unknown`/`recommend`/`describe` intent that is a bare car reply completes the pending fuel estimate from the remembered route. Two sub-bugs found while testing: (a) phrases like "what about a suv?" hit the `about` keyword and routed to `describe` first — added `describe` to the override set; (b) the context scan window (6 messages) was too shallow, so after a few vehicle-switch exchanges the original route-setting message fell out of memory and "and the distance?" lost the route — widened the window to 10 messages (~5 exchanges), with early-stop once route+car+topic are all found. Verified: transcript flow now returns the full Calle → Bantay estimate with Motorcycle, vehicle switches ("and with a van?", "what about a suv?") re-estimate, distance follow-ups keep the route, and real car-containing requests ("recommend motorcycle spots") are never hijacked. |
| 18 | **Bot replies full of emojis and over-commented code** — every reply was decorated with map/compass/fuel/fuel-pump emojis and smileys, and the code carried decorative comment dividers and verbose docstrings. | Removed all emojis from bot replies, code comments, README, and this log (reply headers now plain text like `Distance Estimate` / `Trip:`); trimmed comments across every module to minimal — removed decorative dividers and redundant docstrings, kept only comments that explain non-obvious logic. Engine behavior and reply content otherwise unchanged (verified by re-running the transcript flow and core regressions). |

### Feature deep-dive — directions responses ("paano pumunta sa…?")

**Problem**

A user asked `"paano pumunta sa university of northern phillipines?"` (how to
get to UNP) and the bot replied with the generic **Top Must-See Spots** list
instead of anything about UNP. Three separate defects stacked up to produce
that answer.

**Diagnosis (three layers)**

1. **Intent misrouting.** The model scored the query `recommend` at **0.585**
   — just over the 0.5 confidence threshold. Why: the word `pumunta` appears
   in the recommend seed examples (`"saan maganda pumunta"`), and the training
   data contained **zero** "how to get to X" examples, so the model had
   nothing to map the phrase to `describe`.
2. **Fragile spot matching.** Even when describe wins, `_fuzzy_match_spot`
   failed on long, typo'd phrases: `get_close_matches` (difflib) rejected a
   literal 14-character match (`"bantay church "` scored 0.48 < its 0.5
   cutoff) and instead returned **Santa Maria Church** for
   `"paano pumunta sa bantay church"`; and `"how to get to unp"` matched
   nothing at all because `unp` is only a single overlapping token.
3. **No directions capability.** Even with the right spot, the `describe`
   handler only knew how to print the static spot card — no getting-there info.

**Approach (three-part fix)**

1. **Train the model** — added 9 `describe` seed examples covering the
   getting-there pattern in English and Taglish (`paano pumunta sa unp`,
   `how to get to unp`, `paano makarating sa plaza salcedo`, `directions to
   sta maria church`, …). Seed grew 89 → **98 examples**; the failing query
   now scores `describe` at 0.85.
2. **Fix spot matching** — reworked `_fuzzy_match_spot` fallback order:
   spot-id word check (e.g. `unp`) → token-overlap ≥ 2 (most reliable on long
   phrases) → `get_close_matches` demoted to last resort (still useful for
   short typos like `balwarte` → Baluarte).
3. **Add a directions reply** — in the `describe` handler, detect
   getting-there phrasing (`_is_directions_query`) and return a dedicated
   card: location, distance from Vigan city center, nearest landmark, best
   time. Plain info queries keep the regular spot card.

**Design decisions**

- **City-center reference = Calle Crisologo.** It is the iconic heart of Vigan
  *and* the `DISTANCES_KM` table happens to have entries from it to **all 12
  spots**, so the headline distance is never an estimate.
- **Nearest landmark computed on the fly** (`_nearest_spot`): scans the other
  11 spots using the distance table, falling back to Haversine (×1.3 road
  factor). Values from Haversine are marked with `~`. Cheap: ~11 lookups per
  reply.
- **Detection is phrase-based, not intent-based.** No new intent was added —
  getting-there questions stay `describe` (the model layer decides the *kind*
  of question), and the directions card is a *handler variation* (the answer
  layer). This keeps the two-layer architecture intact.
- **English + Taglish patterns**: `how (do i|can i|to) get/go/reach`,
  `directions to`, `way to`, `paano (pumunta|makarating|makapunta|…)`,
  `papunta`.

**Verification**

- 6 directions phrasings (EN + Taglish) → directions card with correct
  distances (UNP 2.5 km, Bantay 2.8 km, Pinsal Falls 28 km, Sta Maria 55 km).
- 3 plain describe queries (`tell me about X`, `what is X`, `describe X`) →
  regular spot card (no false positives).
- Regression: distance, fuel, recommend, greeting intents unchanged.
- Live API check: `POST /api/v1/chat/message` with the original query returns
  the UNP directions card (2.5 km from city center, nearest landmark Burnay
  Pottery ~1.7 km).

**Known limitation** — "nearest landmark" figures outside the road table are
straight-line estimates; extend `DISTANCES_KM` in `knowledge_base.py` for
precise road distances.

### Algorithms used

**Intent classification (the trainable model)**

- **TF-IDF vectorizer** (`TfidfVectorizer`, sklearn): character-level token
  unigrams **and** bigrams (`ngram_range=(1,2)`), `sublinear_tf=True` (1 + log
  tf), lowercase, Unicode accent stripping. Handles English + Taglish.
- **Multinomial Logistic Regression** (`LogisticRegression`, solver lbfgs):
  7 classes (the intents), `C=10.0`, `max_iter=2000`. Softmax probabilities
  from `predict_proba` give the confidence score.
- **Confidence gating**: argmax probability ≥ **0.5** → model decides;
  otherwise fall back to the rule-based detector.
- **Evaluation**: train accuracy + `StratifiedKFold` cross-validation accuracy
  (when every class has ≥ 2 examples). Reported via `/ai/model`.

**Rule-based fallback layer** (unchanged logic, still the safety net)

- Regex `INTENT_PATTERNS` per intent (English + Tagalog/Taglish patterns).
- Fuzzy spot matching via `difflib.get_close_matches` + name/location
  substring checks.
- Category detection for recommendations — now word-boundary matched (fix #7).
- Haversine straight-line distance (×1.3 road factor) when a spot pair isn't
  in the distance table; fuel cost = distance × consumption / price.

**Clarifying-question fallback** (when model AND rules are unsure)

1. Spot reference detection (`_find_spot_reference`): fuzzy match → spot-name
   first-word word-boundary match → location mention. If found, ask what the
   user wants to know about that spot.
2. Hint from `predict_top()` (top-1 intent even below threshold) → targeted
   nudge per intent (`_CLARIFY_HINTS`).
3. General "what would you like to know?" menu.

**Persistence**

- Model → `backend/data/intent_model.joblib` (joblib).
- Dataset → `backend/data/training_data.csv` (CSV, human-editable).
- Startup priority: dataset CSV → joblib → seed CSV.

### How to verify (what was run)

```bash
cd backend
.venv/Scripts/python scripts/train_model.py --seed          # trains seed, prints metrics
.venv/Scripts/python -m uvicorn app.main:app --reload       # run the API

# Smoke tests (FastAPI TestClient, run ad-hoc):
#   GET  /api/v1/ai/model            → source: seed | examples: 98
#   POST /api/v1/ai/predict          → {"text": "best to go on vigan"} → recommend 0.86
#   POST /api/v1/ai/train/file       → upload CSV → dataset saved to data/training_data.csv
#   POST /api/v1/chat/message        → "best to go on vigan" → must-see list
#   POST /api/v1/ai/model/reset      → back to seed, deletes training_data.csv
```

### Possible next steps

- Feedback loop: thumbs-down in chat → collect misclassified queries into a
  labeling CSV.
- Data-labeling web UI served by FastAPI.
- More seed coverage for `describe` / `distance` / `fuel_cost` phrasings.
- Fine-tune a transformer (`sentence-transformers` / `torch` are installed)
  behind the same training API for higher accuracy.
