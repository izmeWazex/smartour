# Smartour Backend

Tourist-spot chatbot backend for Vigan / Ilocos: keyword intent classification plus a
TF-IDF + cosine similarity recommender over `data/raw/tourist_spots.csv`.

## How the recommender works

The recommender (`src/recommender/recommend_tfidf.py`) is a classical ML pipeline built
on **pandas** and **scikit-learn** — and it uses **no pre-trained AI model**. The only
"model" is a `TfidfVectorizer` fitted on this project's own spot descriptions, so every
weight is learned from your data and can be explained line by line:

1. **TF** — term frequency: how often a word appears in a spot's description.
2. **IDF** — inverse document frequency: `idf(t) = ln((1 + N) / (1 + df(t))) + 1`, so
   rare words ("bagnet") weigh more than common ones ("the").
3. **TF-IDF** — each description becomes a vector of TF × IDF word weights.
4. **Cosine similarity** — `cos(q, d) = (q · d) / (|q| × |d|)` between the query vector
   and each spot vector; the highest-scoring spots are returned.

`sentence-transformers` is deliberately **not** a dependency: that library exists only to
load pre-trained neural models, which is exactly what this project avoids. scikit-learn
ships algorithm implementations, not learned weights, so nothing is downloaded and the
API works offline.

Because the algorithm runs on word weights, the bot understands queries like
`gusto ko kumain ng bagnet` even when the CSV never contains those exact words.

## How intent classification works

`src/intent/classify.py` is a hybrid classifier that also uses no pre-trained model:

1. **Keyword pass** — the message is scanned for `CATEGORY_KEYWORDS` substrings
   (e.g. "bagnet" → food, "museum" → historical); every category that hits is
   returned.
2. **TF-IDF fallback** — only when no keyword matches, the message is vectorized
   with a word-level `TfidfVectorizer(stop_words="english")` fitted on the spot
   descriptions grouped by category, and scored against each category's
   "document" with cosine similarity. The best category is returned when it
   clears `SIMILARITY_THRESHOLD`; random text scores ~0 and returns nothing.

Because the fallback learns from the real spot descriptions, it catches wording
the keyword lists miss — "water" and "clean" point to nature, "swimming" to
beach — without any pre-trained model. Stop words are stripped so common words
can't dominate the four tiny category documents.

## Setup

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # macOS/Linux
```

## Run the API

```bash
cd backend
.venv/Scripts/python -m uvicorn app:app --reload
```

The recommender reads the CSV and fits the TF-IDF vectorizer on it in memory at
startup — nothing is downloaded and the API works offline.

## Endpoints

- `GET /health` — liveness check
- `POST /recommend` — returns the top matching spots

```json
{
  "query": "gusto ko kumain ng bagnet",
  "top_n": 3,
  "category": "food"
}
```

Response:

```json
{
  "query": "gusto ko kumain ng bagnet",
  "detected_categories": ["food"],
  "results": [
    {"spot_id": 19, "name": "Bantay Camarin Bagnet House", "category": "food", "city": "Bantay", "similarity": 0.2275},
    {"spot_id": 17, "name": "Grandpa's Inn Restaurant", "category": "food", "city": "Vigan", "similarity": 0.2134},
    {"spot_id": 15, "name": "Lingsat Public Market Food Row", "category": "food", "city": "Vigan", "similarity": 0.2071}
  ]
}
```

`category` is optional. When omitted, any categories detected in the chat message
(see "How intent classification works") hard-filter the candidates, and the top
spots **within those categories** are then ranked by content match (TF-IDF cosine
similarity). Design note: the filter only decides *which* spots are eligible, and
content match still decides *which one* wins — so `gusto ko kumain ng bagnet`
returns the bagnet-specific eatery (Bantay Camarin Bagnet House), not just any
food spot. When neither an explicit `category` nor a detected category exists,
every spot is ranked purely by content match. If a filter matches no spots — a
client-sent `category` name the CSV doesn't use, or chat-detected categories that
drifted from the CSV — the API retries with the chat-detected categories and then
with no filter, so it never returns an empty list while spots exist. Pass
`category` explicitly to hard-filter to a single category (it overrides the
detected ones).

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest tests -q
```

The recommender tests run against the real algorithm on a tiny CSV (built once per
test module), and the API tests swap in a stub recommender — so the suite is fast,
deterministic, and needs no downloads.
