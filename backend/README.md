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

`category` is optional. When omitted, **every** spot is ranked purely by content match
(TF-IDF cosine similarity) — the bot recommends real spots, not spots of a detected
category. Category keywords found in the query are reported as `detected_categories`
and used only as a fallback when the query shares no words with any description
(e.g. Tagalog-only phrasing). Pass `category` explicitly to hard-filter to one category.

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest tests -q
```

The recommender tests run against the real algorithm on a tiny CSV (built once per
test module), and the API tests swap in a stub recommender — so the suite is fast,
deterministic, and needs no downloads.
