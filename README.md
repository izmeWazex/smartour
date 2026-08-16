# Smartour Backend

FastAPI backend for the Smartour Ilocos Sur tourism chatbot. Fully offline —
no external AI API required.

> **Development log** — what was built, problems encountered, fixes, and the
> algorithms used: [`docs/DEVELOPMENT_LOG.md`](docs/DEVELOPMENT_LOG.md)

## Dependencies

Everything needed to run the API is declared in [`requirements.txt`](requirements.txt):

| Package             | Used for                                                                       |
| ------------------- | ------------------------------------------------------------------------------ |
| `fastapi`           | Web framework — all API routes, request/response models                        |
| `uvicorn[standard]` | ASGI server that runs the app (`app/main.py` entry point)                      |
| `pydantic`          | Request/response schemas and validation (`app/models/`)                        |
| `python-multipart`  | File uploads — required by `POST /api/v1/ai/train/file`                        |
| `scikit-learn`      | The trainable intent model: `TfidfVectorizer` + `LogisticRegression`, metrics  |
| `joblib`            | Persists the trained model to `data/intent_model.joblib`                       |
| `numpy`             | Probability/array math inside the model's `predict()`                          |
| `pymysql`           | Optional — store the knowledge base (spots, distances, fuel data) in MySQL     |

Notes:

- Everything else the code imports (`csv`, `json`, `re`, `math`, `logging`, …)
  is Python's standard library — no extra installs.
- `httpx` is only needed for development-time API checks via
  `fastapi.testclient`; it is commented out in `requirements.txt` since the
  server itself doesn't use it.
- The venv also happens to contain `torch` / `transformers` /
  `sentence-transformers`, but the app does **not** use them — the intent model
  runs on scikit-learn so it stays light and fully offline.

```bash
cd backend
.venv/Scripts/python -m pip install -r requirements.txt
```

## Trainable Intent Model

The chat engine ships with rule-based intent detection (`app/ai/engine.py`),
plus a **trainable ML intent classifier** (TF-IDF + Logistic Regression) that
you can train with your own labeled examples. When the trained model is
confident (≥ 50%) it takes priority; otherwise the engine falls back to the
rules.

### Training data format

Datasets live in **CSV files** (the default dataset is
`app/ai/seed_training_data.csv`). Both JSON and CSV are accepted for uploads
and the CLI; CSV is the recommended format:

```csv
text,intent,response
Where can I rent a kalesa?,kalesa_rental,Kalesa rides start at Plaza Salcedo, ~₱200.
How much does a kalesa ride cost?,kalesa_rental,Kalesa rides start at Plaza Salcedo, ~₱200.
Recommend food spots,recommend,
Tell me about Calle Crisologo,describe,
```

- Columns: `text,intent[,response]`, UTF-8, header row optional.
- **Built-in intents**: `greeting`, `help`, `fuel_cost`, `recommend`,
  `describe`, `distance`, `unknown`.
- Any **new intent** must include a canned `response` (that's how you teach the
  bot new answers without touching code).
- Use ≥ 2 examples per intent (≥ 2 intents total); 5–10 per intent is better.
- Training **replaces** the previous model and copies the dataset to
  `backend/data/training_data.csv`, which is reloaded automatically on server
  restart — so you can edit the dataset by hand and restart to apply it.

### API endpoints (all under `/api/v1`)

| Method | Path             | Description                                   |
| ------ | ---------------- | --------------------------------------------- |
| GET    | `/ai/model`      | Model status, intents, metrics                |
| POST   | `/ai/predict`    | Debug: intent + confidence scores for a text  |
| POST   | `/ai/train`      | Train with JSON examples (body above)         |
| POST   | `/ai/train/file` | Train from an uploaded `.json` or `.csv` file |
| POST   | `/ai/model/reset`| Retrain from built-in seed data               |

Examples:

```bash
# Status
curl http://localhost:8000/api/v1/ai/model

# Train from a CSV dataset (recommended)
curl -X POST http://localhost:8000/api/v1/ai/train/file \
  -F "file=@training_data.csv"

# Train from JSON
curl -X POST http://localhost:8000/api/v1/ai/train \
  -H "Content-Type: application/json" \
  -d @training.json

# Reset
curl -X POST http://localhost:8000/api/v1/ai/model/reset
```

### Offline training (CLI)

```bash
cd backend
.venv/Scripts/python scripts/train_model.py path/to/training.csv   # CSV or JSON
.venv/Scripts/python scripts/train_model.py --seed                  # back to seed data
```

The dataset is copied to `backend/data/training_data.csv` and the model to
`backend/data/intent_model.joblib`; the server reloads both on startup.
Delete `data/training_data.csv` (or hit `/ai/model/reset`) to restore the
default seed model.

### How the engine uses it

`app/ai/engine.py` → `SmartourAI.respond()` asks the trained model for an
intent first. If it's confident (≥ 50%), the request is routed to the matching
handler (custom intents return their canned `response`). Otherwise the original
regex intent detection handles it, so the bot keeps working even with an empty
model.

**When the bot is unsure** (model below threshold *and* rules miss), it asks a
clarifying question instead of saying "I don't know":

- a known spot is mentioned → *"I think you're asking about Baluarte Zoo —
  what would you like to know?"* with example prompts;
- the model almost guessed an intent → a targeted nudge
  (e.g. *"sounds like you want a fuel estimate — tell me your route and car"*);
- otherwise → a short menu of what it can do.

**Directions:** describe queries phrased as *"how do I get to X?"* (English or
Taglish — `paano pumunta sa X`, `paano makarating sa X`) get a directions
reply instead of the plain spot card: the spot's location, its distance from
Vigan city center (Calle Crisologo), and its nearest known landmark.

**Combined requests:** distance questions that also ask for fuel (or name a
vehicle, e.g. `how far is vigan to santa maria using a sedan`) automatically
include the fuel estimate too.

**Conversation memory:** the bot remembers the last few messages of the
session, so follow-ups can skip repeating context. It keeps the last route,
car type, spot, and topic:

```
You:  fuel cost from calle crisologo to santa maria church using a sedan
Bot:  …₱289.85…
You:  and the fuel cost?          → same route + car → same estimate
You:  and the distance?           → distance for the same route
You:  and to unp?                 → swaps the destination → Calle → UNP
You:  tell me about baluarte
Bot:  …Baluarte Zoo card…
You:  what about its history?     → reuses the last spot → Baluarte card

You:  fuel cot calle crisologo to bantay church
Bot:  …Distance… Tell me your vehicle type…
You:  motorcycle                  → completes the estimate → Calle → Bantay, motorcycle
```

This works because the chat API returns a `session_id` — keep sending it with
follow-up messages (`POST /api/v1/chat/message` with `"session_id": …`). A
fresh session (or a first message) has no memory and behaves exactly as
before. Short follow-ups that name a spot but carry no intent keywords
("and to unp?") reuse the previous topic instead of falling back to a
clarifying question. Answering the vehicle prompt with just a car word
("motorcycle", "a sedan", "suv po") also completes the pending fuel estimate
from the remembered route.

### Training guide — how to make the bot smarter

The model only knows the examples you give it. If the bot doesn't answer a
question, the phrase isn't covered (or is too unlike anything it saw). The fix
is always: **add examples, retrain, retest.**

**The loop** (keep one master CSV that grows over time):

```bash
cd backend
# 1. Start from the seed dataset
cp app/ai/seed_training_data.csv my_dataset.csv

# 2. Add rows for the questions the bot got wrong — use the exact words
#    people type, plus a few variations:
#    text,intent,response
#    best to go on vigan,recommend,
#    where should i go in vigan,recommend,
#    saan maganda sa vigan,recommend,

# 3. Train (this replaces the old model with your CSV — which is why you
#    keep building one growing master file)
.venv/Scripts/python scripts/train_model.py my_dataset.csv

# 4. Check what the model now thinks about the exact phrase that failed
curl -X POST http://localhost:8000/api/v1/ai/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "best to go on vigan"}'

# 5. Chat-test it
curl -X POST http://localhost:8000/api/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "best to go on vigan"}'
```

**Best practices**

- **5–10+ examples per intent**, not 1–2. The model generalizes from them.
- **Cover the ways people actually phrase things**: typos, Taglish, short
  forms (`saan maganda`, `vigan food`, `magkano gas`).
- **Add a few `unknown` examples** too — that's how the bot learns to say
  "I don't know" for off-topic stuff.
- **Never train with a tiny file that drops your old data.** Train with your
  whole master dataset (seed + additions). Training replaces, it doesn't merge.
- **New topic?** Give it a new intent name + a canned `response`:
  `Where can I rent a kalesa?,kalesa_rental,Kalesa rides start at Plaza Salcedo.`
- **Retest after every change** using `/ai/predict` and the chat endpoint.
- Use `/ai/model` to see your dataset size and per-intent example counts.

**Why the bot still gets things wrong** — two common traps:

1. *It never saw the phrase.* The model scores it below the 50% confidence
   cut-off and falls back to the regex rules, which have their own gaps —
   in that case the bot now asks a clarifying question. Fix: add the phrase +
   variations to your dataset so it stops needing to ask.
2. *The intent is right but the answer comes from the handler.* The model only
   decides *what kind* of question it is (recommend / describe / fuel_cost…);
   the actual reply text comes from the hard-coded handlers in `engine.py`.
   To change answers, edit the handlers or use a custom intent with a canned
   `response`.

## Knowledge base: MySQL (optional)

The tourist-spot data (spots, distances, car types, fuel prices,
recommendations) can live in **MySQL** instead of code. The engine loads the
tables at startup; if MySQL isn't configured or reachable it automatically
falls back to the embedded dataset, so the app always runs.

**1. Install & start MySQL**, then create a database:

```sql
CREATE DATABASE smartour CHARACTER SET utf8mb4;
```

**2. Create the tables and seed them from the embedded dataset:**

```bash
cd backend
SMART_DB_HOST=localhost SMART_DB_USER=root SMART_DB_PASSWORD=yourpass \
  .venv/Scripts/python scripts/import_seed_to_db.py
```

**3. Start the server with the same env vars** (Windows PowerShell: prefix
`$env:SMART_DB_HOST="localhost"; ...`):

```bash
SMART_DB_HOST=localhost SMART_DB_USER=root SMART_DB_PASSWORD=yourpass \
  .venv/Scripts/python -m uvicorn app.main:app --reload
```

| Variable             | Default     | Meaning                                  |
| -------------------- | ----------- | ---------------------------------------- |
| `SMART_DB_HOST`      | *(unset)*   | Set it to enable MySQL (e.g. `localhost`) |
| `SMART_DB_PORT`      | `3306`      | MySQL port                               |
| `SMART_DB_USER`      | `root`      | DB user                                  |
| `SMART_DB_PASSWORD`  | *(empty)*   | DB password                              |
| `SMART_DB_NAME`      | `smartour`  | Database name                            |

Notes:

- The knowledge base is loaded **once at startup** — edit the DB and restart
  the server to apply changes.
- If MySQL is unreachable (or `SMART_DB_HOST` is unset), the embedded data in
  `app/ai/embedded_data.py` is used and a warning is logged.
- `GET /health` includes the current data source
  (`mysql` or `embedded`) so you can confirm which one is active.

## Run the server

```bash
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload
# or: .venv/Scripts/python app/main.py
```

Interactive API docs: http://localhost:8000/docs
# smartour
