"""
Trainable Intent Model
----------------------
TF-IDF + Logistic Regression intent classifier (scikit-learn), fully offline.
Custom intents can carry a canned ``response`` so the bot can answer questions
it was never hard-coded to handle. Persisted to disk with joblib.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.training.csv_data import DATASET_CSV_PATH

logger = logging.getLogger("smartour.model")

# Paths & constants

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "data"
MODEL_PATH = DATA_DIR / "intent_model.joblib"

# Below this confidence the engine falls back to rule-based detection
CONFIDENCE_THRESHOLD = 0.5

# Intents with built-in handlers; any other trained intent needs a canned response
BUILTIN_INTENTS = {
    "greeting",
    "help",
    "fuel_cost",
    "recommend",
    "describe",
    "distance",
    "unknown",
}

MODEL_BACKEND = "tfidf + logistic_regression"


class TrainingError(ValueError):
    """Raised when training data is invalid; surfaces as a 400 to the caller."""


class TrainableIntentModel:
    """A persisted TF-IDF + Logistic Regression intent classifier."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        self.path = path
        self.pipeline: Optional[Pipeline] = None
        self.intents: List[str] = []
        self.example_count: int = 0
        self.metrics: Dict = {}
        self.custom_responses: Dict[str, str] = {}
        self.trained_at: Optional[datetime] = None
        self.source: Optional[str] = None  # "seed" or "custom"

    # Training

    def fit(self, examples: List[dict], source: str = "custom") -> Dict:
        """Train (or retrain) on labeled examples.
        Each: {"text", "intent", "response"?}; ≥2 distinct intents required."""
        validated = self._validate(examples)

        texts = [ex["text"] for ex in validated]
        labels = [ex["intent"] for ex in validated]

        # Deduplicate identical (text, intent) pairs
        seen, unique = set(), []
        for ex in validated:
            key = (ex["text"].strip().lower(), ex["intent"])
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        if len(unique) < 2:
            raise TrainingError("Need at least two examples to train.")

        texts = [ex["text"] for ex in unique]
        labels = [ex["intent"] for ex in unique]

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                # C=10 (vs sklearn's default 1.0): with small labeled sets, the
                # default regularization makes every prediction timid (~0.3-0.4),
                # so exact matches rarely beat the confidence threshold. C=10
                # gives well-calibrated confidences: exact matches ~0.8-0.9,
                # fuzzy matches ~0.5, out-of-scope stays low.
                ("clf", LogisticRegression(max_iter=2000, C=10.0)),
            ]
        )
        pipeline.fit(texts, labels)

        train_acc = accuracy_score(labels, pipeline.predict(texts))

        metrics: Dict = {
            "accuracy": round(float(train_acc), 4),
            "cv_accuracy": None,
            "per_intent": {},
        }
        for intent in sorted(set(labels)):
            metrics["per_intent"][intent] = {
                "support": labels.count(intent),
                "examples": [t for t, l in zip(texts, labels) if l == intent][:3],
            }

        # Cross-validation only when every class has ≥ 2 examples (StratifiedKFold)
        min_support = min(labels.count(i) for i in set(labels))
        if min_support >= 2 and len(set(labels)) >= 2 and len(texts) >= 6:
            try:
                skf = StratifiedKFold(n_splits=min(3, min_support), shuffle=True, random_state=42)
                scores = cross_val_score(pipeline, texts, labels, cv=skf, scoring="accuracy")
                metrics["cv_accuracy"] = round(float(scores.mean()), 4)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Cross-validation skipped: %s", exc)

        self.pipeline = pipeline
        self.intents = sorted(set(labels))
        self.example_count = len(unique)
        self.metrics = metrics
        self.custom_responses = {
            ex["intent"]: ex["response"]
            for ex in unique
            if ex.get("response")
        }
        self.trained_at = datetime.now(timezone.utc)
        self.source = source

        self.save()
        logger.info(
            "Trained intent model: %d examples, %d intents, acc=%.3f",
            self.example_count,
            len(self.intents),
            metrics["accuracy"],
        )
        return metrics

    def _validate(self, examples: List[dict]) -> List[dict]:
        if not examples or not isinstance(examples, list):
            raise TrainingError("'examples' must be a non-empty list.")

        validated: List[dict] = []
        intents_seen = set()
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                raise TrainingError(f"Example #{i + 1} must be an object with 'text' and 'intent'.")
            text = str(ex.get("text", "")).strip()
            intent = str(ex.get("intent", "")).strip()
            if not text:
                raise TrainingError(f"Example #{i + 1} has an empty 'text'.")
            if not intent:
                raise TrainingError(f"Example #{i + 1} has an empty 'intent'.")
            response = ex.get("response")
            if response is not None:
                response = str(response).strip()
            if intent not in BUILTIN_INTENTS and not response:
                raise TrainingError(
                    f"Intent '{intent}' (example #{i + 1}) is not a built-in intent and has "
                    f"no canned 'response'. Provide a 'response' or use a built-in intent: "
                    f"{sorted(BUILTIN_INTENTS)}."
                )
            intents_seen.add(intent)
            validated.append({"text": text, "intent": intent, "response": response})

        if len(intents_seen) < 2:
            raise TrainingError(
                "Need at least two distinct intents to train a classifier. "
                f"Found only: {sorted(intents_seen)}"
            )
        return validated

    # Prediction

    def predict(self, text: str) -> Tuple[Optional[str], float]:
        """Return (intent, confidence); (None, confidence) when below threshold."""
        if not self.is_trained:
            return None, 0.0
        probs = self.pipeline.predict_proba([text])[0]  # type: ignore[union-attr]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        if confidence < CONFIDENCE_THRESHOLD:
            return None, confidence
        return self.intents[best], confidence

    def predict_top(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Top-k (intent, confidence) predictions, ignoring the threshold."""
        if not self.is_trained:
            return []
        probs = self.pipeline.predict_proba([text])[0]  # type: ignore[union-attr]
        ranked = sorted(zip(self.intents, probs), key=lambda x: -x[1])[:k]
        return [(intent, float(conf)) for intent, conf in ranked]

    # Persistence

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.path)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            loaded = joblib.load(self.path)
            if isinstance(loaded, TrainableIntentModel) and loaded.is_trained:
                self.__dict__.update(loaded.__dict__)
                logger.info("Loaded intent model from %s", self.path)
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load %s (%s); will retrain from seed.", self.path, exc)
        return False

    def status(self) -> Dict:
        return {
            "is_trained": self.is_trained,
            "backend": MODEL_BACKEND,
            "source": self.source,
            "intents": self.intents,
            "example_count": self.example_count,
            "metrics": self.metrics,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "model_path": str(self.path),
            "dataset_csv": str(DATASET_CSV_PATH),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        }


# Singleton

_model: Optional[TrainableIntentModel] = None


def get_intent_model() -> TrainableIntentModel:
    """App-wide model; trained from the dataset CSV, saved joblib, or seed."""
    global _model
    if _model is None:
        from app.training.csv_data import DATASET_CSV_PATH, read_examples_csv
        from app.training.seed_training_data import SEED_TRAINING_EXAMPLES

        _model = TrainableIntentModel()
        if DATASET_CSV_PATH.exists():
            examples = read_examples_csv(DATASET_CSV_PATH)
            logger.info("Found dataset %s — training from it (%d examples).",
                        DATASET_CSV_PATH, len(examples))
            _model.fit(examples, source="custom")
        elif not _model.load():
            logger.info("No saved model found — training from seed data (%d examples).",
                        len(SEED_TRAINING_EXAMPLES))
            _model.fit(SEED_TRAINING_EXAMPLES, source="seed")
    return _model
