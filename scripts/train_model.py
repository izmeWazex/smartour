"""
Train the Smartour intent model from a JSON or CSV file (offline).

Usage:
    python scripts/train_model.py path/to/training.json|csv
    python scripts/train_model.py --seed

JSON: a bare list of examples or {"examples": [...]}.
CSV: columns text,intent[,response], header optional.
On success the dataset is copied to backend/data/training_data.csv and the
model saved to backend/data/intent_model.joblib.
"""

import json
import sys
from pathlib import Path

# Windows consoles default to cp1252 and can't print non-ASCII (emoji, ₱) — force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Allow running as `python scripts/train_model.py` from the backend/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.csv_data import DATASET_CSV_PATH, read_examples_csv, write_examples_csv  # noqa: E402
from app.training.trainable_model import TrainingError, get_intent_model, MODEL_PATH  # noqa: E402
from app.training.seed_training_data import SEED_TRAINING_EXAMPLES  # noqa: E402


def _print_help() -> None:
    print(__doc__)
    print("Options:")
    print("  -h, --help    Show this help message")
    print("  --seed        Train from the built-in seed dataset (resets custom data)")
    print("\nExamples:")
    print("  python scripts/train_model.py my_dataset.csv")
    print("  python scripts/train_model.py training.json")
    print("  python scripts/train_model.py --seed")


def main() -> int:
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        _print_help()
        return 0 if ("-h" in args or "--help" in args) else 1

    if "--seed" in args:
        examples = SEED_TRAINING_EXAMPLES
        source = "seed"
        print(f"Using built-in seed data ({len(examples)} examples).")
        # Drop any custom dataset so the app falls back to seed on restart
        if DATASET_CSV_PATH.exists():
            DATASET_CSV_PATH.unlink()
    else:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}")
            return 1
        if path.suffix.lower() == ".csv":
            examples = read_examples_csv(path)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        source = "custom"
        print(f"Loaded {len(examples)} examples from {path}.")

    try:
        metrics = get_intent_model().fit(examples, source=source)
        if source == "custom":
            write_examples_csv(examples)
            print(f"Dataset copied to {DATASET_CSV_PATH}")
    except (TrainingError, ValueError) as exc:
        print(f"Training failed: {exc}")
        return 1

    model = get_intent_model()
    print("\nModel trained and saved to:")
    print(f"   {MODEL_PATH}")
    print(f"\n{model.example_count} examples | {len(model.intents)} intents")
    print(f"   accuracy: {metrics['accuracy']}")
    if metrics.get("cv_accuracy"):
        print(f"   cross-validation accuracy: {metrics['cv_accuracy']}")
    print("\nIntents:")
    for intent in model.intents:
        info = metrics["per_intent"][intent]
        extra = " (custom response)" if intent in model.custom_responses else ""
        print(f"   - {intent}: {info['support']} examples{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
