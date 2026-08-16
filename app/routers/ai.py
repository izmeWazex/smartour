import json
from io import StringIO

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ai.csv_data import DATASET_CSV_PATH, read_examples_csv, write_examples_csv
from app.ai.trainable_model import CONFIDENCE_THRESHOLD, TrainingError, get_intent_model
from app.ai.seed_training_data import SEED_TRAINING_EXAMPLES
from app.models.ai import (
    IntentScore,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainResponse,
    TrainingExample,
)

router = APIRouter(prefix="/ai", tags=["AI Training"])


def _train_and_respond(examples: list, source: str, persist_dataset: bool = True) -> TrainResponse:
    try:
        metrics = get_intent_model().fit(examples, source=source)
        # Keep the dataset as an editable CSV so it reloads on restart
        if persist_dataset:
            write_examples_csv(examples)
    except TrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    model = get_intent_model()
    return TrainResponse(
        status="trained",
        message=f"Model trained on {model.example_count} examples across "
                f"{len(model.intents)} intents.",
        intents=model.intents,
        example_count=model.example_count,
        metrics=metrics,
        trained_at=model.trained_at,
    )


@router.get(
    "/model",
    response_model=ModelStatusResponse,
    summary="Get the current intent model status",
)
def model_status() -> ModelStatusResponse:
    return ModelStatusResponse(**get_intent_model().status())


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train the intent model with labeled examples",
)
def train(body: TrainRequest) -> TrainResponse:
    """(Re)train the intent classifier with labeled examples.
    JSON body: {"examples": [{"text", "intent", "response"?}, ...]}
    Intents outside the built-in set must include a canned `response`."""
    examples = [ex.model_dump() for ex in body.examples]
    return _train_and_respond(examples, source="custom")


@router.post(
    "/train/file",
    response_model=TrainResponse,
    summary="Train the intent model from an uploaded JSON or CSV file",
)
async def train_from_file(file: UploadFile = File(..., description="JSON or CSV file of training examples")) -> TrainResponse:
    """Train from an uploaded JSON or CSV file.
    JSON: a list of examples or {"examples": [...]}.
    CSV: columns text,intent[,response], header optional.
    On success the dataset is saved to data/training_data.csv."""
    raw = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith(".csv"):
            # CSV: columns text,intent[,response] (header optional)
            examples = read_examples_csv(StringIO(raw.decode("utf-8-sig")))
        else:
            # JSON: list of examples or {"examples": [...]}
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("examples", [])
            if not isinstance(payload, list):
                raise ValueError("JSON must be a list of examples or an object with an 'examples' key.")
            examples = [TrainingExample(**ex).model_dump() for ex in payload]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse '{file.filename}': {exc}. Use JSON or CSV.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training file: {exc}",
        )
    return _train_and_respond(examples, source="custom")


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Debug: show the intent scores for a message",
)
def predict(body: PredictRequest) -> PredictResponse:
    """Show the model's top intent scores for a message (debugging aid)."""
    model = get_intent_model()
    if not model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not trained yet.",
        )
    ranked = model.predict_top(body.text, body.top_k)
    intent, confidence = model.predict(body.text)
    return PredictResponse(
        text=body.text,
        intent=intent,
        confidence=round(confidence, 4),
        threshold=CONFIDENCE_THRESHOLD,
        top_intents=[
            IntentScore(intent=i, confidence=round(c, 4)) for i, c in ranked
        ],
    )


@router.post(
    "/model/reset",
    response_model=TrainResponse,
    summary="Reset the model back to the built-in seed training data",
)
def reset_model() -> TrainResponse:
    # Drop the custom dataset so the next startup falls back to seed as well
    if DATASET_CSV_PATH.exists():
        DATASET_CSV_PATH.unlink()
    return _train_and_respond(SEED_TRAINING_EXAMPLES, source="seed", persist_dataset=False)
