from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class TrainingExample(BaseModel):
    text: str = Field(..., min_length=1, max_length=1024, description="Example user message")
    intent: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",
        description="Intent label (lowercase, letters/digits/underscore)",
    )
    response: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Canned reply for custom intents. Required for intents "
                    "that are not built-in.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Where can I rent a kalesa?",
                    "intent": "kalesa_rental",
                    "response": "You can rent a kalesa at Plaza Salcedo — rides start around ₱200.",
                }
            ]
        }
    }


class TrainRequest(BaseModel):
    examples: List[TrainingExample] = Field(
        ...,
        min_length=2,
        description="Labeled examples used to (re)train the intent model. "
                    "Replaces the previous model.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "examples": [
                    {"text": "Where can I rent a kalesa?", "intent": "kalesa_rental",
                     "response": "Kalesa rides start at Plaza Salcedo, ~₱200 for 30 min."},
                    {"text": "How much does a kalesa ride cost?", "intent": "kalesa_rental",
                     "response": "Kalesa rides start at Plaza Salcedo, ~₱200 for 30 min."},
                    {"text": "Where can I eat longanisa?", "intent": "recommend"},
                    {"text": "Recommend food spots", "intent": "recommend"},
                ]
            }
        }
    }


class TrainResponse(BaseModel):
    status: str
    message: str
    intents: List[str]
    example_count: int
    metrics: dict
    trained_at: datetime


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1024, description="Message to inspect")
    top_k: int = Field(default=3, ge=1, le=7, description="How many candidate intents to return")

    model_config = {
        "json_schema_extra": {
            "example": {"text": "best to go on vigan"}
        }
    }


class IntentScore(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    text: str
    intent: Optional[str] = Field(
        default=None,
        description="Chosen intent (None when confidence is below the threshold)",
    )
    confidence: float
    threshold: float
    top_intents: List[IntentScore]


class ModelStatusResponse(BaseModel):
    is_trained: bool
    backend: str
    source: Optional[str]
    intents: List[str]
    example_count: int
    metrics: dict
    trained_at: Optional[datetime]
    model_path: str
    dataset_csv: str
    confidence_threshold: float
