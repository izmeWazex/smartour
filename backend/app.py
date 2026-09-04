from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import RAW_CSV_PATH
from src.intent.classify import extract_category
from src.recommender.recommend_tfidf import TfidfRecommender

app = FastAPI(title="Smartour API", description="Vigan / Ilocos tourist spot chatbot backend")

_recommender: TfidfRecommender | None = None


def get_recommender() -> TfidfRecommender:
    """Lazily build the recommender on first use so imports (and tests) stay fast."""
    global _recommender
    if _recommender is None:
        _recommender = TfidfRecommender(RAW_CSV_PATH)
    return _recommender


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User message, e.g. 'gusto ko kumain ng bagnet'")
    top_n: int = Field(3, ge=1, le=20, description="Number of spots to return")
    category: str | None = Field(None, description="Optional hard filter to one category; when omitted, all spots are ranked by content match")


class Spot(BaseModel):
    spot_id: int
    name: str
    category: str
    city: str
    similarity: float


class RecommendResponse(BaseModel):
    query: str
    detected_categories: list[str]
    results: list[Spot]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    detected = extract_category(req.query) or []

    recommender = get_recommender()
    results = recommender.recommend(
        req.query,
        top_n=req.top_n,
        category=req.category,
        preferred_categories=detected or None,
    )

    spots = [
        Spot(
            spot_id=int(row.spot_id),
            name=row.name,
            category=row.category,
            city=row.city,
            similarity=round(float(row.similarity), 4),
        )
        for row in results.itertuples(index=False)
    ]
    return RecommendResponse(query=req.query, detected_categories=detected, results=spots)
