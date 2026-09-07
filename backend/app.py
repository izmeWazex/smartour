from fastapi import FastAPI, HTTPException
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
        print(
            f"[app] recommender ready: {len(_recommender.df)} spots loaded "
            f"from {RAW_CSV_PATH}"
        )
    return _recommender


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User message, e.g. 'gusto ko kumain ng bagnet'")
    top_n: int = Field(3, ge=1, le=20, description="Number of spots to return")
    category: str | None = Field(None, description="Optional hard filter to one category; when omitted, all spots are ranked by content match")
    preferred_categories: list[str] | None = Field(
        None,
        description="Optional soft bonus to spots whose category appears here; does not filter out other categories",
    )


class Spot(BaseModel):
    spot_id: int
    name: str
    category: str
    city: str
    description: str | None = None
    similarity: float


class RecommendResponse(BaseModel):
    query: str
    detected_categories: list[str]
    results: list[Spot]


@app.get("/health")
def health() -> dict:
    recommender = get_recommender()
    return {
        "status": "ok",
        "csv_path": str(RAW_CSV_PATH),
        "spots": len(recommender.df),
        "categories": sorted(recommender.df["category"].str.lower().unique().tolist()),
    }


@app.get("/spot/{place_name}")
def spot_detail(place_name: str) -> dict:
    """Get details and description for a specific tourist spot."""
    recommender = get_recommender()
    spot = recommender.get_spot_description(place_name)
    if spot is None:
        raise HTTPException(status_code=404, detail=f"Spot '{place_name}' not found")
    return spot


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    # Check for specific place name match first - clarify what user is asking
    recommender = get_recommender()
    spot = recommender.get_spot_description(req.query)
    
    if spot:
        # User is asking about a specific place - clarify and return description
        # This keeps get_spot_description() separate for future features
        return RecommendResponse(
            query=req.query,
            detected_categories=[],
            results=[Spot(
                spot_id=spot["spot_id"],
                name=str(spot["name"]),
                category=spot["category"],
                city=spot["city"],
                description=spot["description"],
                similarity=1.0,
            )]
        )
    
    # No specific place match - proceed with normal intent classification
    detected = extract_category(req.query) or []

    recommender = get_recommender()
    filter_category = req.category or detected or None
    results = recommender.recommend(
        req.query,
        top_n=req.top_n,
        category=filter_category,
        preferred_categories=req.preferred_categories,
    )

    if results.empty:
        if req.category and detected:
            results = recommender.recommend(
                req.query,
                top_n=req.top_n,
                category=detected,
                preferred_categories=req.preferred_categories,
            )
        if results.empty:
            results = recommender.recommend(
                req.query,
                top_n=req.top_n,
                preferred_categories=req.preferred_categories,
            )

    spots = [
        Spot(
            spot_id=int(row.spot_id),
            name=row.name,
            category=row.category,
            city=row.city,
            description=recommender.df[recommender.df["spot_id"] == int(row.spot_id)]["description"].values[0] if int(row.spot_id) in recommender.df["spot_id"].values else None,
            similarity=round(float(row.similarity), 4),
        )
        for row in results.itertuples(index=False)
    ]

    print(
        f"[app] recommend: query={req.query!r} category={req.category!r} "
        f"preferred_categories={req.preferred_categories!r} detected={detected} "
        f"filter={filter_category!r} "
        f"-> {len(spots)} result(s): {[s.name for s in spots]}"
    )
    return RecommendResponse(query=req.query, detected_categories=detected, results=spots)