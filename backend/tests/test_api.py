import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module


class StubRecommender:
    def __init__(self, *args, **kwargs):
        pass

    def recommend(self, user_query, top_n=3, category=None, preferred_categories=None):
        return pd.DataFrame(
            {
                "spot_id": [1, 2],
                "name": ["Calle Crisologo", "Mindoro Beach"],
                "category": ["historical", "beach"],
                "city": ["Vigan", "Cabugao"],
                "similarity": [0.912345, 0.812345],
            }
        ).head(top_n)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "TfidfRecommender", StubRecommender)
    app_module._recommender = None
    return TestClient(app_module.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_recommend_returns_spots_and_detected_categories(client):
    resp = client.post("/recommend", json={"query": "gusto ko kumain ng bagnet"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "gusto ko kumain ng bagnet"
    assert "food" in body["detected_categories"]
    assert len(body["results"]) == 2
    assert body["results"][0]["name"] == "Calle Crisologo"
    assert body["results"][0]["similarity"] == 0.9123


def test_recommend_explicit_category_override(client):
    resp = client.post("/recommend", json={"query": "anything", "category": "beach"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_categories"] == []
    assert len(body["results"]) == 2


def test_recommend_top_n_is_respected(client):
    resp = client.post("/recommend", json={"query": "beach", "top_n": 1})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_recommend_missing_query_is_rejected(client):
    resp = client.post("/recommend", json={})
    assert resp.status_code == 422
