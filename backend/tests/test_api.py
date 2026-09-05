import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as app_module


class StubRecommender:
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.fail_on = None
        self.df = pd.DataFrame({"category": ["historical", "beach"]})

    def recommend(self, user_query, top_n=3, category=None, preferred_categories=None):
        self.calls.append((category, preferred_categories))
        if self.fail_on is not None and category == self.fail_on:
            return pd.DataFrame(
                columns=["spot_id", "name", "category", "city", "similarity"]
            )
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
    body = resp.json()
    assert body["status"] == "ok"
    assert body["csv_path"]
    assert body["spots"] == 2
    assert body["categories"] == ["beach", "historical"]


def test_recommend_returns_spots_and_detected_categories(client):
    resp = client.post("/recommend", json={"query": "gusto ko kumain ng bagnet"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "gusto ko kumain ng bagnet"
    assert "food" in body["detected_categories"]
    assert len(body["results"]) == 2
    assert body["results"][0]["name"] == "Calle Crisologo"
    assert body["results"][0]["similarity"] == 0.9123


def test_recommend_uses_detected_category_as_filter(client):
    resp = client.post("/recommend", json={"query": "gusto ko kumain ng bagnet"})
    assert resp.status_code == 200
    assert app_module.get_recommender().calls == [(["food"], None)]


def test_recommend_falls_back_when_detected_filter_finds_nothing(client):
    recommender = app_module.get_recommender()
    recommender.fail_on = ["food"]
    resp = client.post("/recommend", json={"query": "gusto ko kumain ng bagnet"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert recommender.calls == [(["food"], None), (None, None)]


def test_recommend_falls_back_to_detected_when_explicit_category_matches_nothing(client):
    recommender = app_module.get_recommender()
    recommender.fail_on = "restaurant"
    resp = client.post(
        "/recommend", json={"query": "i want to eat bagnet", "category": "restaurant"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2
    assert recommender.calls == [("restaurant", None), (["food"], None)]


def test_recommend_falls_back_to_content_ranking_when_nothing_matches(client):
    recommender = app_module.get_recommender()
    recommender.fail_on = "shopping"
    resp = client.post(
        "/recommend", json={"query": "anything", "category": "shopping"}
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2
    assert recommender.calls == [("shopping", None), (None, None)]


def test_recommend_explicit_category_override(client):
    resp = client.post("/recommend", json={"query": "anything", "category": "beach"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected_categories"] == []
    assert len(body["results"]) == 2
    assert app_module.get_recommender().calls == [("beach", None)]


def test_recommend_without_category_uses_full_ranking(client):
    resp = client.post("/recommend", json={"query": "anything"})
    assert resp.status_code == 200
    assert app_module.get_recommender().calls == [(None, None)]


def test_recommend_top_n_is_respected(client):
    resp = client.post("/recommend", json={"query": "beach", "top_n": 1})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_recommend_passes_preferred_categories_to_recommender(client):
    resp = client.post(
        "/recommend",
        json={"query": "bagnet", "preferred_categories": ["food"]},
    )
    assert resp.status_code == 200
    # No explicit category was sent, so detected categories become the filter.
    assert app_module.get_recommender().calls == [(["food"], ["food"])]


def test_recommend_missing_query_is_rejected(client):
    resp = client.post("/recommend", json={})
    assert resp.status_code == 422
