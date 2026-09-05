import pytest

from src.recommender.recommend_tfidf import TfidfRecommender

CSV = """spot_id,name,category,description,city
1,Old Church,historical,old church and museum with heritage walls,Vigan
2,Mindoro Beach,beach,sand and shore with calm water,Cabugao
3,Lingsat Food Row,food,bagnet and empanada street food,Vigan
4,Pinsal Falls,nature,waterfall hike in the mountains,Burgos
"""


@pytest.fixture(scope="module")
def recommender(tmp_path_factory):
    """Built once per module: the recommender is read-only after __init__."""
    csv_path = tmp_path_factory.mktemp("data") / "spots.csv"
    csv_path.write_text(CSV)
    return TfidfRecommender(csv_path)


def test_recommend_returns_top_n(recommender):
    results = recommender.recommend("bagnet", top_n=2)
    assert len(results) == 2
    assert list(results.columns) == ["spot_id", "name", "category", "city", "similarity"]


def test_recommend_sorts_by_similarity_descending(recommender):
    results = recommender.recommend("waterfall hike mountains", top_n=4)
    assert list(results["similarity"]) == sorted(results["similarity"], reverse=True)


def test_recommend_best_match_comes_first(recommender):
    results = recommender.recommend("waterfall hike in the mountains", top_n=4)
    # The query is word-for-word the Pinsal Falls description -> highest score.
    assert results.iloc[0]["name"] == "Pinsal Falls"
    assert results.iloc[0]["category"] == "nature"


def test_recommend_matches_by_word_weights_not_just_keywords(recommender):
    results = recommender.recommend("craving bagnet", top_n=4)
    assert results.iloc[0]["name"] == "Lingsat Food Row"


def test_recommend_category_filter(recommender):
    results = recommender.recommend("something", top_n=4, category="beach")
    assert len(results) == 1
    assert results.iloc[0]["name"] == "Mindoro Beach"


def test_recommend_category_is_case_insensitive(recommender):
    results = recommender.recommend("something", top_n=4, category="BEACH")
    assert len(results) == 1
    assert results.iloc[0]["name"] == "Mindoro Beach"


def test_recommend_category_list_filter(recommender):
    results = recommender.recommend("something", top_n=4, category=["beach", "food"])
    assert set(results["category"]) == {"beach", "food"}
    assert len(results) == 2


def test_recommend_category_list_is_case_insensitive(recommender):
    results = recommender.recommend("something", top_n=4, category=["BEACH", "NATURE"])
    assert set(results["category"]) == {"beach", "nature"}


def test_recommend_empty_category_list_means_no_filter(recommender):
    results = recommender.recommend("something", top_n=4, category=[])
    assert len(results) == 4


def test_recommend_unknown_category_returns_empty(recommender):
    results = recommender.recommend("something", top_n=4, category="shopping")
    assert len(results) == 0


def test_recommend_query_without_corpus_words_does_not_crash(recommender):
    results = recommender.recommend("zzzz qqqq", top_n=2)
    assert len(results) == 2
    assert results["similarity"].notna().all()
    assert list(results["similarity"]) == [0.0, 0.0]
