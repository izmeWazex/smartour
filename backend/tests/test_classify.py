from src.intent.classify import extract_category


def test_historical():
    assert extract_category("I want a peaceful historical place") == ["historical"]


def test_beach():
    assert extract_category("suggest me a good beach") == ["beach"]


def test_food():
    assert extract_category("gusto ko kumain ng bagnet") == ["food"]


def test_nature():
    assert extract_category("Pwede ba mag hike papunta sa waterfalls?") == ["nature"]


def test_multiple_categories():
    result = extract_category("Where can I dine near the ancestral houses?")
    assert "food" in result
    assert "historical" in result


def test_no_match_returns_none():
    assert extract_category("asdkjaskjd random text") is None


def test_empty_string_returns_none():
    assert extract_category("") is None


def test_case_insensitive():
    assert extract_category("HISTORICAL PLACES") == ["historical"]


def test_substring_matching():
    assert extract_category("beachhh vibes only") == ["beach"]


def test_fallback_catches_unlisted_words():
    assert extract_category("Is the water clean there?") == ["nature"]


def test_fallback_catches_word_forms():
    assert extract_category("looking for a calm seaside spot") == ["beach"]


def test_fallback_rejects_noise():
    assert extract_category("asdkjaskjd random text") is None