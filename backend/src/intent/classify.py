

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import RAW_CSV_PATH

CATEGORY_KEYWORDS = {
    "historical": [
        "historical", "heritage", "ancestral", "old", "church", "museum",
        "cathedral", "plaza", "tower", "pottery", "colonial", "ruins",
        "landmark", "antique",
    ],
    "beach": [
        "beach", "swim", "shore", "sand", "sea", "ocean", "coast",
        "sunset", "island", "waves", "seaside",
    ],
    "food": [
        "food", "eat", "dine", "restaurant", "bagnet", "empanada",
        "hungry", "craving", "cafe", "meal", "lunch", "dinner",
        "breakfast", "cuisine", "kain", "ulam", "snack", "street food",
    ],
    "nature": [
        "waterfall", "falls", "nature", "hike", "mountain", "trek",
        "forest", "river", "view", "scenic", "trail", "rock formation",
        "cliff", "greenery",
    ],
}

SIMILARITY_THRESHOLD = 0.1

_vectorizer: TfidfVectorizer | None = None
_category_vectors: csr_matrix | None = None


def _get_fallback_model() -> tuple[TfidfVectorizer, csr_matrix]:
    global _vectorizer, _category_vectors
    if _vectorizer is None:
        df = pd.read_csv(RAW_CSV_PATH)
        documents = [
            " ".join(df.loc[df["category"].str.lower() == category, "description"])
            for category in CATEGORY_KEYWORDS
        ]
        _vectorizer = TfidfVectorizer(stop_words="english")
        _category_vectors = _vectorizer.fit_transform(documents)
        print(f"[classify] built TF-IDF fallback model from {len(documents)} category documents")
    return _vectorizer, _category_vectors


def _fallback_categories(message: str) -> list[str] | None:
    vectorizer, category_vectors = _get_fallback_model()

    query_vector = vectorizer.transform([message])
    similarities = cosine_similarity(query_vector, category_vectors).flatten()
    similarities = np.nan_to_num(similarities, nan=0.0)

    scores = {
        category: round(float(score), 3)
        for category, score in zip(CATEGORY_KEYWORDS, similarities)
    }
    best_index = int(similarities.argmax())
    if similarities[best_index] >= SIMILARITY_THRESHOLD:
        return [list(CATEGORY_KEYWORDS)[best_index]]
    return None


def extract_category(user_message: str) -> list[str] | None:
    message = user_message.lower()

    if not message.strip():
            return None

    match = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            match.append(category)

    result = match if match else _fallback_categories(message)
    print(f"[classify] extract_category({user_message!r}) -> {result}")
    return result


# if __name__ == "__main__":
#     print(extract_category("Where can I dine near the ancestral houses?"))
#     print(extract_category("I'm craving something near the shore"))
#     print(extract_category("Is there a restaurant inside an old museum?"))
#     print(extract_category("Pwede ba mag hike papunta sa waterfalls?"))
#     print(extract_category("HISTORICAL PLACES i can eat PLEASE"))
#     print(extract_category(""))
#     print(extract_category("beachhh vibes only"))
#     print(extract_category("I heard bagnet is famous near the beach"))
#     print(extract_category("is there a waterfall in vigan city"))
#     print(extract_category("Is the water clean there?"))
#     print(extract_category("I want to go swimming"))
#     print(extract_category("asdkjaskjd random text"))