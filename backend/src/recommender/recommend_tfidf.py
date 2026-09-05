

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import RAW_CSV_PATH

CATEGORY_MATCH_BONUS = 0.25


class TfidfRecommender:
    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path)

        self.vectorizer = TfidfVectorizer()
        self.spot_vectors = self.vectorizer.fit_transform(self.df["description"])
        print(
            f"[recommend] TfidfRecommender loaded {len(self.df)} spots from {csv_path} "
            f"| categories: {sorted(self.df['category'].str.lower().unique().tolist())}"
        )

    def recommend(
        self,
        user_query: str,
        top_n: int = 3,
        category: str | list[str] | None = None,
        preferred_categories: list[str] | None = None,
    ):

        query_vector = self.vectorizer.transform([user_query])

        similarity_scores = cosine_similarity(query_vector, self.spot_vectors).flatten()
        similarity_scores = np.nan_to_num(similarity_scores, nan=0.0)

        results = self.df.copy()
        results["similarity"] = similarity_scores

        if category:
            wanted = [category] if isinstance(category, str) else category
            wanted = {c.lower() for c in wanted}
            results = results[results["category"].str.lower().isin(wanted)]

        if preferred_categories:
            wanted = {c.lower() for c in preferred_categories}
            matches = results["category"].str.lower().isin(wanted)
            results["similarity"] = (
                results["similarity"] + matches.astype(float) * CATEGORY_MATCH_BONUS
            )

        results = results.sort_values("similarity", ascending=False).head(top_n)
        results = results[["spot_id", "name", "category", "city", "similarity"]]

        return results


# if __name__ == "__main__":
#     recommender = TfidfRecommender(RAW_CSV_PATH)
#     print(recommender.recommend("I want a peaceful historical place", top_n=3))
