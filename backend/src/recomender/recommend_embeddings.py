import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRecommender:
    def __init__(self , csv_path: str):
        self.df = pd.read_csv(csv_path)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.spot_vectors = self.model.encode(self.df["description"].tolist())

    def recommend(self, user_query: str , top_n : int = 3 , category : str = None):
                  
        query_vector = self.model.encode([user_query])

        similarity_scores = cosine_similarity ( query_vector , self.spot_vectors).flatten()

        results = self.df.copy()
        results["similarity"] = similarity_scores

       
        if category:
            results = results[results["category"].str.lower() == category.lower()]

        results = results.sort_values("similarity", ascending=False).head(top_n)

        return results[["spot_id", "name", "category", "city", "similarity"]]

if __name__ == "__main__":
    recommender = EmbeddingRecommender("backend/data/raw/tourist_spots.csv")
    print(recommender.recommend("I want a peaceful historical place", top_n=3))