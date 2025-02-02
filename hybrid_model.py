"""Hybrid Recommendation System with Contextual Enrichment"""

import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from nltk.stem import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommender:
    """Hybrid recommendation engine combining ALS collaborative filtering
    with contextual content-based filtering"""

    def __init__(self, data_path: str = "lastfmdata/"):
        self.data_path = data_path
        self._load_data()
        self._prepare_models()

    def _load_data(self):
        """
        Load and preprocess datasets.


        Merge user-tagged artists with timestamps, calculate days_old and apply an
        exponential decay for time_weight.
        """
        logger.info("Loading and preprocessing data...")

        # Core datasets
        self.user_artists = pd.read_csv(
            f"{self.data_path}user_artists.dat", sep="\t"
        )
        self.artists = pd.read_csv(f"{self.data_path}artists.dat", sep="\t")

        # Tag processing with temporal decay
        tags = pd.read_csv(
            f"{self.data_path}tags.dat", sep="\t", encoding="latin1"
        )
        user_tags = pd.read_csv(
            f"{self.data_path}user_taggedartists.dat", sep="\t"
        )

        # Load timestamps with proper type conversion
        timestamps = pd.read_csv(
            f"{self.data_path}user_taggedartists-timestamps.dat",
            sep="\t",
            dtype={"timestamp": "int64"},
        )
        timestamps["timestamp"] = pd.to_datetime(
            timestamps["timestamp"], unit="ms", utc=True
        ).dt.tz_localize(None)

        # Merge datasets
        self.tag_data = user_tags.merge(
            timestamps, on=["userID", "artistID", "tagID"]
        )
        self.tag_data = self.tag_data.merge(tags, on="tagID")

        # Calculate temporal weights
        current_time = datetime.now()
        self.tag_data["days_old"] = (
            current_time - self.tag_data["timestamp"]
        ).dt.days
        self.tag_data["time_weight"] = np.exp(
            -self.tag_data["days_old"] * 0.002
        )

    def _prepare_models(self):
        """
        Initialize and train recommendation models


        Set up the collaborative filtering using ALS from implicit.
        Create a user-artist matrix in CSR format,
        apply BM25 weighting, then fit the ALS model.
        """
        logger.info("Preparing recommendation models...")

        # --- Collaborative Filtering Setup ---
        # Create CSR matrix and convert to CSR after BM25 weighting
        user_artist_matrix = csr_matrix(
            (
                self.user_artists["weight"].astype(float),
                (self.user_artists["userID"], self.user_artists["artistID"]),
            )
        )
        weighted_matrix = bm25_weight(user_artist_matrix, K1=100, B=0.8)
        self.user_artist_matrix = weighted_matrix.tocsr()  # Ensure CSR format

        self.cf_model = AlternatingLeastSquares(
            factors=128, regularization=0.08, iterations=25, random_state=42
        )
        self.cf_model.fit(self.user_artist_matrix)

        # --- Content-Based Setup ---
        self._prepare_tfidf_model()
        self._build_artist_similarity_index()

    def _prepare_tfidf_model(self):
        """Create enhanced TF-IDF vectorizer with stemming and temporal weights"""
        logger.info("Building enhanced TF-IDF model...")

        # Custom stemmer with fallback
        stemmer = SnowballStemmer("english")

        class StemmedTfidfVectorizer(TfidfVectorizer):
            def build_analyzer(self):
                analyzer = super().build_analyzer()
                return lambda doc: [
                    stemmer.stem(word)
                    for word in analyzer(doc)
                    if len(word) > 2  # Filter out short tokens
                ]

        # Temporal-weighted tag aggregation with validation
        artist_tags = (
            self.tag_data[["artistID", "tagValue", "time_weight"]]
            .groupby("artistID", group_keys=False, observed=True)
            .apply(
                lambda x: " ".join(
                    np.repeat(
                        x["tagValue"].astype(str),
                        np.clip((x["time_weight"] * 10).astype(int), 1, 100),
                    )
                ),
                include_groups=False,
            )
            .reindex(self.artists["id"])
            .fillna("")
            .apply(lambda x: x if x.strip() else "unknown")
        )

        # TF-IDF with safety parameters
        self.tfidf_vectorizer = StemmedTfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2,
            max_features=2000,
            stop_words=None,
            token_pattern=r"(?u)\b[\w\-]+\b",
        )

        # Validate input data
        if artist_tags.str.contains(r"\w").sum() == 0:
            raise ValueError("No valid text data for TF-IDF processing")

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(artist_tags)

    def _build_artist_similarity_index(self):
        """Create similarity index for fast lookups"""
        logger.info("Building similarity index...")
        self.artist_similarities = cosine_similarity(self.tfidf_matrix)

    def _get_user_context(self, user_id: int) -> Tuple[csr_matrix, List[int]]:
        """Create user-specific context vector that represents their preferences."""
        # Get user's favorite artists (top 10 by listening weight)
        user_favorites = (
            self.user_artists[self.user_artists["userID"] == user_id]
            .nlargest(10, "weight")["artistID"]
            .values
        )

        # Get user's personal tag profile with temporal weights
        user_tags = self.tag_data[self.tag_data["userID"] == user_id]
        tag_profile = " ".join(
            np.repeat(
                user_tags["tagValue"].astype(str),
                np.clip((user_tags["time_weight"] * 10).astype(int), 1, 100),
            )
        )

        # Transform to TF-IDF space
        if tag_profile.strip():
            user_vector = self.tfidf_vectorizer.transform([tag_profile])
        else:
            user_vector = csr_matrix((1, self.tfidf_matrix.shape[1]))

        return user_vector, user_favorites

    def _dynamic_alpha(self, user_id: int) -> float:
        """Calculate dynamic weighting between CF and CB"""
        tag_count = len(self.tag_data[self.tag_data["userID"] == user_id])
        return np.clip(0.8 - 0.6 * np.log1p(tag_count) / np.log(100), 0.2, 0.8)

    def recommend(
        self, user_id: int, top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Generate hybrid recommendations for a user.


        Get CF recommendations using ALS, then compute CB scores using the user's TF-IDF vector and
        similarity to favorite artists. The combined score uses alpha to blend CF, CB, and favorite similarities.
        The CB part is split into 70% from the user's tags and 30% from similarity to favorites.
        Then they take the top N recommendations.
        """
        logger.info(f"Generating recommendations for user {user_id}...")

        # Get collaborative filtering candidates
        artist_ids, cf_scores = self.cf_model.recommend(
            user_id,
            self.user_artist_matrix[user_id],  # Now works with CSR matrix
            N=top_n * 3,
        )

        # Get user context
        user_vector, user_favorites = self._get_user_context(user_id)

        # Calculate content-based scores
        if user_vector.sum() > 0:
            cb_scores = cosine_similarity(
                self.tfidf_matrix[artist_ids], user_vector
            ).flatten()
        else:
            cb_scores = np.zeros(len(artist_ids))

        # Calculate similarity to favorite artists
        if len(user_favorites) > 0:
            fav_similarities = cosine_similarity(
                self.tfidf_matrix[artist_ids],
                self.tfidf_matrix[user_favorites],
            ).max(axis=1)
        else:
            fav_similarities = np.zeros(len(artist_ids))

        # Combine scores
        alpha = self._dynamic_alpha(user_id)
        combined_scores = (
            alpha * cf_scores
            + (0.7 * (1 - alpha) * cb_scores)
            + (0.3 * (1 - alpha) * fav_similarities)
        )

        # Get top recommendations
        ranked_indices = np.argsort(-combined_scores)[:top_n]
        recommendations = [
            (
                self.artists.loc[
                    self.artists["id"] == artist_ids[i], "name"
                ].iloc[0],
                combined_scores[i],
            )
            for i in ranked_indices
        ]

        return recommendations

    def partial_fit(self, new_interactions: pd.DataFrame):
        """
        Update model with new interactions.


        Resizes the user-item matrix if needed, updates it with new data,
        and then partially retrains the ALS model on the affected users.
        """
        logger.info("Updating model with new data...")

        if new_interactions.empty:
            return

        # Convert to CSR first for efficient operations
        current_matrix = self.user_artist_matrix.tocsr()

        # Get existing dimensions
        n_users, n_items = current_matrix.shape

        # Find required new dimensions
        max_user = max(new_interactions["userID"].max(), n_users - 1)
        max_item = max(new_interactions["artistID"].max(), n_items - 1)
        new_shape = (max_user + 1, max_item + 1)

        # Resize matrices using efficient LIL format
        if current_matrix.shape != new_shape:
            from scipy.sparse import lil_matrix

            resized = lil_matrix(new_shape, dtype=np.float32)
            resized[:n_users, :n_items] = current_matrix
            current_matrix = resized.tocsr()

        # Create update matrix with same dimensions
        update_matrix = csr_matrix(
            (
                new_interactions["weight"].astype(np.float32),
                (new_interactions["userID"], new_interactions["artistID"]),
            ),
            shape=new_shape,
        )

        # Merge matrices using element-wise maximum
        self.user_artist_matrix = current_matrix.maximum(update_matrix).tocsr()

        # Get unique users with new interactions
        unique_users = np.unique(new_interactions["userID"])

        # Verify matrix contains all users
        if (unique_users >= self.user_artist_matrix.shape[0]).any():
            raise ValueError("Matrix shape doesn't cover all user IDs")

        # Perform partial fit using only affected users
        self.cf_model.partial_fit_users(
            unique_users, self.user_artist_matrix[unique_users]
        )


# --- Usage Example ---
if __name__ == "__main__":
    recommender = HybridRecommender()

    # Get recommendations for user 2
    recommendations = recommender.recommend(user_id=2)
    print("\nTop Recommendations:")
    for artist, score in recommendations:
        print(f"{artist}: {score:.3f}")

    # Simulate model update with new data
    new_data = pd.DataFrame(
        {"userID": [2, 2], "artistID": [123, 456], "weight": [5000, 3000]}
    )
    recommender.partial_fit(new_data)

    # Get updated recommendations for user 2
    recommendations = recommender.recommend(user_id=2)
    print("\nUpdated Recommendations:")
    for artist, score in recommendations:
        print(f"{artist}: {score:.3f}")
