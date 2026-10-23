"""Models share one explicit user/item coordinate system, built from training only."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Recommender:
    """Global mean, item mean, popularity, SVD, content, and hybrid scorers.

    SVD reconstructs the zero-filled TRAIN matrix only. Content profiles are
    rating-weighted genre vectors. Hybrid combines rating-scale SVD and content
    scores; it never compares user coordinates with item coordinates.
    """

    def __init__(self, kind="hybrid", factors=20, seed=42, alpha=0.7):
        if kind not in {"global_mean", "item_mean", "popularity", "svd", "content", "hybrid"}:
            raise ValueError("Unknown model kind")
        if factors < 1 or not 0 <= alpha <= 1:
            raise ValueError("factors must be positive and alpha must be in [0, 1]")
        self.kind, self.factors, self.seed, self.alpha = kind, factors, seed, alpha

    def fit(self, train, movies):
        if train.empty:
            raise ValueError("Training ratings must not be empty")
        self.movie_ids = np.sort(movies.movie_id.unique())
        self.user_ids = np.sort(train.user_id.unique())
        self.item_index = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.mean = float(train.rating.mean())
        self.item_means = (
            train.groupby("movie_id")
            .rating.mean()
            .reindex(self.movie_ids)
            .fillna(self.mean)
            .to_numpy()
        )
        self.counts = (
            train.groupby("movie_id").size().reindex(self.movie_ids, fill_value=0).to_numpy()
        )
        self.seen = {uid: set(group.movie_id) for uid, group in train.groupby("user_id")}
        matrix = np.zeros((len(self.user_ids), len(self.movie_ids)))
        for row in train.itertuples():
            matrix[self.user_index[row.user_id], self.item_index[row.movie_id]] = row.rating
        if self.kind in {"svd", "hybrid"}:
            if min(matrix.shape) < 2:
                self.cf = np.broadcast_to(self.item_means, matrix.shape).copy()
            else:
                rank = min(self.factors, min(matrix.shape) - 1)
                svd = TruncatedSVD(n_components=rank, random_state=self.seed)
                self.cf = svd.inverse_transform(svd.fit_transform(matrix))
            # No evidence for a cold item: use training global mean, not arbitrary factors.
            self.cf[:, self.counts == 0] = self.mean
        if self.kind in {"content", "hybrid"}:
            genres = movies.set_index("movie_id").loc[self.movie_ids, "genres"]
            # Explicit empty token supports catalogs with no genre annotations.
            vectors = TfidfVectorizer(
                tokenizer=str.split, token_pattern=None, lowercase=False
            ).fit_transform(genres.replace("", "unknown"))
            profiles = normalize(matrix @ vectors)
            self.content = np.asarray(profiles @ vectors.T.toarray())
        return self

    def scores(self, user_id):
        """Return catalog-aligned scores; unknown users fall back to item means."""
        if user_id not in self.user_index:
            return self.item_means.copy()
        if self.kind == "global_mean":
            return np.full(len(self.movie_ids), self.mean)
        if self.kind == "item_mean":
            return self.item_means.copy()
        if self.kind == "popularity":
            return self.counts.astype(float).copy()
        row = self.user_index[user_id]
        if self.kind == "svd":
            return np.clip(self.cf[row], 1, 5)
        content = 1 + 4 * self.content[row]
        if self.kind == "content":
            return content
        return self.alpha * np.clip(self.cf[row], 1, 5) + (1 - self.alpha) * content

    def predict(self, user_id, movie_id):
        """Predict an observed pair; unknown items fall back to training global mean."""
        index = self.item_index.get(movie_id)
        return self.mean if index is None else float(self.scores(user_id)[index])

    def recommend(self, user_id, k=10):
        """Rank all unseen catalog items, breaking score ties by ascending movie ID."""
        if k < 1:
            raise ValueError("k must be positive")
        scores = self.scores(user_id)
        seen = self.seen.get(user_id, set())
        candidates = [i for i, mid in enumerate(self.movie_ids) if mid not in seen]
        candidates.sort(key=lambda i: (-scores[i], self.movie_ids[i]))
        return [int(self.movie_ids[i]) for i in candidates[:k]]
