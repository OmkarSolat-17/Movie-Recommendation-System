"""MovieLens loading, validation, and deterministic interaction splits."""

from pathlib import Path

import numpy as np
import pandas as pd

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def validate(ratings, movies):
    """Fail instead of silently dropping malformed or unmatched interactions."""
    if ratings.empty or movies.empty:
        raise ValueError("Ratings and movie catalog must not be empty")
    if ratings[["user_id", "movie_id", "rating", "timestamp"]].isna().any().any():
        raise ValueError("Ratings contain missing values")
    for frame, columns in ((ratings, ["user_id", "movie_id", "timestamp"]), (movies, ["movie_id"])):
        for column in columns:
            values = frame[column]
            if not np.isfinite(values).all() or not (values % 1 == 0).all():
                raise ValueError(f"{column} must contain finite integer IDs/timestamps")
    if not ratings.rating.between(1, 5).all():
        raise ValueError("Ratings must be finite and between 1 and 5")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate user/movie interactions are not supported")
    if movies.movie_id.duplicated().any() or movies.movie_id.isna().any():
        raise ValueError("Movie IDs must be unique and nonmissing")
    if not set(ratings.movie_id).issubset(set(movies.movie_id)):
        raise ValueError("Ratings reference movies missing from catalog")
    if movies[["title", "genres"]].isna().any().any():
        raise ValueError("Movie titles and genres must be nonmissing")


def load_movielens(path):
    """Read official MovieLens 100K u.data and Latin-1 u.item; no user PII needed."""
    path = Path(path).expanduser()
    ratings = pd.read_csv(
        path / "u.data", sep="\t", header=None, names=["user_id", "movie_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        path / "u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        names=["movie_id", "title", "release", "video", "url", *GENRES],
    )
    # Validate before converting flags to strings: otherwise NaN, text, or 2
    # silently become an absent genre, corrupting downstream content profiles.
    if not movies[GENRES].isin([0, 1]).all().all():
        raise ValueError("Genre flags must be nonmissing binary values (0 or 1)")
    movies["genres"] = movies[GENRES].apply(
        lambda row: " ".join(genre for genre in GENRES if row[genre] == 1), axis=1
    )
    movies = movies[["movie_id", "title", "genres"]]
    validate(ratings, movies)
    return ratings, movies


def split_interactions(ratings, test_fraction=0.2, seed=42, strategy="temporal"):
    """Hold out per-user interactions, preserving at least one training observation.

    Single-interaction users remain train-only. Temporal ordering breaks timestamp
    ties by movie ID. Random splitting uses a seeded generator and stable ID order.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    if strategy not in {"temporal", "random"}:
        raise ValueError("strategy must be temporal or random")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for _, group in ratings.groupby("user_id", sort=True):
        group = group.sort_values(["timestamp", "movie_id"]).reset_index(drop=True)
        order = np.arange(len(group))
        if strategy == "random":
            order = rng.permutation(order)
        n = min(len(group) - 1, max(1, int(np.ceil(len(group) * test_fraction))))
        train.append(group.iloc[order[:-n]] if n else group)
        test.append(group.iloc[order[-n:]] if n else group.iloc[:0])
    if not train:
        raise ValueError("Cannot split empty ratings")
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)
