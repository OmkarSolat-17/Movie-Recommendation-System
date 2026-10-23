import pandas as pd
import pytest


@pytest.fixture
def frames():
    movies = pd.DataFrame(
        {
            "movie_id": [900, 10, 70, 400, 2000],
            "title": ["Z", "A", "B", "C", "Cold"],
            "genres": ["Drama", "Action", "Drama", "Comedy", ""],
        }
    )
    ratings = pd.DataFrame(
        [
            (51, 10, 5, 1),
            (51, 70, 3, 2),
            (51, 900, 4, 3),
            (800, 70, 4, 1),
            (800, 900, 2, 2),
            (800, 400, 5, 3),
            (3, 400, 3, 1),
        ],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    return ratings, movies
