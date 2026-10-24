"""Regression tests for silently discarded MovieLens genre annotations."""

import pytest

from movie_recommender.data import GENRES, load_movielens


def write_dataset(path, flags):
    (path / "u.data").write_text("51\t900\t4\t1\n", encoding="utf-8")
    row = ["900", "Example", "", "", "", *flags]
    (path / "u.item").write_text("|".join(row) + "\n", encoding="latin-1")


@pytest.mark.parametrize("invalid", ["", "2", "-1", "0.5", "yes", "NaN", "inf"])
def test_loader_rejects_invalid_genre_flags(tmp_path, invalid):
    flags = ["0"] * len(GENRES)
    flags[GENRES.index("Action")] = invalid
    write_dataset(tmp_path, flags)
    with pytest.raises(ValueError, match="Genre flags must be nonmissing binary values"):
        load_movielens(tmp_path)


def test_loader_rejects_truncated_genre_columns(tmp_path):
    write_dataset(tmp_path, ["0"] * (len(GENRES) - 1))
    with pytest.raises(ValueError, match="Genre flags must be nonmissing binary values"):
        load_movielens(tmp_path)


@pytest.mark.parametrize("active", [[], ["unknown"], ["Action", "Comedy"]])
def test_loader_preserves_valid_genres(tmp_path, active):
    write_dataset(tmp_path, [str(int(genre in active)) for genre in GENRES])
    ratings, movies = load_movielens(tmp_path)
    assert len(ratings) == 1
    assert movies.loc[0, "genres"] == " ".join(active)
