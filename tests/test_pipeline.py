import json

import numpy as np
import pandas as pd
import pytest

from movie_recommender.cli import MODELS, main
from movie_recommender.data import GENRES, load_movielens, split_interactions, validate
from movie_recommender.evaluation import evaluate
from movie_recommender.models import Recommender


@pytest.mark.parametrize("strategy", ["temporal", "random"])
def test_split_is_disjoint_reproducible_and_train_covered(frames, strategy):
    ratings, _ = frames
    train, test = split_interactions(ratings, strategy=strategy)
    again = split_interactions(ratings.sample(frac=1, random_state=9), strategy=strategy)
    pd.testing.assert_frame_equal(train, again[0])
    pd.testing.assert_frame_equal(test, again[1])
    assert len(train) + len(test) == len(ratings)
    assert not set(zip(train.user_id, train.movie_id)) & set(zip(test.user_id, test.movie_id))
    assert set(test.user_id) <= set(train.user_id)
    assert 3 not in set(test.user_id)
    if strategy == "temporal":
        for uid in test.user_id.unique():
            assert (
                train[train.user_id == uid].timestamp.max()
                <= test[test.user_id == uid].timestamp.min()
            )


@pytest.mark.parametrize("kind", MODELS)
def test_ids_seen_exclusion_cold_start_and_determinism(frames, kind):
    ratings, movies = frames
    train, test = split_interactions(ratings)
    model = Recommender(kind, factors=50).fit(train, movies)
    assert model.movie_ids.tolist() == [10, 70, 400, 900, 2000]
    for uid in [51, 800, 3, -100]:
        recs = model.recommend(uid, 100)
        assert set(recs) <= set(movies.movie_id)
        assert not set(recs) & model.seen.get(uid, set())
        assert len(recs) == len(set(recs))
        assert np.isfinite(model.scores(uid)).all()
    assert model.predict(51, -123) == model.mean
    np.testing.assert_array_equal(model.scores(-100), model.item_means)
    shuffled = Recommender(kind, factors=50).fit(train, movies.iloc[::-1])
    np.testing.assert_allclose(model.scores(51), shuffled.scores(51))
    assert evaluate(model, test)["test_interactions"] == 2


@pytest.mark.parametrize("kind", MODELS)
def test_holdout_changes_cannot_change_predictions(frames, kind):
    ratings, movies = frames
    train, test = split_interactions(ratings)
    before = Recommender(kind).fit(train, movies).scores(51)
    altered = ratings.copy()
    for row in test.itertuples():
        altered.loc[
            (altered.user_id == row.user_id) & (altered.movie_id == row.movie_id), "rating"
        ] = 1
    altered_train, _ = split_interactions(altered)
    after = Recommender(kind).fit(altered_train, movies).scores(51)
    np.testing.assert_array_equal(before, after)


def test_ranking_and_rating_metrics_by_hand(frames):
    _, movies = frames
    train = pd.DataFrame([(51, 10, 3, 1)], columns=["user_id", "movie_id", "rating", "timestamp"])
    test = pd.DataFrame([(51, 400, 5, 2), (51, 900, 4, 3)], columns=train.columns)
    model = Recommender("global_mean").fit(train, movies)
    # Equal scores rank IDs 70,400,900,2000. One of two relevant items at K=2.
    metrics = evaluate(model, test, k=2)
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 0.5
    assert metrics["ndcg@2"] == pytest.approx((1 / np.log2(3)) / (1 + 1 / np.log2(3)))
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["mae"] == 1.5
    test.rating = 1
    assert evaluate(model, test)["ndcg@10"] is None
    with pytest.raises(ValueError, match="overlap"):
        evaluate(model, train)
    with pytest.raises(ValueError, match="held-out"):
        evaluate(model, test.iloc[:0])


@pytest.mark.parametrize("problem", ["duplicate", "missing_movie", "range", "nan", "fractional"])
def test_validation_rejects_bad_data(frames, problem):
    ratings, movies = frames
    if problem == "duplicate":
        ratings = pd.concat([ratings, ratings.iloc[:1]])
    elif problem == "missing_movie":
        movies = movies[movies.movie_id != 10]
    elif problem == "range":
        ratings.loc[0, "rating"] = 6
    elif problem == "nan":
        ratings.loc[0, "rating"] = np.nan
    else:
        ratings = ratings.astype({"user_id": float})
        ratings.loc[0, "user_id"] = 1.5
    with pytest.raises(ValueError):
        validate(ratings, movies)


def test_tiny_catalog_and_invalid_parameters(frames):
    ratings, movies = frames
    tiny = Recommender("hybrid").fit(ratings.iloc[:1], movies.iloc[[1]])
    assert tiny.recommend(51) == []
    assert np.isfinite(tiny.scores(51)).all()
    for fraction in [0, 1, -1]:
        with pytest.raises(ValueError):
            split_interactions(ratings, fraction)
    with pytest.raises(ValueError):
        Recommender(alpha=2)
    with pytest.raises(ValueError):
        tiny.recommend(51, 0)


def test_movielens_loader_and_cli(tmp_path, frames, capsys):
    ratings, movies = frames
    ratings.to_csv(tmp_path / "u.data", sep="\t", header=False, index=False)
    rows = []
    for movie in movies.itertuples():
        rows.append(
            [
                movie.movie_id,
                movie.title + "é",
                "",
                "",
                "",
                *[int(g == movie.genres) for g in GENRES],
            ]
        )
    pd.DataFrame(rows).to_csv(
        tmp_path / "u.item", sep="|", header=False, index=False, encoding="latin-1"
    )
    loaded, catalog = load_movielens(tmp_path)
    assert len(loaded) == 7
    assert catalog.title.str.endswith("é").all()
    output = tmp_path / "nested" / "report.json"
    args = ["--data-dir", str(tmp_path), "--output", str(output), "--user-id", "51"]
    main(args)
    first = json.loads(output.read_text())
    main(args)
    assert first == json.loads(output.read_text())
    assert set(first["models"]) == set(MODELS)
    assert "rmse" not in first["models"]["popularity"]
    assert len(first["data_sha256"]["u.data"]) == 64
    capsys.readouterr()
