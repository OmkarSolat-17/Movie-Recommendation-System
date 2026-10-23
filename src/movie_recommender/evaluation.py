"""Held-out rating errors and macro-averaged full-catalog ranking metrics."""

import numpy as np


def evaluate(model, test, k=10, relevance_threshold=4):
    if test.empty:
        raise ValueError("Evaluation requires held-out interactions")
    if k < 1 or not 1 <= relevance_threshold <= 5:
        raise ValueError("k must be positive and relevance threshold must be in [1, 5]")
    for row in test.itertuples():
        if row.movie_id in model.seen.get(row.user_id, set()):
            raise ValueError("Train/test overlap detected")
    result = {"test_interactions": len(test), "test_users": int(test.user_id.nunique())}
    # Popularity counts are not ratings; reporting their RMSE would be misleading.
    if model.kind != "popularity":
        errors = np.array(
            [model.predict(r.user_id, r.movie_id) - r.rating for r in test.itertuples()]
        )
        result.update(rmse=float(np.sqrt(np.mean(errors**2))), mae=float(np.mean(np.abs(errors))))
    values = []
    for uid, group in test.groupby("user_id"):
        relevant = set(group.loc[group.rating >= relevance_threshold, "movie_id"])
        if not relevant:
            continue
        ranked = model.recommend(uid, k)
        hits = np.array([mid in relevant for mid in ranked], dtype=float)
        discounts = 1 / np.log2(np.arange(len(hits)) + 2)
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(relevant))) + 2))
        values.append(
            (hits.sum() / k, hits.sum() / len(relevant), float(np.sum(hits * discounts) / ideal))
        )
    result["ranking_users"] = len(values)
    for i, name in enumerate(("precision", "recall", "ndcg")):
        result[f"{name}@{k}"] = float(np.mean([v[i] for v in values])) if values else None
    return result
