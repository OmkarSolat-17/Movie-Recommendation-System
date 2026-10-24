"""Reporting preserves observed metrics and safely renders external titles."""

from movie_recommender.reporting import render_markdown


def test_markdown_metrics_missing_values_and_escaping():
    report = {
        "config": {"k": 3, "seed": 42, "split": "temporal", "user_id": 51},
        "train_interactions": 10,
        "test_interactions": 2,
        "environment": {"python": "3.11"},
        "data_sha256": {"u.data": "abc"},
        "models": {
            "popularity": {
                "precision@3": 0.0,
                "recall@3": None,
                "ndcg@3": None,
                "ranking_users": 0,
                "recommendations": [{"movie_id": 900, "title": "A|B\n<script>"}],
            },
        },
    }
    text = render_markdown(report)
    assert "| popularity | — | — | 0.000000 | — | — | 0 |" in text
    assert "| 900 | A&#124;B &lt;script&gt; |" in text
    assert "SHA-256 u.data | abc" in text
    assert "smoke tests" in text
    assert "<script>" not in text
