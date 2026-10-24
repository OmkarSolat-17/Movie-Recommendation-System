"""Reproducible comparison CLI; no data downloads or work at import time."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from .data import load_movielens, split_interactions
from .evaluation import evaluate
from .models import Recommender
from .reporting import render_markdown

MODELS = ["global_mean", "item_mean", "popularity", "svd", "content", "hybrid"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results.json"))
    parser.add_argument("--markdown", type=Path, help="Also write a shareable Markdown report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--factors", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--relevance-threshold", type=float, default=4)
    parser.add_argument("--user-id", type=int, help="Optional training-only recommendations")
    args = parser.parse_args(argv)
    ratings, movies = load_movielens(args.data_dir)
    train, test = split_interactions(ratings, args.test_fraction, args.seed, args.split)
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "python": platform.python_version(),
            **{name: version(name) for name in ["numpy", "pandas", "scikit-learn"]},
        },
        "data_sha256": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ["u.data", "u.item"]
        },
        "train_interactions": len(train),
        "test_interactions": len(test),
        "cold_test_items": len(set(test.movie_id) - set(train.movie_id)),
        "models": {},
    }
    for kind in MODELS:
        model = Recommender(kind, args.factors, args.seed, args.alpha).fit(train, movies)
        metrics = evaluate(model, test, args.k, args.relevance_threshold)
        if args.user_id is not None:
            titles = movies.set_index("movie_id").title
            metrics["recommendations"] = [
                {"movie_id": mid, "title": titles.loc[mid]}
                for mid in model.recommend(args.user_id, args.k)
            ]
        report["models"][kind] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
