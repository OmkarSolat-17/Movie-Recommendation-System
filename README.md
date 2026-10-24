# Movie Recommendation System

A reproducible, train-only MovieLens 100K experiment suite. Compare simple baselines,
collaborative filtering, genre-based content filtering, and a hybrid on both rating
prediction and recommendation ranking. Runs on CPU; no TensorFlow, GUI, user
personal information, API keys, or network access is required at runtime.

**Start here:** [guided walkthrough](docs/walkthrough.md) ·
[actual synthetic example output](docs/demo-report.md) ·
[evaluation protocol](#evaluation-protocol)

## Quick start

Python **3.11 or newer** (CI targets 3.11 and 3.12). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
pytest -q
```

The lock file pins the complete development/runtime dependency set from the tested
Python 3.11 environment. `pip install -e '.[dev]'` is an alternative when intentionally
resolving newer versions within `pyproject.toml` bounds. Regenerate the lock in a
clean environment and rerun checks before changing it.

### Run without downloading data

```bash
python examples/make_demo.py --output data/demo
movie-recommender --data-dir data/demo --k 3 --user-id 101 \
  --markdown artifacts/demo.md
```

This deliberately tiny, deterministic **synthetic smoke test is not a benchmark**.
The CLI prints JSON and writes `artifacts/results.json`. Data and generated results
are ignored by Git. Repeating a run with the same environment, data and arguments
produces the same report. JSON includes configuration, dependency versions, input
SHA-256 hashes, split counts, cold-item counts, per-model metrics and optional
recommendations with original movie IDs and titles.

Use `--markdown PATH` with either synthetic or real data to export the same
measurements as a readable comparison table with recommendations and provenance.
JSON remains the full-precision source. [See an actual generated example](docs/demo-report.md);
these toy results do not establish real-data performance. The
[walkthrough](docs/walkthrough.md) explains how to interpret and share them responsibly.

### Use MovieLens 100K

Obtain and extract **MovieLens 100K** from the
[GroupLens dataset page](https://grouplens.org/datasets/movielens/100k/), observing
its usage and attribution terms. Do not commit or redistribute the dataset here.
Point to the directory containing `u.data` and `u.item` (not their parent):

```bash
movie-recommender --data-dir data/ml-100k --split temporal \
  --test-fraction 0.2 --seed 42 --factors 20 --alpha 0.7 \
  --k 10 --relevance-threshold 4 --user-id 2 \
  --output artifacts/movielens.json
# Equivalent compatibility entry point:
python main.py --data-dir data/ml-100k
movie-recommender --help
```

No absolute paths are embedded. Missing files and malformed data fail explicitly;
duplicate user/movie pairs, invalid ratings/IDs, and missing catalog references
are rejected rather than silently merged, imputed or dropped. All 19 genre flags
in `u.item` must be present and binary (0 or 1); missing, nonnumeric, or nonbinary
flags raise `ValueError` before content features are built. An all-zero genre row
is valid and uses the model's unknown-genre fallback. `u.user` is not used.

## Models and comparison

Every invocation evaluates all six models on the **same split and candidate catalog**:

| Model | Scores | Rating error reported? |
| --- | --- | --- |
| Global mean | Training global rating mean | Yes |
| Item mean | Training mean per movie, global fallback | Yes |
| Popularity | Training interaction count per movie | No: counts are not ratings |
| SVD | Truncated reconstruction of the zero-filled training matrix, clipped to [1, 5] | Yes |
| Content | Cosine similarity of rating-weighted user genre profile and movie TF-IDF, scaled to [1, 5] | Yes |
| Hybrid | `alpha * SVD + (1-alpha) * content` (default alpha 0.7) | Yes |

IDs are dictionary-mapped independently for users and movies. Matrix coordinates
are never inferred with `id - 1`; recommendations return **movie IDs**, not neighbor
user IDs. Non-contiguous IDs and shuffled catalog order are supported. Unknown
users fall back to training item means; unknown items to the global training mean.
Catalog movies with no training observations use the global mean in SVD and can
still receive genre-based content scores. Seen training items are always excluded;
score ties use ascending movie ID. Fewer than K candidates returns fewer than K items.

## Evaluation protocol

1. Split **interactions before fitting**. Default: hold out the latest `ceil(n * fraction)`
   events per user, keeping at least one training event. Single-event users are
   training-only. Timestamp ties break by movie ID. `--split random` uses stable
   ordering and a seeded generator instead.
2. Fit all rating-derived state, SVD factors and content profiles from **training
   ratings only**. Genre TF-IDF uses the supplied catalog: this is a transductive
   metadata assumption, not a fit on held-out ratings. SVD never transforms a test
   matrix. Evaluation rejects overlapping train/test user/movie pairs.
3. RMSE and MAE use only observed held-out pairs and the actual named model's
   predictions. There is no mislabeled "hybrid RMSE" from SVD outputs.
4. Rank the **entire catalog minus training-seen items**, without negative sampling.
   Held-out ratings >= the relevance threshold (default 4) are relevant. Macro-average
   Precision@K, Recall@K and binary NDCG@K across test users with at least one relevant
   item. Report the number of eligible users; when zero, ranking metrics are JSON null.
   Precision divides by K even if fewer candidates exist; recall divides by the
   user's held-out relevant count; ideal DCG uses `min(K, relevant_count)`.

### Interpretation and limitations

- Per-user temporal splitting is **not** a global time cutoff: another user's later
  training activity can precede neither a particular user's test event nor deployment.
  Use a global cutoff for deployment-time studies. This implementation evaluates
  known-user offline recommendations, not a full future-production simulation.
- Zero-filled SVD treats missing ratings as zero and is a transparent baseline, not
  observed-only matrix factorization. Content score scaling is heuristic, not calibrated.
- The implementation uses dense user-item score matrices for MovieLens 100K-scale
  experiments; it is not intended for million-user production catalogs.
- Unobserved items are not known negatives. Offline ranking scores reflect exposure
  bias. Compare accuracy alongside coverage, diversity and online behavior before deployment.
- No validation/tuning loop is included. Defaults are fixed, not optimized. If tuning
  factors, alpha or thresholds, introduce a separate validation split and keep test
  data untouched. No quality improvement over baselines is asserted.
- The legacy autoencoder was removed: it trained on all ratings, added heavyweight
  dependencies and mixed user-neighbor indices with movie IDs. Reintroducing it
  requires train-only fitting and a genuine item-scoring objective with regression tests.

## Repository layout

```text
src/movie_recommender/
  data.py          # MovieLens input validation and interaction splits
  models.py        # Explicit ID maps and six train-only scorers
  evaluation.py    # Observed-pair errors and full-catalog ranking
  cli.py           # Configuration, comparison and JSON reports
  reporting.py     # Shareable Markdown tables from actual evaluation results
examples/make_demo.py
main.py            # Thin backwards-compatible CLI launcher
tests/            # Offline unit, regression and CLI integration tests
.github/workflows/ci.yml
pyproject.toml
requirements-dev.lock
```

## Development and checks

```bash
ruff check .
ruff format --check .
pytest -q
movie-recommender --help
```

CI runs these commands on Python 3.11 and 3.12 without downloading data. Tests cover
split disjointness/determinism, held-out-rating invariance, non-contiguous IDs,
catalog alignment, unseen-item ranking, cold starts, tiny matrices, malformed data,
hand-calculated metrics, Latin-1 loading and repeatable CLI output. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and
[docs/validation.md](docs/validation.md) for locally executed checks and their scope.
