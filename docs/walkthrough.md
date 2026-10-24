# From a clean checkout to an inspectable recommendation experiment

This walkthrough is designed for classmates and developers who want to inspect
an offline recommender, not just see a claimed accuracy number.

## 1. Install in an isolated environment

```bash
git clone https://github.com/OmkarSolat-17/Movie-Recommendation-System.git
cd Movie-Recommendation-System
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
```

Python 3.11 or 3.12 is recommended. Installation downloads dependencies; the
synthetic experiment itself needs no internet access, API key, GPU or dataset download.

## 2. Run the synthetic example

```bash
python examples/make_demo.py --output data/demo
movie-recommender --data-dir data/demo --k 3 --user-id 101 \
  --output artifacts/demo.json --markdown artifacts/demo.md
```

Open `artifacts/demo.md` in a Markdown viewer. The full machine-readable report
remains in `artifacts/demo.json`. Both are generated from the same evaluation;
Markdown rounds metrics to six decimal places. See the checked-in
[example report](demo-report.md), produced by an actual local run.

The generator produces 8 users, 12 movies and 48 ratings with deliberately
non-contiguous IDs. The default split leaves 32 training and 16 test interactions.
Recommendations use original movie IDs and synthetic titles. Only four test users
have relevant held-out items, so ranking metrics average over four users, not eight.

## 3. Read the result critically

- Compare global mean and item mean before interpreting more complex models.
- In this toy run global mean has RMSE 1.463421, while hybrid has 1.771445:
  complexity does not automatically improve rating prediction.
- Hybrid's NDCG@3 of 0.815465 is **not a MovieLens benchmark**. It is a smoke-test
  measurement on a tiny artificial dataset and says nothing about real users.
- Popularity has no RMSE because its scores are counts rather than rating predictions.
  A dash is an unavailable metric, not a zero error.
- Inspect recommendation IDs against training-seen IDs. The tests verify their
  exclusion and that changing held-out ratings cannot change fitted predictions.

## 4. Evaluate real data separately

Follow the README's MovieLens 100K download instructions and dataset terms, then run:

```bash
movie-recommender --data-dir data/ml-100k --k 10 --user-id 2 \
  --output artifacts/movielens.json --markdown artifacts/movielens.md
```

Keep the JSON alongside the Markdown: it records package versions, input hashes,
configuration and unrounded measurements. Do not replace the synthetic example's
label with “MovieLens” unless those results actually came from MovieLens inputs.
No real-data experiment was executed for this walkthrough.

Per-user temporal splitting is not a global deployment-time cutoff. Dense matrices
are intended for small experiments; metadata availability is assumed. The README
explains candidate construction, metric denominators and further limitations.
Use a separate validation set if selecting hyperparameters rather than tuning on test scores.

## 5. Reproduce the checks and offer useful feedback

```bash
pytest -q
ruff check .
ruff format --check .
```

Useful feedback includes an installation failure with Python/OS versions, an ID
alignment counterexample, a metric calculation discrepancy, or a concrete protocol
improvement. Share a minimal synthetic example rather than redistributing data.

## Optional community post draft

Adapt this only after running the commands yourself. Follow the community's
self-promotion rules and do not repeatedly post or ask for reciprocal stars.

> I built a small, CPU-only movie-recommendation experiment suite to learn how to
> evaluate recommendations without feeding held-out ratings into the model.
> It compares six models and reports rating error plus full-catalog ranking metrics.
> There's an offline synthetic demo, a readable example report, and regression tests
> for non-contiguous IDs and train/test separation. The toy results are not a claim
> of real-world accuracy; I would especially welcome feedback on the evaluation
> protocol and first-time installation experience.
>
> Repository: https://github.com/OmkarSolat-17/Movie-Recommendation-System

Publishing this draft is up to you; no community post has been sent automatically.
Stars should be an independent reader decision, not a purchase, exchange or a test requirement.
