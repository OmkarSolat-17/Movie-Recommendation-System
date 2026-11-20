<h1 align="center">🎬 Movie Recommendation System</h1>

<p align="center">
  <img src="https://github.com/OmkarSolat-17/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Tests-Passing-2EA44F?style=flat-square&logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/Models-6-blueviolet?style=flat-square" alt="Models">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Metrics-RMSE%20%7C%20NDCG%40K%20%7C%20P%40K-orange?style=flat-square" alt="Metrics">
</p>

<p align="center">
  <b>Compare 6 recommendation algorithms under a leakage-safe evaluation protocol.</b><br>
  SVD · TF-IDF · Hybrid · Popularity · Item Mean · Global Mean<br>
  <sub>Reproducible. CPU-only. No data download needed to try the demo.</sub>
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/⚡_Quick_Start-0A66C2?style=for-the-badge" alt="Quick Start"></a>
  <a href="#-results"><img src="https://img.shields.io/badge/📊_See_Results-2EA44F?style=for-the-badge" alt="Results"></a>
  <a href="#-how-it-works"><img src="https://img.shields.io/badge/🔍_How_It_Works-FFA116?style=for-the-badge" alt="How It Works"></a>
</p>

---

## 📊 Results

> ⚠️ These are **synthetic smoke-test results** — not a real benchmark. Run with MovieLens 100K for meaningful numbers. The point is that all 6 models are evaluated identically, on the same split, with no data leakage.

| Model | RMSE | MAE | Precision@3 | Recall@3 | NDCG@3 |
|:------|----:|----:|------------:|---------:|-------:|
| Global Mean | 1.463 | 1.262 | 0.333 | 0.750 | 0.545 |
| Item Mean | 2.101 | 1.896 | 0.000 | 0.000 | 0.000 |
| Popularity | — | — | 0.083 | 0.250 | 0.125 |
| SVD | 2.264 | 1.750 | 0.333 | 0.750 | 0.545 |
| Content | 2.025 | 1.719 | 0.417 | 1.000 | 0.815 |
| **Hybrid** | **1.771** | **1.352** | **0.417** | **1.000** | **0.815** |

<sub>Missing values (—) are not zero. Popularity counts have no rating error. Full report: [docs/demo-report.md](docs/demo-report.md)</sub>

---

## 🔍 How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                   Movie Recommendation Pipeline                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐    ┌──────────────┐    ┌───────────────────┐     │
│  │ MovieLens │───▶│  data.py     │───▶│  Temporal Split   │     │
│  │   100K    │    │  Validation   │    │  Train / Test     │     │
│  └───────────┘    └──────────────┘    └────────┬──────────┘     │
│                                                │                 │
│               ┌────────────────────────────────┼───────┐        │
│               ▼                                ▼       ▼        │
│  ┌──────────────┐  ┌───────────┐  ┌──────┐  ┌────────┐         │
│  │  Global Mean │  │ Item Mean │  │ SVD  │  │Content │         │
│  └──────┬───────┘  └─────┬─────┘  └──┬───┘  └───┬────┘         │
│         │                │           │           │              │
│         └────────────────┴─────┬─────┴───────────┘              │
│                                ▼                                 │
│                     ┌───────────────────┐                       │
│                     │   Hybrid Model    │                       │
│                     │ α·SVD + (1-α)·CT │                       │
│                     └────────┬──────────┘                       │
│                              ▼                                   │
│                ┌───────────────────────────┐                    │
│                │     evaluation.py         │                    │
│                │  RMSE · MAE · P@K · NDCG  │                    │
│                └─────────────┬─────────────┘                    │
│                              ▼                                   │
│                ┌───────────────────────────┐                    │
│                │      reporting.py         │                    │
│                │    JSON + Markdown        │                    │
│                └───────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
```

**The Pipeline in 5 Steps:**

1. 📥 **Load & Validate** — Parse MovieLens `u.data` and `u.item`, reject malformed entries, validate genre flags
2. ✂️ **Split** — Temporal split (train on past, test on future) or random with seeded generator
3. 🧠 **Fit** — Train 6 models on **training data only** — no leakage from test ratings
4. 📊 **Evaluate** — RMSE/MAE for rating prediction; Precision@K, Recall@K, NDCG@K for full-catalog ranking
5. 📝 **Report** — Export JSON (full precision) + Markdown (human-readable comparison table)

---

## 🚀 Quick Start

Python **3.11 or newer**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
pytest -q                          # Run tests
```

### Run the Demo (No Data Download Needed)

```bash
python examples/make_demo.py --output data/demo
movie-recommender --data-dir data/demo --k 3 --user-id 101 \
  --markdown artifacts/demo.md
```

This generates a tiny, deterministic **synthetic smoke test**. The CLI prints JSON and writes `artifacts/results.json`. Repeating a run with the same environment, data, and arguments produces the same report.

### Run with MovieLens 100K

Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and extract it. Point to the directory containing `u.data` and `u.item`:

```bash
movie-recommender --data-dir data/ml-100k --split temporal \
  --test-fraction 0.2 --seed 42 --factors 20 --alpha 0.7 \
  --k 10 --relevance-threshold 4 --user-id 2 \
  --output artifacts/movielens.json

# Legacy entry point:
python main.py --data-dir data/ml-100k
```

---

## 🧠 Models

Every run evaluates **all 6 models** on the same split and candidate catalog:

| Model | Approach | Rating Error? |
|:------|:---------|:------------:|
| **Global Mean** | Training global rating mean | ✅ |
| **Item Mean** | Training mean per movie, global fallback | ✅ |
| **Popularity** | Training interaction count per movie | ❌ (counts ≠ ratings) |
| **SVD** | Truncated SVD of zero-filled training matrix, clipped to [1, 5] | ✅ |
| **Content** | Cosine similarity of rating-weighted user genre profile × movie TF-IDF | ✅ |
| **Hybrid** | `α × SVD + (1-α) × Content` (default α = 0.7) | ✅ |

**Key design decisions:**
- IDs are dictionary-mapped independently for users and movies — never inferred with `id - 1`
- Non-contiguous IDs and shuffled catalog order are supported
- Unknown users fall back to training item means; unknown items to global mean
- Training-seen items are always excluded from recommendations
- Score ties use ascending movie ID

---

## 📐 Evaluation Protocol

### Splitting

1. **Temporal split** (default): hold out the latest `ceil(n × fraction)` events per user, keeping at least one training event. Single-event users are training-only. Timestamp ties break by movie ID.
2. **Random split**: stable ordering with a seeded generator.

### Fitting

All rating-derived state, SVD factors, and content profiles are fit from **training ratings only**. Genre TF-IDF uses the supplied catalog (a transductive metadata assumption, not a fit on held-out ratings). SVD never transforms a test matrix. Evaluation rejects overlapping train/test user/movie pairs.

### Rating Metrics

RMSE and MAE use only observed held-out pairs and the actual named model's predictions. There is no mislabeled "hybrid RMSE" from SVD outputs.

### Ranking Metrics

Rank the **entire catalog minus training-seen items**, without negative sampling. Held-out ratings ≥ the relevance threshold (default 4) are relevant. Macro-average Precision@K, Recall@K, and binary NDCG@K across test users with at least one relevant item.

- **Precision** divides by K even if fewer candidates exist
- **Recall** divides by the user's held-out relevant count
- **Ideal DCG** uses `min(K, relevant_count)`

---

## ⚠️ Interpretation & Limitations

- Per-user temporal splitting is **not** a global time cutoff — another user's later training activity can precede a particular user's test event. Use a global cutoff for deployment-time studies.
- Zero-filled SVD treats missing ratings as zero — a transparent baseline, not observed-only matrix factorization.
- Content score scaling is heuristic, not calibrated.
- Dense user-item score matrices are used for MovieLens 100K-scale experiments — not intended for million-user production catalogs.
- Unobserved items are not known negatives. Offline ranking scores reflect exposure bias.
- **No validation/tuning loop is included.** Defaults are fixed, not optimized. If tuning factors, alpha, or thresholds, introduce a separate validation split and keep test data untouched.
- **No quality improvement over baselines is asserted.**

---

## 📂 Repository Layout

```
src/movie_recommender/
  data.py          # MovieLens input validation and interaction splits
  models.py        # Explicit ID maps and six train-only scorers
  evaluation.py    # Observed-pair errors and full-catalog ranking
  cli.py           # Configuration, comparison and JSON reports
  reporting.py     # Shareable Markdown tables from actual evaluation results

examples/
  make_demo.py     # Generates synthetic smoke-test data

docs/
  walkthrough.md   # Step-by-step guided walkthrough
  demo-report.md   # Actual generated example report
  validation.md    # Locally executed checks and their scope

tests/
  conftest.py      # Shared fixtures
  test_pipeline.py # Split, model, evaluation, and CLI tests
  test_genre_validation.py  # Genre flag validation tests
  test_reporting.py         # Markdown reporting tests

main.py            # Thin backwards-compatible CLI launcher
.github/workflows/ci.yml
pyproject.toml
requirements-dev.lock
```

---

## 🛠️ Development & Checks

```bash
ruff check .
ruff format --check .
pytest -q
movie-recommender --help
```

CI runs these commands on **Python 3.11 and 3.12** without downloading data. Tests cover:

- ✅ Split disjointness and determinism
- ✅ Held-out-rating invariance
- ✅ Non-contiguous IDs
- ✅ Catalog alignment
- ✅ Unseen-item ranking
- ✅ Cold starts and tiny matrices
- ✅ Malformed data rejection
- ✅ Hand-calculated metric verification
- ✅ Latin-1 loading
- ✅ Repeatable CLI output

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidance and [docs/validation.md](docs/validation.md) for locally executed checks.

---

## 📚 Further Reading

- [📖 Guided Walkthrough](docs/walkthrough.md) — how to interpret and share results responsibly
- [📊 Demo Report](docs/demo-report.md) — actual generated example with reproducibility info
- [✅ Validation Notes](docs/validation.md) — what tests cover and their scope

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/OmkarSolat-17">Omkar Solat</a> · B.Tech CSE (AI/ML)</sub>
</p>
