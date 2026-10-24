# Local validation record

Executed on 2026-10-23 with Python 3.11.2 and the pinned dependencies in
`requirements-dev.lock`. These are local results, not a claim that hosted CI passed.

| Command | Observed result |
| --- | --- |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 11 files already formatted |
| `pytest -q` | 22 passed in 1.08s |
| `movie-recommender --help` | Exit 0; CLI options displayed |
| `python -m pip check` | No broken requirements found |
| Synthetic CLI experiment below | Exit 0; all six models produced finite results |

## Synthetic end-to-end smoke test

```bash
python examples/make_demo.py
movie-recommender --data-dir data/demo --k 3 --user-id 101
```

8 users, 12 movies, 48 interactions; 32 training and 16 held out using the default
per-user temporal split. Four test users have a relevant held-out item. Results
below are rounded to six decimal places; the CLI writes full precision JSON with
input hashes and package versions.

| Model | RMSE | MAE | Precision@3 | Recall@3 | NDCG@3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global mean | 1.463421 | 1.261719 | 0.333333 | 0.750000 | 0.545395 |
| Item mean | 2.100760 | 1.895833 | 0.000000 | 0.000000 | 0.000000 |
| Popularity | — | — | 0.083333 | 0.250000 | 0.125000 |
| SVD | 2.263846 | 1.750000 | 0.333333 | 0.750000 | 0.545395 |
| Content | 2.025271 | 1.719443 | 0.416667 | 1.000000 | 0.815465 |
| Hybrid | 1.771445 | 1.352351 | 0.416667 | 1.000000 | 0.815465 |

**These are synthetic functional checks, not MovieLens accuracy evidence.** The
hybrid does not beat global mean on RMSE even on this toy fixture. No real
MovieLens benchmark was run in this environment. Python 3.12 is configured for
hosted CI but was not executed locally. Real-data performance, scalability and
production recommendation quality remain unverified.

## Shareable-report follow-up (2026-10-24)

Executed locally with Python 3.11 and the pinned dependencies:

- `pytest -q`: **34 passed in 1.12s**.
- `ruff check .`: all checks passed.
- `ruff format --check .`: 17 files already formatted.
- `python -m pip check`: no broken requirements.
- `git diff --check`: passed.
- Synthetic generation and CLI with `--markdown docs/demo-report.md`: exit 0;
  produced the checked-in example report from 32 training / 16 test interactions.

Reporting tests check missing versus zero metrics, external-title escaping, and
Markdown generation through the real CLI. No new MovieLens benchmark or local
Python 3.12 run was performed for this follow-up. Hosted CI results are separate.
