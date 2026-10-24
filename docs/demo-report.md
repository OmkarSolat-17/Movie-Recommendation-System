# Recommendation experiment

> Dataset identity is determined by the input hashes below. Synthetic inputs
> are smoke tests, not evidence of real-world recommendation quality.

Split: **temporal**; seed: **42**; training interactions: **32**; held-out interactions: **16**.

| Model | RMSE | MAE | Precision@3 | Recall@3 | NDCG@3 | Ranking users |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| global_mean | 1.463421 | 1.261719 | 0.333333 | 0.750000 | 0.545395 | 4 |
| item_mean | 2.100760 | 1.895833 | 0.000000 | 0.000000 | 0.000000 | 4 |
| popularity | — | — | 0.083333 | 0.250000 | 0.125000 | 4 |
| svd | 2.263846 | 1.750000 | 0.333333 | 0.750000 | 0.545395 | 4 |
| content | 2.025271 | 1.719443 | 0.416667 | 1.000000 | 0.815465 | 4 |
| hybrid | 1.771445 | 1.352351 | 0.416667 | 1.000000 | 0.815465 | 4 |

Missing values (—) are not zero. Popularity counts have no rating error.
Ranking averages include only users with relevant held-out items; precision
divides by K even when fewer candidates exist. See README for the full protocol.

## Recommendations for user 101

### global_mean

| Movie ID | Title |
| ---: | --- |
| 17 | Synthetic movie 1 |
| 31 | Synthetic movie 3 |
| 45 | Synthetic movie 5 |

### item_mean

| Movie ID | Title |
| ---: | --- |
| 87 | Synthetic movie 11 |
| 31 | Synthetic movie 3 |
| 80 | Synthetic movie 10 |

### popularity

| Movie ID | Title |
| ---: | --- |
| 59 | Synthetic movie 7 |
| 45 | Synthetic movie 5 |
| 66 | Synthetic movie 8 |

### svd

| Movie ID | Title |
| ---: | --- |
| 17 | Synthetic movie 1 |
| 31 | Synthetic movie 3 |
| 45 | Synthetic movie 5 |

### content

| Movie ID | Title |
| ---: | --- |
| 80 | Synthetic movie 10 |
| 66 | Synthetic movie 8 |
| 17 | Synthetic movie 1 |

### hybrid

| Movie ID | Title |
| ---: | --- |
| 80 | Synthetic movie 10 |
| 66 | Synthetic movie 8 |
| 17 | Synthetic movie 1 |


## Reproducibility

| Setting | Value |
| --- | --- |
| data_dir | data/demo |
| output | artifacts/results.json |
| markdown | docs/demo-report.md |
| seed | 42 |
| split | temporal |
| test_fraction | 0.2 |
| k | 3 |
| factors | 20 |
| alpha | 0.7 |
| relevance_threshold | 4 |
| user_id | 101 |
| python | 3.11.2 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| SHA-256 u.data | 90ad4ba9218a5a887837c6573525c1e5f965c9aac0f885ac9adf726f23111d7d |
| SHA-256 u.item | 8646bde2a48759eb3975a09113ed12931cfd0438badfe403f5ab19e00caf0ea3 |

Fixed hyperparameters; no quality improvement over baselines is asserted.
