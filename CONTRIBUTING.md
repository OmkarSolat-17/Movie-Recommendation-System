# Contributing

1. Follow the isolated setup in README.md. Keep private data, large artifacts,
   generated reports and environments out of version control.
2. Add a regression test for each behavior change. Fit models only on training
   interactions; never use evaluation ratings as features or tune on test metrics.
3. Preserve original IDs at public interfaces and explicit mappings internally.
4. Run `ruff check .`, `ruff format --check .` and `pytest -q` before opening a PR.
   State which commands actually ran and which datasets were used. Do not label
   synthetic smoke-test scores as MovieLens results.
5. Explain protocol or metric denominator changes in documentation. Use a separate
   validation split for future hyperparameter searches.

Dependency updates should regenerate requirements-dev.lock from a clean virtual
environment with `pip freeze --exclude-editable`, then pass the CI Python matrix.
No repository license was supplied by the original project; this change does not
invent one. Dataset terms are independent of code permissions.
