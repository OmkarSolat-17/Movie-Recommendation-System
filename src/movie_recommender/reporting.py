"""Portable Markdown summaries of evaluated experiment reports."""


def _cell(value):
    """Keep external movie titles from introducing table rows or HTML markup."""
    import html

    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def render_markdown(report):
    """Render existing measurements without rerunning models or inventing scores."""
    config = report["config"]
    k = config["k"]
    lines = [
        "# Recommendation experiment",
        "",
        "> Dataset identity is determined by the input hashes below. Synthetic inputs",
        "> are smoke tests, not evidence of real-world recommendation quality.",
        "",
        (
            f"Split: **{_cell(config['split'])}**; seed: **{config['seed']}**; "
            f"training interactions: **{report['train_interactions']}**; "
            f"held-out interactions: **{report['test_interactions']}**."
        ),
        "",
        f"| Model | RMSE | MAE | Precision@{k} | Recall@{k} | NDCG@{k} | Ranking users |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in report["models"].items():
        cells = []
        for key in ["rmse", "mae", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]:
            value = metrics.get(key)
            cells.append("—" if value is None else f"{value:.6f}")
        lines.append(f"| {_cell(name)} | " + " | ".join(cells) + f" | {metrics['ranking_users']} |")
    lines.extend(
        [
            "",
            "Missing values (—) are not zero. Popularity counts have no rating error.",
            "Ranking averages include only users with relevant held-out items; precision",
            "divides by K even when fewer candidates exist. See README for the full protocol.",
        ]
    )
    if config.get("user_id") is not None:
        lines.extend(["", f"## Recommendations for user {config['user_id']}", ""])
        for name, metrics in report["models"].items():
            lines.extend([f"### {_cell(name)}", "", "| Movie ID | Title |", "| ---: | --- |"])
            for movie in metrics.get("recommendations", []):
                lines.append(f"| {movie['movie_id']} | {_cell(movie['title'])} |")
            lines.append("")
    lines.extend(["", "## Reproducibility", "", "| Setting | Value |", "| --- | --- |"])
    for key, value in config.items():
        lines.append(f"| {_cell(key)} | {_cell(value)} |")
    for name, value in report["environment"].items():
        lines.append(f"| {_cell(name)} | {_cell(value)} |")
    for name, value in report["data_sha256"].items():
        lines.append(f"| SHA-256 {_cell(name)} | {_cell(value)} |")
    lines.extend(
        ["", "Fixed hyperparameters; no quality improvement over baselines is asserted.", ""]
    )
    return "\n".join(lines)
