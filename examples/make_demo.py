"""Create synthetic MovieLens-format smoke-test data, not a benchmark dataset."""

import argparse
from pathlib import Path

GENRES = 19


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    movies, ratings = [], []
    for item in range(12):
        flags = [str(int(i == item % 4 + 1)) for i in range(GENRES)]
        movies.append("|".join([str(10 + item * 7), f"Synthetic movie {item}", "", "", "", *flags]))
    for user in range(8):
        for event in range(6):
            item = (user + event * 2) % 12
            ratings.append(
                f"{101 + user * 13}\t{10 + item * 7}\t{1 + (user + event) % 5}\t{event + 1}"
            )
    (args.output / "u.item").write_text("\n".join(movies) + "\n", encoding="latin-1")
    (args.output / "u.data").write_text("\n".join(ratings) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
