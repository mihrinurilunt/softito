"""Hadoop-style MapReduce Word Count implemented in Python.

Hadoop is a distributed data processing ecosystem designed for large datasets.
HDFS stores files across a cluster, while MapReduce executes computation near
the stored data. A classic Word Count job demonstrates the execution model:

1. Map: emit `(word, 1)` pairs from text chunks.
2. Shuffle: group values that have the same word key.
3. Sort: order grouped keys for deterministic reducer input.
4. Reduce: sum counts for each word.

This script preserves the same execution logic without requiring a Hadoop
cluster. It can run locally for portfolio demonstration, and the same mapper
and reducer ideas can be adapted to Hadoop Streaming.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Iterator, List, Sequence, Tuple

import matplotlib.pyplot as plt
import requests


LOGGER = logging.getLogger(__name__)
DEFAULT_TEXT_URL = "https://www.gutenberg.org/files/11/11-0.txt"
DEFAULT_SAMPLE_TEXT = """
Alice was beginning to get very tired of sitting by her sister on the bank.
The rabbit-hole went straight on like a tunnel for some way, and then dipped
suddenly down. Alice had not a moment to think about stopping herself before
she found herself falling down a very deep well.
"""
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "herself", "his", "i", "in", "is", "it",
    "its", "of", "on", "or", "she", "that", "the", "their", "then", "there",
    "they", "this", "to", "was", "were", "with", "you", "your",
}


def configure_logging() -> None:
    """Configure consistent CLI logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate Hadoop Word Count.")
    parser.add_argument("--input", type=Path, help="Optional local text file.")
    parser.add_argument("--url", default=DEFAULT_TEXT_URL, help="Text URL to download when --input is not provided.")
    parser.add_argument("--output", type=Path, default=Path("output/wordcount.csv"))
    parser.add_argument("--chart", type=Path, default=Path("assets/wordcount_results.png"))
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_text(input_path: Path | None, url: str) -> str:
    """Load text from a local file, URL, or bundled fallback sample."""
    if input_path:
        LOGGER.info("Reading local text file: %s", input_path)
        return input_path.read_text(encoding="utf-8")

    try:
        LOGGER.info("Downloading public-domain text from %s", url)
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        LOGGER.warning("Download failed: %s. Falling back to bundled sample text.", exc)
        return DEFAULT_SAMPLE_TEXT


def tokenize(text: str) -> List[str]:
    """Normalize text and remove punctuation, numbers, and stop words."""
    punctuation_table = str.maketrans({char: " " for char in string.punctuation})
    normalized = text.lower().translate(punctuation_table)
    words = re.findall(r"[a-z]+", normalized)
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def map_phase(words: Iterable[str]) -> Iterator[Tuple[str, int]]:
    """Emit `(word, 1)` pairs, equivalent to the mapper output."""
    for word in words:
        yield word, 1


def shuffle_phase(mapped_pairs: Iterable[Tuple[str, int]]) -> DefaultDict[str, List[int]]:
    """Group mapped values by key, equivalent to the MapReduce shuffle."""
    grouped: DefaultDict[str, List[int]] = defaultdict(list)
    for word, count in mapped_pairs:
        grouped[word].append(count)
    return grouped


def sort_phase(grouped_pairs: DefaultDict[str, List[int]]) -> List[Tuple[str, List[int]]]:
    """Sort grouped keys before reduction for deterministic output."""
    return sorted(grouped_pairs.items(), key=lambda item: item[0])


def reduce_phase(sorted_pairs: Sequence[Tuple[str, List[int]]]) -> Counter[str]:
    """Aggregate grouped counts, equivalent to reducer output."""
    return Counter({word: sum(counts) for word, counts in sorted_pairs})


def save_counts(counts: Counter[str], output_path: Path) -> None:
    """Save word counts to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["word", "count"])
        writer.writerows(counts.most_common())
    LOGGER.info("Saved word counts to %s", output_path)


def save_bar_chart(top_words: Sequence[Tuple[str, int]], chart_path: Path) -> None:
    """Save a bar chart of the most frequent words."""
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    words, counts = zip(*top_words)
    plt.figure(figsize=(12, 7))
    plt.bar(words, counts, color="#2563eb")
    plt.title("Top Word Frequencies")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()
    LOGGER.info("Saved chart to %s", chart_path)


def run_wordcount(text: str) -> Counter[str]:
    """Execute the complete Map -> Shuffle -> Sort -> Reduce workflow."""
    words = tokenize(text)
    LOGGER.info("Tokenized %d meaningful words.", len(words))
    mapped = map_phase(words)
    shuffled = shuffle_phase(mapped)
    sorted_pairs = sort_phase(shuffled)
    return reduce_phase(sorted_pairs)


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()

    try:
        text = load_text(args.input, args.url)
        counts = run_wordcount(text)
        top_words = counts.most_common(args.top_n)
        for word, count in top_words:
            LOGGER.info("%-16s %s", word, count)
        save_counts(counts, args.output)
        save_bar_chart(top_words, args.chart)
    except Exception:
        LOGGER.exception("Word Count job failed.")
        raise


if __name__ == "__main__":
    main()
