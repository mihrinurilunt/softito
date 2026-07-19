"""Production-style PySpark exploratory data analysis.

This script analyzes the Netflix Movies and TV Shows dataset. If a local CSV is
not provided, it creates a small demo dataset so the workflow can still be run
from the terminal. Spark transformations such as `filter` and `groupBy` are
lazy: Spark builds a logical plan and executes it only when an action such as
`count`, `show`, or `collect` is called.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


LOGGER = logging.getLogger(__name__)
ASSETS_DIR = Path("assets")
DEMO_DATA = [
    ("s1", "Movie", "Ocean Days", "Turkey", 2020, "TV-14", 94, "Documentaries"),
    ("s2", "TV Show", "City Lights", "United States", 2019, "TV-MA", 2, "Dramas"),
    ("s3", "Movie", "Hidden Roads", "India", 2021, "PG-13", 121, "Action"),
    ("s4", "TV Show", "Kitchen Stories", "United Kingdom", 2018, "TV-PG", 4, "Reality TV"),
    ("s5", "Movie", "Forest Echo", "United States", 2020, "PG", 88, "Children"),
    ("s6", "TV Show", "Code School", "Turkey", 2022, "TV-G", 1, "Education"),
]


def configure_logging() -> None:
    """Configure structured console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run PySpark EDA on Netflix data.")
    parser.add_argument("--input", type=Path, default=Path("data/netflix_titles.csv"))
    parser.add_argument("--app-name", default="NetflixPySparkEDA")
    return parser.parse_args()


def create_spark(app_name: str) -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def create_demo_dataframe(spark: SparkSession) -> DataFrame:
    """Create a small Netflix-like DataFrame for local demonstrations."""
    columns = ["show_id", "type", "title", "country", "release_year", "rating", "duration", "listed_in"]
    return spark.createDataFrame(DEMO_DATA, columns)


def load_dataset(spark: SparkSession, input_path: Path) -> DataFrame:
    """Load CSV data with inferred schema or create demo data if missing."""
    if input_path.exists():
        LOGGER.info("Loading dataset from %s", input_path)
        return spark.read.csv(str(input_path), header=True, inferSchema=True, multiLine=True, escape='"')
    LOGGER.warning("Dataset not found at %s. Using generated demo data.", input_path)
    return create_demo_dataframe(spark)


def clean_data(df: DataFrame) -> DataFrame:
    """Handle missing values and normalize selected fields."""
    return (
        df.fillna({"country": "Unknown", "rating": "Unrated", "listed_in": "Unknown"})
        .withColumn("release_year", F.col("release_year").cast("int"))
        .withColumn("content_type", F.col("type"))
    )


def run_summary_analysis(df: DataFrame) -> None:
    """Run basic EDA actions."""
    LOGGER.info("Total rows: %d", df.count())
    LOGGER.info("Schema:")
    df.printSchema()
    df.select("release_year").summary("count", "min", "max", "mean").show()
    df.filter(F.col("release_year") >= 2020).select("title", "type", "release_year").show(10, truncate=False)


def run_grouped_analysis(df: DataFrame) -> dict[str, DataFrame]:
    """Create reusable grouped analytical DataFrames."""
    by_year = df.groupBy("release_year").count().orderBy("release_year")
    by_country = df.groupBy("country").count().orderBy(F.desc("count")).limit(10)
    by_type = df.groupBy("content_type").count().orderBy(F.desc("count"))
    return {"year": by_year, "country": by_country, "type": by_type}


def run_sql_queries(df: DataFrame) -> None:
    """Run Spark SQL examples."""
    df.createOrReplaceTempView("netflix")
    queries = [
        "SELECT content_type, COUNT(*) AS total FROM netflix GROUP BY content_type",
        "SELECT release_year, COUNT(*) AS titles FROM netflix GROUP BY release_year ORDER BY release_year DESC LIMIT 10",
        "SELECT country, COUNT(*) AS titles FROM netflix GROUP BY country ORDER BY titles DESC LIMIT 10",
    ]
    for query in queries:
        LOGGER.info("SQL query: %s", query)
        df.sparkSession.sql(query).show(truncate=False)


def save_bar_chart(rows: Iterable, x_field: str, y_field: str, title: str, output_path: Path) -> None:
    """Save a Matplotlib bar chart from collected Spark rows."""
    collected = list(rows)
    if not collected:
        LOGGER.warning("No rows available for chart %s", output_path)
        return
    x_values = [str(row[x_field]) for row in collected]
    y_values = [row[y_field] for row in collected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    plt.bar(x_values, y_values, color="#0f766e")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_field)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Saved chart to %s", output_path)


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    start = time.perf_counter()
    spark = create_spark(args.app_name)

    try:
        df = clean_data(load_dataset(spark, args.input))
        run_summary_analysis(df)
        grouped = run_grouped_analysis(df)
        run_sql_queries(df)
        save_bar_chart(grouped["type"].collect(), "content_type", "count", "Netflix Titles by Type", ASSETS_DIR / "netflix_by_type.png")
        save_bar_chart(grouped["country"].collect(), "country", "count", "Top Countries by Title Count", ASSETS_DIR / "netflix_by_country.png")
        save_bar_chart(grouped["year"].orderBy(F.desc("release_year")).limit(15).collect(), "release_year", "count", "Recent Titles by Release Year", ASSETS_DIR / "netflix_by_year.png")
    except Exception:
        LOGGER.exception("PySpark data analysis failed.")
        raise
    finally:
        spark.stop()
        LOGGER.info("Execution time: %.2f seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
