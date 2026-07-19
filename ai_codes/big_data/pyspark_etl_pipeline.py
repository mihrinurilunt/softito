"""Production-inspired PySpark ETL pipeline for Online Retail data.

The pipeline extracts transactions from CSV, cleans missing and duplicate rows,
casts invoice dates, creates calculated sales columns, aggregates business
metrics, saves curated data as Parquet, compares CSV and Parquet sizes, and
exports portfolio charts to the assets folder.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path("output")
ASSETS_DIR = Path("assets")
DEMO_ROWS = [
    ("536365", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 6, "12/1/2010 8:26", 2.55, "17850", "United Kingdom"),
    ("536365", "71053", "WHITE METAL LANTERN", 6, "12/1/2010 8:26", 3.39, "17850", "United Kingdom"),
    ("536366", "22633", "HAND WARMER UNION JACK", 6, "12/1/2010 8:28", 1.85, "17850", "United Kingdom"),
    ("536367", "84879", "ASSORTED COLOUR BIRD ORNAMENT", 32, "12/1/2010 8:34", 1.69, "13047", "United Kingdom"),
    ("536368", "22960", "JAM MAKING SET WITH JARS", 6, "12/1/2010 8:34", 4.25, "13047", "United Kingdom"),
    ("536369", "21756", "BATH BUILDING BLOCK WORD", 3, "12/1/2010 8:35", 5.95, "13047", "France"),
]
RETAIL_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run a PySpark Online Retail ETL pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/online_retail.csv"))
    parser.add_argument("--parquet-output", type=Path, default=OUTPUT_DIR / "online_retail_parquet")
    return parser.parse_args()


def create_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.appName("OnlineRetailETL")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def extract_data(spark: SparkSession, input_path: Path) -> DataFrame:
    """Extract data from CSV or create a local demo DataFrame."""
    if input_path.exists():
        LOGGER.info("Extracting Online Retail data from %s", input_path)
        return spark.read.csv(str(input_path), header=True, inferSchema=True)
    LOGGER.warning("Dataset not found at %s. Using generated demo data.", input_path)
    return spark.createDataFrame(DEMO_ROWS, RETAIL_COLUMNS)


def transform_data(df: DataFrame) -> DataFrame:
    """Clean data and create business-friendly calculated columns."""
    return (
        df.dropDuplicates()
        .dropna(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"])
        .withColumn("Quantity", F.col("Quantity").cast("int"))
        .withColumn("UnitPrice", F.col("UnitPrice").cast("double"))
        .filter((F.col("Quantity") > 0) & (F.col("UnitPrice") > 0))
        .withColumn("InvoiceTimestamp", F.to_timestamp("InvoiceDate", "M/d/yyyy H:mm"))
        .withColumn("InvoiceDateOnly", F.to_date("InvoiceTimestamp"))
        .withColumn("SalesAmount", F.round(F.col("Quantity") * F.col("UnitPrice"), 2))
    )


def aggregate_sales(df: DataFrame) -> dict[str, DataFrame]:
    """Create sales aggregations for reporting."""
    by_country = df.groupBy("Country").agg(F.round(F.sum("SalesAmount"), 2).alias("total_sales")).orderBy(F.desc("total_sales"))
    by_date = df.groupBy("InvoiceDateOnly").agg(F.round(F.sum("SalesAmount"), 2).alias("daily_sales")).orderBy("InvoiceDateOnly")
    top_products = df.groupBy("Description").agg(F.round(F.sum("SalesAmount"), 2).alias("product_sales")).orderBy(F.desc("product_sales")).limit(10)
    return {"country": by_country, "date": by_date, "products": top_products}


def save_parquet(df: DataFrame, output_path: Path) -> None:
    """Save processed data in Parquet format."""
    df.write.mode("overwrite").parquet(str(output_path))
    LOGGER.info("Saved processed Parquet data to %s", output_path)


def directory_size(path: Path) -> int:
    """Return total size for a file or directory."""
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return 0


def compare_storage_sizes(csv_path: Path, parquet_path: Path) -> None:
    """Log CSV and Parquet storage sizes."""
    csv_size = directory_size(csv_path)
    parquet_size = directory_size(parquet_path)
    LOGGER.info("CSV size: %.2f KB", csv_size / 1024)
    LOGGER.info("Parquet size: %.2f KB", parquet_size / 1024)


def save_bar_chart(df: DataFrame, x_col: str, y_col: str, title: str, output_path: Path) -> None:
    """Collect a small aggregate and save a bar chart."""
    rows = df.limit(15).collect()
    if not rows:
        LOGGER.warning("No rows available for chart %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [str(row[x_col]) for row in rows]
    y_values = [row[y_col] for row in rows]
    plt.figure(figsize=(12, 6))
    plt.bar(x_values, y_values, color="#7c3aed")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Saved chart to %s", output_path)


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    start = time.perf_counter()
    spark = create_spark()

    try:
        raw_df = extract_data(spark, args.input)
        processed_df = transform_data(raw_df)
        processed_df.cache()
        LOGGER.info("Processed rows: %d", processed_df.count())
        aggregates = aggregate_sales(processed_df)
        save_parquet(processed_df, args.parquet_output)
        compare_storage_sizes(args.input, args.parquet_output)
        save_bar_chart(aggregates["country"], "Country", "total_sales", "Sales by Country", ASSETS_DIR / "retail_sales_by_country.png")
        save_bar_chart(aggregates["products"], "Description", "product_sales", "Top Product Sales", ASSETS_DIR / "retail_top_products.png")
    except Exception:
        LOGGER.exception("PySpark ETL pipeline failed.")
        raise
    finally:
        spark.stop()
        LOGGER.info("Execution time: %.2f seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
