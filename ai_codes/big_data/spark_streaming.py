"""Spark Structured Streaming demo with simulated IoT sensor data.

The script uses Spark's built-in `rate` source to simulate an event stream.
Each generated row becomes an IoT sensor measurement with event time, sensor
ID, temperature, and an anomaly flag. The streaming query computes windowed
rolling averages, applies watermarking for late data handling, and prints
aggregated results to the console.
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
ASSETS_DIR = Path("assets")


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run a Spark Structured Streaming IoT demo.")
    parser.add_argument("--duration", type=int, default=30, help="Seconds to keep the stream running.")
    parser.add_argument("--rows-per-second", type=int, default=5)
    return parser.parse_args()


def create_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.appName("IoTStructuredStreaming")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def create_sensor_stream(spark: SparkSession, rows_per_second: int) -> DataFrame:
    """Generate a simulated IoT stream from Spark's rate source."""
    raw_stream = spark.readStream.format("rate").option("rowsPerSecond", rows_per_second).load()
    return (
        raw_stream.withColumn("sensor_id", F.concat(F.lit("sensor_"), (F.col("value") % 5).cast("string")))
        .withColumn("temperature", F.round(F.lit(22) + (F.rand(seed=42) * 18) + F.when(F.col("value") % 23 == 0, 20).otherwise(0), 2))
        .withColumn("event_time", F.col("timestamp"))
        .withColumn("is_abnormal", F.col("temperature") > 38)
        .select("event_time", "sensor_id", "temperature", "is_abnormal")
    )


def compute_windowed_averages(stream_df: DataFrame) -> DataFrame:
    """Apply watermarking and compute rolling window aggregations."""
    return (
        stream_df.withWatermark("event_time", "30 seconds")
        .groupBy(F.window("event_time", "20 seconds", "10 seconds"), "sensor_id")
        .agg(
            F.round(F.avg("temperature"), 2).alias("avg_temperature"),
            F.max(F.col("is_abnormal").cast("int")).alias("has_abnormal_reading"),
            F.count("*").alias("events"),
        )
        .orderBy("window", "sensor_id")
    )


def save_sample_visualization(output_path: Path) -> None:
    """Save a static example chart that explains the streaming anomaly pattern."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = list(range(1, 31))
    temperatures = [23, 24, 25, 24, 26, 27, 25, 24, 41, 40, 28, 27, 26, 25, 24, 23, 39, 42, 27, 26, 25, 24, 23, 24, 25, 26, 27, 28, 29, 30]
    plt.figure(figsize=(11, 6))
    plt.plot(timestamps, temperatures, marker="o", linewidth=2, color="#dc2626")
    plt.axhline(38, linestyle="--", color="#111827", label="Anomaly threshold")
    plt.title("Simulated IoT Temperature Stream")
    plt.xlabel("Event Number")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    LOGGER.info("Saved streaming visualization to %s", output_path)


def run_stream(query_df: DataFrame, duration_seconds: int) -> None:
    """Start the streaming query and stop it after a fixed duration."""
    query = (
        query_df.writeStream.outputMode("complete")
        .format("console")
        .option("truncate", "false")
        .option("numRows", "50")
        .start()
    )
    try:
        LOGGER.info("Streaming for %d seconds...", duration_seconds)
        time.sleep(duration_seconds)
    finally:
        query.stop()
        LOGGER.info("Streaming query stopped.")


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    spark = create_spark()

    try:
        save_sample_visualization(ASSETS_DIR / "streaming_temperature_demo.png")
        sensor_stream = create_sensor_stream(spark, args.rows_per_second)
        rolling_averages = compute_windowed_averages(sensor_stream)
        run_stream(rolling_averages, args.duration)
    except Exception:
        LOGGER.exception("Structured Streaming demo failed.")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
