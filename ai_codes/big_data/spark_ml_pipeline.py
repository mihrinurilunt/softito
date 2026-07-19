"""Spark ML classification pipeline for the Adult Income dataset.

The Adult Income task predicts whether a person earns more than 50K per year.
This script demonstrates a complete Spark ML workflow: loading data, cleaning
missing values, indexing categorical features, one-hot encoding, assembling a
feature vector, training Logistic Regression, evaluating metrics, saving a
confusion matrix, and persisting the trained Spark ML model.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


LOGGER = logging.getLogger(__name__)
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models/adult_income_logistic_regression")
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
DEMO_ROWS = [
    (39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States", "<=50K"),
    (50, "Self-emp-not-inc", 83311, "Bachelors", 13, "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", 0, 0, 13, "United-States", "<=50K"),
    (38, "Private", 215646, "HS-grad", 9, "Divorced", "Handlers-cleaners", "Not-in-family", "White", "Male", 0, 0, 40, "United-States", "<=50K"),
    (53, "Private", 234721, "11th", 7, "Married-civ-spouse", "Handlers-cleaners", "Husband", "Black", "Male", 0, 0, 40, "United-States", "<=50K"),
    (28, "Private", 338409, "Bachelors", 13, "Married-civ-spouse", "Prof-specialty", "Wife", "Black", "Female", 0, 0, 40, "Cuba", ">50K"),
    (37, "Private", 284582, "Masters", 14, "Married-civ-spouse", "Exec-managerial", "Wife", "White", "Female", 0, 0, 40, "United-States", ">50K"),
    (49, "Private", 160187, "9th", 5, "Married-spouse-absent", "Other-service", "Not-in-family", "Black", "Female", 0, 0, 16, "Jamaica", "<=50K"),
    (52, "Self-emp-not-inc", 209642, "HS-grad", 9, "Married-civ-spouse", "Exec-managerial", "Husband", "White", "Male", 0, 0, 45, "United-States", ">50K"),
]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Train a Spark ML Adult Income classifier.")
    parser.add_argument("--input", type=Path, default=Path("data/adult.csv"))
    parser.add_argument("--metrics-output", type=Path, default=OUTPUT_DIR / "spark_ml_metrics.json")
    parser.add_argument("--model-output", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def create_spark() -> SparkSession:
    """Create a local SparkSession."""
    return (
        SparkSession.builder.appName("AdultIncomeSparkML")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def load_dataset(spark: SparkSession, input_path: Path) -> DataFrame:
    """Load Adult Income CSV or create a demo dataset."""
    if input_path.exists():
        LOGGER.info("Loading Adult Income data from %s", input_path)
        return spark.read.csv(str(input_path), header=True, inferSchema=True)
    LOGGER.warning("Dataset not found at %s. Using generated demo data.", input_path)
    return spark.createDataFrame(DEMO_ROWS, ADULT_COLUMNS)


def clean_dataset(df: DataFrame) -> DataFrame:
    """Replace missing markers, drop incomplete rows, and create a binary label."""
    cleaned = df.replace("?", None).dropna()
    return cleaned.withColumn("label", F.when(F.col("income").contains(">50K"), 1.0).otherwise(0.0))


def build_pipeline(df: DataFrame) -> Pipeline:
    """Build a Spark ML pipeline with categorical encoding and Logistic Regression."""
    categorical_cols = [
        field.name for field in df.schema.fields
        if field.dataType.simpleString() == "string" and field.name != "income"
    ]
    numeric_cols = [
        field.name for field in df.schema.fields
        if field.dataType.simpleString() in {"int", "bigint", "double"} and field.name != "label"
    ]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        for col in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec")
        for col in categorical_cols
    ]
    assembler = VectorAssembler(
        inputCols=numeric_cols + [f"{col}_vec" for col in categorical_cols],
        outputCol="features",
        handleInvalid="keep",
    )
    classifier = LogisticRegression(featuresCol="features", labelCol="label", maxIter=25)
    return Pipeline(stages=[*indexers, *encoders, assembler, classifier])


def evaluate_model(predictions: DataFrame) -> dict[str, float]:
    """Evaluate Accuracy, Precision, Recall, F1, and ROC AUC."""
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    return {
        "accuracy": evaluator.setMetricName("accuracy").evaluate(predictions),
        "precision": evaluator.setMetricName("weightedPrecision").evaluate(predictions),
        "recall": evaluator.setMetricName("weightedRecall").evaluate(predictions),
        "f1": evaluator.setMetricName("f1").evaluate(predictions),
        "roc_auc": binary_evaluator.setMetricName("areaUnderROC").evaluate(predictions),
    }


def show_confusion_matrix(predictions: DataFrame) -> None:
    """Display the confusion matrix in the console."""
    LOGGER.info("Confusion matrix:")
    predictions.groupBy("label", "prediction").count().orderBy("label", "prediction").show()


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save evaluation metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", output_path)


def save_model(model: PipelineModel, output_path: Path) -> None:
    """Persist the trained Spark ML pipeline."""
    model.write().overwrite().save(str(output_path))
    LOGGER.info("Saved model to %s", output_path)


def main() -> None:
    """CLI entry point."""
    configure_logging()
    args = parse_args()
    start = time.perf_counter()
    spark = create_spark()

    try:
        df = clean_dataset(load_dataset(spark, args.input))
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        pipeline = build_pipeline(df)
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        metrics = evaluate_model(predictions)
        LOGGER.info("Evaluation metrics: %s", metrics)
        show_confusion_matrix(predictions)
        save_metrics(metrics, args.metrics_output)
        save_model(model, args.model_output)
    except Exception:
        LOGGER.exception("Spark ML pipeline failed.")
        raise
    finally:
        spark.stop()
        LOGGER.info("Execution time: %.2f seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
