# Big Data

[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]() [![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)]() [![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?logo=apachespark&logoColor=white)]() [![Hadoop](https://img.shields.io/badge/Hadoop-66CCFF?logo=apachehadoop&logoColor=black)]()

This folder contains production-style Big Data scripts. Unlike the notebook-heavy NLP, LLM, and Deep Learning sections, these projects are written as executable Python jobs that resemble real ETL, Spark analytics, streaming, and ML workflows.

> Goal: show practical ability with Hadoop-style MapReduce, PySpark analytics, Spark ML pipelines, Structured Streaming, and production-inspired ETL.

---

## Projects

| Project | Script | Dataset | Main Skills |
|---|---|---|---|
| Hadoop Word Count | [hadoop_wordcount.py](hadoop_wordcount.py) | Project Gutenberg text | Map, shuffle, sort, reduce, text processing |
| PySpark Data Analysis | [pyspark_data_analysis.py](pyspark_data_analysis.py) | Netflix Movies and TV Shows | Spark SQL, EDA, grouping, visualization |
| Spark ML Pipeline | [spark_ml_pipeline.py](spark_ml_pipeline.py) | Adult Income | StringIndexer, OneHotEncoder, VectorAssembler, Logistic Regression |
| Spark Structured Streaming | [spark_streaming.py](spark_streaming.py) | Simulated IoT sensors | Watermarking, windows, rolling averages, anomaly detection |
| PySpark ETL Pipeline | [pyspark_etl_pipeline.py](pyspark_etl_pipeline.py) | Online Retail | Extract, clean, transform, Parquet, size comparison |

---

## Repository Structure

```text
big_data/
├─ README.md
├─ hadoop_wordcount.py
├─ pyspark_data_analysis.py
├─ spark_ml_pipeline.py
├─ spark_streaming.py
├─ pyspark_etl_pipeline.py
├─ requirements.txt
├─ .gitignore
└─ assets/
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run any script directly:

```bash
python hadoop_wordcount.py
python pyspark_data_analysis.py --input data/netflix_titles.csv
python spark_ml_pipeline.py --input data/adult.csv
python spark_streaming.py --duration 30
python pyspark_etl_pipeline.py --input data/online_retail.csv
```

Each script creates missing output folders automatically. Spark-based scripts can also generate small demo datasets when no local CSV is provided, so the workflow remains easy to inspect.

---

## What This Portfolio Demonstrates

- Writing command-line data processing jobs instead of notebook-only demos
- Structuring Spark code with reusable functions
- Using logging, error handling, and configuration variables
- Saving artifacts such as charts, JSON metrics, trained models, and Parquet outputs
- Explaining Big Data concepts through code comments and clear module documentation

