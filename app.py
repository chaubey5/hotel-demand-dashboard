from __future__ import annotations

import datetime as dt
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Demand Forecast | PySpark Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme Constants (Clean Light Hospitality & Navy Palette)
# ---------------------------------------------------------------------------
NAVY = "#0F172A"
PRIMARY_BLUE = "#1E3A8A"
ACCENT_TEAL = "#0D9488"
ACCENT_AMBER = "#D97706"
SLATE = "#475569"
BORDER_COLOR = "#E2E8F0"
CARD_BG = "#FFFFFF"

REQUIRED_COLUMNS = [
    "hotel",
    "is_canceled",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "reserved_room_type",
]

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "weekday",
    "month",
    "hotel_code",
    "room_code",
]


# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #F8FAFC;
                color: {NAVY};
            }}
            section[data-testid="stSidebar"] {{
                background-color: #0F172A;
                color: #F8FAFC;
            }}
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span {{
                color: #E2E8F0 !important;
            }}
            section[data-testid="stSidebar"] .stRadio label {{
                padding: 0.35rem 0.5rem;
                border-radius: 0.4rem;
            }}
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
                background-color: #1E293B;
                border: 1px dashed #475569;
            }}
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
                background-color: #1E3A8A;
                color: #FFFFFF !important;
                border: none;
            }}
            .hero-banner {{
                background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%);
                padding: 1.5rem 2rem;
                border-radius: 12px;
                color: #FFFFFF;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
            }}
            .hero-banner h1 {{
                margin: 0;
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: #FFFFFF;
            }}
            .hero-banner p {{
                margin: 0.4rem 0 0 0;
                font-size: 0.96rem;
                color: #CBD5E1;
            }}
            .kpi-card {{
                background-color: {CARD_BG};
                border: 1px solid {BORDER_COLOR};
                border-top: 3px solid {PRIMARY_BLUE};
                border-radius: 10px;
                padding: 1rem 1.15rem;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
                height: 100%;
            }}
            .kpi-card.teal {{
                border-top-color: {ACCENT_TEAL};
            }}
            .kpi-card.amber {{
                border-top-color: {ACCENT_AMBER};
            }}
            .kpi-label {{
                font-size: 0.78rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                color: {SLATE};
                margin-bottom: 0.35rem;
            }}
            .kpi-value {{
                font-size: 1.65rem;
                font-weight: 700;
                color: {NAVY};
                line-height: 1.2;
            }}
            .kpi-sub {{
                font-size: 0.8rem;
                color: #64748B;
                margin-top: 0.3rem;
            }}
            .section-header {{
                font-size: 1.18rem;
                font-weight: 700;
                color: {NAVY};
                margin: 1.1rem 0 0.6rem 0;
                padding-bottom: 0.35rem;
                border-bottom: 2px solid {BORDER_COLOR};
            }}
            .info-badge {{
                display: inline-block;
                background-color: #DBEAFE;
                color: #1E3A8A;
                font-weight: 600;
                font-size: 0.82rem;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str = "", accent: str = "blue") -> None:
    accent_class = "" if accent == "blue" else f" {accent}"
    st.markdown(
        f"""
        <div class="kpi-card{accent_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PySpark Environment & Session Management
# ---------------------------------------------------------------------------
def configure_spark_env() -> None:
    """Ensure PySpark worker Python matches driver Python and JAVA_HOME is set."""
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    if not os.environ.get("JAVA_HOME"):
        candidate_java_homes = [
            "/usr/lib/jvm/default-java",
            "/usr/lib/jvm/java-17-openjdk-amd64",
            "/usr/lib/jvm/java-17-openjdk-arm64",
            "/usr/lib/jvm/java-11-openjdk-amd64",
        ]
        for candidate in candidate_java_homes:
            if os.path.isdir(candidate):
                os.environ["JAVA_HOME"] = candidate
                break

        if not os.environ.get("JAVA_HOME"):
            java_bin = shutil.which("java")
            if java_bin:
                resolved = Path(java_bin).resolve()
                if resolved.parent.name == "bin":
                    os.environ["JAVA_HOME"] = str(resolved.parent.parent)


@st.cache_resource(show_spinner="Initializing Apache Spark session...")
def get_spark_session() -> SparkSession:
    """Create and cache a single SparkSession for the Streamlit application."""
    configure_spark_env()
    spark = (
        SparkSession.builder.appName("Hotel Demand Forecast")
        .master("local[*]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "1536m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def spark_to_pandas(sdf: DataFrame) -> pd.DataFrame:
    """Convert a small aggregated Spark DataFrame to Pandas without Arrow/worker issues."""
    columns = sdf.columns
    rows = [row.asDict(recursive=True) for row in sdf.collect()]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Core PySpark Data Processing, ML Training & Recursive Forecasting
# ---------------------------------------------------------------------------
def _pyspark_dayofweek(date_val: dt.date) -> int:
    """Match PySpark dayofweek(): 1 = Sunday, 2 = Monday, ..., 7 = Saturday."""
    return ((date_val.weekday() + 1) % 7) + 1


def _to_date(val: Any) -> dt.date:
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    return pd.to_datetime(val).date()


def run_pyspark_pipeline(csv_bytes: bytes, file_name: str) -> dict[str, Any]:
    """
    Execute the complete PySpark hotel demand forecasting pipeline:
    1. CSV ingestion & schema validation
    2. Missing-value handling & cleaning
    3. Daily room-demand calculation (non-cancelled bookings expanded by stay_nights)
    4. Lag & calendar feature engineering + StringIndexer encoding
    5. Chronological train/test split (80/20 by date)
    6. Linear Regression & Decision Tree training + MAE/RMSE evaluation + auto-selection
    7. Recursive day-by-day forecasting for next 7 and 14 days
    8. Weekday analysis & busiest/quietest day identification
    """
    spark = get_spark_session()
    tmp_dir = tempfile.mkdtemp(prefix="hotel_forecast_")
    csv_path = os.path.join(tmp_dir, "uploaded_bookings.csv")

    try:
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        raw_df = spark.read.csv(
            csv_path,
            header=True,
            inferSchema=True,
            nullValue="NULL",
        )

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing_cols:
            raise ValueError(
                "Uploaded CSV is missing required column(s): "
                + ", ".join(missing_cols)
            )

        raw_count = raw_df.count()
        if raw_count == 0:
            raise ValueError("The uploaded CSV file contains zero rows.")

        raw_columns = list(raw_df.columns)
        raw_preview_pd = spark_to_pandas(raw_df.limit(15))

        # Missing value audit on key columns before cleaning
        audit_cols = [
            c
            for c in [
                "hotel",
                "is_canceled",
                "arrival_date_year",
                "arrival_date_month",
                "arrival_date_day_of_month",
                "stays_in_weekend_nights",
                "stays_in_week_nights",
                "children",
                "reserved_room_type",
            ]
            if c in raw_df.columns
        ]
        null_exprs = [
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in audit_cols
        ]
        null_counts_row = raw_df.agg(*null_exprs).first().asDict()
        missing_summary_pd = pd.DataFrame(
            [
                {
                    "Column": col_name,
                    "Missing Values": int(null_counts_row[col_name] or 0),
                    "Missing %": round(
                        100.0 * (null_counts_row[col_name] or 0) / raw_count, 4
                    ),
                }
                for col_name in audit_cols
            ]
        )

        # -------------------------------------------------------------------
        # Step 2: Clean Data & Basic Booking Analysis (matches notebook Cell 4)
        # -------------------------------------------------------------------
        df = raw_df.dropna(
            subset=[
                "hotel",
                "arrival_date_year",
                "arrival_date_month",
                "arrival_date_day_of_month",
                "is_canceled",
            ]
        )

        fill_map: dict[str, Any] = {
            "stays_in_week_nights": 0,
            "stays_in_weekend_nights": 0,
            "reserved_room_type": "Unknown",
        }
        if "children" in df.columns:
            fill_map["children"] = 0
        df = df.fillna(fill_map)

        df = df.withColumn(
            "arrival_date",
            F.to_date(
                F.concat_ws(
                    "-",
                    F.col("arrival_date_year"),
                    F.col("arrival_date_month"),
                    F.col("arrival_date_day_of_month"),
                ),
                "yyyy-MMMM-d",
            ),
        )
        df = df.withColumn(
            "stay_nights",
            F.col("stays_in_week_nights").cast("int")
            + F.col("stays_in_weekend_nights").cast("int"),
        )
        df = df.withColumnRenamed("reserved_room_type", "room_category")
        df = df.filter(F.col("arrival_date").isNotNull()).filter(
            F.col("stay_nights") > 0
        )
        df.cache()

        cleaned_count = df.count()
        if cleaned_count == 0:
            raise ValueError(
                "No valid booking records remained after date parsing and stay_nights > 0 filtering."
            )

        hotel_stats_pd = spark_to_pandas(
            df.groupBy("hotel")
            .agg(
                F.count("*").alias("total_bookings"),
                F.sum(F.when(F.col("is_canceled") == 0, 1).otherwise(0)).alias(
                    "active_bookings"
                ),
                (F.avg("is_canceled") * 100.0).alias("cancellation_percent"),
                F.avg("stay_nights").alias("average_stay_nights"),
            )
            .orderBy("hotel")
        )

        overall_stats = (
            df.select(
                (F.avg("is_canceled") * 100.0).alias("cancellation_percent"),
                F.avg("stay_nights").alias("average_stay_nights"),
            )
            .first()
            .asDict()
        )
        cancellation_percent = float(overall_stats["cancellation_percent"] or 0.0)
        average_stay_nights = float(overall_stats["average_stay_nights"] or 0.0)

        # -------------------------------------------------------------------
        # Step 3: Daily Room Demand Calculation (matches notebook Cell 6)
        # -------------------------------------------------------------------
        bookings = df.filter(F.col("is_canceled") == 0)
        if bookings.count() == 0:
            raise ValueError(
                "All valid bookings are marked as cancelled; cannot compute room demand."
            )

        bookings = bookings.withColumn(
            "date",
            F.explode(
                F.sequence(
                    F.col("arrival_date"),
                    F.date_add(F.col("arrival_date"), F.col("stay_nights") - 1),
                )
            ),
        )

        daily = (
            bookings.groupBy("hotel", "room_category", "date")
            .count()
            .withColumnRenamed("count", "demand")
        )
        daily.cache()

        daily_pd = spark_to_pandas(daily.orderBy("date", "hotel", "room_category"))
        daily_pd["date"] = pd.to_datetime(daily_pd["date"]).dt.date
        daily_pd["demand"] = daily_pd["demand"].astype(float)

        # -------------------------------------------------------------------
        # Step 4: Lag & Calendar Feature Engineering (matches notebook Cell 8)
        # -------------------------------------------------------------------
        date_range = daily.groupBy("hotel", "room_category").agg(
            F.min("date").alias("first_date"), F.max("date").alias("last_date")
        )
        calendar = date_range.withColumn(
            "date", F.explode(F.sequence("first_date", "last_date"))
        )
        data = calendar.join(daily, ["hotel", "room_category", "date"], "left")
        data = data.withColumn("demand", F.coalesce(F.col("demand"), F.lit(0)))

        hotel_indexer = StringIndexer(
            inputCol="hotel", outputCol="hotel_code", handleInvalid="keep"
        )
        room_indexer = StringIndexer(
            inputCol="room_category", outputCol="room_code", handleInvalid="keep"
        )
        hotel_indexer_fit = hotel_indexer.fit(data)
        data = hotel_indexer_fit.transform(data)
        room_indexer_fit = room_indexer.fit(data)
        data = room_indexer_fit.transform(data)

        w = Window.partitionBy("hotel", "room_category").orderBy("date")
        data = data.withColumn("lag_1", F.lag("demand", 1).over(w))
        data = data.withColumn("lag_7", F.lag("demand", 7).over(w))
        data = data.withColumn("lag_14", F.lag("demand", 14).over(w))
        data = data.withColumn("weekday", F.dayofweek("date"))
        data = data.withColumn("month", F.month("date"))
        data = data.dropna(subset=["lag_1", "lag_7", "lag_14"])
        data.cache()

        total_model_rows = data.count()
        if total_model_rows < 10:
            raise ValueError(
                "Not enough historical dates (need > 14 consecutive days per hotel/room) "
                "to engineer lag_1, lag_7, and lag_14 features."
            )

        # Chronological 80/20 split (never random split for time series)
        data = data.withColumn(
            "date_number", F.datediff(F.col("date"), F.lit("1970-01-01"))
        )
        split_number = data.approxQuantile("date_number", [0.8], 0.01)[0]
        split_date = (
            data.filter(F.col("date_number") == split_number)
            .agg(F.min("date"))
            .first()[0]
        )

        train = data.filter(F.col("date") < F.lit(split_date))
        test = data.filter(F.col("date") >= F.lit(split_date))

        train_count = train.count()
        test_count = test.count()

        # Fallback if all rows fell on one side due to very few unique dates
        if train_count == 0 or test_count == 0:
            unique_dates = [
                r["date"]
                for r in data.select("date").distinct().orderBy("date").collect()
            ]
            if len(unique_dates) < 2:
                raise ValueError(
                    "Need at least 2 distinct dates after lag creation for chronological splitting."
                )
            cutoff_idx = max(1, int(len(unique_dates) * 0.8))
            if cutoff_idx >= len(unique_dates):
                cutoff_idx = len(unique_dates) - 1
            split_date = unique_dates[cutoff_idx]
            train = data.filter(F.col("date") < F.lit(split_date))
            test = data.filter(F.col("date") >= F.lit(split_date))
            train_count = train.count()
            test_count = test.count()

        assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="features")
        train_vec = assembler.transform(train).cache()
        test_vec = assembler.transform(test).cache()

        # -------------------------------------------------------------------
        # Step 5: Train Linear Regression & Decision Tree (matches notebook Cell 10)
        # -------------------------------------------------------------------
        linear_model = LinearRegression(featuresCol="features", labelCol="demand")
        linear_fit = linear_model.fit(train_vec)
        linear_predictions = linear_fit.transform(test_vec)

        tree_model = DecisionTreeRegressor(
            featuresCol="features", labelCol="demand", maxDepth=5
        )
        tree_fit = tree_model.fit(train_vec)
        tree_predictions = tree_fit.transform(test_vec)

        mae_evaluator = RegressionEvaluator(
            labelCol="demand", predictionCol="prediction", metricName="mae"
        )
        rmse_evaluator = RegressionEvaluator(
            labelCol="demand", predictionCol="prediction", metricName="rmse"
        )

        linear_mae = float(mae_evaluator.evaluate(linear_predictions))
        linear_rmse = float(rmse_evaluator.evaluate(linear_predictions))
        tree_mae = float(mae_evaluator.evaluate(tree_predictions))
        tree_rmse = float(rmse_evaluator.evaluate(tree_predictions))

        # -------------------------------------------------------------------
        # Step 6: Automatic Best-Model Selection (matches notebook Cell 12)
        # -------------------------------------------------------------------
        if linear_mae <= tree_mae:
            best_model = linear_fit
            best_predictions = linear_predictions
            best_name = "Linear Regression"
        else:
            best_model = tree_fit
            best_predictions = tree_predictions
            best_name = "Decision Tree"

        scores_pd = pd.DataFrame(
            [
                {
                    "Model": "Linear Regression",
                    "MAE": round(linear_mae, 4),
                    "RMSE": round(linear_rmse, 4),
                    "Selected": "Best Model" if best_name == "Linear Regression" else "-",
                },
                {
                    "Model": "Decision Tree",
                    "MAE": round(tree_mae, 4),
                    "RMSE": round(tree_rmse, 4),
                    "Selected": "Best Model" if best_name == "Decision Tree" else "-",
                },
            ]
        )

        test_comparison_pd = spark_to_pandas(
            best_predictions.select(
                "date",
                "hotel",
                "room_category",
                F.col("demand").cast("double").alias("actual_demand"),
                F.greatest(F.col("prediction"), F.lit(0.0)).alias("predicted_demand"),
            ).orderBy("date", "hotel", "room_category")
        )
        test_comparison_pd["date"] = pd.to_datetime(test_comparison_pd["date"]).dt.date

        # -------------------------------------------------------------------
        # Step 7: Recursive 7-Day & 14-Day Forecasting (Requirement 5)
        # -------------------------------------------------------------------
        last_date = _to_date(data.agg(F.max("date")).first()[0])
        lookback_start = last_date - dt.timedelta(days=20)

        # Build full zero-filled history window from calendar join daily up to last_date
        pairs_df = data.select(
            "hotel", "room_category", "hotel_code", "room_code"
        ).distinct()
        pairs_list = [row.asDict() for row in pairs_df.collect()]

        recent_history_sdf = (
            calendar.join(daily, ["hotel", "room_category", "date"], "left")
            .withColumn("demand", F.coalesce(F.col("demand"), F.lit(0)))
            .filter(F.col("date") >= F.lit(lookback_start))
            .select("hotel", "room_category", "date", F.col("demand").cast("double"))
        )
        recent_rows = recent_history_sdf.collect()

        history_map: dict[tuple[str, str], dict[dt.date, float]] = {
            (p["hotel"], p["room_category"]): {} for p in pairs_list
        }
        for row in recent_rows:
            key = (row["hotel"], row["room_category"])
            if key in history_map:
                history_map[key][_to_date(row["date"])] = float(row["demand"])

        forecast_records: list[dict[str, Any]] = []
        for step in range(1, 15):
            forecast_date = last_date + dt.timedelta(days=step)
            wkday = _pyspark_dayofweek(forecast_date)
            mth = forecast_date.month

            for pair in pairs_list:
                hotel_name = pair["hotel"]
                room_cat = pair["room_category"]
                hotel_code = float(pair["hotel_code"])
                room_code = float(pair["room_code"])
                series = history_map[(hotel_name, room_cat)]

                lag_1 = float(series.get(forecast_date - dt.timedelta(days=1), 0.0))
                lag_7 = float(series.get(forecast_date - dt.timedelta(days=7), 0.0))
                lag_14 = float(series.get(forecast_date - dt.timedelta(days=14), 0.0))

                feature_vec = Vectors.dense(
                    [lag_1, lag_7, lag_14, float(wkday), float(mth), hotel_code, room_code]
                )
                raw_pred = float(best_model.predict(feature_vec))
                pred_demand = max(0.0, raw_pred)

                # Recursively store prediction so subsequent forecast days use updated lags
                series[forecast_date] = pred_demand

                forecast_records.append(
                    {
                        "date": forecast_date,
                        "horizon_day": step,
                        "hotel": hotel_name,
                        "room_category": room_cat,
                        "lag_1_used": round(lag_1, 2),
                        "lag_7_used": round(lag_7, 2),
                        "lag_14_used": round(lag_14, 2),
                        "predicted_demand": pred_demand,
                    }
                )

        forecast_14_granular_pd = pd.DataFrame(forecast_records)
        forecast_7_granular_pd = forecast_14_granular_pd[
            forecast_14_granular_pd["horizon_day"] <= 7
        ].copy()

        forecast_14_daily_pd = (
            forecast_14_granular_pd.groupby("date", as_index=False)["predicted_demand"]
            .sum()
            .sort_values("date")
        )
        forecast_7_daily_pd = (
            forecast_7_granular_pd.groupby("date", as_index=False)["predicted_demand"]
            .sum()
            .sort_values("date")
        )

        # -------------------------------------------------------------------
        # Step 8: Weekday Demand Analysis & Busiest/Quietest Day (notebook Cell 14)
        # -------------------------------------------------------------------
        weekday_summary = (
            daily.withColumn("weekday", F.date_format("date", "EEEE"))
            .withColumn("day_number", F.dayofweek("date"))
            .groupBy("weekday", "day_number")
            .agg(F.avg("demand").alias("average_demand"))
            .orderBy("day_number")
        )
        weekday_pd = spark_to_pandas(weekday_summary)

        # Also compute total daily demand averaged by weekday across all hotels/rooms
        total_daily_sdf = daily.groupBy("date").agg(
            F.sum("demand").alias("total_demand")
        )
        weekday_total_pd = spark_to_pandas(
            total_daily_sdf.withColumn("weekday", F.date_format("date", "EEEE"))
            .withColumn("day_number", F.dayofweek("date"))
            .groupBy("weekday", "day_number")
            .agg(F.avg("total_demand").alias("average_total_daily_demand"))
            .orderBy("day_number")
        )
        weekday_pd = weekday_pd.merge(
            weekday_total_pd[["weekday", "average_total_daily_demand"]],
            on="weekday",
            how="left",
        )

        busiest_row = weekday_pd.loc[weekday_pd["average_demand"].idxmax()]
        quietest_row = weekday_pd.loc[weekday_pd["average_demand"].idxmin()]

        df.unpersist()
        daily.unpersist()
        data.unpersist()
        train_vec.unpersist()
        test_vec.unpersist()

        return {
            "file_name": file_name,
            "raw_count": int(raw_count),
            "cleaned_count": int(cleaned_count),
            "removed_count": int(raw_count - cleaned_count),
            "raw_columns": raw_columns,
            "raw_preview": raw_preview_pd,
            "missing_summary": missing_summary_pd,
            "hotel_stats": hotel_stats_pd,
            "cancellation_percent": cancellation_percent,
            "average_stay_nights": average_stay_nights,
            "daily_demand": daily_pd,
            "train_count": int(train_count),
            "test_count": int(test_count),
            "split_date": _to_date(split_date),
            "last_date": last_date,
            "linear_mae": linear_mae,
            "linear_rmse": linear_rmse,
            "tree_mae": tree_mae,
            "tree_rmse": tree_rmse,
            "best_name": best_name,
            "scores": scores_pd,
            "test_comparison": test_comparison_pd,
            "forecast_7_granular": forecast_7_granular_pd,
            "forecast_14_granular": forecast_14_granular_pd,
            "forecast_7_daily": forecast_7_daily_pd,
            "forecast_14_daily": forecast_14_daily_pd,
            "weekday_summary": weekday_pd,
            "busiest_day": {
                "weekday": str(busiest_row["weekday"]),
                "average_demand": float(busiest_row["average_demand"]),
                "average_total_daily_demand": float(
                    busiest_row["average_total_daily_demand"]
                ),
            },
            "quietest_day": {
                "weekday": str(quietest_row["weekday"]),
                "average_demand": float(quietest_row["average_demand"]),
                "average_total_daily_demand": float(
                    quietest_row["average_total_daily_demand"]
                ),
            },
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Plotly Chart Builders
# ---------------------------------------------------------------------------
def _apply_plotly_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY)),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color=NAVY),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1")
    return fig


# ---------------------------------------------------------------------------
# Page Renderers
# ---------------------------------------------------------------------------
def render_overview_page(res: dict[str, Any]) -> None:
    st.markdown('<div class="section-header">Key Operational Metrics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card(
            "Total Raw Records",
            f"{res['raw_count']:,}",
            f"{len(res['raw_columns'])} dataset columns",
            accent="blue",
        )
    with c2:
        render_kpi_card(
            "Cleaned Records",
            f"{res['cleaned_count']:,}",
            f"{res['removed_count']:,} invalid/zero-night rows removed",
            accent="teal",
        )
    with c3:
        render_kpi_card(
            "Cancellation Rate",
            f"{res['cancellation_percent']:.2f}%",
            "Excluded from room-demand totals",
            accent="amber",
        )
    with c4:
        render_kpi_card(
            "Avg. Length of Stay",
            f"{res['average_stay_nights']:.2f} nights",
            f"Latest date: {res['last_date']}",
            accent="blue",
        )

    st.markdown('<div class="section-header">Hotel Portfolio Summary</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        hotel_df = res["hotel_stats"].copy()
        hotel_df["cancellation_percent"] = hotel_df["cancellation_percent"].round(2)
        hotel_df["average_stay_nights"] = hotel_df["average_stay_nights"].round(2)
        hotel_df.columns = [
            "Hotel",
            "Total Bookings",
            "Active (Non-Cancelled)",
            "Cancellation %",
            "Avg Stay Nights",
        ]
        st.dataframe(hotel_df, width="stretch", hide_index=True)

    with col_right:
        fig = px.bar(
            res["hotel_stats"],
            x="hotel",
            y="active_bookings",
            color="hotel",
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL],
            text_auto=",",
            labels={"hotel": "Hotel", "active_bookings": "Non-Cancelled Bookings"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_apply_plotly_layout(fig, "Non-Cancelled Bookings by Hotel", 290), width="stretch")

    st.markdown('<div class="section-header">Data Cleaning & Missing-Value Audit</div>', unsafe_allow_html=True)
    tab_preview, tab_missing, tab_cols = st.tabs(
        ["Dataset Preview (First 15 Rows)", "Missing-Value Handling", "Schema & Columns"]
    )
    with tab_preview:
        st.dataframe(res["raw_preview"], width="stretch", hide_index=True)
    with tab_missing:
        st.caption(
            "Rows missing `hotel`, `arrival_date_*`, or `is_canceled` are dropped. "
            "Missing `children`, `stays_in_week_nights`, and `stays_in_weekend_nights` are filled with `0`; "
            "missing `reserved_room_type` is filled with `'Unknown'`."
        )
        st.dataframe(res["missing_summary"], width="stretch", hide_index=True)
    with tab_cols:
        st.write(f"**Total Columns:** {len(res['raw_columns'])}")
        st.dataframe(
            pd.DataFrame({"Column Index": range(1, len(res["raw_columns"]) + 1), "Column Name": res["raw_columns"]}),
            width="stretch",
            hide_index=True,
        )


def render_historical_page(res: dict[str, Any]) -> None:
    daily_pd = res["daily_demand"]
    min_date = daily_pd["date"].min()
    max_date = daily_pd["date"].max()
    all_hotels = sorted(daily_pd["hotel"].unique().tolist())
    all_rooms = sorted(daily_pd["room_category"].unique().tolist())

    st.markdown('<div class="section-header">Interactive Filters</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([1.2, 1, 1.3])
    with f1:
        date_range = st.date_input(
            "Arrival / Stay Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
    with f2:
        selected_hotels = st.multiselect(
            "Hotel",
            options=all_hotels,
            default=all_hotels,
        )
    with f3:
        selected_rooms = st.multiselect(
            "Room Category",
            options=all_rooms,
            default=all_rooms,
        )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    filtered = daily_pd[
        (daily_pd["date"] >= start_d)
        & (daily_pd["date"] <= end_d)
        & (daily_pd["hotel"].isin(selected_hotels))
        & (daily_pd["room_category"].isin(selected_rooms))
    ]

    if filtered.empty:
        st.warning("No records match the current filter selection. Adjust the date range, hotel, or room categories.")
        return

    daily_totals = filtered.groupby("date", as_index=False)["demand"].sum().sort_values("date")
    peak_row = daily_totals.loc[daily_totals["demand"].idxmax()]
    low_row = daily_totals.loc[daily_totals["demand"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Total Room-Nights",
            f"{int(filtered['demand'].sum()):,}",
            f"{len(daily_totals):,} active days",
            accent="blue",
        )
    with k2:
        render_kpi_card(
            "Avg. Daily Demand",
            f"{daily_totals['demand'].mean():.1f}",
            "Rooms occupied per day",
            accent="teal",
        )
    with k3:
        render_kpi_card(
            "Peak Day Demand",
            f"{int(peak_row['demand']):,}",
            f"Recorded on {peak_row['date']}",
            accent="amber",
        )
    with k4:
        render_kpi_card(
            "Lowest Day Demand",
            f"{int(low_row['demand']):,}",
            f"Recorded on {low_row['date']}",
            accent="blue",
        )

    st.markdown('<div class="section-header">Historical Daily Room-Demand Trend</div>', unsafe_allow_html=True)
    fig_total = px.line(
        daily_totals,
        x="date",
        y="demand",
        labels={"date": "Date", "demand": "Rooms Demanded"},
    )
    fig_total.update_traces(line=dict(color=PRIMARY_BLUE, width=2))
    st.plotly_chart(_apply_plotly_layout(fig_total, "Total Daily Room Demand", 380), width="stretch")

    col_h, col_r = st.columns(2)
    with col_h:
        hotel_daily = (
            filtered.groupby(["date", "hotel"], as_index=False)["demand"]
            .sum()
            .sort_values("date")
        )
        fig_hotel = px.line(
            hotel_daily,
            x="date",
            y="demand",
            color="hotel",
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL, ACCENT_AMBER],
            labels={"date": "Date", "demand": "Rooms", "hotel": "Hotel"},
        )
        st.plotly_chart(_apply_plotly_layout(fig_hotel, "Hotel-Wise Daily Demand", 360), width="stretch")

    with col_r:
        room_agg = (
            filtered.groupby("room_category", as_index=False)["demand"]
            .sum()
            .sort_values("demand", ascending=False)
        )
        fig_room = px.bar(
            room_agg,
            x="room_category",
            y="demand",
            text_auto=",",
            labels={"room_category": "Room Category", "demand": "Total Room-Nights"},
        )
        fig_room.update_traces(marker_color=ACCENT_TEAL)
        st.plotly_chart(
            _apply_plotly_layout(fig_room, "Demand by Room Category", 360),
            width="stretch",
        )

    with st.expander("View Daily Room Demand Table (Hotel & Room Category)"):
        st.dataframe(filtered.head(500), width="stretch", hide_index=True)


def render_weekday_page(res: dict[str, Any]) -> None:
    busiest = res["busiest_day"]
    quietest = res["quietest_day"]
    weekday_pd = res["weekday_summary"]
    daily_pd = res["daily_demand"]

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card(
            "Busiest Day of Week",
            busiest["weekday"],
            f"Avg {busiest['average_demand']:.2f} rooms / segment · {busiest['average_total_daily_demand']:.1f} total/day",
            accent="teal",
        )
    with k2:
        render_kpi_card(
            "Quietest Day of Week",
            quietest["weekday"],
            f"Avg {quietest['average_demand']:.2f} rooms / segment · {quietest['average_total_daily_demand']:.1f} total/day",
            accent="amber",
        )
    with k3:
        spread = busiest["average_demand"] - quietest["average_demand"]
        pct = 100.0 * spread / quietest["average_demand"] if quietest["average_demand"] > 0 else 0.0
        render_kpi_card(
            "Weekly Swing",
            f"+{pct:.1f}%",
            f"{spread:.2f} more rooms/segment on {busiest['weekday']}",
            accent="blue",
        )

    st.markdown('<div class="section-header">Weekday Demand Patterns</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        bar_colors = [
            ACCENT_TEAL
            if w == busiest["weekday"]
            else (ACCENT_AMBER if w == quietest["weekday"] else PRIMARY_BLUE)
            for w in weekday_pd["weekday"]
        ]
        fig_wk = go.Figure(
            data=[
                go.Bar(
                    x=weekday_pd["weekday"],
                    y=weekday_pd["average_demand"],
                    marker_color=bar_colors,
                    text=[f"{v:.2f}" for v in weekday_pd["average_demand"]],
                    textposition="outside",
                )
            ]
        )
        fig_wk.update_xaxes(title_text="Weekday")
        fig_wk.update_yaxes(title_text="Average Demand (per Hotel-Room Group)")
        st.plotly_chart(
            _apply_plotly_layout(fig_wk, "Average Room Demand by Weekday", 390),
            width="stretch",
        )

    with col_r:
        hotel_wk = daily_pd.copy()
        hotel_wk["weekday"] = pd.to_datetime(hotel_wk["date"]).dt.day_name()
        hotel_wk["day_num"] = pd.to_datetime(hotel_wk["date"]).dt.dayofweek
        hotel_wk_agg = (
            hotel_wk.groupby(["hotel", "weekday", "day_num"], as_index=False)["demand"]
            .mean()
            .sort_values("day_num")
        )
        fig_hwk = px.line(
            hotel_wk_agg,
            x="weekday",
            y="demand",
            color="hotel",
            markers=True,
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL],
            labels={"weekday": "Weekday", "demand": "Avg Demand", "hotel": "Hotel"},
        )
        st.plotly_chart(
            _apply_plotly_layout(fig_hwk, "Hotel-Wise Weekday Pattern", 390),
            width="stretch",
        )

    st.markdown('<div class="section-header">Monthly Seasonality & Export</div>', unsafe_allow_html=True)
    monthly_df = daily_pd.copy()
    monthly_df["month_num"] = pd.to_datetime(monthly_df["date"]).dt.month
    monthly_df["month_name"] = pd.to_datetime(monthly_df["date"]).dt.strftime("%b")
    monthly_agg = (
        monthly_df.groupby(["month_num", "month_name", "hotel"], as_index=False)["demand"]
        .mean()
        .sort_values("month_num")
    )
    fig_m = px.bar(
        monthly_agg,
        x="month_name",
        y="demand",
        color="hotel",
        barmode="group",
        color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL],
        labels={"month_name": "Month", "demand": "Avg Daily Segment Demand", "hotel": "Hotel"},
    )
    st.plotly_chart(_apply_plotly_layout(fig_m, "Average Demand by Month & Hotel", 340), width="stretch")

    display_wk = weekday_pd.copy()
    display_wk["average_demand"] = display_wk["average_demand"].round(2)
    display_wk["average_total_daily_demand"] = display_wk["average_total_daily_demand"].round(2)
    st.dataframe(display_wk, width="stretch", hide_index=True)

    csv_wk = weekday_pd[["weekday", "day_number", "average_demand"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Weekday Analysis CSV",
        data=csv_wk,
        file_name="weekday_analysis.csv",
        mime="text/csv",
    )


def render_ml_page(res: dict[str, Any]) -> None:
    st.markdown(
        f'<span class="info-badge">Selected Best Model: {res["best_name"]}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Linear Regression MAE",
            f"{res['linear_mae']:.2f}",
            f"RMSE: {res['linear_rmse']:.2f}",
            accent="teal" if res["best_name"] == "Linear Regression" else "blue",
        )
    with k2:
        render_kpi_card(
            "Decision Tree MAE",
            f"{res['tree_mae']:.2f}",
            f"RMSE: {res['tree_rmse']:.2f} (maxDepth=5)",
            accent="teal" if res["best_name"] == "Decision Tree" else "blue",
        )
    with k3:
        render_kpi_card(
            "Training Records",
            f"{res['train_count']:,}",
            f"Chronological < {res['split_date']}",
            accent="blue",
        )
    with k4:
        render_kpi_card(
            "Testing Records",
            f"{res['test_count']:,}",
            f"Chronological ≥ {res['split_date']}",
            accent="amber",
        )

    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    col_table, col_chart = st.columns([1, 1.2])
    with col_table:
        st.dataframe(res["scores"], width="stretch", hide_index=True)
        st.caption(
            "Selection criterion: the PySpark model with the lower test **MAE** "
            "(Mean Absolute Error) is automatically selected for forecasting. "
            "Features used: `lag_1`, `lag_7`, `lag_14`, `weekday`, `month`, `hotel_code`, `room_code`."
        )
        csv_scores = res["scores"][["Model", "MAE", "RMSE"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Model Comparison CSV",
            data=csv_scores,
            file_name="model_comparison.csv",
            mime="text/csv",
        )

    with col_chart:
        scores_melted = res["scores"].melt(
            id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error"
        )
        fig_scores = px.bar(
            scores_melted,
            x="Model",
            y="Error",
            color="Metric",
            barmode="group",
            text_auto=".2f",
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_AMBER],
        )
        st.plotly_chart(_apply_plotly_layout(fig_scores, "MAE vs RMSE by Model", 330), width="stretch")

    st.markdown('<div class="section-header">Actual vs Predicted Room Demand (Test Set)</div>', unsafe_allow_html=True)
    comp_df = res["test_comparison"]

    view_mode = st.radio(
        "Comparison Granularity",
        options=["Aggregated Daily Total (All Hotels & Rooms)", "Filter by Hotel & Room Category"],
        horizontal=True,
    )

    if view_mode == "Aggregated Daily Total (All Hotels & Rooms)":
        plot_df = (
            comp_df.groupby("date", as_index=False)[["actual_demand", "predicted_demand"]]
            .sum()
            .sort_values("date")
        )
        chart_title = f"Test Set Daily Actual vs Predicted Demand ({res['best_name']})"
    else:
        c_h, c_r = st.columns(2)
        with c_h:
            sel_hotel = st.selectbox("Hotel", sorted(comp_df["hotel"].unique().tolist()))
        with c_r:
            avail_rooms = sorted(
                comp_df[comp_df["hotel"] == sel_hotel]["room_category"].unique().tolist()
            )
            sel_room = st.selectbox("Room Category", avail_rooms)
        plot_df = comp_df[
            (comp_df["hotel"] == sel_hotel) & (comp_df["room_category"] == sel_room)
        ].sort_values("date")
        chart_title = f"Actual vs Predicted: {sel_hotel} — Room {sel_room}"

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["actual_demand"],
            mode="lines+markers",
            name="Actual Demand",
            line=dict(color=PRIMARY_BLUE, width=2.2),
            marker=dict(size=5),
        )
    )
    fig_comp.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["predicted_demand"],
            mode="lines+markers",
            name="Predicted Demand",
            line=dict(color=ACCENT_TEAL, width=2.2, dash="dot"),
            marker=dict(size=5),
        )
    )
    fig_comp.update_xaxes(title_text="Test Date")
    fig_comp.update_yaxes(title_text="Room Demand")
    st.plotly_chart(_apply_plotly_layout(fig_comp, chart_title, 400), width="stretch")

    with st.expander("Inspect Granular Test Predictions"):
        st.dataframe(comp_df.head(300), width="stretch", hide_index=True)


def render_forecast_page(res: dict[str, Any]) -> None:
    st.info(
        f"**Recursive Multi-Step Forecasting Active ({res['best_name']})** — "
        f"Forecasts start from `{res['last_date'] + dt.timedelta(days=1)}`. "
        "Each predicted day feeds back into `lag_1`, `lag_7`, and `lag_14` for subsequent forecast dates."
    )

    horizon_choice = st.radio(
        "Select Forecast Horizon",
        options=["Next 7 Days", "Next 14 Days"],
        horizontal=True,
    )
    horizon_days = 7 if horizon_choice == "Next 7 Days" else 14

    daily_fc = (
        res["forecast_7_daily"] if horizon_days == 7 else res["forecast_14_daily"]
    ).copy()
    granular_fc = (
        res["forecast_7_granular"] if horizon_days == 7 else res["forecast_14_granular"]
    ).copy()

    peak_fc = daily_fc.loc[daily_fc["predicted_demand"].idxmax()]
    low_fc = daily_fc.loc[daily_fc["predicted_demand"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            f"{horizon_days}-Day Total Demand",
            f"{daily_fc['predicted_demand'].sum():.1f}",
            "Total predicted room-nights",
            accent="blue",
        )
    with k2:
        render_kpi_card(
            "Avg. Daily Forecast",
            f"{daily_fc['predicted_demand'].mean():.1f}",
            f"Across {horizon_days} days",
            accent="teal",
        )
    with k3:
        render_kpi_card(
            "Forecast Peak Day",
            f"{peak_fc['predicted_demand']:.1f}",
            f"Expected on {peak_fc['date']}",
            accent="amber",
        )
    with k4:
        render_kpi_card(
            "Forecast Quietest Day",
            f"{low_fc['predicted_demand']:.1f}",
            f"Expected on {low_fc['date']}",
            accent="blue",
        )

    st.markdown(
        f'<div class="section-header">Predicted Daily Room Demand — {horizon_choice}</div>',
        unsafe_allow_html=True,
    )
    col_trend, col_bar = st.columns(2)
    with col_trend:
        fig_line = px.line(
            daily_fc,
            x="date",
            y="predicted_demand",
            markers=True,
            labels={"date": "Forecast Date", "predicted_demand": "Predicted Rooms"},
        )
        fig_line.update_traces(line=dict(color=PRIMARY_BLUE, width=2.5), marker=dict(size=7))
        st.plotly_chart(
            _apply_plotly_layout(fig_line, f"Demand Forecast Trend ({horizon_choice})", 370),
            width="stretch",
        )

    with col_bar:
        fig_bar = px.bar(
            daily_fc,
            x="date",
            y="predicted_demand",
            text_auto=".1f",
            labels={"date": "Forecast Date", "predicted_demand": "Predicted Rooms"},
        )
        fig_bar.update_traces(marker_color=ACCENT_TEAL)
        st.plotly_chart(
            _apply_plotly_layout(fig_bar, f"Daily Predicted Volume ({horizon_choice})", 370),
            width="stretch",
        )

    st.markdown(
        '<div class="section-header">Hotel-Wise & Room-Category-Wise Forecast</div>',
        unsafe_allow_html=True,
    )
    col_h, col_r = st.columns(2)
    with col_h:
        hotel_fc = (
            granular_fc.groupby(["date", "hotel"], as_index=False)["predicted_demand"]
            .sum()
            .sort_values("date")
        )
        fig_hf = px.line(
            hotel_fc,
            x="date",
            y="predicted_demand",
            color="hotel",
            markers=True,
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL],
            labels={"date": "Date", "predicted_demand": "Predicted Demand", "hotel": "Hotel"},
        )
        st.plotly_chart(
            _apply_plotly_layout(fig_hf, f"Hotel-Wise Forecast ({horizon_choice})", 360),
            width="stretch",
        )

    with col_r:
        room_fc = (
            granular_fc.groupby(["room_category", "hotel"], as_index=False)["predicted_demand"]
            .sum()
            .sort_values("predicted_demand", ascending=False)
        )
        fig_rf = px.bar(
            room_fc,
            x="room_category",
            y="predicted_demand",
            color="hotel",
            barmode="group",
            color_discrete_sequence=[PRIMARY_BLUE, ACCENT_TEAL],
            labels={
                "room_category": "Room Category",
                "predicted_demand": f"Total Predicted ({horizon_days}d)",
                "hotel": "Hotel",
            },
        )
        st.plotly_chart(
            _apply_plotly_layout(fig_rf, f"Room-Category Forecast ({horizon_choice})", 360),
            width="stretch",
        )

    st.markdown('<div class="section-header">Forecast Tables & CSV Downloads</div>', unsafe_allow_html=True)
    tab_daily, tab_granular = st.tabs(["Daily Forecast Summary", "Hotel & Room-Category Breakdown (with Recursive Lags)"])
    with tab_daily:
        disp_daily = daily_fc.copy()
        disp_daily["predicted_demand"] = disp_daily["predicted_demand"].round(2)
        st.dataframe(disp_daily, width="stretch", hide_index=True)
    with tab_granular:
        disp_gran = granular_fc.copy()
        disp_gran["predicted_demand"] = disp_gran["predicted_demand"].round(2)
        st.dataframe(disp_gran, width="stretch", hide_index=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="Download Next 7 Days (Daily CSV)",
            data=res["forecast_7_daily"].to_csv(index=False).encode("utf-8"),
            file_name="forecast_next_7_days.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl2:
        st.download_button(
            label="Download Next 14 Days (Daily CSV)",
            data=res["forecast_14_daily"].to_csv(index=False).encode("utf-8"),
            file_name="forecast_next_14_days.csv",
            mime="text/csv",
            width="stretch",
        )
    with dl3:
        st.download_button(
            label=f"Download Detailed {horizon_days}-Day CSV",
            data=granular_fc.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_next_{horizon_days}_days_detailed.csv",
            mime="text/csv",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Main Application Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()

    st.markdown(
        """
        <div class="hero-banner">
            <h1>Hotel Occupancy & Room-Demand Forecasting</h1>
            <p>Apache Spark (PySpark MLlib) analytics & recursive 7-day / 14-day demand forecasting dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Dashboard Section",
            options=[
                "Overview & Data Pipeline",
                "Historical Demand Explorer",
                "Weekday & Seasonality Analysis",
                "PySpark ML Evaluation",
                "7 & 14-Day Demand Forecast",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("## Dataset Upload")
        uploaded_file = st.file_uploader(
            "Upload hotel_bookings.csv",
            type=["csv"],
            help="Upload a CSV with standard hotel booking columns (hotel, is_canceled, arrival_date_*, stays_in_*_nights, reserved_room_type).",
        )

        local_default = Path(__file__).resolve().parent / "hotel_bookings.csv"
        use_local_file = False
        if uploaded_file is None and local_default.is_file():
            use_local_file = st.checkbox(
                "Use local hotel_bookings.csv in app folder",
                value=True,
            )

    csv_bytes: bytes | None = None
    file_label: str = ""

    if uploaded_file is not None:
        csv_bytes = uploaded_file.getvalue()
        file_label = uploaded_file.name
    elif use_local_file and local_default.is_file():
        csv_bytes = local_default.read_bytes()
        file_label = local_default.name

    if csv_bytes is None:
        st.info("Upload your `hotel_bookings.csv` dataset from the sidebar to run the PySpark ETL, model training, and forecasting pipeline.")
        st.markdown(
            """
            ### Expected CSV Columns
            | Column | Description |
            |---|---|
            | `hotel` | Hotel name (e.g., `City Hotel`, `Resort Hotel`) |
            | `is_canceled` | Cancellation indicator (`0` = active booking, `1` = cancelled) |
            | `arrival_date_year` | Year of arrival (e.g., `2016`) |
            | `arrival_date_month` | Full month name (e.g., `July`) |
            | `arrival_date_day_of_month` | Day of month (`1`–`31`) |
            | `stays_in_weekend_nights` | Number of weekend nights |
            | `stays_in_week_nights` | Number of weekday nights |
            | `reserved_room_type` | Room category code (e.g., `A`, `D`, `E`) |
            """
        )
        return

    file_hash = hashlib.sha256(csv_bytes).hexdigest()
    cache_store = st.session_state.setdefault("pipeline_cache", {})

    if file_hash not in cache_store:
        try:
            with st.spinner("Running PySpark ETL, feature engineering, and ML training..."):
                cache_store.clear()
                cache_store[file_hash] = run_pyspark_pipeline(csv_bytes, file_label)
        except ValueError as exc:
            st.error(f"Dataset validation error: {exc}")
            return
        except Exception as exc:
            st.error(f"PySpark pipeline execution failed: {exc}")
            return

    results = cache_store[file_hash]

    with st.sidebar:
        st.divider()
        st.markdown("### Active Pipeline")
        st.caption(f"File: `{results['file_name']}`")
        st.caption(f"Cleaned Rows: `{results['cleaned_count']:,}`")
        st.caption(f"Best Model: `{results['best_name']}`")
        st.caption(f"Last Historical Date: `{results['last_date']}`")

    if page == "Overview & Data Pipeline":
        render_overview_page(results)
    elif page == "Historical Demand Explorer":
        render_historical_page(results)
    elif page == "Weekday & Seasonality Analysis":
        render_weekday_page(results)
    elif page == "PySpark ML Evaluation":
        render_ml_page(results)
    elif page == "7 & 14-Day Demand Forecast":
        render_forecast_page(results)


if __name__ == "__main__":
    main()
