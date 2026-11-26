# Hotel Occupancy & Room-Demand Forecasting (PySpark + Streamlit)

An interactive web application built with **Apache Spark (PySpark MLlib)**, **Streamlit**, and **Plotly** that analyzes hotel booking records, computes daily room-night demand, trains and evaluates time-series regression models, and produces **recursive 7-day and 14-day room-demand forecasts**.

---

## Project Structure

```text
hotel-demand-forecast/
│
├── app.py                  # Complete Streamlit + PySpark ML application
├── requirements.txt        # Python dependencies for local & Streamlit Cloud
├── packages.txt            # System OpenJDK package for Streamlit Cloud
├── README.md               # Documentation and deployment guide
├── .gitignore              # Git ignore rules
└── .streamlit/
    └── config.toml         # Theme and upload configuration
```

---

## Key Features

### 1. Data Ingestion & Preprocessing (PySpark)
- Upload any hotel booking CSV (`hotel_bookings.csv`) directly in the sidebar.
- Schema validation, dataset preview, and missing-value audit.
- Drops records missing critical booking/date fields and fills missing `children`, `stays_in_week_nights`, `stays_in_weekend_nights`, and `reserved_room_type` (`Unknown`).
- Constructs `arrival_date`, computes `stay_nights`, filters invalid/zero-night stays, and calculates hotel-level cancellation rates and average stay duration.

### 2. Daily Room-Demand Explorer
- Expands non-cancelled bookings (`is_canceled == 0`) across every night of stay using Spark `sequence()` + `explode()`.
- Aggregates daily room demand by `hotel`, `room_category`, and `date`.
- Interactive date-range, hotel, and room-category filters with responsive KPI cards and Plotly charts.

### 3. Weekday & Seasonality Analysis
- Calculates average room demand across all 7 days of the week (`Sunday`–`Saturday`).
- Automatically identifies the **Busiest Day** and **Quietest Day**.
- Displays hotel-wise weekday curves, monthly seasonality, and downloadable `weekday_analysis.csv`.

### 4. PySpark MLlib Pipeline
- Zero-fills continuous daily calendars per `(hotel, room_category)` pair.
- Encodes categorical features with `StringIndexer` (`hotel_code`, `room_code`).
- Engineers chronological lag and calendar features: `lag_1`, `lag_7`, `lag_14`, `weekday`, and `month`.
- Performs a strict **chronological 80/20 train-test split** (no random shuffling).
- Trains and compares:
  - **Linear Regression**
  - **Decision Tree Regressor** (`maxDepth=5`)
- Evaluates both models using **MAE** and **RMSE**, automatically selects the model with lower MAE, and visualizes **Actual vs Predicted** demand.

### 5. Recursive 7-Day & 14-Day Forecasting
- Uses the winning PySpark model to generate **Next 7 Days** and **Next 14 Days** forecasts.
- **Recursive lag updating:** each predicted day is fed back into `lag_1`, `lag_7`, and `lag_14` for subsequent forecast days instead of reusing static historical lags.
- Interactive daily trend charts, hotel-wise and room-category-wise breakdowns, and downloadable CSV exports (`forecast_next_7_days.csv`, `forecast_next_14_days.csv`, and detailed granular CSVs).

---

## Local Setup & Execution

### Prerequisites
- **Python 3.10 – 3.12** (recommended for PySpark compatibility; Python 3.13 also supported)
- **Java 11 or Java 17 (OpenJDK)** installed and available on `PATH` / `JAVA_HOME`

### Run Locally
```bash
cd hotel-demand-forecast
pip install -r requirements.txt
streamlit run app.py
```
Then open the displayed local URL (typically `http://localhost:8501`) and upload `hotel_bookings.csv` from the sidebar.

---

## Push to GitHub & Deploy on Streamlit Community Cloud

### Step 1: Initialize Git & Commit the Project
```bash
cd hotel-demand-forecast
git init
git add .
git commit -m "Initial commit: PySpark Hotel Demand Forecast Streamlit app"
```

### Step 2: Create a GitHub Repository & Push
1. Go to [GitHub](https://github.com/new) and create a new repository (e.g., `hotel-demand-forecast`). Do **not** initialize it with a README or `.gitignore` (you already have them).
2. Connect your local folder and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/hotel-demand-forecast.git
   git push -u origin main
   ```

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Select:
   - **Repository:** `<your-username>/hotel-demand-forecast`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Streamlit Community Cloud will automatically:
   - Install **OpenJDK** from `packages.txt` (`default-jre`)
   - Install Python packages from `requirements.txt` (`streamlit`, `pyspark`, `pandas`, `numpy`, `plotly`)
   - Apply `.streamlit/config.toml`
5. Click **Deploy!** and wait for the build to complete, then upload your `hotel_bookings.csv` in the app sidebar.
