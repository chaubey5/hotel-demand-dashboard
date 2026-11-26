import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Hotel Demand Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {background: #f5f7fb;}
    [data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e7ebf3;
    }
    div[data-testid="stSidebar"] {background: #101c35;}
    div[data-testid="stSidebar"] * {color: white !important;}
    h1, h2, h3 {color: #14213d;}
</style>
""", unsafe_allow_html=True)

st.title("🏨 Hotel Room Demand Forecasting")
st.caption("An interactive dashboard to explore hotel bookings, compare machine-learning models, and forecast room demand.")

@st.cache_data(show_spinner=False)
def prepare_data(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))

    required = [
        "hotel", "arrival_date_year", "arrival_date_month",
        "arrival_date_day_of_month", "is_canceled",
        "stays_in_week_nights", "stays_in_weekend_nights"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df = df.dropna(subset=required).copy()
    df["stays_in_week_nights"] = pd.to_numeric(df["stays_in_week_nights"], errors="coerce").fillna(0)
    df["stays_in_weekend_nights"] = pd.to_numeric(df["stays_in_weekend_nights"], errors="coerce").fillna(0)
    df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce")
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(int).astype(str) + "-" +
        df["arrival_date_month"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(int).astype(str),
        format="%Y-%B-%d", errors="coerce"
    )
    df["stay_nights"] = (df["stays_in_week_nights"] + df["stays_in_weekend_nights"]).astype(int)
    if "reserved_room_type" in df.columns:
        df["room_category"] = df["reserved_room_type"].fillna("Unknown").astype(str)
    else:
        df["room_category"] = "Unknown"
    df = df.dropna(subset=["arrival_date"])
    df = df[df["stay_nights"] > 0].copy()

    bookings = df[df["is_canceled"] == 0].copy()
    daily_rows = []
    for row in bookings[["hotel", "room_category", "arrival_date", "stay_nights"]].itertuples(index=False):
        dates = pd.date_range(row.arrival_date, periods=int(row.stay_nights), freq="D")
        daily_rows.extend((row.hotel, row.room_category, d) for d in dates)

    if not daily_rows:
        raise ValueError("No valid non-cancelled stays were found in this file.")

    daily = pd.DataFrame(daily_rows, columns=["hotel", "room_category", "date"])
    daily = daily.groupby(["hotel", "room_category", "date"]).size().reset_index(name="demand")
    return df, daily

@st.cache_data(show_spinner=False)
def make_model_data(daily):
    parts = []
    for (hotel, room), group in daily.groupby(["hotel", "room_category"]):
        series = group.set_index("date")["demand"].sort_index().asfreq("D", fill_value=0)
        part = pd.DataFrame({"date": series.index, "demand": series.values})
        part["hotel"] = hotel
        part["room_category"] = room
        part["lag_1"] = part["demand"].shift(1)
        part["lag_7"] = part["demand"].shift(7)
        part["lag_14"] = part["demand"].shift(14)
        part["weekday"] = part["date"].dt.dayofweek
        part["month"] = part["date"].dt.month
        parts.append(part)
    if not parts:
        raise ValueError("Not enough demand data to build model features.")
    data = pd.concat(parts, ignore_index=True).dropna()
    return data.sort_values("date").reset_index(drop=True)

def train_models(data):
    features = ["lag_1", "lag_7", "lag_14", "weekday", "month"]
    split_date = data["date"].quantile(0.8)
    train = data[data["date"] < split_date]
    test = data[data["date"] >= split_date]
    if train.empty or test.empty:
        raise ValueError("Not enough date history to split into training and testing data.")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    results, predictions = [], {}
    for name, model in models.items():
        model.fit(train[features], train["demand"])
        pred = np.maximum(model.predict(test[features]), 0)
        predictions[name] = pred
        mae = mean_absolute_error(test["demand"], pred)
        rmse = np.sqrt(mean_squared_error(test["demand"], pred))
        wape = np.abs(test["demand"].to_numpy() - pred).sum() / max(test["demand"].sum(), 1) * 100
        results.append([name, mae, rmse, wape])

    scores = pd.DataFrame(results, columns=["Algorithm", "MAE", "RMSE", "WAPE (%)"])
    best_name = scores.sort_values("MAE").iloc[0]["Algorithm"]
    comparison = pd.DataFrame({
        "Date": test["date"].to_numpy(),
        "Actual": test["demand"].to_numpy(),
        "Predicted": predictions[best_name]
    })
    final_model = models[best_name]
    final_model.fit(data[features], data["demand"])
    return scores, best_name, comparison, final_model, features

def create_forecast(daily, model, features, days=14):
    future_rows = []
    for (hotel, room), group in daily.groupby(["hotel", "room_category"]):
        series = group.set_index("date")["demand"].sort_index().asfreq("D", fill_value=0)
        history = list(series.values)
        last_date = series.index.max()
        if len(history) < 14:
            continue
        for step in range(1, days + 1):
            next_date = last_date + pd.Timedelta(days=step)
            row = pd.DataFrame([[
                history[-1], history[-7], history[-14],
                next_date.dayofweek, next_date.month
            ]], columns=features)
            value = max(0, float(model.predict(row)[0]))
            history.append(value)
            future_rows.append([next_date, hotel, room, value, step])
    return pd.DataFrame(
        future_rows,
        columns=["date", "hotel", "room_category", "predicted_demand", "day_ahead"]
    )

with st.sidebar:
    st.header("Dashboard controls")
    uploaded = st.file_uploader("Upload hotel_bookings.csv", type=["csv"])
    st.caption("Upload the Hotel Booking Demand CSV used in the notebook.")
    st.divider()
    st.markdown("**Project models**")
    st.write("• Linear Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")

if uploaded is None:
    st.info("Upload `hotel_bookings.csv` from the sidebar to generate the live dashboard.")
    st.markdown("### What this dashboard includes")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Booking overview**\\n\\nBookings, cancellation rate and average stay.")
    c2.markdown("**Model comparison**\\n\\nMAE, RMSE and WAPE for three algorithms.")
    c3.markdown("**Demand forecast**\\n\\nRoom-demand estimates for the next 7 and 14 days.")
    st.stop()

try:
    with st.spinner("Preparing booking data and training forecasting models..."):
        df, daily = prepare_data(uploaded.getvalue())
        data = make_model_data(daily)
        scores, best_name, comparison, best_model, features = train_models(data)
        forecast = create_forecast(daily, best_model, features, 14)
except Exception as e:
    st.error(f"Could not process this dataset: {e}")
    st.stop()

if forecast.empty:
    st.error("There is not enough history in the dataset to create a 14-day forecast for the selected room groups.")
    st.stop()

with st.sidebar:
    hotel_options = ["All hotels"] + sorted(daily["hotel"].astype(str).unique().tolist())
    selected_hotel = st.selectbox("Hotel", hotel_options)
    room_options = ["All room categories"] + sorted(daily["room_category"].astype(str).unique().tolist())
    selected_room = st.selectbox("Room category", room_options)
    horizon = st.radio("Forecast horizon", [7, 14], horizontal=True)

filtered_daily = daily.copy()
filtered_forecast = forecast.copy()
if selected_hotel != "All hotels":
    filtered_daily = filtered_daily[filtered_daily["hotel"].astype(str) == selected_hotel]
    filtered_forecast = filtered_forecast[filtered_forecast["hotel"].astype(str) == selected_hotel]
if selected_room != "All room categories":
    filtered_daily = filtered_daily[filtered_daily["room_category"].astype(str) == selected_room]
    filtered_forecast = filtered_forecast[filtered_forecast["room_category"].astype(str) == selected_room]

forecast_view = filtered_forecast[filtered_forecast["day_ahead"] <= horizon]
total_bookings = len(df)
cancel_rate = df["is_canceled"].mean() * 100
avg_stay = df["stay_nights"].mean()
avg_daily_demand = filtered_daily.groupby("date")["demand"].sum().mean()

st.success(f"Dashboard ready · Best model by MAE: {best_name}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Booking records", f"{total_bookings:,}")
m2.metric("Cancellation rate", f"{cancel_rate:.1f}%")
m3.metric("Average stay", f"{avg_stay:.1f} nights")
m4.metric("Average daily room demand", f"{avg_daily_demand:,.1f}" if pd.notna(avg_daily_demand) else "N/A")

st.divider()
left, right = st.columns([1.5, 1])
with left:
    st.subheader("Daily room demand")
    demand_chart = filtered_daily.groupby("date", as_index=False)["demand"].sum()
    fig = px.line(demand_chart, x="date", y="demand", markers=True,
                  labels={"date": "Date", "demand": "Rooms"},
                  title="Historical daily demand")
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Bookings by hotel")
    hotel_chart = df["hotel"].value_counts().rename_axis("Hotel").reset_index(name="Bookings")
    fig = px.pie(hotel_chart, names="Hotel", values="Bookings", hole=0.55)
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Machine-learning model comparison")
score_display = scores.copy()
score_display[["MAE", "RMSE", "WAPE (%)"]] = score_display[["MAE", "RMSE", "WAPE (%)"]].round(2)
st.dataframe(score_display, use_container_width=True, hide_index=True)
score_long = scores.melt(id_vars="Algorithm", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Error")
fig = px.bar(score_long, x="Algorithm", y="Error", color="Metric", barmode="group",
             title="Model errors (lower is better)")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Actual vs predicted demand")
fig = px.line(comparison.head(100), x="Date", y=["Actual", "Predicted"],
              title=f"Test-period performance · {best_name}")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader(f"Room-demand forecast · Next {horizon} days")
if forecast_view.empty:
    st.warning("No forecast rows match the selected hotel and room category.")
else:
    daily_forecast = forecast_view.groupby("date", as_index=False)["predicted_demand"].sum()
    fig = px.bar(daily_forecast, x="date", y="predicted_demand",
                 labels={"date": "Date", "predicted_demand": "Estimated rooms"},
                 title=f"Estimated total room demand for the next {horizon} days")
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    table = forecast_view.copy()
    table["predicted_demand"] = table["predicted_demand"].round(1)
    st.dataframe(table, use_container_width=True, hide_index=True)
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast CSV", csv, f"hotel_demand_forecast_{horizon}_days.csv", "text/csv")

st.caption("Method note: Cancelled bookings are excluded. Each non-cancelled booking is counted as one room for each night of its stay. This is an estimate of room demand, not verified physical occupancy.")
