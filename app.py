)

plt.title("Hotel Demand Forecast - Next 7 Days")
plt.xlabel("Date")
plt.ylabel("Predicted Room Demand")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------
# 7. Next 14 days graph
# -----------------------------------

plt.figure(figsize=(12, 5))
plt.bar(
    forecast_pd["date"].astype(str),
    forecast_pd["predicted_demand"]
)

plt.title("Hotel Demand Forecast - Next 14 Days")
plt.xlabel("Date")
plt.ylabel("Predicted Room Demand")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# -----------------------------------
# 8. Weekday demand analysis
# -----------------------------------

weekday_summary = (
    daily
    .withColumn("weekday", F.date_format("date", "EEEE"))
    .withColumn("day_number", F.dayofweek("date"))
    .groupBy("weekday", "day_number")
    .agg(
        F.avg("demand").alias("average_demand")
    )
    .orderBy("day_number")
)

weekday_pd = weekday_summary.toPandas()

print("WEEKDAY DEMAND ANALYSIS")
display(weekday_pd)

plt.figure(figsize=(10, 5))
plt.bar(
    weekday_pd["weekday"],
    weekday_pd["average_demand"]
)

plt.title("Average Hotel Demand by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Average Demand")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# -----------------------------------
# 9. Busiest and quietest days
# -----------------------------------

busiest_day = weekday_pd.loc[
    weekday_pd["average_demand"].idxmax()
]

quietest_day = weekday_pd.loc[
    weekday_pd["average_demand"].idxmin()
]

print("BUSIEST DAY:", busiest_day["weekday"])
print(
    "Average Demand:",
    builtins.round(float(busiest_day["average_demand"]), 2)
)

print("\nQUIETEST DAY:", quietest_day["weekday"])
print(
    "Average Demand:",
    builtins.round(float(quietest_day["average_demand"]), 2)
)


# -----------------------------------
# 10. Model performance
# -----------------------------------

print("\nBEST MODEL:", best_name)
display(scores.toPandas())


# -----------------------------------
# 11. Export CSV files
# -----------------------------------

forecast_7_pd.to_csv(
    "forecast_next_7_days.csv",
    index=False
)

forecast_pd.to_csv(
    "forecast_next_14_days.csv",
    index=False
)

weekday_pd.to_csv(
    "weekday_analysis.csv",
    index=False
)

scores.toPandas().to_csv(
    "model_comparison.csv",
    index=False
)

print("\nALL FORECAST FILES EXPORTED SUCCESSFULLY!")
