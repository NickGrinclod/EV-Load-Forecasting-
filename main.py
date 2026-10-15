import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("energydata_complete.csv")

data["date"] = pd.to_datetime(data["date"])


# ============================================================
# 2. TIME FEATURES
# ============================================================

data["hour"] = data["date"].dt.hour

data["day"] = data["date"].dt.date

data["minute_of_day"] = (
    data["date"].dt.hour * 60
    + data["date"].dt.minute
)


# ============================================================
# 3. CREATE 30-MINUTE FUTURE TARGET
# ============================================================

# Dataset interval = 10 minutes
# 3 rows = 30 minutes

data["target_30min"] = data["Appliances"].shift(-3)


# ============================================================
# 4. CREATE POWER HISTORY
# ============================================================

data["Appliances_lag1"] = data["Appliances"].shift(1)   # 10 min ago
data["Appliances_lag2"] = data["Appliances"].shift(2)   # 20 min ago
data["Appliances_lag3"] = data["Appliances"].shift(3)   # 30 min ago


# Same time one day earlier
# 6 measurements/hour × 24 hours = 144 rows

data["Appliances_1day_ago"] = data["Appliances"].shift(144)


# Power-change experiments

data["power_change_10min"] = (
    data["Appliances"] - data["Appliances_lag1"]
)

data["power_change_20min"] = (
    data["Appliances"] - data["Appliances_lag2"]
)

data["power_change_30min"] = (
    data["Appliances"] - data["Appliances_lag3"]
)


# ============================================================
# 5. REMOVE ROWS WITHOUT REQUIRED HISTORY/TARGET
# ============================================================

data = data.dropna(
    subset=[
        "target_30min",
        "Appliances_lag1",
        "Appliances_lag2",
        "Appliances_lag3"
    ]
)


# ============================================================
# 6. MODEL FEATURES
# ============================================================

# Best simple feature set found so far:
# current power + previous 30 minutes

features = [
    "Appliances",
    "Appliances_lag1",
    "Appliances_lag2",
    "Appliances_lag3"
]

target = "target_30min"

X = data[features]
y = data[target]


print("\n========================================")
print("MODEL INPUT FEATURES")
print("========================================")

print(X.head())

print("\nTARGET")
print(y.head())


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

# Keep time order.
# First 80% = training
# Last 20% = testing

split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# ============================================================
# 8. RANDOM FOREST REGRESSION
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf
)

rmse_rf = (
    mean_squared_error(
        y_test,
        y_pred_rf
    ) ** 0.5
)

print("\n========================================")
print("RANDOM FOREST RESULTS")
print("========================================")

print("MAE:", mae_rf)
print("RMSE:", rmse_rf)


# ============================================================
# 9. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

y_pred_linear = linear_model.predict(X_test)

mae_linear = mean_absolute_error(
    y_test,
    y_pred_linear
)

rmse_linear = (
    mean_squared_error(
        y_test,
        y_pred_linear
    ) ** 0.5
)

print("\n========================================")
print("LINEAR REGRESSION RESULTS")
print("========================================")

print("MAE:", mae_linear)
print("RMSE:", rmse_linear)


# ============================================================
# 10. NEURAL NETWORK
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential([
    Input(shape=(4,)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1)
])

nn_model.compile(
    optimizer="adam",
    loss="mse"
)

nn_model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred_nn = (
    nn_model
    .predict(X_test_scaled)
    .flatten()
)

mae_nn = mean_absolute_error(
    y_test,
    y_pred_nn
)

rmse_nn = (
    mean_squared_error(
        y_test,
        y_pred_nn
    ) ** 0.5
)

print("\n========================================")
print("NEURAL NETWORK RESULTS")
print("========================================")

print("MAE:", mae_nn)
print("RMSE:", rmse_nn)


print("\nSAMPLE NEURAL NETWORK PREDICTIONS")

for actual, predicted in zip(
    y_test[:10],
    y_pred_nn[:10]
):
    print(
        f"Actual: {actual:.1f} W   "
        f"Predicted: {predicted:.1f} W"
    )


# ============================================================
# 11. GRAPH:
#     SAME CLOCK TIMES ON DIFFERENT DAYS
# ============================================================

data_12pm = data[
    data["minute_of_day"] == 12 * 60
]

data_1pm = data[
    data["minute_of_day"] == 13 * 60
]

data_3pm = data[
    data["minute_of_day"] == 15 * 60
]

plt.figure(figsize=(12, 6))

plt.plot(
    data_12pm["day"],
    data_12pm["Appliances"],
    label="12:00 PM"
)

plt.plot(
    data_1pm["day"],
    data_1pm["Appliances"],
    label="1:00 PM"
)

plt.plot(
    data_3pm["day"],
    data_3pm["Appliances"],
    label="3:00 PM"
)

plt.xlabel("Date")
plt.ylabel("Power Consumption (W)")

plt.title(
    "Power Consumption at Fixed Times "
    "Across Different Days"
)

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# 12. CORRELATION WITH CURRENT POWER
# ============================================================

correlations = (
    data
    .corr(numeric_only=True)["Appliances"]
    .sort_values(ascending=False)
)

print("\n========================================")
print("CORRELATION WITH CURRENT POWER")
print("========================================")

print(correlations)


# ============================================================
# 13. CORRELATION WITH 30-MINUTE FUTURE TARGET
# ============================================================

correlations_target = (
    data
    .corr(numeric_only=True)["target_30min"]
    .sort_values(ascending=False)
)

print("\n========================================")
print("CORRELATION WITH 30-MINUTE FUTURE POWER")
print("========================================")

print(correlations_target)


# ============================================================
# 14. NEW EXPERIMENT:
#     ANALYZE ALL HIGH-LOAD EVENTS
# ============================================================

HIGH_LOAD_THRESHOLD = 300


# Current row represents what is known NOW.
# target_30min happens 30 minutes later.

data["event_time"] = (
    data["date"]
    + pd.Timedelta(minutes=30)
)


high_load = data[
    data["target_30min"]
    > HIGH_LOAD_THRESHOLD
]

normal_load = data[
    data["target_30min"]
    <= HIGH_LOAD_THRESHOLD
]


print("\n========================================")
print("ALL HIGH-LOAD EVENT ANALYSIS")
print("========================================")

print(
    "High-load threshold:",
    HIGH_LOAD_THRESHOLD,
    "W"
)

print(
    "Number of high-load events:",
    len(high_load)
)

print(
    "Number of normal events:",
    len(normal_load)
)


# ============================================================
# 15. AVERAGE HISTORY BEFORE HIGH-LOAD EVENTS
# ============================================================

history_columns = [
    "Appliances_lag3",
    "Appliances_lag2",
    "Appliances_lag1",
    "Appliances"
]

high_average = (
    high_load[history_columns]
    .mean()
)

normal_average = (
    normal_load[history_columns]
    .mean()
)


print("\n========================================")
print("AVERAGE HISTORY BEFORE HIGH-LOAD EVENTS")
print("========================================")

print(high_average)


print("\n========================================")
print("AVERAGE HISTORY BEFORE NORMAL EVENTS")
print("========================================")

print(normal_average)


# ============================================================
# 16. COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "High_Load": high_average,

    "Normal_Load": normal_average

})

comparison["Difference"] = (
    comparison["High_Load"]
    - comparison["Normal_Load"]
)


print("\n========================================")
print("HIGH LOAD VS NORMAL")
print("========================================")

print(
    comparison.round(2)
)


# ============================================================
# 17. GRAPH:
#     POWER HISTORY BEFORE HIGH LOAD VS NORMAL
# ============================================================

# IMPORTANT:
#
# At the prediction point:
#
# lag3 = 30 min before NOW
#       = 60 min before future event
#
# lag2 = 20 min before NOW
#       = 50 min before future event
#
# lag1 = 10 min before NOW
#       = 40 min before future event
#
# current = NOW
#         = 30 min before future event


history_labels = [
    "60 min before event",
    "50 min before event",
    "40 min before event",
    "30 min before event"
]

high_history = [
    high_average["Appliances_lag3"],
    high_average["Appliances_lag2"],
    high_average["Appliances_lag1"],
    high_average["Appliances"]
]

normal_history = [
    normal_average["Appliances_lag3"],
    normal_average["Appliances_lag2"],
    normal_average["Appliances_lag1"],
    normal_average["Appliances"]
]


plt.figure(figsize=(9, 5))

plt.plot(
    history_labels,
    high_history,
    marker="o",
    label="Future high-load event"
)

plt.plot(
    history_labels,
    normal_history,
    marker="o",
    label="Future normal-load event"
)

plt.xlabel(
    "Time Before Actual Event"
)

plt.ylabel(
    "Power Consumption (W)"
)

plt.title(
    "Average Power History Before "
    "High-Load and Normal Events"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 18. SAMPLE OF HIGH-LOAD EVENTS
# ============================================================

print("\n========================================")
print("FIRST 20 HIGH-LOAD EVENTS")
print("========================================")

high_load_sample = high_load[
    [
        "event_time",
        "target_30min",
        "Appliances",
        "Appliances_lag1",
        "Appliances_lag2",
        "Appliances_lag3"
    ]
].head(20)

print(
    high_load_sample.to_string(
        index=False
    )
)

import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# TIME-OF-DAY ANALYSIS OF HIGH-LOAD EVENTS
# -------------------------------------------------

HIGH_LOAD_THRESHOLD = 300
NUMBER_OF_DAYS = 3

# Make sure Date is datetime
data["date"] = pd.to_datetime(data["date"])

# target_30min represents consumption 30 minutes later,
# so create the actual time of that future event
data["event_time"] = data["date"] + pd.Timedelta(minutes=30)

# Take only the first 3 days
start_date = data["event_time"].min().normalize()
end_date = start_date + pd.Timedelta(days=NUMBER_OF_DAYS)

data_3days = data[
    (data["event_time"] >= start_date) &
    (data["event_time"] < end_date)
].copy()

# Mark high-load events
data_3days["high_load"] = (
    data_3days["target_30min"] >= HIGH_LOAD_THRESHOLD
)

# -------------------------------------------------
# GRAPH 1:
# Target 30 minutes vs actual event time
# -------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    data_3days["event_time"],
    data_3days["target_30min"],
    marker="o",
    markersize=3,
    label="Target 30 min"
)

# Highlight only high-load events
high_events = data_3days[data_3days["high_load"]]

plt.scatter(
    high_events["event_time"],
    high_events["target_30min"],
    s=60,
    label="High-load event"
)

plt.axhline(
    HIGH_LOAD_THRESHOLD,
    linestyle="--",
    label="300 W threshold"
)

plt.xlabel("Actual event time")
plt.ylabel("Target 30 min")
plt.title("High-load events over 3 days")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------------------------
# GRAPH 2:
# Number of high-load events by hour of day
# -------------------------------------------------

high_events = high_events.copy()
high_events["hour"] = high_events["event_time"].dt.hour

hourly_counts = (
    high_events.groupby("hour")
    .size()
    .reindex(range(24), fill_value=0)
)

plt.figure(figsize=(12, 5))

plt.bar(
    hourly_counts.index,
    hourly_counts.values
)

plt.xlabel("Hour of day")
plt.ylabel("Number of high-load events")
plt.title("High-load events by hour of day — first 3 days")
plt.xticks(range(24))
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# Print the events so we can inspect them directly
print(
    high_events[
        ["event_time", "target_30min"]
    ].to_string(index=False)
)

# -------------------------------------------------
# HIGH-LOAD PROBABILITY BY HOUR — FULL DATASET
# -------------------------------------------------

HIGH_LOAD_THRESHOLD = 300

# Hour when the future target event actually occurs
data["event_hour"] = data["event_time"].dt.hour

# Mark whether each target is a high-load event
data["high_load"] = (
    data["target_30min"] >= HIGH_LOAD_THRESHOLD
)

# Percentage of observations that are high-load,
# calculated separately for every hour of the day
hourly_probability = (
    data.groupby("event_hour")["high_load"]
    .mean()
    .mul(100)
    .reindex(range(24), fill_value=0)
)

plt.figure(figsize=(12, 5))

plt.bar(
    hourly_probability.index,
    hourly_probability.values
)

plt.xlabel("Hour of day")
plt.ylabel("High-load event probability (%)")
plt.title(
    "Probability of High-Load Event by Hour of Day — Full Dataset"
)

plt.xticks(range(24))
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\nHigh-load probability by hour:")
print(hourly_probability.round(2))