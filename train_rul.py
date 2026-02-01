import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ---------- Load data ----------
df = pd.read_csv("data/train_FD001.txt", sep=" ", header=None)
df = df.loc[:, df.notna().any()]

# ---------- Column names ----------
columns = ["engine_id", "cycle"] + \
          [f"setting_{i}" for i in range(1, 4)] + \
          [f"sensor_{i}" for i in range(1, 22)]
df.columns = columns

# ---------- RUL calculation ----------
max_cycle = df.groupby("engine_id")["cycle"].max()
df["RUL"] = df.apply(
    lambda row: max_cycle[row["engine_id"]] - row["cycle"],
    axis=1
)

# ---------- RUL capping ----------
RUL_CAP = 125
df["RUL"] = df["RUL"].clip(upper=RUL_CAP)

# ---------- Features ----------
sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
X = df[sensor_cols]
y = df["RUL"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Train / Test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------- Model ----------
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

# ---------- Evaluation ----------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# ---------- Save model ----------
joblib.dump(model, "final_rul_model.pkl")
print("Model saved as final_rul_model.pkl")
