import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_parquet("data/dataset.parquet")

print(df["target_win"].value_counts(normalize=True))  # 👈 QUI

features = [
    "dist_to_tp_at_entry",
    "mae_1m"
]

X = df[features]
y = df["target_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # ⚠️ time series safe
)

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print(f"Accuracy: {score:.2f}")

joblib.dump(model, "models/model_v1.pkl")