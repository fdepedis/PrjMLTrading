import joblib
import pandas as pd

model = joblib.load("models/model_v1.pkl")

df = pd.read_parquet("data/dataset.parquet")

features = [
    "dist_to_tp_at_entry",
    "mae_1m"
]

X = df[features]

proba = model.predict_proba(X)

print(proba[:10])

print(model.feature_importances_)