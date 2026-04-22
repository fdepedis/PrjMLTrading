import joblib

model = joblib.load("models/model_v1.pkl")

print(model.feature_importances_)