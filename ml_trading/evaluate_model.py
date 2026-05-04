import duckdb
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from config.settings import DB_PATH, MODEL_PATH


def load_dataset():
    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT *
        FROM v_trade_features_ml
        WHERE target_win IS NOT NULL
    """).df()

    con.close()

    return df

def prepare_features(df: pd.DataFrame):
    features = [
        "entry_hour",
        "distance_sl",
        "distance_tp",
        "rr_ratio",
        "momentum_1m",
        "momentum_5m",
    ]

    # Verifica se vuoi usare altre colonne presenti nell'immagine, come 'r_mult' o 'qty'

    df = df.dropna(subset=features + ["target_win"])
    X = df[features]
    y = df["target_win"]

    return X, y

def evaluate():
    df = load_dataset()

    print(f"Dataset size: {len(df)}")

    print("\nTarget distribution:")
    print(df["target_win"].value_counts(normalize=True))

    X, y = prepare_features(df)

    # split identico al training
    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # carica modello
    model = joblib.load(MODEL_PATH)

    # predizioni
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 🔥 BASELINE (importantissimo)
    baseline_pred = np.full_like(y_test, round(y_test.mean()))
    baseline_acc = accuracy_score(y_test, baseline_pred)

    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")

    # 🔥 probabilità
    y_prob = model.predict_proba(X_test)

    print("\nSample probabilities (first 10):")
    print(y_prob[:10])


if __name__ == "__main__":
    evaluate()