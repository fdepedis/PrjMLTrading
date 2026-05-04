import duckdb
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

def train():
    df = load_dataset()

    print(f"Dataset size: {len(df)}")

    # distribuzione target
    print("\nTarget distribution:")
    print(df["target_win"].value_counts(normalize=True))

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTrain Accuracy: {acc:.4f}")

    # feature importance
    print("\nFeature importance:")
    for name, val in zip(X.columns, model.feature_importances_):
        print(f"{name}: {val:.4f}")

    # salva modello
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel salvato in: {MODEL_PATH}")


if __name__ == "__main__":
    train()