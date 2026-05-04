import duckdb
import pandas as pd

from config.settings import DB_PATH, PARQUET_PATH

def build_dataset():
    con = duckdb.connect(DB_PATH)

    df = con.execute("""
        SELECT *
        FROM v_trade_features_ml
        WHERE r_mult IS NOT NULL
          AND entry_price IS NOT NULL
    """).df()

    # 🎯 target binario (prima versione semplice)
    df["target_win"] = (df["r_mult"] > 0).astype(int)

    # ⚠️ pulizia minima
    df = df.dropna(subset=[
        "dist_to_tp_at_entry",
        "mae_1m"
    ])

    df.to_parquet(PARQUET_PATH, index=False)

    print(f"Dataset salvato: {len(df)} righe")

if __name__ == "__main__":
    build_dataset()