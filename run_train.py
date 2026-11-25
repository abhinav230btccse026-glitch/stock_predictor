# run_train.py

import os
import joblib
import numpy as np
from datetime import datetime
import pandas as pd

from data_loader import fetch_data
from features.indicators import calculate_indicators
from pipeline import build_sequences, time_series_train_val_split
from config import FEATURES_SEQ, SEQ_WINDOWS, TARGET_KIND, UP_THRESHOLD, DOWN_THRESHOLD
from models.lstm_seq import LSTMSeqModel
from backtest.evaluate import eval_binary


def main():
    symbol = "RELIANCE.NS"   # you can loop over multiple stocks later
    print(f"🚀 Training LSTM models for {symbol}")

    # Step 1: fetch data
    daily_data, hourly_data = fetch_data(symbol)
    if daily_data is None or daily_data.empty:
        print("❌ No data fetched")
        return

    # Step 2: calculate indicators
    features = calculate_indicators(daily_data, hourly_data)
    if features.empty:
        print("❌ No features generated")
        return

    # Step 3: merge OHLCV + features
    df = daily_data.copy()
    #flattern multiIndex columns if present
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=['_'.join([str(c) for c in col if c]) for col in df.columns.values]
    else:df.columns=df.columns.astype(str)
    df.columns = df.columns.str.lower()
    df = df.join(features, how="left")
    # Rename OHLCV columns to standard names
    rename_map = {
        "open_reliance.ns": "open",
        "high_reliance.ns": "high",
        "low_reliance.ns": "low",
        "close_reliance.ns": "close",
        "volume_reliance.ns": "volume",
    }
    df = df.rename(columns=rename_map)
    
    

    os.makedirs("outputs", exist_ok=True)

    # Step 4: train models for each SEQ_WINDOW
    for win in SEQ_WINDOWS:
        print(f"\n🟦 Training window={win}")

        X, y, idx = build_sequences(
            df,
            window=win,
            features=FEATURES_SEQ,
            target_kind=TARGET_KIND,
            up_threshold=UP_THRESHOLD,
            down_threshold=DOWN_THRESHOLD
        )

        if len(X) < 50:
            print(f"⚠️ Not enough samples for window={win}")
            continue

        # Chronological split
        X_train, X_val, y_train, y_val = time_series_train_val_split(X, y, val_size=0.2)

        # Train
        model = LSTMSeqModel(seq_window=win, feature_size=X.shape[2], target_kind=TARGET_KIND)
        history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                            val_data=(X_val, y_val))

        # Evaluate
        preds = model.predict_proba(X_val)
        metrics = eval_binary(y_val, preds)
        print(f"📊 Window={win} metrics: {metrics}")

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"outputs/lstm_seq_{win}_{ts}.h5"
        model.save(model_path)
        joblib.dump(metrics, f"outputs/metrics_{win}_{ts}.pkl")
        print(f"✅ Saved model + metrics for window={win}")


if __name__ == "__main__":
    main()
