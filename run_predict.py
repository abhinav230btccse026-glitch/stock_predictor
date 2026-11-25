# # run_predict.py

# import os
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# import csv
# from datetime import datetime
# import glob
# from config import SEQ_WINDOWS, FEATURES, TARGET_KIND, UP_THRESHOLD, DOWN_THRESHOLD
# from features.indicators import calculate_indicators
# from features.commentary import generate_commentary
# from pipeline import build_sequences

# # ========== CONFIG ==========
# TICKER = "RELIANCE.NS"      # change to any NSE stock
# LOOKBACK_YEARS = 3
# MODEL_WINDOW = 90            # choose from SEQ_WINDOW (30, 60, 90)
# # MODEL_PATH = f"outputs/lstm_seq_90_20250831_120925.h5"
# # find the latest model for this window
# model_files = glob.glob(f"outputs/lstm_seq_{MODEL_WINDOW}_*.h5")
# if not model_files:
#     raise FileNotFoundError(f"No trained model found for window={MODEL_WINDOW}. Train first.")
# MODEL_PATH = max(model_files, key=os.path.getctime)  # pick the most recent one
# print(f"📂 Loading model from {MODEL_PATH}")
# model = load_model(MODEL_PATH)



# # =============================

# def fetch_data(ticker, years=3):
#     """Fetch daily + hourly data for stock"""
#     daily = yf.download(ticker, period=f"{years}y", interval="1d")
#     hourly = yf.download(ticker, period="60d", interval="1h")
#     return daily, hourly

# def main():
#     print(f"📥 Fetching data for {TICKER}...")
#     daily, hourly = fetch_data(TICKER, LOOKBACK_YEARS)

#     if daily is None or daily.empty:
#         print("❌ No daily data fetched, aborting.")
#         return

#     # Calculate indicators
#     df = calculate_indicators(daily, hourly)

#     # Add OHLCV base cols
#     df["open"] = daily["Open"].squeeze()
#     df["high"] = daily["High"].squeeze()
#     df["low"] = daily["Low"].squeeze()
#     df["close"] = daily["Close"].squeeze()
#     df["volume"] = daily["Volume"].squeeze()

#     # Lowercase columns to match config
#     df.columns = df.columns.str.lower()

#     # Build sequences for chosen window
#     X, y, idx = build_sequences(
#         df,
#         window=SEQ_WINDOWS[-1],   # ✅ correct param name
#         features=FEATURES,   # ✅ your function expects "features"
#         target_kind=TARGET_KIND,
#         up_threshold=UP_THRESHOLD,
#         down_threshold=DOWN_THRESHOLD
#     )


#     if len(X) == 0:
#         print("❌ Not enough data to build sequences.")
#         return

#     # Load trained LSTM model
#     # if not os.path.exists(model_files):
#     #     print(f"❌ Model not found at {model_files}. Train first with run_train.py")
#     #     return
#     # model = load_model(model_files)

#     # Take the latest sequence for prediction
#     latest_X = X[-1:].copy()
#     # pred_probs = model.predict(latest_X)[0]
#     # pred_class = pred_probs.argmax()
#     pred_probs=model.predict(latest_X)[0][0]
#     signal="BUY" if pred_probs> 0.5 else "SELL"
#     pred_class = int(pred_probs.round())

#     # Convert to Buy/Sell/Hold
#     # if pred_class == 1:
#     #     signal = "BUY"
#     # elif pred_class == 2:
#     #     signal = "SELL"
#     # else:
#     #     signal = "HOLD"

#     print(f"\n📈 Prediction for {TICKER} → {signal}")
#     print(f"Probabilities → BUY={pred_probs:.2f},SELL={1 - pred_probs:.2f}")

#     # Generate commentary on latest indicators
#     latest_features = df.iloc[-1].to_dict()
#     commentary = generate_commentary(latest_features,pred_class)

#     print("\n📝 Explanation:")
#     for line in commentary:
#         print("-", line)

    
#     print(f"\n📈 Prediction for {TICKER} → {signal}")
#     #print(f"Probabilities → Hold={pred_probs[0]:.2f}, Buy={pred_probs[1]:.2f}, Sell={pred_probs[2]:.2f}")
#     # For binary model (only 2 outputs)
#     print(f"Probabilities → Sell={pred_probs[0]:.2f}, Buy={pred_probs[1]:.2f}")


#     # Generate commentary on latest indicators
#     latest_features = df.iloc[-1].to_dict()
#     commentary = generate_commentary(latest_features)

#     print("\n📝 Explanation:")
#     for line in commentary:
#         print("-", line)

#     # ================= SAVE OUTPUT =================
#     os.makedirs("outputs", exist_ok=True)
#     save_path = "outputs/predictions.csv"
#     now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     with open(save_path, mode="a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         # Write header only if file is new
#         if f.tell() == 0:
#             writer.writerow(["timestamp", "ticker", "window", "signal",
#                              "prob_hold", "prob_buy", "prob_sell", "commentary"])
#         writer.writerow([
#             now, TICKER, MODEL_WINDOW, signal,
#             f"{pred_probs[0]:.4f}", f"{pred_probs[1]:.4f}", f"{pred_probs[2]:.4f}",
#             " | ".join(commentary)
#         ])

#     print(f"\n💾 Saved prediction log to {save_path}")


# if __name__ == "__main__":
#     main()

#----------------- fixed the issue of index coming in the above code --------------------------------------------------
# run_predict.py

import os
import pandas as pd
import yfinance as yf
import csv
import glob
from datetime import datetime
from keras.models import load_model
from config import SEQ_WINDOWS, FEATURES, TARGET_KIND
from features.indicators import calculate_indicators
from features.commentary import generate_commentary
from pipeline import build_sequences

# ========== CONFIG ==========
TICKER = "RELIANCE.NS"      # change to any NSE stock
LOOKBACK_YEARS = 3
MODEL_WINDOW = 90            # choose from SEQ_WINDOWS (30, 60, 90)

# Find latest trained model for this window
model_files = glob.glob(f"outputs/lstm_seq_{MODEL_WINDOW}_*.h5")
if not model_files:
    raise FileNotFoundError(f"No trained model found for window={MODEL_WINDOW}. Train first.")
MODEL_PATH = max(model_files, key=os.path.getctime)
print(f"📂 Loading model from {MODEL_PATH}")

model = load_model(MODEL_PATH)


# ========== FETCH DATA ==========
def fetch_data(ticker, years=3):
    """Fetch daily + hourly data for stock"""
    daily = yf.download(ticker, period=f"{years}y", interval="1d")
    hourly = yf.download(ticker, period="60d", interval="1h")
    return daily, hourly


# ========== MAIN PREDICTION FLOW ==========
def main():
    print(f"📥 Fetching data for {TICKER}...")
    daily, hourly = fetch_data(TICKER, LOOKBACK_YEARS)

    if daily is None or daily.empty:
        print("❌ No daily data fetched, aborting.")
        return

    # Compute indicators
    df = calculate_indicators(daily, hourly)

    # Add OHLCV base columns
    df["open"] = daily["Open"]
    df["high"] = daily["High"]
    df["low"] = daily["Low"]
    df["close"] = daily["Close"]
    df["volume"] = daily["Volume"]

    # Lowercase columns to match config
    df.columns = df.columns.str.lower()

    # Build sequences
    X, y, idx = build_sequences(
        df,
        window=SEQ_WINDOWS[-1],
        features=FEATURES,
        target_kind=TARGET_KIND,
        up_threshold=0.0,
        down_threshold=0.0
    )

    if len(X) == 0:
        print("❌ Not enough data to build sequences.")
        return

    # Take latest sequence for prediction
    latest_X = X[-1:].copy()
    pred_probs = float(model.predict(latest_X, verbose=0)[0][0])

    # ==== Rule-based 3-signal decision ====
    # Thresholds to convert binary probability into 3 classes
    if pred_probs > 0.55:
        signal = "BUY"
    elif pred_probs < 0.45:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Derived probabilities
    prob_buy = pred_probs
    prob_sell = 1 - pred_probs
    prob_hold = 1 - abs(0.5 - pred_probs) * 2  # uncertainty-based HOLD strength

    # ===== Output summary =====
    print(f"\n📈 Prediction for {TICKER} → {signal}")
    print(f"Probabilities → BUY={prob_buy:.2f}, SELL={prob_sell:.2f}, HOLD={prob_hold:.2f}")

    # Generate commentary based on latest indicators
    latest_features = df.iloc[-1].to_dict()
    commentary = generate_commentary(latest_features, int(round(pred_probs)))
    

# 🔥 FIX: Ensure it's always a list
    if isinstance(commentary, str):
        commentary = [commentary]

    print("\n📝 Explanation:")
    for line in commentary:
        print("-", line)

    # ===== Save Output =====
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/predictions.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(save_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "timestamp", "ticker", "window", "signal",
                "prob_hold", "prob_buy", "prob_sell", "commentary"
            ])
        writer.writerow([
            now, TICKER, MODEL_WINDOW, signal,
            f"{prob_hold:.4f}", f"{prob_buy:.4f}", f"{prob_sell:.4f}",
            " | ".join(commentary)
        ])

    print(f"\n💾 Saved prediction log to {save_path}")

    return {
    "ticker": TICKER,
    "signal": signal,
    "prob_buy": prob_buy,
    "prob_sell": prob_sell,
    "prob_hold": prob_hold,
    "commentary": commentary
    }



 