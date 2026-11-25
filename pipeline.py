# pipeline.py
"""
Utilities to build sequence windows and snapshot datasets for training.
Assumes the DataFrame `df` has already had indicators added (no NaNs in feature columns).
Index should be chronological (oldest -> newest).
"""

import numpy as np
import pandas as pd


def _validate_df_for_features(df: pd.DataFrame, features: list):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in df: {missing}")


def build_sequences(
    df: pd.DataFrame,
    window: int,
    features: list,
    target_kind: str = "binary",
    up_threshold: float = 0.0,
    down_threshold: float = 0.0,
):
    """
    Build X (nsamples, window, nfeatures) and y (nsamples,) from df.
    - df: chronological dataframe (oldest -> newest)
    - window: lookback length (e.g., 60)
    - features: list of column names used as features (must be present in df)
    - target_kind: "binary" or "ternary"
      * binary: y in {0,1} where 1 means next-day return > up_threshold
      * ternary: y in {0,1,2} mapped to {sell, hold, buy}
    - thresholds: fractional returns (e.g., 0.0 for any positive return = up)
    Returns: X, y, idx_map
      idx_map: list of dataframe indices corresponding to the prediction date for each sample (index of the row used to compute the target)
    """
    _validate_df_for_features(df, features)
    # Drop rows with NaNs in required features
    df2 = df.dropna(subset=features + ["close"]).copy()
    if len(df2) < window + 1:
        raise ValueError("Not enough rows after dropping NaNs to build one sequence.")

    # compute next-day fractional return aligned to current row i
    # r_i = (close_{i+1} / close_i) - 1  -> target attached to row i
    close = df2["close"]
    ret1 = close.shift(-1) / close - 1.0

    values = df2[features].values
    X = []
    y = []
    idx_map = []  # keeps the index label from df2 corresponding to the sample's prediction date

    n = len(df2)
    for i in range(window, n - 0):  # i is the index of the row whose last window ends at i-1 and target is at i
        # window covers rows [i-window, ..., i-1] and target uses row i (next-day return from i -> i+1)
        x_win = values[i - window:i]
        r = ret1.iloc[i - 1]  # careful alignment: we want the return *after* the window; using i-1 if you want window->predict next
        # Alternative alignment: if you want window ending at i and target = return at i -> i+1, use ret1.iloc[i]
        # We choose window ending at i-1 then target = r at i-1+1 = i (works consistently).
        if pd.isna(r):
            # if the return is NaN (last row), skip
            continue

        if target_kind == "binary":
            label = 1 if r > up_threshold else 0
        else:  # ternary
            if r > up_threshold:
                label = 2  # buy
            elif r < down_threshold:
                label = 0  # sell
            else:
                label = 1  # hold

        X.append(x_win)
        y.append(label)
        idx_map.append(df2.index[i])  # index corresponding to the prediction moment

    return np.array(X, dtype=float), np.array(y, dtype=int), idx_map


def build_snapshots(
    df: pd.DataFrame,
    features_snap: list,
    target_kind: str = "binary",
    up_threshold: float = 0.0,
    down_threshold: float = 0.0,
):
    """
    Build snapshot dataset: X_snap (nsamples, nfeatures_snap) and y aligned to next-day return.
    Each row i uses df.loc[i] as features and next-day return (i -> i+1) as target.
    """
    _validate_df_for_features(df, features_snap)
    df2 = df.dropna(subset=features_snap + ["close"]).copy()
    if len(df2) < 2:
        raise ValueError("Not enough rows after dropping NaNs to build snapshots.")

    close = df2["close"]
    ret1 = close.shift(-1) / close - 1.0

    X = df2[features_snap].values.astype(float)
    y = []
    idx_map = []

    for i in range(len(df2)):
        r = ret1.iloc[i]
        if pd.isna(r):
            # last row: no future return -> skip
            continue
        if target_kind == "binary":
            label = 1 if r > up_threshold else 0
        else:
            if r > up_threshold:
                label = 2
            elif r < down_threshold:
                label = 0
            else:
                label = 1
        y.append(label)
        idx_map.append(df2.index[i])

    # trim X to match y length (drop the final row which had no return)
    X = X[: len(y)]
    return np.array(X, dtype=float), np.array(y, dtype=int), idx_map


def time_series_train_val_split(X, y, val_size: float = 0.2):
    """
    Chronological split: first (1 - val_size) for training, last val_size for validation.
    Returns: X_train, X_val, y_train, y_val
    """
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    n = len(X)
    split = int(n * (1.0 - val_size))
    if split < 1:
        raise ValueError("Not enough samples for the requested val_size.")
    X_train = X[:split]
    X_val = X[split:]
    y_train = y[:split]
    y_val = y[split:]
    return X_train, X_val, y_train, y_val


# quick demo usage (uncomment to run)
if __name__ == "__main__":
    import pandas as pd
    from config import FEATURES_SEQ

    # demo assumes you have data/data.csv with datetime index and required columns
    try:
        df = pd.read_csv("data/sample.csv", parse_dates=True, index_col=0).sort_index()
    except Exception as e:
        print("No sample file found. Replace with your own data load. Error:", e)
        exit(0)

    X, y, idx = build_sequences(df, window=60, features=FEATURES_SEQ, target_kind="binary")
    print("Sequences:", X.shape, "Labels:", y.shape)
