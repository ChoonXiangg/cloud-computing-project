"""
script1_data_preparation.py
============================
- Loads raw 1-minute OHLCV CSVs from rawSP500data/
- Filters to market hours 09:30–16:00 only
- Computes intraday log returns per (Ticker, Day) — first bar of each day gets
  NaN and is dropped so no overnight gaps appear in the return series
- Splits into train (≤ 2023-12-31) / val (2024) / test (2025)
- Fits a z-score scaler on training log returns only, then applies to all splits
- Saves one parquet per ticker per split in processed/{split}/
  Each file columns: Date, Ticker, log_return, normalized_return
- Saves processed/scaler.json with mean and std
"""

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DIR       = "rawSP500data"
PROCESSED_DIR = "processed"

TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_END   = pd.Timestamp("2024-12-31 23:59:59")
# Test: 2025-01-01 onwards

MARKET_OPEN  = pd.Timestamp("09:30").time()
MARKET_CLOSE = pd.Timestamp("16:00").time()


# ── Helpers ─────────────────────────────────────────────────────────────────
def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    t = df["Date"].dt.time
    return df[(t >= MARKET_OPEN) & (t <= MARKET_CLOSE)].copy()


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns within each (Ticker, calendar-date) group.
    The first bar of each day becomes NaN (no previous bar in the same day)
    and is dropped, ensuring no overnight gap contaminates the return series.
    """
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["_date"] = df["Date"].dt.date
    df["log_return"] = df.groupby(["Ticker", "_date"])["Close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    df = df.drop(columns=["_date", "Close"])
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(PROCESSED_DIR, split), exist_ok=True)

    files = sorted(glob(os.path.join(RAW_DIR, "*_1min.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}/")
    print(f"Found {len(files)} ticker files.")

    # ── Pass 1: load, filter, compute log returns ────────────────────────────
    ticker_dfs: dict[str, pd.DataFrame] = {}
    train_log_returns: list[np.ndarray] = []

    print("\nPass 1 — loading and computing log returns …")
    for fpath in tqdm(files, ncols=80):
        df = pd.read_csv(fpath, parse_dates=["Date"])
        df = df[["Date", "Close", "Ticker"]]
        df = filter_market_hours(df)
        if df.empty:
            continue
        df = compute_log_returns(df)
        if df.empty:
            continue

        ticker = df["Ticker"].iloc[0]
        ticker_dfs[ticker] = df

        train_lr = df.loc[df["Date"] <= TRAIN_END, "log_return"].values
        if len(train_lr):
            train_log_returns.append(train_lr)

    print(f"Loaded {len(ticker_dfs)} tickers with data.")

    # ── Fit scaler on training log returns only ───────────────────────────────
    all_train_lr = np.concatenate(train_log_returns)
    lr_mean = float(np.mean(all_train_lr))
    lr_std  = float(np.std(all_train_lr, ddof=1))
    print(f"\nScaler  mean={lr_mean:.8f}  std={lr_std:.8f}")

    scaler_path = os.path.join(PROCESSED_DIR, "scaler.json")
    with open(scaler_path, "w") as fh:
        json.dump({"mean": lr_mean, "std": lr_std}, fh, indent=2)
    print(f"Saved scaler → {scaler_path}")

    # ── Pass 2: normalise and split ──────────────────────────────────────────
    print("\nPass 2 — normalising and splitting …")
    split_counts = {"train": 0, "val": 0, "test": 0}

    for ticker, df in tqdm(ticker_dfs.items(), ncols=80):
        df = df.copy()
        df["normalized_return"] = (df["log_return"] - lr_mean) / lr_std

        splits = {
            "train": df["Date"] <= TRAIN_END,
            "val":   (df["Date"] > TRAIN_END) & (df["Date"] <= VAL_END),
            "test":  df["Date"] > VAL_END,
        }
        for split_name, mask in splits.items():
            sub = df[mask]
            if sub.empty:
                continue
            out = os.path.join(PROCESSED_DIR, split_name, f"{ticker}.parquet")
            sub.to_parquet(out, index=False)
            split_counts[split_name] += len(sub)

    print("\nRow counts per split:")
    for split_name, cnt in split_counts.items():
        print(f"  {split_name:6s}: {cnt:>12,}")
    print("\nDone — script 1.")


if __name__ == "__main__":
    main()
