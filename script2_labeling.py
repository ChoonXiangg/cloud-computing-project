"""
script2_labeling.py
====================
For each ticker / split:
  1. Load processed/{split}/{ticker}.parquet (log_return, normalized_return)
  2. Compute trailing rolling volatility:
       vol_t = std(log_return[t-60 … t-1])
     using .shift(1) so the current bar's return is NOT included.
     Computed per ticker (window may cross day boundaries — overnight gaps are
     absent because script 1 dropped first-bar-of-day returns).
  3. Assign Buy / Hold / Sell labels per bar, looking forward at most
     LOOKAHEAD bars within the same calendar day:
       TP = vol * TP_MUL   (positive threshold → Buy  = 0)
       SL = vol * SL_MUL   (negative threshold → Sell = 2)
     Whichever is hit first wins; if neither is hit → Hold = 1.
     Bars with NaN vol OR the last LOOKAHEAD bars of each day are
     marked -1 (unlabelled) and excluded downstream.
  4. Save labeled/{split}/{ticker}.parquet
     Extra columns: vol, label (int8, -1 = unlabelled)

Design notes
- The shift(1) on log_return before rolling std ensures the volatility
  estimate at time t uses only data up to t-1 (no look-ahead bias).
- The forward cumulative return from bar i to bar i+k within a day is
  computed as cs[i+k+1] - cs[i+1] where cs is the day's cumsum padded
  with a leading 0.  Only within-day bars are used (no cross-day lookahead).
- A bar is labelled only when it has the full LOOKAHEAD window available
  (i.e. it is not one of the last LOOKAHEAD bars of its trading day).
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"
LABELED_DIR   = "labeled"

SEQ_LEN     = 60    # rolling vol window (bars)
LOOKAHEAD   = 30    # bars to look forward for TP / SL
TP_MUL      = 1.5   # TP = vol × TP_MUL
SL_MUL      = 1.5   # SL = vol × SL_MUL
MIN_PERIODS = 20    # minimum bars for a valid rolling-std estimate

SPLITS = ("train", "val", "test")


# ── Core labelling ───────────────────────────────────────────────────────────
def label_day(
    log_ret: np.ndarray,
    vol:     np.ndarray,
    tp_mul:  float,
    sl_mul:  float,
    lookahead: int,
) -> np.ndarray:
    """
    Vectorised label assignment for a single (ticker, calendar-day) block.

    Returns int8 array, same length as log_ret:
        0  = Buy
        1  = Hold
        2  = Sell
       -1  = unlabelled (NaN vol OR bar is within last `lookahead` of day)

    Forward cumulative return from bar i to bar i+k (within the day):
        cum(i, k) = sum(log_ret[i+1 : i+k+1])
                  = cs[i+k+1] - cs[i+1]
    where cs = [0, cumsum(log_ret)] has length n+1.
    """
    n = len(log_ret)
    labels = np.full(n, -1, dtype=np.int8)

    if n < lookahead + 1:
        return labels

    # Padded cumulative sum  (length n+1, cs[0]=0)
    cs = np.empty(n + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(log_ret, out=cs[1:])

    i_arr = np.arange(n, dtype=np.int32)

    # Build forward cumulative matrix  shape (n, lookahead)
    # fwd_cum[i, k-1] = cum return from close_i to close_{i+k}  (k = 1..lookahead)
    fwd_cum = np.full((n, lookahead), np.nan, dtype=np.float32)
    for k in range(1, lookahead + 1):
        end_cs = i_arr + k + 1          # cs index needed: i + k + 1
        valid  = end_cs <= n            # stay within the day
        rows   = i_arr[valid]
        fwd_cum[rows, k - 1] = cs[rows + k + 1] - cs[rows + 1]

    tp = (vol * tp_mul).astype(np.float32)
    sl = (vol * sl_mul).astype(np.float32)

    # Detect TP / SL hits
    buy_hits  = fwd_cum >= tp[:, np.newaxis]   # shape (n, lookahead)
    sell_hits = fwd_cum <= -sl[:, np.newaxis]

    # Index of first hit (lookahead = sentinel for "no hit")
    first_buy  = np.where(buy_hits.any(axis=1),
                          np.argmax(buy_hits,  axis=1), lookahead)
    first_sell = np.where(sell_hits.any(axis=1),
                          np.argmax(sell_hits, axis=1), lookahead)

    result = np.ones(n, dtype=np.int8)   # default: Hold
    result[(first_buy  < first_sell) & (first_buy  < lookahead)] = 0   # Buy
    result[(first_sell < first_buy)  & (first_sell < lookahead)] = 2   # Sell

    # Valid bars: non-NaN vol AND full LOOKAHEAD window available within day
    valid_mask = ~np.isnan(vol) & (i_arr <= n - 1 - lookahead)
    labels[valid_mask] = result[valid_mask]
    return labels


def label_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'vol' and 'label' columns to a single ticker's DataFrame.
    df must contain columns: Date, log_return, normalized_return.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Trailing rolling vol — shift(1) excludes the current bar's return
    df["vol"] = (
        df["log_return"]
        .shift(1)
        .rolling(SEQ_LEN, min_periods=MIN_PERIODS)
        .std()
    )

    df["label"] = np.int8(-1)

    for date_val, grp in df.groupby(df["Date"].dt.date, sort=False):
        lr  = grp["log_return"].values.astype(np.float64)
        vol = grp["vol"].values.astype(np.float64)
        lbl = label_day(lr, vol, TP_MUL, SL_MUL, LOOKAHEAD)
        df.loc[grp.index, "label"] = lbl

    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for split in SPLITS:
        os.makedirs(os.path.join(LABELED_DIR, split), exist_ok=True)

    for split in SPLITS:
        files = sorted(glob(os.path.join(PROCESSED_DIR, split, "*.parquet")))
        if not files:
            print(f"[{split}] No files found — skipping.")
            continue

        print(f"\nLabelling {split} ({len(files)} tickers) …")
        counts = {0: 0, 1: 0, 2: 0, -1: 0}

        for fpath in tqdm(files, ncols=80):
            df = pd.read_parquet(fpath)
            df = label_ticker(df)

            out = os.path.join(LABELED_DIR, split,
                               os.path.basename(fpath))
            df.to_parquet(out, index=False)

            for lbl, cnt in zip(*np.unique(df["label"].values,
                                           return_counts=True)):
                counts[int(lbl)] = counts.get(int(lbl), 0) + int(cnt)

        total = sum(v for k, v in counts.items() if k >= 0)
        print(f"  [Buy={counts[0]:,}  Hold={counts[1]:,}  Sell={counts[2]:,}]"
              f"  unlabelled={counts[-1]:,}  labelled={total:,}")
        if total:
            print(f"  Buy%={100*counts[0]/total:.1f}  "
                  f"Hold%={100*counts[1]/total:.1f}  "
                  f"Sell%={100*counts[2]/total:.1f}")

    print("\nDone — script 2.")


if __name__ == "__main__":
    main()
