"""
script3_dataset_builder.py
===========================
Converts the labeled per-ticker parquet files into compact numpy archives
ready for the LSTM trainer.

For each ticker / split:
  - Load labeled/{split}/{ticker}.parquet
  - Drop rows with label == -1 (unlabelled)
  - Save datasets/{split}/{ticker}.npz  with arrays:
        normalized_return  float32  (N,)
        label              int8     (N,)   0=Buy 1=Hold 2=Sell
        (N is the number of labelled bars for this ticker / split)

The LSTM trainer (script 4) builds (seq_len=60)-length input windows on the
fly from these arrays using a custom PyTorch Dataset, so we do NOT
materialise the full (samples × 60) tensor here.

Also writes datasets/stats.json containing per-split label counts and
class weights (inverse-frequency, normalised) for the loss function.

Memory note
-----------
Storing raw float32 arrays in .npz is very compact:
  ~500 tickers × 300 k bars × 4 bytes ≈ 600 MB across all splits.
The Dataset at training time keeps every ticker's array in RAM
(~600 MB total) and slices windows on demand — no per-sample copy.
"""

import os
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
LABELED_DIR  = "labeled"
DATASETS_DIR = "datasets"
SPLITS       = ("train", "val", "test")
N_CLASSES    = 3


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)

    all_stats: dict = {}

    for split in SPLITS:
        out_dir = os.path.join(DATASETS_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob(os.path.join(LABELED_DIR, split, "*.parquet")))
        if not files:
            print(f"[{split}] No labeled files found — skipping.")
            continue

        print(f"\nBuilding {split} ({len(files)} tickers) …")
        label_counts = np.zeros(N_CLASSES, dtype=np.int64)
        n_saved = 0

        for fpath in tqdm(files, ncols=80):
            df = pd.read_parquet(fpath,
                                 columns=["normalized_return", "label"])

            # Drop unlabelled rows
            df = df[df["label"] >= 0].reset_index(drop=True)
            if df.empty:
                continue

            norm_ret = df["normalized_return"].values.astype(np.float32)
            labels   = df["label"].values.astype(np.int8)

            ticker = os.path.splitext(os.path.basename(fpath))[0]
            npz_path = os.path.join(out_dir, f"{ticker}.npz")
            np.savez_compressed(npz_path,
                                normalized_return=norm_ret,
                                label=labels)

            for lbl in range(N_CLASSES):
                label_counts[lbl] += int((labels == lbl).sum())
            n_saved += 1

        total = int(label_counts.sum())
        class_weights = (1.0 / np.maximum(label_counts, 1)).astype(np.float64)
        class_weights /= class_weights.sum()

        print(f"  Tickers saved : {n_saved}")
        print(f"  Labelled bars : {total:,}")
        for i, name in enumerate(["Buy", "Hold", "Sell"]):
            pct = 100.0 * label_counts[i] / max(total, 1)
            print(f"    {name:4s}: {label_counts[i]:>10,}  ({pct:.1f}%)  "
                  f"weight={class_weights[i]:.4f}")

        all_stats[split] = {
            "n_tickers":     n_saved,
            "total_bars":    total,
            "label_counts":  label_counts.tolist(),
            "class_weights": class_weights.tolist(),
        }

    stats_path = os.path.join(DATASETS_DIR, "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nSaved dataset stats → {stats_path}")
    print("Done — script 3.")


if __name__ == "__main__":
    main()
