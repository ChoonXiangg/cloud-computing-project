



# S&P 500 Day-Trading LSTM Classifier

A pooled LSTM classifier trained on 5 years of 1-minute S&P 500 OHLCV data.
The model predicts **Buy / Hold / Sell** for each bar using a rolling 60-bar
normalised log-return sequence as input.  Take-Profit and Stop-Loss thresholds
are set dynamically from trailing volatility, so label frequency stays balanced
across different market regimes.

## Project layout

```
rawSP500data/          Raw 1-minute CSVs  (TICKER_1min.csv)
processed/             Log returns, normalised — one parquet per ticker per split
labeled/               As above, with vol and label columns added
datasets/              Compact .npz arrays ready for the DataLoader
models/                Best checkpoint and training history
results/               Evaluation artefacts (report, plots, JSON summary)
script1_data_preparation.py
script2_labeling.py
script3_dataset_builder.py
script4_train_lstm.py
script5_evaluation.py
```

## Pipeline description

| Script | Input | Output | What it does |
|--------|-------|--------|--------------|
| script1 | `rawSP500data/` | `processed/` | Filter market hours, compute intraday log returns, z-score normalise (training stats only), split train / val / test |
| script2 | `processed/` | `labeled/` | Assign Buy / Hold / Sell labels using volatility-scaled TP/SL thresholds; within-day lookahead only; no look-ahead bias in vol estimate |
| script3 | `labeled/` | `datasets/` | Drop unlabelled rows, save compact `.npz` per ticker, compute class weights |
| script4 | `datasets/` | `models/` | Train 2-layer LSTM with weighted loss, AdamW, early stopping |
| script5 | `datasets/test/`, `models/` | `results/` | Classification report, confusion matrix, training curves, return analysis |

### Date splits
| Split | Date range |
|-------|------------|
| Train | 2020-12-28 – 2023-12-31 |
| Val   | 2024-01-01 – 2024-12-31 |
| Test  | 2025-01-01 – 2025-12-23 |

### Label logic
For each bar *t*:
- `vol_t = std(log_return[t-60 … t-1])` — trailing 60-bar window, shifted by 1 so the current bar is excluded
- `TP = vol_t × 1.5`,  `SL = vol_t × 1.5`
- Look ahead up to 30 bars **within the same trading day**
- First cumulative return to reach +TP → **Buy (0)**
- First cumulative return to reach −SL → **Sell (2)**
- Neither hit → **Hold (1)**

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Run

Execute the scripts in order:

```bash
python script1_data_preparation.py
python script2_labeling.py
python script3_dataset_builder.py
python script4_train_lstm.py
python script5_evaluation.py
```

Each script prints progress and a row-count / metric summary on completion.
GPU is used automatically if available (CUDA); falls back to CPU otherwise.
