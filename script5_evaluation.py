"""
script5_evaluation.py
======================
Evaluates the best saved LSTM checkpoint on the held-out test split.

Outputs
-------
results/
  classification_report.txt    — per-class precision / recall / F1
  confusion_matrix.png         — heatmap
  training_curves.png          — loss & accuracy over epochs
  return_simulation.txt        — simple forward-return analysis per prediction
  evaluation_summary.json      — machine-readable metrics

Usage
-----
  python script5_evaluation.py

Expects
-------
  models/best_model.pth        — saved by script 4
  models/training_history.json — saved by script 4
  datasets/test/               — .npz files from script 3
  datasets/stats.json          — class weights / counts
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from glob import glob
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Reuse Dataset and Model from script 4 ────────────────────────────────────
# (defined inline to keep the script self-contained)

DATASETS_DIR = "datasets"
MODELS_DIR   = "models"
RESULTS_DIR  = "results"

SEQ_LEN      = 60
BATCH_SIZE   = 1024
NUM_WORKERS  = 0
CLASS_NAMES  = ["Buy", "Hold", "Sell"]
N_CLASSES    = 3


# ── Dataset (copied from script 4 for self-containment) ──────────────────────
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.ticker_returns: list[np.ndarray] = []
        self.ticker_labels:  list[np.ndarray] = []
        idx_list: list[tuple[int, int]] = []

        files = sorted(glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        for t_idx, fpath in enumerate(files):
            data = np.load(fpath)
            ret  = data["normalized_return"].astype(np.float32)
            lbl  = data["label"].astype(np.int64)
            self.ticker_returns.append(ret)
            self.ticker_labels.append(lbl)
            for pos in range(seq_len, len(ret)):
                idx_list.append((t_idx, pos))

        self.indices = np.array(idx_list, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t_idx, pos = int(self.indices[idx, 0]), int(self.indices[idx, 1])
        ret = self.ticker_returns[t_idx]
        x   = ret[pos - self.seq_len : pos]
        y   = self.ticker_labels[t_idx][pos]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.long)


# ── Model (identical to script 4) ────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 n_classes, lstm_dropout, fc_dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ── Helpers ───────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_curves(history: list[dict], save_path: str):
    epochs     = [r["epoch"]      for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss   = [r["val_loss"]   for r in history]
    train_acc  = [r["train_acc"]  for r in history]
    val_acc    = [r["val_acc"]    for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train loss")
    ax1.plot(epochs, val_loss,   label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(epochs, train_acc, label="Train acc")
    ax2.plot(epochs, val_acc,   label="Val acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def forward_return_analysis(
    all_true: np.ndarray,
    all_pred: np.ndarray,
    all_returns: np.ndarray,
) -> dict:
    """
    For each predicted class, compute the distribution of the actual
    normalised log-return at the predicted bar.
    This is a rough indicator of directional signal quality — a good
    Buy signal should correlate with positive future returns.

    NOTE: all_returns here is the normalised return at position `pos`
    (the bar being classified), NOT a look-ahead return.  It serves
    as a contemporaneous sanity check, not a backtest.
    """
    result = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = all_pred == cls_idx
        if mask.sum() == 0:
            result[cls_name] = {}
            continue
        rets = all_returns[mask]
        result[cls_name] = {
            "count": int(mask.sum()),
            "mean_return":   float(np.mean(rets)),
            "median_return": float(np.median(rets)),
            "std_return":    float(np.std(rets)),
        }
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = os.path.join(MODELS_DIR, "best_model.pth")
    ckpt      = torch.load(ckpt_path, map_location=device)
    cfg       = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f})")

    model = LSTMClassifier(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Load test dataset ─────────────────────────────────────────────────────
    print("\nLoading test dataset …")
    test_ds     = StockDataset(os.path.join(DATASETS_DIR, "test"),
                               seq_len=cfg["seq_len"])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    print(f"  {len(test_ds):,} test samples")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_true    = []
    all_pred    = []
    all_prob    = []
    all_returns = []   # normalised return at the predicted bar (last element of window)

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_true.extend(y.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.append(probs)
            # last bar of window = x[:, -1, 0]
            all_returns.extend(x[:, -1, 0].numpy().tolist())

    all_true    = np.array(all_true,    dtype=np.int64)
    all_pred    = np.array(all_pred,    dtype=np.int64)
    all_prob    = np.concatenate(all_prob, axis=0)   # (N, 3)
    all_returns = np.array(all_returns, dtype=np.float32)

    # ── Classification metrics ────────────────────────────────────────────────
    acc     = accuracy_score(all_true, all_pred)
    f1_mac  = f1_score(all_true, all_pred, average="macro",    zero_division=0)
    f1_wt   = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    cm      = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
    report  = classification_report(all_true, all_pred,
                                    target_names=CLASS_NAMES, zero_division=0)

    print(f"\nTest accuracy         : {acc:.4f}")
    print(f"Macro   F1            : {f1_mac:.4f}")
    print(f"Weighted F1           : {f1_wt:.4f}")
    print("\nClassification report:")
    print(report)

    # ── Save text report ──────────────────────────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as fh:
        fh.write(f"Test accuracy  : {acc:.4f}\n")
        fh.write(f"Macro   F1     : {f1_mac:.4f}\n")
        fh.write(f"Weighted F1    : {f1_wt:.4f}\n\n")
        fh.write(report)
        fh.write("\nConfusion matrix (rows=true, cols=predicted):\n")
        fh.write("          Buy   Hold   Sell\n")
        for i, name in enumerate(CLASS_NAMES):
            fh.write(f"  {name:4s}  {cm[i, 0]:6d} {cm[i, 1]:6d} {cm[i, 2]:6d}\n")
    print(f"  Saved: {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        cm, os.path.join(RESULTS_DIR, "confusion_matrix.png"))

    hist_path = os.path.join(MODELS_DIR, "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            history = json.load(fh)
        plot_training_curves(
            history, os.path.join(RESULTS_DIR, "training_curves.png"))

    # ── Return simulation ─────────────────────────────────────────────────────
    ret_analysis = forward_return_analysis(all_true, all_pred, all_returns)
    sim_path = os.path.join(RESULTS_DIR, "return_simulation.txt")
    with open(sim_path, "w") as fh:
        fh.write("Return distribution at predicted bar (normalised log-return)\n")
        fh.write("=" * 60 + "\n")
        for cls_name, stats in ret_analysis.items():
            if not stats:
                continue
            fh.write(f"\n{cls_name}  (n={stats['count']:,})\n")
            fh.write(f"  mean   : {stats['mean_return']:+.6f}\n")
            fh.write(f"  median : {stats['median_return']:+.6f}\n")
            fh.write(f"  std    : {stats['std_return']:.6f}\n")
    print(f"  Saved: {sim_path}")

    # ── Confusion matrix per true class (precision / recall summary) ──────────
    print("\nPer-class confusion (rows=true, cols=predicted):")
    header = "".join(f"{n:>8}" for n in CLASS_NAMES)
    print(f"{'':>6}{header}")
    for i, name in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i, j]:>8d}" for j in range(N_CLASSES))
        print(f"  {name:4s}{row}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "accuracy":    round(acc,    4),
        "macro_f1":    round(f1_mac, 4),
        "weighted_f1": round(f1_wt,  4),
        "n_test_samples": int(len(all_true)),
        "confusion_matrix": cm.tolist(),
        "return_analysis": ret_analysis,
    }
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved summary → {summary_path}")
    print("Done — script 5.")


if __name__ == "__main__":
    main()
