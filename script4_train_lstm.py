"""
script4_train_lstm.py
======================
Trains a two-layer unidirectional LSTM classifier on the pooled
multi-ticker dataset built by scripts 1–3.

Architecture
------------
  Input  : (batch, SEQ_LEN=60, 1)   — normalised log-return sequence
  LSTM   : 2 layers, hidden=128, dropout=0.3
  FC head: Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→3)
  Loss   : CrossEntropyLoss with inverse-frequency class weights
           (loaded from datasets/stats.json)

Dataset / DataLoader
---------------------
StockDataset keeps every ticker's float32 array in RAM and builds
60-bar windows on the fly via __getitem__ — no pre-materialised tensor.
A pre-computed list of (ticker_idx, position) pairs is built at init;
the list itself is stored as a numpy int32 array to minimise overhead.

Training loop
-------------
  - AdamW optimiser, lr=1e-3, weight_decay=1e-5
  - ReduceLROnPlateau (factor=0.5, patience=3) on val loss
  - Early stopping (patience=7 epochs, tracking val loss)
  - Up to MAX_EPOCHS epochs
  - Best checkpoint saved to models/best_model.pth
  - Training history saved to models/training_history.json
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASETS_DIR = "datasets"
MODELS_DIR   = "models"

SEQ_LEN      = 60
INPUT_SIZE   = 1
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
LSTM_DROPOUT = 0.3
FC_DROPOUT   = 0.2
N_CLASSES    = 3

BATCH_SIZE   = 512
MAX_EPOCHS   = 50
LR           = 1e-3
WEIGHT_DECAY = 1e-5
LR_PATIENCE  = 3
LR_FACTOR    = 0.5
ES_PATIENCE  = 7    # early-stopping patience (epochs)

NUM_WORKERS  = 0    # set >0 on Linux for faster data loading
SEED         = 42


# ── Dataset ───────────────────────────────────────────────────────────────────
class StockDataset(Dataset):
    """
    Loads all per-ticker .npz files for a split into RAM.
    Each __getitem__ call slices a SEQ_LEN window ending at `position`
    and returns (x, y) where x shape = (SEQ_LEN, 1).

    Valid positions satisfy:
      - position >= SEQ_LEN          (enough history)
      - label[position] in {0,1,2}   (already guaranteed by script 3,
                                      which dropped label == -1 rows)
    """

    def __init__(self, data_dir: str, seq_len: int = SEQ_LEN):
        self.seq_len = seq_len
        self.ticker_returns: list[np.ndarray] = []
        self.ticker_labels:  list[np.ndarray] = []

        # indices[k] = (ticker_idx, position) stored as int32 pair
        idx_list: list[tuple[int, int]] = []

        files = sorted(glob(os.path.join(data_dir, "*.npz")))
        if not files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")

        for t_idx, fpath in enumerate(files):
            data     = np.load(fpath)
            ret      = data["normalized_return"].astype(np.float32)
            lbl      = data["label"].astype(np.int64)
            n        = len(ret)

            self.ticker_returns.append(ret)
            self.ticker_labels.append(lbl)

            # All positions >= seq_len are valid (label == -1 already removed)
            for pos in range(seq_len, n):
                idx_list.append((t_idx, pos))

        # Store as contiguous int32 array  shape (N, 2)
        self.indices = np.array(idx_list, dtype=np.int32)
        print(f"  Loaded {len(files)} tickers, {len(self.indices):,} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t_idx, pos = int(self.indices[idx, 0]), int(self.indices[idx, 1])
        ret = self.ticker_returns[t_idx]
        x   = ret[pos - self.seq_len : pos]          # (SEQ_LEN,)
        y   = self.ticker_labels[t_idx][pos]
        # x shape → (SEQ_LEN, 1) for LSTM
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor(y, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size:   int = INPUT_SIZE,
        hidden_size:  int = HIDDEN_SIZE,
        num_layers:   int = NUM_LAYERS,
        n_classes:    int = N_CLASSES,
        lstm_dropout: float = LSTM_DROPOUT,
        fc_dropout:   float = FC_DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)        # (batch, seq_len, hidden)
        last   = out[:, -1, :]       # (batch, hidden)  — last time step
        return self.head(last)       # (batch, n_classes)


# ── Training utilities ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Class weights from stats.json ────────────────────────────────────────
    stats_path = os.path.join(DATASETS_DIR, "stats.json")
    with open(stats_path) as fh:
        stats = json.load(fh)

    cw = torch.tensor(stats["train"]["class_weights"], dtype=torch.float32).to(device)
    print(f"Class weights (Buy/Hold/Sell): {cw.cpu().numpy().round(4)}")

    # ── Datasets / loaders ───────────────────────────────────────────────────
    print("\nLoading training dataset …")
    train_ds = StockDataset(os.path.join(DATASETS_DIR, "train"))
    print("Loading validation dataset …")
    val_ds   = StockDataset(os.path.join(DATASETS_DIR, "val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=(device.type == "cuda"))

    # ── Model / optimiser / scheduler ────────────────────────────────────────
    model     = LSTMClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR,
        patience=LR_PATIENCE, verbose=True)
    criterion = nn.CrossEntropyLoss(weight=cw)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    history         = []
    best_val_loss   = float("inf")
    es_counter      = 0
    best_ckpt_path  = os.path.join(MODELS_DIR, "best_model.pth")

    print(f"\nTraining for up to {MAX_EPOCHS} epochs "
          f"(early-stop patience={ES_PATIENCE}) …\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.0f}s)")

        record = dict(epoch=epoch,
                      train_loss=train_loss, train_acc=train_acc,
                      val_loss=val_loss,     val_acc=val_acc)
        history.append(record)

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            es_counter    = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config": dict(
                    input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS, n_classes=N_CLASSES,
                    lstm_dropout=LSTM_DROPOUT, fc_dropout=FC_DROPOUT,
                    seq_len=SEQ_LEN,
                ),
            }, best_ckpt_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            es_counter += 1
            if es_counter >= ES_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = os.path.join(MODELS_DIR, "training_history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nBest val_loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {best_ckpt_path}")
    print(f"History       : {hist_path}")
    print("Done — script 4.")


if __name__ == "__main__":
    main()
