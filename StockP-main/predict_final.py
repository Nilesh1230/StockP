#!/usr/bin/env python3
"""
FINAL PREDICT SCRIPT - ARCHITECTURE MATCHED TO TRAINING
Training used:
 - gat_h = 32
 - hidden = 128
"""

import os, sys, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from datetime import date, timedelta

SEQ_LENGTH = 30
MODEL_FILE = "gat_lstm_model.pth"
DATA_FILE = "data/model_ready/combined_denoised_prices.csv"
ADJ_FILE  = "data/model_ready/adjacency_matrix.npy"
OUT_DIR = "data/model_ready"
EPS = 1e-9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)

# ======================================================
# MODEL (MATCHES TRAINING EXACTLY)
# ======================================================
class HybridGATLSTM_for_predict(nn.Module):
    def __init__(self, num_nodes, gat_h=32, hidden=128):
        super().__init__()
        self.gat_h = gat_h
        self.gat = GATConv(1, gat_h, heads=1, concat=False)

        # LSTM hidden size matches TRAINING hidden
        self.lstm = nn.LSTM(gat_h * num_nodes, hidden, batch_first=True)

        # return head now matches hidden=128
        self.return_head = nn.Linear(hidden, num_nodes)

    def forward(self, x, edge_index):
        x = x.float()
        B, S, N = x.shape
        seq_emb = []
        edge_index = edge_index.long().to(x.device)

        for t in range(S):
            features = []
            for b in range(B):
                xt = x[b, t, :].reshape(N, 1)
                h_b = torch.relu(self.gat(xt, edge_index))
                features.append(h_b.reshape(1, -1))
            seq_emb.append(torch.cat(features, dim=0))

        lstm_in = torch.stack(seq_emb, dim=1)
        out, _ = self.lstm(lstm_in)
        z = out[:, -1, :]
        return self.return_head(z)

# ======================================================
def next_business_day(d):
    wd = d.weekday()
    return d + timedelta(days=1) if wd < 4 else d + timedelta(days=7 - wd)

# ======================================================
def resolve_ticker(ticker, cols):
    raw = ticker.upper().strip()
    candidates = [raw, raw + ".NS", raw.replace(".NS","") + ".NS"]
    for c in candidates:
        if c in cols:
            return c
    return None

# ======================================================
def predict(ticker):

    if not os.path.exists(MODEL_FILE):
        print("❌ Model missing:", MODEL_FILE)
        return

    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    cols = list(df.columns)
    mu = np.load(os.path.join(OUT_DIR, "logret_mu.npy"))
    sd = np.load(os.path.join(OUT_DIR, "logret_sd.npy"))

    resolved = resolve_ticker(ticker, cols)
    if resolved is None:
        print("❌ Ticker not found:", ticker)
        return

    print(f"\n>> Resolved Ticker = {resolved}")

    idx = cols.index(resolved)
    prices = df.values.astype(float)

    # Build logreturns
    rets = pd.DataFrame(prices).pct_change().fillna(0).values
    logrets = np.log1p(rets)

    # Last 30 window
    X_last = logrets[-SEQ_LENGTH:]
    X_norm = (X_last - mu.reshape(1,-1)) / sd.reshape(1,-1)

    X_input = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Adjacency
    adj = np.load(ADJ_FILE)
    ei = np.vstack(np.where(adj == 1))
    edge_index = torch.tensor(ei, dtype=torch.long)

    # Load model EXACTLY like training
    model = HybridGATLSTM_for_predict(
        num_nodes=len(cols), gat_h=32, hidden=128
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred_lr = model(X_input, edge_index.to(DEVICE)).cpu().numpy()[0]

    pred_lr_t = pred_lr[idx]
    last_price = df.iloc[-1, idx]
    pred_price = last_price * np.exp(pred_lr_t)
    pct = (pred_price - last_price)/last_price * 100
    next_day = next_business_day(df.index[-1].date())

    print("\n================= PREDICTION =================")
    print("Ticker         :", resolved)
    print("Last Close     :", f"{last_price:.4f}")
    print("Predicted Close:", f"{pred_price:.4f}", f"({pct:+.2f}%)")
    print("Next Day       :", next_day)
    print("================================================\n")

# ======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_final.py TCS")
    else:
        predict(sys.argv[1])
