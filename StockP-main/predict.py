#!/usr/bin/env python3
"""
predict_v2.py
- Uses log-returns (same feature used during training)
- Auto-detect model / meta / denoised file names (tolerant)
- Injects target's denoised log-returns into context
- Normalizes with saved mu / sd (from training)
- Runs Group model and converts predicted log-return -> price
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from datetime import timedelta

# -------------------------
# CONFIG
# -------------------------
SEQ_LENGTH = 30
MODEL_DIR = "models"
META_DIR = "models/meta"
DENOISED_DIR = "data/denoised_universe"
COMBINED = "data/model_ready/combined_denoised_prices.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
                      "cpu")

EPS = 1e-9

# -------------------------
# Small helpers
# -------------------------
def next_business_day(d):
    wd = d.weekday()
    if wd >= 4:
        return d + timedelta(days=7 - wd)
    return d + timedelta(days=1)

def find_file_with_token(folder, token, exts=None):
    """Return first file path in folder whose filename contains token (case-insensitive)."""
    token = token.lower()
    files = os.listdir(folder) if os.path.exists(folder) else []
    if exts:
        files = [f for f in files if any(f.lower().endswith(e) for e in exts)]
    for f in files:
        if token in f.lower():
            return os.path.join(folder, f)
    return None

# -------------------------
# Model (must match training model)
# -------------------------
class GroupGATLSTM(nn.Module):
    def __init__(self, num_nodes, gat_h=32, lstm_h=128, dropout=0.2):
        super().__init__()
        self.N = num_nodes
        self.gat_h = gat_h
        self.lstm_h = lstm_h
        self.gat = GATConv(1, gat_h, heads=1, concat=False, dropout=0.2)
        self.lstm = nn.LSTM(num_nodes * gat_h, lstm_h, num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.return_head = nn.Linear(lstm_h, num_nodes)

    def forward(self, x, edge_index):
        # x: [B, S, N]
        B, S, N = x.shape
        device = x.device
        seq_node_emb = []
        if edge_index.device != device:
            edge_index = edge_index.to(device)

        E = edge_index.size(1)
        for t in range(S):
            xt = x[:, t, :].reshape(B * N, 1).to(device)             # [B*N, 1]
            edge_rep = edge_index.repeat(1, B)                       # [2, E*B]
            offset = (torch.arange(B, device=device).repeat_interleave(E) * N).unsqueeze(0)
            edge_batch = edge_rep + offset                           # [2, E*B]
            h = torch.relu(self.gat(xt, edge_batch))                 # [B*N, gat_h]
            h = h.view(B, N, self.gat_h)                             # [B, N, gat_h]
            seq_node_emb.append(h)
        lstm_in = torch.stack([h.reshape(B, -1) for h in seq_node_emb], dim=1)  # [B, S, N*gat_h]
        out, _ = self.lstm(lstm_in)
        z = out[:, -1, :]
        z = self.dropout(z)
        ret = self.return_head(z)
        return ret

# -------------------------
# MAIN PREDICT FUNCTION
# -------------------------
def predict(ticker):
    ticker_clean = ticker.upper().replace(".NS", "")

    # find model & meta
    model_path = find_file_with_token(MODEL_DIR, ticker_clean, exts=[".pth"])
    meta_path  = find_file_with_token(META_DIR, ticker_clean, exts=[".json"])
    denoised_path = find_file_with_token(DENOISED_DIR, ticker_clean, exts=[".csv"])

    if model_path is None:
        print("ERROR: model file not found for:", ticker_clean, "in", MODEL_DIR)
        return
    if meta_path is None:
        print("ERROR: meta file not found for:", ticker_clean, "in", META_DIR)
        return
    if denoised_path is None:
        print("ERROR: denoised price file not found for:", ticker_clean, "in", DENOISED_DIR)
        return

    # load meta
    meta = json.load(open(meta_path, "r"))
    group_tickers = meta.get("group_tickers")
    mu = np.array(meta.get("mu"))
    sd = np.array(meta.get("sd"))
    edge_index_list = np.array(meta.get("edge_index"))   # expected shape [2, E]

    if group_tickers is None or mu is None or sd is None:
        print("Meta missing required fields (group_tickers / mu / sd).")
        return

    M = len(group_tickers)

    # load denoised series (target)
    df_den = pd.read_csv(denoised_path, parse_dates=["Date"] if "Date" in pd.read_csv(denoised_path, nrows=0).columns else None)
    if "Date" in df_den.columns:
        df_den["Date"] = pd.to_datetime(df_den["Date"])
        df_den = df_den.set_index("Date")

    # pick price column robustly
    if "Denoised_Close" in df_den.columns:
        series = df_den["Denoised_Close"].astype(float)
    elif "Close" in df_den.columns:
        series = df_den["Close"].astype(float)
    else:
        # take last numeric column
        numeric_cols = df_den.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("No numeric column in denoised file:", denoised_path)
            return
        series = df_den[numeric_cols[-1]].astype(float)

    if len(series) < SEQ_LENGTH:
        print("ERROR: not enough denoised rows:", len(series), "need at least", SEQ_LENGTH)
        return

    last_price = float(series.iloc[-1])

    # compute log-returns for target exactly as training did:
    # returns = pct_change().fillna(0) ; lr = log1p(returns)  -> shape same as prices
    series_returns = series.pct_change().fillna(0).values
    series_lr = np.log1p(series_returns)   # length = len(series)
    seq_target_lr = series_lr[-SEQ_LENGTH:]  # shape (SEQ_LENGTH,)

    # load combined universe and convert to log-returns (same pipeline)
    if not os.path.exists(COMBINED):
        print("ERROR: combined matrix missing at:", COMBINED)
        return
    combined_df = pd.read_csv(COMBINED, index_col=0, parse_dates=True)
    # ensure group tickers exist in combined; meta likely stores 'TCS.NS' etc
    missing = [g for g in group_tickers if g not in combined_df.columns and g.replace(".NS","") not in combined_df.columns]
    if missing:
        print("WARNING: these group tickers missing in combined matrix:", missing)
    # align columns: prefer exact match, otherwise try without .NS
    cols_in_combined = []
    for g in group_tickers:
        if g in combined_df.columns:
            cols_in_combined.append(g)
        elif g.replace(".NS","") in combined_df.columns:
            cols_in_combined.append(g.replace(".NS",""))
        else:
            # fallback: try any column that contains token
            cand = find_file_with_token("data/model_ready", g.replace(".NS",""), exts=[".csv"])  # not ideal but skip
            cols_in_combined.append(g)  # keep original and hope it exists; we already warned
    # build lr matrix for combined (same transform)
    returns_df = combined_df.pct_change().fillna(0)
    lr_df = np.log1p(returns_df)
    ctx_last = lr_df.tail(SEQ_LENGTH)[group_tickers].values.astype(float)  # [SEQ_LENGTH, M]

    # find target index inside group_tickers robustly
    possible_names = [ticker_clean + ".NS", ticker_clean]
    t_idx = None
    for i, name in enumerate(group_tickers):
        if name in possible_names or name.replace(".NS","") in possible_names:
            t_idx = i
            break
    if t_idx is None:
        # fallback index 0 but warn
        t_idx = 0
        print("WARNING: target not found in group_tickers, using index 0. group_tickers:", group_tickers)

    # inject target's log-returns (important: we inject LR not prices)
    if ctx_last.shape[0] != SEQ_LENGTH:
        print("WARNING: combined context has fewer rows than SEQ_LENGTH; padding using last row.")
        last_row = ctx_last[-1:]
        pad_count = SEQ_LENGTH - ctx_last.shape[0]
        ctx_last = np.vstack([np.tile(last_row, (pad_count, 1)), ctx_last])

    ctx_last = ctx_last.copy()
    ctx_last[:, t_idx] = seq_target_lr  # inject log-returns

    # ensure mu/sd shapes align with ctx_last columns
    mu = mu.reshape(-1)
    sd = sd.reshape(-1)
    if mu.shape[0] != ctx_last.shape[1] or sd.shape[0] != ctx_last.shape[1]:
        print("ERROR: mu/sd shape mismatch -> meta mu/sd shapes:", mu.shape, sd.shape, "vs ctx cols:", ctx_last.shape[1])
        print("Try inspecting meta file:", meta_path)
        return

    # normalize using training mu/sd
    X_norm = (ctx_last - mu.reshape(1, -1)) / (sd.reshape(1, -1) + EPS)
    X_norm = X_norm.astype(np.float32)
    x_tensor = torch.tensor(X_norm).unsqueeze(0).to(DEVICE)  # [1, S, M]

    # load edge_index
    try:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    except Exception:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    # load model
    model = GroupGATLSTM(num_nodes=len(group_tickers)).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # predict
    with torch.no_grad():
        preds = model(x_tensor, edge_index.to(DEVICE)).cpu().numpy()   # shape [1, M] -> take [0]
    preds = preds[0]
    pred_lr_target = float(preds[t_idx])

    # sanity: compare predicted lr magnitude with historical
    hist_lr_target = seq_target_lr
    hist_mean = float(np.mean(hist_lr_target))
    hist_std  = float(np.std(hist_lr_target))
    out_of_dist = abs(pred_lr_target - hist_mean) > 6 * (hist_std + 1e-12)

    # convert predicted log-return -> price
    pred_price = last_price * float(np.exp(pred_lr_target))
    pct = (pred_price - last_price) / (last_price + 1e-12) * 100
    next_day = next_business_day(pd.to_datetime(df_den.index[-1]).date()) if hasattr(df_den.index, "__len__") else None

    # print summary
    print("\n=======================================================")
    print(f"Ticker: {ticker_clean}.NS")
    print(f"Last Close: {last_price}")
    print(f"Predicted log-return (target): {pred_lr_target:.6f}")
    print(f"Predicted Close: {pred_price:.6f} ({pct:+.2f}%)")
    print("Next Day:", next_day)
    if out_of_dist:
        print("WARNING: predicted log-return is far from recent history (possible OOD).")
        print(f" recent mean={hist_mean:.6e} std={hist_std:.6e}")
    print("=======================================================\n")

    # save CSV
    out_csv = os.path.join(RESULTS_DIR, f"pred_{ticker_clean}_v2.csv")
    pd.DataFrame({
        "Date": [next_day],
        "Ticker": [ticker_clean + ".NS"],
        "Predicted_Close": [pred_price],
        "Last_Close": [last_price],
        "Expected_Change(%)": [pct]
    }).to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_v2.py TCS")
    else:
        predict(sys.argv[1])
