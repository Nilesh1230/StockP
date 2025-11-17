# predict_final_fixed.py
import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ---------------- CONFIG ----------------
SEQ_LENGTH = 30
# If you have per-stock models saved in models/<SAFE>.pth, we prefer that.
MODEL_UNIVERSAL = "gat_lstm_model.pth"     # fallback universal model (if exists)
MODELS_DIR = "models"                      # per-stock models (optional)
DATA_READY = "data/model_ready"
DENOISED_DIR = "data/denoised_universe"    # adjust to your denoised folder
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
EPS = 1e-9

# ---------------- helpers ----------------
def next_business_day(d):
    wd = d.weekday()
    if wd >= 4:
        return d + timedelta(days=7 - wd)
    return d + timedelta(days=1)

def normalize_ticker_input(t):
    t = t.upper().strip()
    if not t.endswith(".NS"):
        t = t + ".NS"
    return t

def safe_name_from_full(ticker_full):
    return ticker_full.replace(".", "_")

def find_denoised_file(safe):
    """
    try multiple filename patterns used in different scripts:
      - SAFE.csv
      - SAFE_NS.csv
      - SAFE_denoised.csv
      - SAFE_den oised.csv (common variants)
    returns full path or None
    """
    candidates = [
        f"{safe}.csv",
        f"{safe}_NS.csv",
        f"{safe}_denoised.csv",
        f"{safe}_denoised.csv".replace("_denoised.csv","_denoised.csv"),
        f"{safe}.csv"
    ]
    for c in candidates:
        p = os.path.join(DENOISED_DIR, c)
        if os.path.exists(p):
            return p
    # fallback: try uppercase/lowercase variants
    for fname in os.listdir(DENOISED_DIR):
        if fname.upper().startswith(safe.upper()):
            return os.path.join(DENOISED_DIR, fname)
    return None

# ---------------- MODEL (old architecture replicating your old code) ----------------
class HybridGATLSTM(nn.Module):
    def __init__(self, num_nodes, hidden=128, use_pool=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden = hidden
        self.use_pool = use_pool
        self.gat = GATConv(1, hidden, heads=1, concat=False, dropout=0.2)
        if use_pool:
            self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True)
            self.price_head = nn.Linear(hidden, num_nodes)
            self.return_head = nn.Linear(hidden, num_nodes)
        else:
            self.lstm = nn.LSTM(hidden * num_nodes, hidden * num_nodes,
                                num_layers=2, batch_first=True)
            self.price_head = nn.Linear(hidden * num_nodes, num_nodes)
            self.return_head = nn.Linear(hidden * num_nodes, num_nodes)

    def forward(self, x, edge_index):
        B, S, N = x.shape
        device = x.device
        seq_emb = []
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        num_edges = edge_index.size(1)
        for t in range(S):
            xt = x[:, t, :].reshape(B * N, 1).to(device)
            edge_rep = edge_index.repeat(1, B)
            offset = (torch.arange(B, device=device).repeat_interleave(num_edges) * N).unsqueeze(0)
            edge_batch = edge_rep + offset
            h = torch.relu(self.gat(xt, edge_batch))
            h = h.view(B, N, self.hidden)
            if self.use_pool:
                seq_emb.append(h.mean(dim=1))
            else:
                seq_emb.append(h)
        if self.use_pool:
            lstm_in = torch.stack(seq_emb, dim=1)
        else:
            lstm_in = torch.stack(seq_emb, dim=1).view(B, S, N * self.hidden)
        out, _ = self.lstm(lstm_in)
        z = out[:, -1, :]
        p = self.price_head(z)
        r = self.return_head(z)
        return p, r

# ---------------- PREDICT ----------------
def predict(ticker_raw):
    ticker_full = normalize_ticker_input(ticker_raw)   # e.g. 'ACC.NS'
    safe = safe_name_from_full(ticker_full)            # e.g. 'ACC_NS'

    # --- combined context ---
    combined_path = os.path.join(DATA_READY, "combined_denoised_prices.csv")
    if not os.path.exists(combined_path):
        print("ERROR: Combined matrix not found:", combined_path); return
    combined = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    cols = list(combined.columns)   # e.g. ['ACC.NS', 'TCS.NS', ...]
    num_nodes = len(cols)

    # --- scalers: must match combined column order ---
    pm_path = os.path.join(DATA_READY, "price_min.npy")
    pM_path = os.path.join(DATA_READY, "price_max.npy")
    if not os.path.exists(pm_path) or not os.path.exists(pM_path):
        print("ERROR: price_min/price_max scalers missing in", DATA_READY); return
    price_min = np.load(pm_path)
    price_max = np.load(pM_path)
    if price_min.shape[0] != num_nodes or price_max.shape[0] != num_nodes:
        print("ERROR: scaler shape mismatch. price_min shape:", price_min.shape, "num_nodes:", num_nodes)
        return
    prng = np.maximum(price_max - price_min, EPS)

    # --- adjacency (global) ---
    adj_path = os.path.join(DATA_READY, "adjacency_matrix.npy")
    if not os.path.exists(adj_path):
        print("ERROR: adjacency matrix not found:", adj_path); return
    adj = np.load(adj_path)
    # build edge_index robustly
    ei = np.argwhere(adj == 1).T.astype(np.int64)
    edge_index = torch.tensor(ei, dtype=torch.long).to(DEVICE)

    # --- check target exists in combined columns ---
    if ticker_full not in cols:
        print(f"ERROR: ticker {ticker_full} not found in combined columns.")
        print("Available sample:", cols[:12])
        return
    t_idx = cols.index(ticker_full)

    # --- find denoised file for ticker ---
    denoised_path = find_denoised_file(safe)
    if denoised_path is None:
        print("ERROR: denoised file not found for", safe, "in", DENOISED_DIR); return

    dfd = pd.read_csv(denoised_path, parse_dates=["Date"], index_col="Date" if "Date" in pd.read_csv(denoised_path, nrows=1).columns else None)
    # try to get Close column
    if "Denoised_Close" in dfd.columns:
        prices = dfd["Denoised_Close"].values
    elif "Close" in dfd.columns:
        prices = dfd["Close"].values
    else:
        prices = dfd.iloc[:, 0].values

    if len(prices) < SEQ_LENGTH:
        print("ERROR: Not enough denoised history for", ticker_full); return

    # --- construct input matrix: last SEQ_LENGTH rows of combined (context) and replace target column with denoised series
    ctx_last = combined.tail(SEQ_LENGTH).values.astype(float)
    if ctx_last.shape[0] < SEQ_LENGTH:
        # fallback: tile last row if combined has fewer rows
        lastrow = combined.tail(1).values[0]
        ctx_last = np.tile(lastrow, (SEQ_LENGTH, 1))
    X = ctx_last.copy()
    if X.shape[1] != num_nodes:
        print("ERROR: context width mismatch:", X.shape, "expected:", num_nodes); return
    # use last SEQ_LENGTH values of denoised target (align shapes)
    seq_target = np.array(prices[-SEQ_LENGTH:], dtype=float)
    X[:, t_idx] = seq_target

    # --- scale X using price_min/price_max (same as your old code)
    X_scaled = (X - price_min.reshape(1, -1)) / prng.reshape(1, -1)
    X_scaled = X_scaled.astype(np.float32)
    x_tensor = torch.tensor(X_scaled).unsqueeze(0).to(DEVICE)   # [1, S, N]

    # --- decide which model to load: per-stock or universal fallback ---
    model_file_per = os.path.join(MODELS_DIR, f"{safe}.pth")
    if os.path.exists(model_file_per):
        model_path = model_file_per
        print("Loading per-stock model:", model_path)
        # For per-stock per-group models you may need a different model class — this script assumes universal architecture
    elif os.path.exists(MODEL_UNIVERSAL):
        model_path = MODEL_UNIVERSAL
        print("Loading universal model:", model_path)
    else:
        print("ERROR: No model found. Check", model_file_per, "or", MODEL_UNIVERSAL); return

    # --- load model (assumes universal HybridGATLSTM architecture) ---
    model = HybridGATLSTM(num_nodes=num_nodes, hidden=128, use_pool=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        p_scaled, r_scaled = model(x_tensor, edge_index)

    p_scaled = p_scaled.cpu().numpy()[0]   # [N]
    preds_real = p_scaled * prng + price_min
    pred_price = float(preds_real[t_idx])
    last_price = float(seq_target[-1])
    pct = (pred_price - last_price) / (last_price + 1e-12) * 100

    # --- output ---
    last_day = None
    try:
        last_day = pd.to_datetime(dfd.index[-1]).date()
    except Exception:
        last_day = date.today()
    next_day = next_business_day(last_day)

    out_csv = os.path.join(RESULTS_DIR, f"predicted_next_day_{safe}.csv")
    pd.DataFrame({
        "Date":[next_day],
        "Ticker":[ticker_full],
        "Predicted_Close":[pred_price],
        "Last_Close":[last_price],
        "Expected_Change(%)":[pct]
    }).to_csv(out_csv, index=False)

    # plot (save)
    try:
        import matplotlib
        matplotlib.use('Agg')
        hist = dfd.iloc[:, 0].tail(60)
        plt.figure(figsize=(10,5))
        plt.plot(hist.index, hist.values, label=f"{ticker_full} (denoised)")
        plt.scatter([pd.Timestamp(next_day)], [pred_price], color="red", label=f"Predicted: {pred_price:.2f}")
        plt.title(f"{ticker_full} — Next Day Prediction")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plot_file = os.path.join(RESULTS_DIR, f"prediction_plot_{safe}.png")
        plt.savefig(plot_file, dpi=150); plt.close()
    except Exception as e:
        plot_file = None
        print("Plot failed:", e)

    print("\n==============================================================")
    print(f" FINAL PREDICTION FOR {ticker_full}")
    print("==============================================================")
    print(f"Last Close:      {last_price:.2f}")
    print(f"Predicted Close: {pred_price:.2f}")
    print(f"Expected Change: {pct:+.2f}%")
    print("--------------------------------------------------------------")
    print("CSV saved to:", out_csv)
    if plot_file:
        print("Plot saved to:", plot_file)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict_final_fixed.py TCS")
    else:
        predict(sys.argv[1])
