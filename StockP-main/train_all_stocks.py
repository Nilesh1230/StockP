# train_all_stocks.py
import os, time, json
import numpy as np
import pandas as pd
import torch, random
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.nn import GATConv

# -------------------------
# CONFIG
# -------------------------
DENOISED_DIR = "data/denoised_universe"
MODEL_READY = "data/model_ready"
COMBINED = os.path.join(MODEL_READY, "combined_denoised_prices.csv")
OUT_MODELS = "models"                  # will contain per-ticker model files
META_DIR = os.path.join(OUT_MODELS, "meta")
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

SEQ_LENGTH = 30
TOP_K = 5              # top peers (report uses 5)
BATCH = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 6
GAT_H = 32             # node embedding dim from GAT (small/medium)
LSTM_H = 128
DROPOUT = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# -------------------------
# MODEL (per-group Full-node flatten approach)
# -------------------------
class GroupGATLSTM(nn.Module):
    def __init__(self, num_nodes, gat_h=GAT_H, lstm_h=LSTM_H, dropout=DROPOUT):
        super().__init__()
        self.N = num_nodes
        self.gat_h = gat_h
        self.lstm_h = lstm_h
        self.gat = GATConv(1, gat_h, heads=1, concat=False, dropout=0.2)
        self.lstm = nn.LSTM(input_size=num_nodes * gat_h, hidden_size=lstm_h,
                            num_layers=2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.return_head = nn.Linear(lstm_h, num_nodes)  # predict log-returns for all nodes (we will only use target)
    def forward(self, x, edge_index):
        # x: [B, S, N]
        B, S, N = x.shape
        device = x.device
        seq_node_emb = []
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        E = edge_index.size(1)
        for t in range(S):
            xt = x[:, t, :].reshape(B * N, 1).to(device)
            edge_rep = edge_index.repeat(1, B)
            offset = (torch.arange(B, device=device).repeat_interleave(E) * N).unsqueeze(0)
            edge_batch = edge_rep + offset
            h = torch.relu(self.gat(xt, edge_batch))
            h = h.view(B, N, self.gat_h)
            seq_node_emb.append(h)
        lstm_in = torch.stack([h.reshape(B, -1) for h in seq_node_emb], dim=1)  # [B, S, N*gat_h]
        out, _ = self.lstm(lstm_in)
        z = out[:, -1, :]
        z = self.dropout(z)
        ret = self.return_head(z)
        return ret


# -------------------------
# HELPERS
# -------------------------
def safe_name(ticker):
    return ticker.replace(".", "_")

def ticker_from_safe(s):
    return s.replace("_", ".")

def load_combined():
    if not os.path.exists(COMBINED):
        raise FileNotFoundError(f"{COMBINED} missing. Run combine_all.py first.")
    df = pd.read_csv(COMBINED, index_col=0, parse_dates=True)
    return df

def compute_logreturns(prices):   # prices: [T, N]
    p = prices.astype(float)
    rets = pd.DataFrame(p).pct_change().fillna(0).values
    lr = np.log1p(rets)
    return lr

def prepare_group_data(context_df, tickers_group):
    """
    context_df: combined full DataFrame [T, N_total] with columns as 'ACC.NS' etc
    tickers_group: list of tickers (full form with .NS) selected for this group
    returns:
      X: np.array [num_seq, SEQ_LENGTH, M]
      Y: np.array [num_seq, M]   (next-day log-return for group nodes)
      mu, sd: per-node mu/sd for X normalization (shape M,)
    """
    sub = context_df[tickers_group].copy()
    prices = sub.values.astype(float)   # [T, M]
    lr = compute_logreturns(prices)     # [T, M]
    # windows
    Xs, Ys = [], []
    for i in range(len(lr) - SEQ_LENGTH - 1):
        Xs.append(lr[i:i+SEQ_LENGTH])
        Ys.append(lr[i+SEQ_LENGTH])
    X = np.stack(Xs).astype(np.float32)
    Y = np.stack(Ys).astype(np.float32)
    mu = X.mean(axis=(0,1))
    sd = X.std(axis=(0,1))
    sd[sd==0] = 1.0
    Xs = (X - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)
    return Xs, Y, mu, sd


# -------------------------
# TRAIN per single target stock
# -------------------------
def train_for_target(target, combined_df, corr_matrix):
    """
    target: string like 'TCS.NS'
    combined_df: full combined DataFrame
    corr_matrix: full correlation matrix (numpy) for returns
    """
    print("\n" + "="*60)
    print(f"TRAINING FOR: {target}")
    print("="*60)
    cols = combined_df.columns.tolist()
    if target not in cols:
        print("Target not in combined columns -> skipping:", target)
        return {"ticker": target, "status": "missing"}

    # choose top-K peers (excluding itself)
    idx = cols.index(target)
    row = np.abs(corr_matrix[idx]).copy()
    row[idx] = -1.0
    topk_idx = np.argsort(-row)[:TOP_K]
    group_idx = [idx] + topk_idx.tolist()
    group_tickers = [cols[i] for i in group_idx]
    print("Group tickers:", group_tickers)

    # prepare data for this group
    X, Y, mu, sd = prepare_group_data(combined_df, group_tickers)
    if X.shape[0] < 10:
        print("Too few sequences for", target, "-> skipping")
        return {"ticker": target, "status": "too_few"}

    # split
    total = len(X)
    split = int(total * 0.8)
    val_from = int(split * 0.85)
    trainX = torch.tensor(X[:val_from], dtype=torch.float32)
    trainY = torch.tensor(Y[:val_from], dtype=torch.float32)
    valX = torch.tensor(X[val_from:split], dtype=torch.float32)
    valY = torch.tensor(Y[val_from:split], dtype=torch.float32)
    testX = torch.tensor(X[split:], dtype=torch.float32)
    testY = torch.tensor(Y[split:], dtype=torch.float32)

    # build adjacency for group: use corr between group nodes (full correlation matrix provided)
    adj = np.zeros((len(group_idx), len(group_idx)), dtype=int)
    for i, gi in enumerate(group_idx):
        rowg = np.abs(corr_matrix[gi, group_idx]).copy()
        rowg[i] = -1.0
        top = np.argsort(-rowg)[:TOP_K]   # neighbors within group
        adj[i, top] = 1
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    ei = np.argwhere(adj==1).T.astype(np.int64)
    edge_index = torch.tensor(ei, dtype=torch.long)

    # model, optimizer
    model = GroupGATLSTM(num_nodes=len(group_idx), gat_h=GAT_H, lstm_h=LSTM_H).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)

    # dataloaders
    train_ds = TensorDataset(trainX, trainY)
    val_ds = TensorDataset(valX, valY)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False) if len(val_ds) >= BATCH else None

    best = float("inf"); best_epoch = 0
    model_file = os.path.join(OUT_MODELS, f"{safe_name(target)}.pth")
    meta_file = os.path.join(META_DIR, f"{safe_name(target)}.json")

    for epoch in range(1, EPOCHS+1):
        model.train()
        tot, cnt = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            preds = model(xb, edge_index.to(DEVICE))
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item(); cnt += 1
        train_loss = tot / max(1, cnt)

        # validation
        if val_loader is None:
            val_loss = train_loss
        else:
            model.eval()
            vt, vc = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    p = model(xb, edge_index.to(DEVICE))
                    vt += loss_fn(p, yb).item(); vc += 1
            val_loss = vt / max(1, vc)
            scheduler.step(val_loss)

        if val_loss < best:
            best = val_loss; best_epoch = epoch
            torch.save(model.state_dict(), model_file)

        if epoch - best_epoch >= PATIENCE:
            break

    # save meta
    meta = {
        "target": target,
        "group_tickers": group_tickers,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "best_val": best,
        "best_epoch": best_epoch,
        "edge_index": ei.tolist()
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f)

    # compute a quick test score (load best model)
    try:
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            preds = model(testX.to(DEVICE), edge_index.to(DEVICE)).cpu().numpy()
        test_rmse = float(np.sqrt(((preds - testY.numpy())**2).mean()))
    except Exception:
        test_rmse = None

    print(f"Finished {target} | best_val={best:.6f} | best_epoch={best_epoch} | test_rmse={test_rmse}")
    return {"ticker": target, "status": "ok", "best_val": float(best), "best_epoch": int(best_epoch), "test_rmse": test_rmse}


# -------------------------
# ORCHESTRATOR: train all tickers
# -------------------------
def main():
    print("Device:", DEVICE)
    combined = load_combined()
    cols = combined.columns.tolist()
    # compute full correlation on full universe returns (used to pick peers)
    full_returns = compute_logreturns(combined.values)
    corr = np.corrcoef(full_returns.T)
    results = []
    start = time.time()

    # optional resume: skip tickers which already have a model file
    todos = [c for c in cols]
    print("Total tickers:", len(todos))
    for i, t in enumerate(todos, 1):
        print(f"\n[{i}/{len(todos)}] Processing {t}")
        model_path = os.path.join(OUT_MODELS, f"{safe_name(t)}.pth")
        if os.path.exists(model_path):
            print("Model exists -> skipping:", model_path)
            continue
        r = train_for_target(t, combined, corr)
        results.append(r)
        # save intermediate summary
        with open(os.path.join(OUT_MODELS, "summary_partial.json"), "w") as f:
            json.dump(results, f, indent=2)
    total_time = time.time() - start
    print("All done. elapsed(s):", total_time)
    # finalize summary
    with open(os.path.join(OUT_MODELS, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
