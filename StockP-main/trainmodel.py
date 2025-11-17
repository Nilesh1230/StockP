#!/usr/bin/env python3
"""
trainmodel.py
- Trains a RETURN-based GAT+LSTM over the *global* combined universe produced by run_pipeline.py
- Saves:
   - model checkpoint: gat_lstm_model.pth
   - metadata: model_meta.npz  (cols, logret_mu, logret_sd, adjacency)
"""

import os, argparse, time, math, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv

# -------------------------
# Defaults (tweakable)
# -------------------------
SEQ_LENGTH = 30
TRAIN_SPLIT = 0.80
DEFAULT_HIDDEN = 64
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 8
LR = 1e-3
MODEL_FILE = "gat_lstm_model.pth"
META_FILE  = "model_meta.npz"
DATA_FILE = "data/model_ready/combined_denoised_prices.csv"
ADJ_FILE  = "data/model_ready/adjacency_matrix.npy"
OUT_DIR = "data/model_ready"
EPS = 1e-9
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto",
                    help="auto / cuda / mps / cpu")
parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--lr", type=float, default=LR)
parser.add_argument("--save-every", type=int, default=10)
parser.add_argument("--pin-memory", action="store_true")
args = parser.parse_args()

# -------------------------
# Device selection
# -------------------------
def choose_device(req):
    if req != "auto":
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = choose_device(args.device)
print("Using device:", DEVICE)

# -------------------------
# Model (return-based)
# -------------------------
class HybridGATLSTM(nn.Module):
    def __init__(self, num_nodes, gat_h=32, lstm_layers=1, hidden=64, use_pool=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.gat_h = gat_h
        self.hidden = hidden
        self.use_pool = use_pool

        self.gat = GATConv(1, gat_h, heads=1, concat=False, dropout=0.1)

        self.lstm = nn.LSTM(
            input_size=gat_h * num_nodes,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0
        )

        self.return_head = nn.Linear(hidden, num_nodes)

    def forward(self, x, edge_index):
        x = x.float()
        B, S, N = x.shape
        device = x.device
        edge_index = edge_index.to(device).long()
        seq_emb = []

        if device.type == "cuda":
            E = edge_index.size(1)
            for t in range(S):
                xt = x[:, t, :].reshape(B * N, 1)
                edge_rep = edge_index.repeat(1, B)
                offsets = (torch.arange(B, device=device)
                           .repeat_interleave(E) * N).unsqueeze(0)
                edge_batched = edge_rep + offsets
                h = torch.relu(self.gat(xt, edge_batched))
                h = h.view(B, N, self.gat_h)
                seq_emb.append(h.view(B, -1))
        else:
            for t in range(S):
                h_list = []
                for b in range(B):
                    xt_b = x[b, t, :].unsqueeze(-1)
                    h_b = torch.relu(self.gat(xt_b.to(device), edge_index))
                    h_list.append(h_b.view(1, -1))
                seq_emb.append(torch.cat(h_list, dim=0))

        lstm_in = torch.stack(seq_emb, dim=1)
        out, _ = self.lstm(lstm_in)
        z = out[:, -1, :]
        return self.return_head(z)


# -------------------------
# Data loader / prepare
# -------------------------
def load_and_prepare():
    if not os.path.exists(DATA_FILE) or not os.path.exists(ADJ_FILE):
        raise FileNotFoundError("Run run_pipeline.py first.")

    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    adj = np.load(ADJ_FILE)
    cols = df.columns.tolist()
    prices = df.values.astype(float)

    rets = pd.DataFrame(prices).pct_change().fillna(0).values
    logrets = np.log1p(rets).astype(np.float32)

    mu = logrets.mean(axis=0)
    sd = logrets.std(axis=0)
    sd[sd == 0] = EPS

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "logret_mu.npy"), mu)
    np.save(os.path.join(OUT_DIR, "logret_sd.npy"), sd)
    np.save(os.path.join(OUT_DIR, "cols.npy"), np.array(cols, dtype=object))

    X, Y = [], []
    T = len(logrets)
    for i in range(T - SEQ_LENGTH - 1):
        X.append(logrets[i:i+SEQ_LENGTH])
        Y.append(logrets[i+SEQ_LENGTH])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    X = (X - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)

    total = len(X)
    split = int(total * TRAIN_SPLIT)
    val_from = int(split * 0.85)

    X_train = torch.tensor(X[:val_from])
    Y_train = torch.tensor(Y[:val_from])

    X_val = torch.tensor(X[val_from:split])
    Y_val = torch.tensor(Y[val_from:split])

    X_test = torch.tensor(X[split:])
    Y_test = torch.tensor(Y[split:])

    ei = np.array(np.where(adj == 1), dtype=np.int64)

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test,
            torch.tensor(ei, dtype=torch.long), cols, adj)


# -------------------------
# Training loop
# -------------------------
def train_loop(model, train_ds, val_ds, edge_index, epochs, batch_size, lr,
               pin_memory=False, save_every=10):

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              pin_memory=pin_memory)

    model = model.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    use_amp = (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val = float("inf")
    start = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        tloss = 0.0; cnt = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb, edge_index)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb, edge_index)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tloss += float(loss.item()); cnt += 1

        train_loss = tloss / max(1, cnt)

        # validation
        model.eval()
        vloss = 0.0; vcnt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb, edge_index)
                loss = loss_fn(pred, yb)
                vloss += float(loss.item()); vcnt += 1

        val_loss = vloss / max(1, vcnt)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{epochs} | Train={train_loss:.6f} | Val={val_loss:.6f} | elapsed={elapsed:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_FILE)
            print("  -> Saved best model:", MODEL_FILE)

        if epoch % save_every == 0:
            ck = f"ck_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ck)
            print("  -> periodic checkpoint:", ck)

    print("Training finished. best_val=", best_val)


# ================================================================
# ✅ TEST EVALUATION FUNCTION ADDED HERE
# ================================================================
def evaluate_test_full(model, X_test, Y_test, edge_index, cols):
    print("\n==================== TEST SET EVALUATION =====================")

    model.eval()
    X_test = X_test.to(DEVICE)
    Y_test = Y_test.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    with torch.no_grad():
        pred = model(X_test, edge_index).cpu().numpy()
        actual = Y_test.cpu().numpy()

    mse = np.mean((pred - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - actual))
    eps = 1e-9
    mape = np.mean(np.abs((actual - pred) / (actual + eps))) * 100

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + eps)

    print(f"Test MSE  : {mse:.6f}")
    print(f"Test RMSE : {rmse:.6f}")
    print(f"Test MAE  : {mae:.6f}")
    print(f"Test MAPE : {mape:.2f}%")
    print(f"Test R²   : {r2:.4f}")

    print("\nPer-Stock RMSE:")
    rmse_stock = np.sqrt(np.mean((pred - actual) ** 2, axis=0))

    for i, ticker in enumerate(cols):
        print(f"{ticker:15s}  RMSE={rmse_stock[i]:.6f}")

    print("==============================================================\n")


# -------------------------
# MAIN
# -------------------------
def main():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, edge_index, cols, adj = load_and_prepare()

    train_ds = TensorDataset(X_train, Y_train)
    val_ds   = TensorDataset(X_val, Y_val)

    model = HybridGATLSTM(num_nodes=len(cols), gat_h=32, hidden=args.hidden)

    train_loop(model, train_ds, val_ds, edge_index,
               epochs=args.epochs, batch_size=args.batch_size,
               lr=args.lr, pin_memory=args.pin_memory,
               save_every=args.save_every)

    # -----------------------------
    # 🧪 RUN TEST ACCURACY HERE
    # -----------------------------
    evaluate_test_full(model, X_test, Y_test, edge_index, cols)

    # Save meta
    meta = {"cols": cols}
    np.savez_compressed(META_FILE, cols=np.array(cols, dtype=object))

    print("\nAll artifacts saved. You can copy model + metadata.")


if __name__ == "__main__":
    main()
