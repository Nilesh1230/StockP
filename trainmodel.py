# # ================================================================
# # FILE: train_model.py (FINAL FAST VERSION — RETURN-BASED)
# # This version includes GPU (MPS) support and Early Stopping for massive speedup.
# # (FIX 1) Train on RETURNS (more sensitive than prices)
# # (FIX 2) Added epsilon to scaling to prevent divide-by-zero (NaN) error.
# # ================================================================

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GATConv
# import os
# import time
# import matplotlib.pyplot as plt

# # =======================================================
# #                  CONTROL PANEL
# # =======================================================
# SEQ_LENGTH = 30
# TRAIN_SPLIT = 0.8
# HIDDEN_CHANNELS = 128
# EPOCHS = 500
# LEARNING_RATE = 0.001
# MODEL_SAVE_PATH = "gat_lstm_model.pth"

# # Data files created by run_pipeline.py
# DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
# ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# # =======================================================


# # =======================================================
# # 1. DATA PREPARATION (LOADING, SCALING, SEQUENCING)
# #    NOW RETURN-BASED (more responsive)
# # =======================================================
# def load_and_prepare_data():
#     print_heading("STEP 1: DATA LOADING & PREPARATION")
#     print(f"  - Using device: {device}")
#     prices_df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
#     adj_matrix = np.load(ADJ_MATRIX_FILE)
#     target_stock_name = prices_df.columns[0]
#     print(f"  - Target stock detected: '{target_stock_name}'")

#     # Compute daily RETURNS (stationary target → better sensitivity)
#     returns_df = prices_df.pct_change().fillna(0)

#     # Scaling on RETURNS
#     min_vals = returns_df.min().values
#     max_vals = returns_df.max().values
#     os.makedirs("data/model_ready", exist_ok=True)
#     np.save(os.path.join("data","model_ready","returns_min.npy"), min_vals)
#     np.save(os.path.join("data","model_ready","returns_max.npy"), max_vals)
#     print("  - Saved return scaler (min/max) for prediction use.")

#     range_vals = max_vals - min_vals
#     range_vals[range_vals == 0] = 1e-9  # Avoid division by zero

#     scaled_data = (returns_df.values - min_vals) / range_vals

#     X, y = [], []
#     # Target = NEXT-DAY RETURN (scaled)
#     # FIX: include the last valid sample (no extra -1)
#     for i in range(len(scaled_data) - SEQ_LENGTH):
#         X.append(scaled_data[i : i + SEQ_LENGTH])
#         y.append(scaled_data[i + SEQ_LENGTH])     # next day return (scaled)
#     X, y = np.array(X), np.array(y)

#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.float32)
#     edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)

#     train_size = int(len(X_tensor) * TRAIN_SPLIT)
#     X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
#     y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

#     # FIX: first y_test item corresponds to returns index (train_size + SEQ_LENGTH)
#     test_dates = prices_df.index[train_size + SEQ_LENGTH:]

#     print(f"  - Data split into {len(X_train)} training and {len(X_test)} testing samples.")

#     return (X_train, y_train, X_test, y_test, edge_index,
#             min_vals, max_vals, prices_df.columns, test_dates, target_stock_name,
#             train_size, prices_df)

# ## =======================================================
# # 2. GAT + LSTM MODEL DEFINITION (STABLE FAST VERSION)
# # =======================================================
# class GATLSTM(nn.Module):
#     def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
#         super(GATLSTM, self).__init__()
#         self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
#         self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes, batch_first=True, num_layers=2)
#         self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

#     def forward(self, x_sequence, edge_index):
#         # x_sequence shape: [batch, sequence_length, num_nodes]
#         gat_out_sequence = []
#         for t in range(x_sequence.size(1)):  # seq_len
#             x_t = x_sequence[:, t, :].unsqueeze(-1)  # [B, N, 1]
#             b, n, i = x_t.shape
#             x_t_flat = x_t.reshape(-1, i)  # [B*N, 1]
#             gat_out = torch.relu(self.gat(x_t_flat, edge_index)).reshape(b, n, -1)  # [B, N, H]
#             gat_out_sequence.append(gat_out)

#         x_gat = torch.stack(gat_out_sequence, dim=1)  # [B, S, N, H]
#         b, s, n, h = x_gat.shape
#         x_gat_reshaped = x_gat.reshape(b, s, n * h)   # [B, S, N*H]
#         lstm_out, _ = self.lstm(x_gat_reshaped)
#         prediction = self.linear(lstm_out[:, -1, :])  # predicts NEXT-DAY RETURN (scaled)
#         return prediction

# # =======================================================
# # 3. MODEL TRAINING LOOP (WITH EARLY STOPPING)
# # =======================================================
# def train_model(model, X_train, y_train, edge_index):
#     print_heading("STEP 2: MODEL TRAINING")
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     patience = 25
#     patience_counter = 0
#     best_loss = float('inf')

#     start_time = time.time()

#     X_train_device = X_train.to(device)
#     y_train_device = y_train.to(device)
#     edge_index_device = edge_index.to(device)

#     for epoch in range(EPOCHS):
#         model.train()

#         predictions = model(X_train_device, edge_index_device)
#         loss = loss_function(predictions, y_train_device)

#         if torch.isnan(loss):
#             print(f"ERROR: Loss is NaN at epoch {epoch+1}. Stopping training.")
#             return None

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (epoch+1) % 20 == 0:
#             print(f"  - Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             patience_counter = 0
#             torch.save(model.state_dict(), MODEL_SAVE_PATH)
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print(f"\n  - Early stopping at epoch {epoch+1}. No improvement in last {patience} epochs.")
#             break

#     print(f"\n  - Training completed in {time.time() - start_time:.2f} seconds.")
#     return model

# # =======================================================
# # 4. MODEL EVALUATION AND PLOTTING
# #    (Convert predicted returns → price path for plot)
# # =======================================================
# def evaluate_model(trained_model, X_test, y_test, edge_index,
#                    min_vals, max_vals, column_names, test_dates, target_stock_name,
#                    train_size, prices_df):
#     print_heading("STEP 3: MODEL EVALUATION ON TEST DATA")
#     trained_model.eval()
#     with torch.no_grad():
#         test_predictions_scaled = trained_model(X_test.to(device), edge_index.to(device))

#     test_predictions_scaled = test_predictions_scaled.cpu().numpy()
#     y_test_np = y_test.numpy()

#     # Inverse-scale RETURNS
#     range_vals = max_vals - min_vals
#     range_vals[range_vals == 0] = 1e-9
#     pred_returns = test_predictions_scaled * range_vals + min_vals
#     true_returns = y_test_np * range_vals + min_vals

#     stock_idx = list(column_names).index(target_stock_name)

#     # FIX: base price = price just BEFORE first test return
#     base_idx = train_size + SEQ_LENGTH - 1
#     if base_idx < 0:
#         base_idx = 0
#     base_price = prices_df.iloc[base_idx][target_stock_name]

#     actual_prices, predicted_prices = [], []
#     p_a, p_p = base_price, base_price
#     for r_true, r_pred in zip(true_returns[:, stock_idx], pred_returns[:, stock_idx]):
#         p_a = p_a * (1.0 + r_true);  actual_prices.append(p_a)
#         p_p = p_p * (1.0 + r_pred);  predicted_prices.append(p_p)

#     actual_arr = np.array(actual_prices)
#     pred_arr = np.array(predicted_prices)
#     rmse = np.sqrt(np.mean((pred_arr - actual_arr) ** 2))
#     print(f"  - RMSE for target stock '{target_stock_name}': {rmse:.4f}")

#     if len(test_dates) != len(actual_arr):
#         print(f"  - Warning: Date and prediction length mismatch. Adjusting dates for plotting.")
#         test_dates = test_dates[:len(actual_arr)]

#     plt.figure(figsize=(15, 7))
#     plt.plot(test_dates, actual_arr, label='Actual Price', color='green')
#     plt.plot(test_dates, pred_arr, label='Predicted Price', color='red', linestyle='--')
#     plt.title(f'Test Performance for {target_stock_name}')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # =======================================================
# # 5. MAIN EXECUTION FLOW
# # =======================================================
# def print_heading(text):
#     print("\n" + "="*80)
#     print(text)
#     print("="*80)

# if __name__ == "__main__":
#     try:
#         (X_train, y_train, X_test, y_test, edge_index,
#          min_vals, max_vals, columns, test_dates, target_stock,
#          train_size, prices_df) = load_and_prepare_data()

#         num_nodes = X_train.shape[2]
#         model = GATLSTM(num_nodes=num_nodes, in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=num_nodes)
#         model.to(device)

#         trained_model = train_model(model, X_train, y_train, edge_index)

#         if trained_model is None:
#             print_heading("PROCESS FAILED: Training resulted in NaN.")
#         else:
#             print(f"  - Loading best model saved at '{MODEL_SAVE_PATH}' for final evaluation.")
#             model.load_state_dict(torch.load(MODEL_SAVE_PATH))

#             evaluate_model(model, X_test, y_test, edge_index,
#                            min_vals, max_vals, columns, test_dates, target_stock,
#                            train_size, prices_df)

#             print(f"\n  - Best Trained Model Brain saved to '{MODEL_SAVE_PATH}'")
#             print_heading("PROCESS COMPLETE")

#     except FileNotFoundError:
#         print("\n" + "="*80)
#         print("ERROR: Could not find 'data/model_ready/combined_denoised_prices.csv'.")
#         print("Please run 'run_pipeline.py <TICKER>' first to generate the necessary files.")
#         print("="*80)
# ================================================================
# FILE: train_model.py (FINAL FAST GPU-OPTIMIZED VERSION)
# GAT + LSTM Hybrid Model for Stock Price Forecasting
# ================================================================
# ✅ FIXES & UPGRADES:
#   1) Vectorized GAT computation (no per-sample loop)
#   2) AMP (Automatic Mixed Precision) with version fallback
#   3) pin_memory DataLoader for fast CPU→GPU transfer
#   4) Works on MPS, CUDA, or CPU
# ================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, TensorDataset
import os, time, matplotlib.pyplot as plt

# =======================================================
# CONTROL PANEL
# =======================================================
SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8
HIDDEN_CHANNELS = 96
EPOCHS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SAVE_PATH = "gat_lstm_model.pth"

DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.set_float32_matmul_precision("high")
# =======================================================


# =======================================================
# 1. DATA PREPARATION
# =======================================================
def load_and_prepare_data():
    print_heading("STEP 1: DATA LOADING & PREPARATION (PRICE-BASED)")
    print(f"  - Using device: {device}")

    prices_df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    adj_matrix = np.load(ADJ_MATRIX_FILE)
    target_stock = prices_df.columns[0]

    eps = 1e-9
    min_vals = prices_df.min().values
    max_vals = prices_df.max().values
    range_vals = np.maximum(max_vals - min_vals, eps)

    scaled_data = (prices_df.values - min_vals) / range_vals

    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i:i + SEQ_LENGTH])
        y.append(scaled_data[i + SEQ_LENGTH])
    X, y = np.array(X), np.array(y)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)

    split = int(len(X_t) * TRAIN_SPLIT)
    X_train, X_test = X_t[:split], X_t[split:]
    y_train, y_test = y_t[:split], y_t[split:]
    test_dates = prices_df.index[split + SEQ_LENGTH:]

    print(f"  - Data split: {len(X_train)} train | {len(X_test)} test")
    return X_train, y_train, X_test, y_test, edge_index, min_vals, max_vals, prices_df.columns, test_dates, target_stock


# =======================================================
# 2. MODEL: GAT + LSTM (FAST PARALLEL)
# =======================================================
class GATLSTM(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels):
        super().__init__()
        self.gat = GATConv(1, hidden_channels, heads=1, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes,
                            num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)
        self.num_nodes = num_nodes
        self.hidden = hidden_channels

    def forward(self, x_seq, edge_index):
        # x_seq: [B, S, N]
        b, s, n = x_seq.shape
        x_seq = x_seq.view(b * s, n, 1)
        x_seq = x_seq.reshape(-1, 1)  # Flatten for batched GAT
        gat_out = torch.relu(self.gat(x_seq, edge_index))
        gat_out = gat_out.view(b, s, n, self.hidden)
        lstm_in = gat_out.view(b, s, n * self.hidden)
        lstm_out, _ = self.lstm(lstm_in)
        return self.linear(lstm_out[:, -1, :])


# =======================================================
# 3. TRAINING (FAST GPU + UNIVERSAL AMP)
# =======================================================
def train_model(model, X_train, y_train, edge_index):
    print_heading("STEP 2: MODEL TRAINING (FAST GPU + AMP)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ✅ Safe AMP initialization (compatible with all PyTorch versions)
    try:
        scaler = torch.amp.GradScaler(device_type=device.type)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler() if device.type in ["cuda", "mps"] else None

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True
    )

    best_loss, patience, counter = float("inf"), 25, 0
    start = time.time()
    edge_index = edge_index.to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if scaler:  # ✅ AMP enabled
                with torch.cuda.amp.autocast(enabled=True):
                    preds = model(xb, edge_index)
                    loss = criterion(preds, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # CPU fallback
                preds = model(xb, edge_index)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            total += loss.item()

        avg_loss = total / len(loader)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  - Epoch [{epoch}/{EPOCHS}] | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss - 1e-8:
            best_loss, counter = avg_loss, 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            counter += 1

        if counter >= patience:
            print(f"\n  - Early stopping at epoch {epoch}. No improvement in last {patience} epochs.")
            break

    print(f"\n  ✅ Training Done in {time.time() - start:.2f}s | Best Loss: {best_loss:.6f}")


# =======================================================
# 4. EVALUATION (PRICE-BASED)
# =======================================================
def evaluate_model(model, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates, target_stock):
    print_heading("STEP 3: MODEL EVALUATION (TEST PERFORMANCE)")
    model.eval()

    preds, reals = [], []
    loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb, edge_index.to(device))
            preds.append(out.cpu())
            reals.append(yb.cpu())

    preds = torch.cat(preds).numpy()
    reals = torch.cat(reals).numpy()

    range_vals = np.maximum(max_vals - min_vals, 1e-9)
    preds_real = preds * range_vals + min_vals
    reals_real = reals * range_vals + min_vals

    i = list(columns).index(target_stock)
    actual, predicted = reals_real[:, i], preds_real[:, i]
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    print(f"  - RMSE for '{target_stock}': {rmse:.4f}")

    plt.figure(figsize=(15, 7))
    plt.plot(test_dates[:len(actual)], actual, label="Actual", color="green")
    plt.plot(test_dates[:len(predicted)], predicted, label="Predicted", linestyle="--", color="red")
    plt.title(f"Test Performance for {target_stock}")
    plt.xlabel("Date"); plt.ylabel("Price")
    plt.legend(); plt.grid(True); plt.show()


# =======================================================
# UTILITIES
# =======================================================
def print_heading(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates, target_stock = load_and_prepare_data()

        model = GATLSTM(num_nodes=X_train.shape[2], hidden_channels=HIDDEN_CHANNELS, out_channels=X_train.shape[2]).to(device)
        train_model(model, X_train, y_train, edge_index)

        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        evaluate_model(model, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates, target_stock)

        print_heading("PROCESS COMPLETE ✅")

    except FileNotFoundError:
        print("\nERROR: Missing files. Run 'run_pipeline.py' first.")
