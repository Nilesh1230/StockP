# # ================================================================
# # FILE: predict_final.py (FINAL RETURN-BASED + LOCAL-DWT + WEEKEND + FIXED SCALING)
# # ================================================================
# # - Respects user ticker (no hardcoded index=0)
# # - Uses pre-denoised CSV if available for target
# # - Ensures scaler alignment with training columns (no .NS mismatch)
# # - Applies correct inverse scaling for target return
# # - DWT runs only on peers
# # - Skips weekends
# # - MPS supported
# # ================================================================

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch_geometric.nn import GATConv
# import yfinance as yf
# import os
# from datetime import timedelta
# import sys

# SEQ_LENGTH = 30
# HIDDEN_CHANNELS = 128
# SENSITIVITY = 0.1
# MODEL_FILE = "gat_lstm_model.pth"

# CONTEXT_DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
# CONTEXT_ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# # ---------------- DWT helpers ----------------
# def dwt_from_scratch(signal):
#     h0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
#     h1 = np.array([-1.0, 1.0]) / np.sqrt(2.0)
#     cA = np.correlate(signal, h0, mode='valid')[::2]
#     cD = np.correlate(signal, h1, mode='valid')[::2]
#     return cA, cD

# def idwt_from_scratch(cA, cD):
#     min_len = min(len(cA), len(cD))
#     cA, cD = cA[:min_len], cD[:min_len]
#     rec_even = (cA + cD) / np.sqrt(2.0)
#     rec_odd  = (cA - cD) / np.sqrt(2.0)
#     recon = np.zeros(2 * min_len, dtype=float)
#     recon[0::2] = rec_even
#     recon[1::2] = rec_odd
#     return recon

# def next_business_day(d):
#     wd = d.weekday()
#     return d + timedelta(days=(7 - wd) if wd >= 4 else 1)

# # ---------------- Model ----------------
# class GATLSTM(nn.Module):
#     def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
#         super(GATLSTM, self).__init__()
#         self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
#         self.lstm = nn.LSTM(hidden_channels * num_nodes,
#                             hidden_channels * num_nodes,
#                             batch_first=True, num_layers=2)
#         self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

#     def forward(self, x_sequence, edge_index):
#         seq = []
#         for t in range(x_sequence.size(1)):
#             x_t = x_sequence[:, t, :].unsqueeze(-1)
#             b, n, i = x_t.shape
#             x_flat = x_t.reshape(-1, i)
#             g = torch.relu(self.gat(x_flat, edge_index))
#             seq.append(g.reshape(b, n, -1))
#         x_gat = torch.stack(seq, dim=1)
#         b, s, n, h = x_gat.shape
#         x_gat = x_gat.reshape(b, s, n*h)
#         out, _ = self.lstm(x_gat)
#         return self.linear(out[:, -1, :])

# # ---------------- Main ----------------
# def predict_with_custom_model(user_ticker):
#     print("="*80)
#     print(f" PREDICTING PRICE FOR '{user_ticker}' ")
#     print("="*80)

#     # 1) Load training context
#     try:
#         context_df = pd.read_csv(CONTEXT_DATA_FILE, index_col=0)
#         cols = list(context_df.columns)
#         adj = np.load(CONTEXT_ADJ_MATRIX_FILE)
#         edge_index = torch.tensor(np.array(np.where(adj == 1)), dtype=torch.long)
#     except FileNotFoundError:
#         print("\n  ERROR: Context files missing. Run pipeline + training first.")
#         return

#     user_clean = user_ticker.upper().replace(".NS", "")
#     if user_clean in cols:
#         target_name = user_clean
#     else:
#         target_name = cols[0]
#         print(f"\n  WARNING: '{user_clean}' not in trained group. Using '{target_name}'.")

#     target_idx = cols.index(target_name)
#     cols_ns = [f"{c}.NS" for c in cols]
#     print(f"  - Context OK. Group size: {len(cols)} | Target: {target_name}")

#     # 2) Get Yahoo data, replace with local denoised if available
#     print("\n--- Downloading latest raw data (6mo) from Yahoo ---")
#     live_raw = yf.download(cols_ns, period="6mo", progress=False, auto_adjust=True)['Close'].dropna()
#     if len(live_raw) < SEQ_LENGTH:
#         print(f"\n  ERROR: Need {SEQ_LENGTH} rows, got {len(live_raw)}.")
#         return

#     local_path = os.path.join("data", "denoised_from_scratch", f"{target_name}_denoised.csv")
#     local_used = False
#     if os.path.exists(local_path):
#         try:
#             print(f"--- Using pre-denoised local CSV for target: {local_path} ---")
#             local_df = pd.read_csv(local_path, parse_dates=["Date"])
#             local_series = local_df.set_index("Date")["Denoised_Close"]
#             aligned = local_series.reindex(live_raw.index).interpolate().bfill().ffill()
#             live_raw[f"{target_name}.NS"] = aligned.values
#             local_used = True
#         except Exception as e:
#             print(f"  - WARNING: Failed to use local denoised file ({e}). Using Yahoo for target.")

#     # 3) Apply DWT denoising to peers (not target)
#     print(f"--- Preparing denoised panel (SENSITIVITY={SENSITIVITY}) ---")
#     live_den = live_raw.copy()
#     window = live_raw.tail(60)

#     for tkr in cols_ns:
#         if local_used and tkr == f"{target_name}.NS":
#             continue
#         prices = window[tkr].to_numpy(float)
#         rets = pd.Series(prices).pct_change().fillna(0).to_numpy(float)
#         cA, cD = dwt_from_scratch(rets)
#         sigma = np.median(np.abs(cD)) / 0.6745 if np.median(np.abs(cD)) > 0 else 0
#         thr = sigma * SENSITIVITY
#         cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD) - thr)
#         den_ret = idwt_from_scratch(cA, cD_clean)
#         if len(den_ret) < len(prices):
#             den_ret = np.pad(den_ret, (0, len(prices)-len(den_ret)), 'edge')
#         den_price = np.zeros_like(prices)
#         den_price[0] = prices[0]
#         for i in range(1, len(prices)):
#             den_price[i] = den_price[i-1] * (1 + den_ret[i])
#         live_den.loc[window.index, tkr] = den_price

#     # 4) Load scaler and ensure correct alignment
#     print("\n--- Loading training-time return scaler ---")
#     ret_min = np.load(os.path.join("data","model_ready","returns_min.npy"))
#     ret_max = np.load(os.path.join("data","model_ready","returns_max.npy"))
#     ret_rng = ret_max - ret_min
#     ret_rng[ret_rng == 0] = 1e-9

#     # Ensure the columns match training columns exactly
#     live_rets = live_den.pct_change().fillna(0)
#     live_rets = live_rets.reindex(columns=cols_ns)  # same order as training
#     scaled = (live_rets.values - ret_min) / ret_rng
#     X = torch.tensor(scaled[-SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0).to(device)

#     # 5) Predict
#     model = GATLSTM(num_nodes=len(cols), in_channels=1,
#                     hidden_channels=HIDDEN_CHANNELS, out_channels=len(cols)).to(device)
#     try:
#         model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
#     except FileNotFoundError:
#         print(f"\n  ERROR: '{MODEL_FILE}' not found. Run training first.")
#         return

#     model.eval()
#     with torch.no_grad():
#         pred_scaled = model(X, edge_index.to(device)).cpu().numpy()[0]

#     # 6) Inverse-scale target return → next price
#     pred_ret = pred_scaled[target_idx] * ret_rng[target_idx] + ret_min[target_idx]
#     last_price = live_raw[f"{target_name}.NS"].iloc[-1]
#     pred_price = last_price * (1 + pred_ret)
#     change_pct = pred_ret * 100

#     # 7) Business-day prediction date
#     last_day = live_raw.index[-1].date()
#     pred_day = next_business_day(last_day)

#     print("\n" + "="*80)
#     print(f" FINAL PREDICTION FOR {target_name}")
#     print("="*80)
#     print(f"  - Last Known Close ({last_day}):  {last_price:.2f}")
#     print(f"  - Predicted Close for Next Trading Day ({pred_day}):  {pred_price:.2f}")
#     print(f"  - Expected Change: {change_pct:+.2f}%")
#     print("="*80)

#     # 8) Save
#     out_file = "data/results/predicted_next_day.csv"
#     os.makedirs(os.path.dirname(out_file), exist_ok=True)
#     pd.DataFrame({
#         "Date": [pred_day],
#         "Ticker": [target_name],
#         "Predicted_Close": [pred_price],
#         "Last_Close": [last_price],
#         "Expected_Change(%)": [change_pct]
#     }).to_csv(out_file, index=False)
#     print(f"  - Saved prediction to '{out_file}'")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("\nERROR: Please provide a stock ticker.")
#         print("   Example: python3 predict_final.py HAVELLS")
#     else:
#         predict_with_custom_model(sys.argv[1])
# FILE: predict_final.py (FINAL PRICE-BASED VERSION)
# This version loads the PRICE-based model and predicts PRICE.
# (Matches the "Slow" train_model.py that worked for LUPIN/HINDUNILVR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import yfinance as yf
import os
from datetime import date, timedelta
import sys

# =======================================================
#                  CONTROL PANEL
# =======================================================
SEQ_LENGTH = 30
HIDDEN_CHANNELS = 96
SENSITIVITY = 0.1 # DWT setting (yeh predict mein bhi zaroori hai)
MODEL_FILE = "gat_lstm_model.pth"

CONTEXT_DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
CONTEXT_ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
# =======================================================


# =======================================================
#   HELPER FUNCTIONS (DWT)
# =======================================================
def dwt_from_scratch(signal):
    h0 = np.array([1.0, 1.0]) / np.sqrt(2.0); h1 = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    # (FIX) DWT uses CORRELATION, not convolution. This was the bug.
    cA = np.correlate(signal, h0, mode='valid')[::2]
    cD = np.correlate(signal, h1, mode='valid')[::2]
    return cA, cD

def idwt_from_scratch(cA, cD):
    min_len = min(len(cA), len(cD)); cA, cD = cA[:min_len], cD[:min_len]
    rec_even = (cA + cD) / np.sqrt(2.0); rec_odd  = (cA - cD) / np.sqrt(2.0)
    recon = np.zeros(2 * min_len, dtype=float); recon[0::2] = rec_even; recon[1::2] = rec_odd
    return recon

def next_business_day(d):
    wd = d.weekday()  # 0=Mon ... 6=Sun
    if wd >= 4:       # Fri/Sat/Sun -> next Monday
        return d + timedelta(days=7 - wd)
    return d + timedelta(days=1)
# =======================================================


# =======================================================
#   MODEL CLASS (Copied from train_model.py - STABLE "SLOW" VERSION)
# =======================================================
class GATLSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GATLSTM, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=1, concat=False, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes,
                            batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    def forward(self, x_sequence, edge_index):
        # x_sequence: [batch, seq_len, num_nodes]
        b, s, n = x_sequence.shape
        edge_index = edge_index.to(self.device) # Ensure edge_index is on correct device
        
        gat_out_sequence = []
        for t in range(s):
            x_t = x_sequence[:, t, :]  # [B, N]
            x_t_feat = x_t.unsqueeze(-1) # [B, N, 1]
            
            # (FIX) Yeh slow loop mathematically correct hai
            batch_node_embeddings = []
            for bi in range(b):
                node_feats = x_t_feat[bi] # [N, 1]
                h = self.gat(node_feats, edge_index)
                h = torch.relu(h)
                batch_node_embeddings.append(h)  # list of [N, H]
            
            gat_out_t = torch.stack(batch_node_embeddings, dim=0) # [B, N, H]
            gat_out_sequence.append(gat_out_t)

        x_gat = torch.stack(gat_out_sequence, dim=1) # [B, S, N, H]

        b, s, n, h = x_gat.shape
        x_gat_reshaped = x_gat.reshape(b, s, n * h)  # [B, S, N*H]
        lstm_out, _ = self.lstm(x_gat_reshaped)  # [B, S, N*H]
        prediction = self.linear(lstm_out[:, -1, :])  # [B, N]
        return prediction

# =======================================================
#      MAIN PREDICTION LOGIC (PRICE-BASED)
# =======================================================
def predict_with_custom_model(target_ticker_from_user):
    print("="*80); print(f" PREDICTING PRICE FOR '{target_ticker_from_user}' "); print("="*80)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # --- Step 1: Load Context from previous steps ---
    try:
        context_df = pd.read_csv(CONTEXT_DATA_FILE, index_col=0) 
        tickers = list(context_df.columns) 
        target_stock_from_file = tickers[0]
        
        adj_matrix = np.load(CONTEXT_ADJ_MATRIX_FILE)
        edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
        
        # (FIX) Add epsilon to avoid divide-by-zero
        eps = 1e-9
        min_vals = context_df.min().values.astype(float)
        max_vals = context_df.max().values.astype(float)
        range_vals = (max_vals - min_vals)
        range_vals[range_vals == 0] = eps

        print(f"  - Context loaded. Model is an expert on the '{target_stock_from_file}' group.")
    except FileNotFoundError:
        print("\n  ERROR: Context files not found. Run 'run_pipeline.py' and 'train_model.py' first.")
        return

    # --- Step 2: Check if user request matches trained model ---
    user_clean_name = target_ticker_from_user.upper().replace(".NS", "")
    if user_clean_name != target_stock_from_file:
        print(f"\n  WARNING: You asked for '{user_clean_name}', but the current model is trained for '{target_stock_from_file}'.")
        print(f"     Showing prediction for '{target_stock_from_file}'.")
        # Set target to what model was trained for
        target_name_for_pred = target_stock_from_file
    else:
        target_name_for_pred = user_clean_name
        
    target_idx = tickers.index(target_name_for_pred)
    
    # --- Step 3: Download latest data for the model's expert group ---
    tickers_with_ns = [f"{t}.NS" for t in tickers]
    print("\n--- Downloading latest raw data for the entire group (6mo)... ---")
    live_data_raw = yf.download(tickers_with_ns, period="6mo", progress=False, auto_adjust=True)['Close'].dropna()
    
    if len(live_data_raw) < SEQ_LENGTH:
        print(f"\n  ERROR: Not enough live data. Need {SEQ_LENGTH} days, got {len(live_data_raw)}.")
        return

    # --- (FIX) Step 3.5: Denoise the live data (Returns-Based) ---
    print(f"--- Denoising live data using SENSITIVITY={SENSITIVITY} (Returns-Based)... ---")
    live_data_denoised = live_data_raw.copy()
    
    denoising_context_data = live_data_raw.tail(60) 
    
    for ticker_ns in tickers_with_ns:
        price_signal = denoising_context_data[ticker_ns].to_numpy(dtype=float).flatten()
        return_signal = denoising_context_data[ticker_ns].pct_change().fillna(0).to_numpy(dtype=float).flatten()

        if len(price_signal) == 0:
            continue
            
        cA, cD = dwt_from_scratch(return_signal) 
        
        if cD.size == 0:
            denoised_return_signal = return_signal.copy()
        else:
            sigma = np.median(np.abs(cD)) / 0.6745 if np.median(np.abs(cD)) > 0 else 0
            threshold = sigma * SENSITIVITY
            cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD) - threshold) 
            denoised_return_signal = idwt_from_scratch(cA, cD_clean)
        
        if len(denoised_return_signal) < len(return_signal): 
            denoised_return_signal = np.pad(denoised_return_signal, (0, len(return_signal)-len(denoised_return_signal)), 'edge')
        
        denoised_return_signal = denoised_return_signal[:len(price_signal)]

        denoised_price_signal = np.zeros_like(price_signal)
        denoised_price_signal[0] = price_signal[0] 
        
        for t in range(1, len(price_signal)):
            denoised_price_signal[t] = denoised_price_signal[t-1] * (1 + denoised_return_signal[t])
        
        live_data_denoised.loc[denoising_context_data.index, ticker_ns] = denoised_price_signal
    
    print("--- Denoising complete. ---")

   # --- Step 4: Prepare the input for prediction (from DENOISED PRICES) ---
    last_sequence_denoised = live_data_denoised[tickers_with_ns].tail(SEQ_LENGTH)
    
    scaled_data = (last_sequence_denoised.values - min_vals) / range_vals
    X_pred = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # --- Step 5: Load model and predict PRICE ---
    model = GATLSTM(num_nodes=len(tickers), in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=len(tickers))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(device)
    model.eval()

    with torch.no_grad():
        all_preds_scaled = model(X_pred, edge_index)

    # --- Step 6: Convert predicted PRICE to actual value ---
    all_preds_actual = (all_preds_scaled.cpu().numpy() * range_vals) + min_vals
    
    predicted_price = all_preds_actual[0][target_idx] # Get the price for the correct stock
    last_known_price = live_data_raw[f"{target_name_for_pred}.NS"].iloc[-1]
    change_pct = (predicted_price - last_known_price) / last_known_price * 100
    
    last_day = live_data_raw.index[-1].date()
    tomorrow = next_business_day(last_day)

    print("\n" + "="*80)
    print(f" FINAL PREDICTION FOR {target_name_for_pred} ")
    print("="*80)
    print(f"  - Last Known Close (Raw) ({last_day}):  {last_known_price:.2f}")
    print(f"  - Predicted Close for Next Trading Day ({tomorrow}):  {predicted_price:.2f}")
    print(f"  - Expected Change: {change_pct:+.2f}%")
    print("="*80)
    
    # --- Step 7: Save prediction ---
    out_file = "data/results/predicted_next_day.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    pd.DataFrame({
        "Date": [tomorrow],
        "Ticker": [target_name_for_pred],
        "Predicted_Close": [predicted_price],
        "Last_Close": [last_known_price],
        "Expected_Change(%)": [change_pct]
    }).to_csv(out_file, index=False)
    print(f"  - Saved prediction to '{out_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nERROR: Please provide a stock ticker.")
        print("   Example: python3 predict_final.py NTPC")
    else:
        predict_with_custom_model(sys.argv[1])