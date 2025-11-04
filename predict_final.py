# FILE: predict_final.py
# STEP 4: Loads the custom-trained model and predicts the next day's price.
# FINAL FIX: Uses "Returns-Based DWT" to match the preprocess pipeline.
# Fully GPU (MPS) supported and production-ready.

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
HIDDEN_CHANNELS = 128
START_DATE = "2022-01-01"
SENSITIVITY = 0.1
MODEL_FILE = "gat_lstm_model.pth"

CONTEXT_DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
CONTEXT_ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# =======================================================


# =======================================================
#   HELPER FUNCTIONS (DWT - Copied from preprocess_universe.py)
# =======================================================
def dwt_from_scratch(signal):
    h0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    h1 = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    # (FIX) DWT uses correlation, not convolution
    cA = np.correlate(signal, h0, mode='valid')[::2]
    cD = np.correlate(signal, h1, mode='valid')[::2]
    return cA, cD


def idwt_from_scratch(cA, cD):
    min_len = min(len(cA), len(cD))
    cA, cD = cA[:min_len], cD[:min_len]
    rec_even = (cA + cD) / np.sqrt(2.0)
    rec_odd = (cA - cD) / np.sqrt(2.0)
    recon = np.zeros(2 * min_len, dtype=float)
    recon[0::2] = rec_even
    recon[1::2] = rec_odd
    return recon


# =======================================================
#   MODEL CLASS (Copied from train_model.py - STABLE VERSION)
# =======================================================
class GATLSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GATLSTM, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes,
                            batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

    def forward(self, x_sequence, edge_index):
        gat_out_sequence = []
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :].unsqueeze(-1)
            b, n, i = x_t.shape
            x_t_flat = x_t.reshape(-1, i)
            gat_out = torch.relu(self.gat(x_t_flat, edge_index)).reshape(b, n, -1)
            gat_out_sequence.append(gat_out)

        x_gat = torch.stack(gat_out_sequence, dim=1)
        b, s, n, h = x_gat.shape
        x_gat_reshaped = x_gat.reshape(b, s, n * h)
        lstm_out, _ = self.lstm(x_gat_reshaped)
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction


# =======================================================
#      MAIN PREDICTION LOGIC (FIXED: RETURNS-BASED DWT)
# =======================================================
def predict_with_custom_model(target_ticker_from_user):
    print("="*80)
    print(f" PREDICTING PRICE FOR '{target_ticker_from_user}' ")
    print("="*80)

    # --- Step 1: Load Context from previous steps ---
    try:
        context_df = pd.read_csv(CONTEXT_DATA_FILE, index_col=0)
        tickers = list(context_df.columns)
        target_stock_from_file = tickers[0]
        adj_matrix = np.load(CONTEXT_ADJ_MATRIX_FILE)
        edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
        min_vals = context_df.min().values
        max_vals = context_df.max().values
        print(f"  - Context loaded. Model trained for '{target_stock_from_file}' group.")
    except FileNotFoundError:
        print("\n  ERROR: Context files not found. Run 'run_pipeline.py' and 'train_model.py' first.")
        return

    # --- Step 2: Check if user request matches trained model ---
    if target_ticker_from_user.upper() != target_stock_from_file:
        print(f"\n  WARNING: You asked for '{target_ticker_from_user}', but model is for '{target_stock_from_file}'.")
        print(f"     To predict for '{target_ticker_from_user}', run:")
        print(f"     1. python3 run_pipeline.py {target_ticker_from_user.upper()}.NS")
        print(f"     2. python3 train_model.py")

    # --- Step 3: Download latest data ---
    tickers_with_ns = [f"{t}.NS" for t in tickers]
    print("\n--- Downloading latest raw data (6 months)... ---")
    live_data_raw = yf.download(tickers_with_ns, period="6mo", progress=False, auto_adjust=True)['Close'].dropna()

    if len(live_data_raw) < SEQ_LENGTH:
        print(f"\n  ERROR: Not enough live data. Need {SEQ_LENGTH} days, got {len(live_data_raw)}.")
        return

    # --- Step 3.5: Denoise live data (Returns-Based) ---
    print(f"--- Denoising live data using SENSITIVITY={SENSITIVITY} ---")
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
            denoised_return_signal = np.pad(denoised_return_signal,
                                            (0, len(return_signal) - len(denoised_return_signal)), 'edge')

        denoised_return_signal = denoised_return_signal[:len(price_signal)]
        denoised_price_signal = np.zeros_like(price_signal)
        denoised_price_signal[0] = price_signal[0]
        for t in range(1, len(price_signal)):
            denoised_price_signal[t] = denoised_price_signal[t - 1] * (1 + denoised_return_signal[t])
        live_data_denoised.loc[denoising_context_data.index, ticker_ns] = denoised_price_signal

    print("--- Denoising complete ---")

    # --- Step 4: Prepare input for prediction ---
    last_sequence_denoised = live_data_denoised[tickers_with_ns].tail(SEQ_LENGTH)
    live_data_clipped = np.clip(last_sequence_denoised.values, min_vals, max_vals)
    scaled_data = (live_data_clipped - min_vals) / (max_vals - min_vals)
    X_pred = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)

    # --- Step 5: Load model and predict ---
    model = GATLSTM(num_nodes=len(tickers), in_channels=1,
                    hidden_channels=HIDDEN_CHANNELS, out_channels=len(tickers)).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    except FileNotFoundError:
        print(f"\n  ERROR: Model file '{MODEL_FILE}' not found. Run 'train_model.py' first.")
        return

    model.eval()
    with torch.no_grad():
        all_preds_scaled = model(X_pred, edge_index.to(device))

    all_preds_actual = (all_preds_scaled.cpu().numpy() * (max_vals - min_vals)) + min_vals
    predicted_price = all_preds_actual[0][0]
    last_known_price = live_data_raw[f"{target_stock_from_file}.NS"].iloc[-1]
    tomorrow = date.today() + timedelta(days=1)
    change_pct = ((predicted_price - last_known_price) / last_known_price) * 100

    print("\n" + "="*80)
    print(f" FINAL PREDICTION FOR {target_stock_from_file} ")
    print("="*80)
    print(f"  - Last Known Close ({live_data_raw.index[-1].strftime('%Y-%m-%d')}):  {last_known_price:.2f}")
    print(f"  - Predicted Close for Tomorrow ({tomorrow.strftime('%Y-%m-%d')}):  {predicted_price:.2f}")
    print(f"  - Expected Change: {change_pct:+.2f}%")
    print("="*80)

    # --- Step 6: Save prediction to CSV ---
    output_file = "data/results/predicted_next_day.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame({
        "Date": [tomorrow],
        "Ticker": [target_stock_from_file],
        "Predicted_Close": [predicted_price],
        "Last_Close": [last_known_price],
        "Expected_Change(%)": [change_pct]
    }).to_csv(output_file, index=False)
    print(f"  - Saved prediction to '{output_file}'")


# =======================================================
# ENTRY POINT
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nERROR: Please provide a stock ticker.")
        print("   Example: python3 predict_final.py NTPC")
    else:
        predict_with_custom_model(sys.argv[1])
