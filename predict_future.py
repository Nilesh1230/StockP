# FILE: predict_future.py (Final Interactive Version)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import yfinance as yf
import os
from datetime import date, timedelta
import sys # NEW: Import the sys library to read command-line arguments

# =======================================================
#                  CONTROL PANEL
# =======================================================
SEQ_LENGTH = 30
HIDDEN_CHANNELS = 64
MODEL_FILE = "gat_lstm_model.pth"
DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
# =======================================================

# (All the helper functions and the model class are the same as before)
# =======================================================
#   HELPER FUNCTIONS & MODEL CLASS
# =======================================================
def manual_dwt(signal):
    h0 = np.array([1, 1]) / np.sqrt(2); h1 = np.array([-1, 1]) / np.sqrt(2)
    cA = np.convolve(signal, h0, mode='valid')[::2]; cD = np.convolve(signal, h1, mode='valid')[::2]
    return cA, cD

def manual_idwt(cA, cD):
    n = min(len(cA), len(cD)); cA, cD = cA[:n], cD[:n]
    even = (cA + cD) / np.sqrt(2); odd = (cA - cD) / np.sqrt(2)
    recon = np.zeros(2 * n); recon[0::2] = even; recon[1::2] = odd
    return recon

class GATLSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GATLSTM, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes, batch_first=True)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

    def forward(self, x_sequence, edge_index):
        gat_out_sequence = []
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :].unsqueeze(-1)
            batch_size, num_nodes, in_channels = x_t.shape
            x_t_flat = x_t.reshape(-1, in_channels)
            gat_out = torch.relu(self.gat(x_t_flat, edge_index))
            gat_out = gat_out.reshape(batch_size, num_nodes, -1)
            gat_out_sequence.append(gat_out)
        x_gat = torch.stack(gat_out_sequence, dim=1)
        batch_size, seq_len, num_nodes, hidden = x_gat.shape
        x_gat_reshaped = x_gat.reshape(batch_size, seq_len, num_nodes * hidden)
        lstm_out, _ = self.lstm(x_gat_reshaped)
        last_step = lstm_out[:, -1, :]
        prediction = self.linear(last_step)
        return prediction

# =======================================================
#      MAIN PREDICTION FUNCTION
# =======================================================
def predict_tomorrow():
    print("="*80); print(" 🔮 PREDICTING TOMORROW'S STOCK PRICES "); print("="*80)

    # ... (The first 4 steps of loading data, downloading, processing, and predicting are the same) ...
    # 1. Load historical data info
    try:
        original_prices_df = pd.read_csv(DATA_FILE)
        tickers = list(original_prices_df.columns[1:])
        adj_matrix = np.load(ADJ_MATRIX_FILE)
        edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
        min_vals = original_prices_df[tickers].min(axis=0).to_numpy()
        max_vals = original_prices_df[tickers].max(axis=0).to_numpy()
        print("  - Loaded historical data for context (scaling, graph).")
    except FileNotFoundError:
        print("\nERROR: Could not find 'data/model_ready' files. Please run 'run_pipeline.py' first.")
        return

    # 2. Download latest data
    print("\n--- Downloading latest 60 days of data... ---")
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    live_data_frames = {t: yf.download(f"{t}.NS", start=start_date, end=end_date, progress=False, auto_adjust=True)[['Close']] for t in tickers}
    live_df = pd.concat(live_data_frames.values(), axis=1, keys=live_data_frames.keys())
    live_df.columns = live_df.columns.droplevel(1)
    live_df.dropna(inplace=True)
    if len(live_df) < SEQ_LENGTH:
        print(f"\nERROR: Not enough recent data. Need {SEQ_LENGTH} days, got {len(live_df)}.")
        return

    # 3. Process the new data
    print("\n--- Processing live data... ---")
    denoised_live_df = live_df.copy()
    for stock in denoised_live_df.columns:
        # (Denoising logic is the same)
        signal = denoised_live_df[stock].to_numpy()
        cA, cD = manual_dwt(signal); mad = np.median(np.abs(cD)); sigma = mad / 0.6745 if mad > 0 else 0
        threshold = sigma * 3.0; cD_clean = np.where(np.abs(cD) > threshold, cD, 0)
        denoised_signal = manual_idwt(cA, cD_clean)
        if len(denoised_signal) < len(signal): denoised_signal = np.pad(denoised_signal, (0, len(signal)-len(denoised_signal)), 'edge')
        denoised_live_df[stock] = denoised_signal[:len(signal)]
    
    last_sequence = denoised_live_df.tail(SEQ_LENGTH)
    scaled_data = (last_sequence.to_numpy() - min_vals) / (max_vals - min_vals)
    X_pred = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0)
    
    # 4. Load trained model and make prediction for ALL stocks
    model = GATLSTM(num_nodes=len(tickers), in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=len(tickers))
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(X_pred, edge_index)

    prediction_actual = (prediction_scaled.numpy() * (max_vals - min_vals)) + min_vals
    results_df = pd.DataFrame(prediction_actual, columns=tickers)

    # 5. NEW: Display results based on user input
    tomorrow = date.today() + timedelta(days=1)
    print("\n" + "="*80)
    print(f" 📈 PREDICTIONS FOR NEXT TRADING DAY ({tomorrow.strftime('%Y-%m-%d')}) ")
    print("="*80)

    # Check if the user gave a specific stock name
    if len(sys.argv) > 1:
        # User provided a stock name
        stock_to_predict = sys.argv[1].upper() # Read the name from command line
        if stock_to_predict in results_df.columns:
            predicted_price = results_df[stock_to_predict].iloc[0]
            print(f"  ➤ Prediction for {stock_to_predict}:  ₹{predicted_price:.2f}")
        else:
            print(f"  ERROR: Stock '{stock_to_predict}' not in our list.")
            print(f"  Please choose from: {tickers}")
    else:
        # User did not provide a name, so show all
        print("  (No specific stock requested, showing all predictions)")
        print(results_df.round(2).to_string(index=False))
    
    print("="*80)

if __name__ == "__main__":
    predict_tomorrow()