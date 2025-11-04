# FILE: train_model.py (FINAL FAST VERSION)
# This version includes GPU (MPS) support and Early Stopping for massive speedup.
# (FIX) Added epsilon to Min-Max scaling to prevent divide-by-zero (NaN) error.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import os
import time
import matplotlib.pyplot as plt

# =======================================================
#                  CONTROL PANEL
# =======================================================
SEQ_LENGTH = 30
TRAIN_SPLIT = 0.8
HIDDEN_CHANNELS = 128
EPOCHS = 500
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "gat_lstm_model.pth"

# Data files created by run_pipeline.py
DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# =======================================================


# =======================================================
# 1. DATA PREPARATION (LOADING, SCALING, SEQUENCING)
# =======================================================
def load_and_prepare_data():
    print_heading("STEP 1: DATA LOADING & PREPARATION")
    print(f"  - Using device: {device}")
    prices_df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    adj_matrix = np.load(ADJ_MATRIX_FILE)
    target_stock_name = prices_df.columns[0]
    print(f"  - Target stock detected: '{target_stock_name}'")

    min_vals = prices_df.min().values
    max_vals = prices_df.max().values
    
    # (FIX) Add epsilon to avoid divide-by-zero if min == max
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-9 # Avoid division by zero
    
    scaled_data = (prices_df.values - min_vals) / range_vals

    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i : i + SEQ_LENGTH])
        y.append(scaled_data[i + SEQ_LENGTH])
    X, y = np.array(X), np.array(y)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)

    train_size = int(len(X_tensor) * TRAIN_SPLIT)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    test_dates = prices_df.index[train_size + SEQ_LENGTH:]
    
    print(f"  - Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    return X_train, y_train, X_test, y_test, edge_index, min_vals, max_vals, prices_df.columns, test_dates, target_stock_name

## =======================================================
# 2. GAT + LSTM MODEL DEFINITION (STABLE FAST VERSION)
# =======================================================
class GATLSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GATLSTM, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes, batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

    def forward(self, x_sequence, edge_index):
        # x_sequence shape: [batch, sequence_length, num_nodes]
        
        gat_out_sequence = []
        # Loop over time (S), not batch (B). This is FAST.
        for t in range(x_sequence.size(1)): # x_sequence.size(1) is seq_len
            x_t = x_sequence[:, t, :]
            x_t = x_t.unsqueeze(-1) # [B, N, 1]
            
            b, n, i = x_t.shape
            x_t_flat = x_t.reshape(-1, i) # [B*N, 1]
            
            # GATConv handles batching automatically if data is [B*N, F]
            gat_out = torch.relu(self.gat(x_t_flat, edge_index)).reshape(b, n, -1) # [B, N, H]
            gat_out_sequence.append(gat_out)
        
        x_gat = torch.stack(gat_out_sequence, dim=1) # [B, S, N, H]
        
        b, s, n, h = x_gat.shape
        x_gat_reshaped = x_gat.reshape(b, s, n * h) # [B, S, N*H]
        
        lstm_out, _ = self.lstm(x_gat_reshaped)
        prediction = self.linear(lstm_out[:, -1, :])
        return prediction
# =======================================================
# 3. MODEL TRAINING LOOP (WITH EARLY STOPPING)
# =======================================================
def train_model(model, X_train, y_train, edge_index):
    print_heading("STEP 2: MODEL TRAINING")
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    patience = 25
    patience_counter = 0
    best_loss = float('inf')
    
    start_time = time.time()
    
    X_train_device = X_train.to(device)
    y_train_device = y_train.to(device)
    edge_index_device = edge_index.to(device)
    
    for epoch in range(EPOCHS):
        model.train()
        
        predictions = model(X_train_device, edge_index_device)
        loss = loss_function(predictions, y_train_device)
        
        # (FIX) Check for NaN loss
        if torch.isnan(loss):
            print(f"ERROR: Loss is NaN at epoch {epoch+1}. Stopping training.")
            print("This might be due to bad data (NaNs) or exploding gradients.")
            return None # Return None to indicate failure

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0:
            print(f"  - Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n  - Early stopping at epoch {epoch+1}. No improvement in last {patience} epochs.")
            break
            
    print(f"\n  - Training completed in {time.time() - start_time:.2f} seconds.")
    return model

# =======================================================
# 4. MODEL EVALUATION AND PLOTTING
# =======================================================
def evaluate_model(trained_model, X_test, y_test, edge_index, min_vals, max_vals, column_names, test_dates, target_stock_name):
    print_heading("STEP 3: MODEL EVALUATION ON TEST DATA")
    trained_model.eval()
    with torch.no_grad():
        test_predictions_scaled = trained_model(X_test.to(device), edge_index.to(device))

    test_predictions_scaled = test_predictions_scaled.cpu()

    # (FIX) Add epsilon to avoid divide-by-zero if min == max
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-9 # Avoid division by zero
    
    test_predictions_actual = (test_predictions_scaled.numpy() * range_vals) + min_vals
    y_test_actual = (y_test.numpy() * range_vals) + min_vals

    stock_idx = list(column_names).index(target_stock_name)
    actual = y_test_actual[:, stock_idx]
    predicted = test_predictions_actual[:, stock_idx]

    rmse = np.sqrt(np.mean((predicted - actual)**2))
    print(f"  - RMSE for target stock '{target_stock_name}': {rmse:.4f}")

    if len(test_dates) != len(actual):
        print(f"  - Warning: Date and prediction length mismatch. Adjusting dates for plotting.")
        test_dates = test_dates[:len(actual)]

    plt.figure(figsize=(15, 7))
    plt.plot(test_dates, actual, label='Actual Price', color='blue')
    plt.plot(test_dates, predicted, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'Test Performance for {target_stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# =======================================================
# 5. MAIN EXECUTION FLOW
# =======================================================
def print_heading(text):
    print("\n" + "="*80)
    print(text)
    print("="*80)

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates, target_stock = load_and_prepare_data()
        
        num_nodes = X_train.shape[2]
        model = GATLSTM(num_nodes=num_nodes, in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=num_nodes)
        
        model.to(device)

        trained_model = train_model(model, X_train, y_train, edge_index)
        
        if trained_model is None:
             print_heading("PROCESS FAILED: Training resulted in NaN.")
        else:
            print(f"  - Loading best model saved at '{MODEL_SAVE_PATH}' for final evaluation.")
            model.load_state_dict(torch.load(MODEL_SAVE_PATH))

            evaluate_model(model, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates, target_stock)
            
            print(f"\n  - Best Trained Model Brain saved to '{MODEL_SAVE_PATH}'")
            print_heading("PROCESS COMPLETE")

    except FileNotFoundError:
        print("\n" + "="*80)
        print("ERROR: Could not find 'data/model_ready/combined_denoised_prices.csv'.")
        print("Please run 'run_pipeline.py <TICKER>' first to generate the necessary files.")
        print("="*80)