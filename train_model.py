# FILE: train_model.py
# STAGE 3 & 4 — MODEL BUILDING, TRAINING & EVALUATION (MANUAL STEP-BY-STEP)

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
HIDDEN_CHANNELS = 64
EPOCHS = 50
LEARNING_RATE = 0.001
STOCK_TO_PLOT = 'TCS'

DATA_FILE = os.path.join("data", "model_ready", "combined_denoised_prices.csv")
ADJ_MATRIX_FILE = os.path.join("data", "model_ready", "adjacency_matrix.npy")
# =======================================================

# =======================================================
# 1️⃣ MANUAL MIN–MAX SCALING FUNCTION (No sklearn)
# =======================================================
def manual_min_max_scaler(data: pd.DataFrame):
    # For each stock (column), compute its min and max over the entire time range
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    print("\n================ MANUAL SCALING ================")
    print(f"  ➤ Example: First stock column Min = {min_vals.iloc[0]:.2f}, Max = {max_vals.iloc[0]:.2f}")
    print("  ➤ Formula applied: scaled_value = (x - min) / (max - min)")

    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data, min_vals, max_vals

# =======================================================
# 2️⃣ LOAD DATA + MANUAL SCALING + SLIDING WINDOWS
# =======================================================
def load_and_prepare_data():
    print("\n" + "="*80)
    print(" STEP 1: DATA LOADING & PREPARATION ")
    print("="*80)

    # Load price data and adjacency matrix
    prices_df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    adj_matrix = np.load(ADJ_MATRIX_FILE)

    print(f"  ➤ Price Data Shape        : {prices_df.shape}")
    print(f"  ➤ Adjacency Matrix Shape  : {adj_matrix.shape}")

    # Manually scale data to [0, 1]
    scaled_data_df, min_vals, max_vals = manual_min_max_scaler(prices_df)
    scaled_data = scaled_data_df.to_numpy()

    # Sliding window to create input sequences (X) and next-day targets (y)
    print("\n================ SEQUENCE BUILDING ================")
    X, y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i : i + SEQ_LENGTH])
        y.append(scaled_data[i + SEQ_LENGTH])
    X, y = np.array(X), np.array(y)

    print(f"  ➤ X shape: {X.shape} → (samples={X.shape[0]}, seq_len={SEQ_LENGTH}, nodes={X.shape[2]})")
    print(f"  ➤ y shape: {y.shape} → (samples={y.shape[0]}, nodes={y.shape[1]})")

    # Convert arrays to tensors
    print("\n================ TENSOR CONVERSION ================")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Convert adjacency matrix to PyTorch geometric edge_index format
    edge_index = torch.tensor(np.array(np.where(adj_matrix == 1)), dtype=torch.long)
    print(f"  ➤ edge_index shape: {edge_index.shape} — represents graph edges")

    # Train-test split
    print("\n================ TRAIN-TEST SPLIT ================")
    train_size = int(len(X_tensor) * TRAIN_SPLIT)
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

    print(f"  ➤ Training samples: {len(X_train)}")
    print(f"  ➤ Testing  samples: {len(X_test)}")

    return X_train, y_train, X_test, y_test, edge_index, min_vals.to_numpy(), max_vals.to_numpy(), prices_df.columns, prices_df.index[train_size+SEQ_LENGTH:]

# =======================================================
# 3️⃣ GAT + LSTM MODEL (BUILT STEP BY STEP)
# =======================================================
class GATLSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GATLSTM, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, heads=1, dropout=0.2)
        self.lstm = nn.LSTM(hidden_channels * num_nodes, hidden_channels * num_nodes, batch_first=True)
        self.linear = nn.Linear(hidden_channels * num_nodes, out_channels)

    def forward(self, x_sequence, edge_index):
        gat_out_sequence = []
        # Process each time step individually with GAT
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :].unsqueeze(-1)   # (batch, nodes, 1)
            batch_size, num_nodes, in_channels = x_t.shape

            x_t_flat = x_t.reshape(-1, in_channels)
            gat_out = torch.relu(self.gat(x_t_flat, edge_index))
            gat_out = gat_out.reshape(batch_size, num_nodes, -1)
            gat_out_sequence.append(gat_out)

        # Stack over the time dimension → (batch, seq_len, nodes, hidden)
        x_gat = torch.stack(gat_out_sequence, dim=1)
        batch_size, seq_len, num_nodes, hidden = x_gat.shape

        # Reshape for LSTM input
        x_gat_reshaped = x_gat.reshape(batch_size, seq_len, num_nodes * hidden)
        lstm_out, _ = self.lstm(x_gat_reshaped)

        # Take the output at the last time step for prediction
        last_step = lstm_out[:, -1, :]
        prediction = self.linear(last_step)
        return prediction

# =======================================================
# 4️⃣ TRAINING LOOP — MANUAL EXPLANATION
# =======================================================
def train_model(model, X_train, y_train, edge_index):
    print("\n" + "="*80)
    print(" STEP 6: MODEL TRAINING ")
    print("="*80)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()

        # Forward pass: Predict next-day prices
        predictions = model(X_train, edge_index)

        # Compute loss between predictions and actual target
        loss = loss_function(predictions, y_train)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss at regular intervals
        if (epoch+1) % 10 == 0:
            print(f"  ➤ Epoch [{epoch+1}/{EPOCHS}] → Loss = {loss.item():.6f}")

    print(f"\n✅ Training completed in {time.time() - start_time:.2f} seconds.")
    return model

# =======================================================
# 5️⃣ EVALUATION + MANUAL RESCALING + RMSE
# =======================================================
def evaluate_model(trained_model, X_test, y_test, edge_index, min_vals, max_vals, column_names, test_dates):
    print("\n" + "="*80)
    print(" STEP 7: MODEL EVALUATION ")
    print("="*80)

    trained_model.eval()
    with torch.no_grad():
        test_predictions_scaled = trained_model(X_test, edge_index)

    # Convert scaled predictions back to original price range
    range_vals = max_vals - min_vals
    test_predictions_actual = (test_predictions_scaled.numpy() * range_vals) + min_vals
    y_test_actual = (y_test.numpy() * range_vals) + min_vals

    # Manual RMSE calculation: sqrt(mean((pred - actual)^2))
    rmse = np.sqrt(np.mean((test_predictions_actual - y_test_actual)**2))
    print(f"  ➤ Overall RMSE = {rmse:.4f}")

    # Plot and compare for a single stock
    try:
        stock_idx = list(column_names).index(STOCK_TO_PLOT)
        actual = y_test_actual[:, stock_idx]
        predicted = test_predictions_actual[:, stock_idx]

        stock_rmse = np.sqrt(np.mean((predicted - actual)**2))
        print(f"  ➤ RMSE for {STOCK_TO_PLOT}: {stock_rmse:.4f}")

        print(f"\n--- Last 5 Days Comparison ({STOCK_TO_PLOT}) ---")
        comparison = pd.DataFrame({
            'Date': test_dates[-5:], 
            'Actual Price': actual[-5:].round(2), 
            'Predicted Price': predicted[-5:].round(2)
        })
        print(comparison.to_string(index=False))

        # Plot the actual vs predicted price curve
        plt.figure(figsize=(15, 7))
        plt.plot(test_dates, actual, label='Actual Price', color='blue')
        plt.plot(test_dates, predicted, label='Predicted Price', color='red', linestyle='--')
        plt.title(f'Actual vs Predicted Prices for {STOCK_TO_PLOT}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
        print(f"Stock '{STOCK_TO_PLOT}' not found in dataset.")

# =======================================================
# 6️⃣ MAIN EXECUTION
# =======================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates = load_and_prepare_data()

    print("\n" + "="*80)
    print(" STEP 5: BUILDING THE MODEL ")
    print("="*80)
    num_nodes, in_channels = X_train.shape[2], 1
    model = GATLSTM(num_nodes=num_nodes, in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, out_channels=num_nodes)
    print(model)

    trained_model = train_model(model, X_train, y_train, edge_index)
    evaluate_model(trained_model, X_test, y_test, edge_index, min_vals, max_vals, columns, test_dates)

    # =======================================================
    # Action 1: Save the trained model brain
    # =======================================================
    torch.save(trained_model.state_dict(), "gat_lstm_model.pth")
    print("\n✅ Trained Model Brain has been saved to 'gat_lstm_model.pth'")
