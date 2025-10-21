# ================================================================
# prepare_model_data.py
# ---------------------------------------------------------------
# Combines denoised stock data with original datasets to create
# a clean, aligned price matrix and an adjacency matrix for
# Graph-based AI models (e.g., GAT, GCN, GraphWaveNet).
# ---------------------------------------------------------------
# Key Features:
#  - Robust file handling and error reporting
#  - Automatic date parsing and alignment
#  - Correlation-based graph construction
#  - Works seamlessly after `clean_datasets.py` step
# ================================================================

import os
import numpy as np
import pandas as pd

# =======================================================
#                  CONTROL PANEL
# =======================================================
SENSITIVITY = 3.0                               # Used to identify denoised file suffix
DENOISED_DIR = "data/denoised_single_level"     # Directory with denoised prices
ORIGINAL_DIR = "datasets"                       # Directory with cleaned original CSVs
OUTPUT_DIR = "data/model_ready"                 # Where final merged files will be stored
CORRELATION_THRESHOLD = 0.7                     # Threshold for adjacency matrix

TICKER_LIST = [
    'TCS', 'INFY', 'WIPRO', 'HCLTECH',
    'RELIANCE', 'ONGC', 'BPCL',
    'HDFCBANK', 'ICICIBANK', 'SBIN'
]
# =======================================================


def prepare_data():
    """
    Merges denoised datasets with original dates to produce
    a final aligned price matrix and adjacency matrix.
    """
    print("=" * 90)
    print("  STEP 2: PREPARING DATA FOR THE AI MODEL ")
    print("=" * 90)

    all_frames = []

    # ---------------------------------------------------
    # TASK 1: Load & Combine Denoised Data
    # ---------------------------------------------------
    print("\n TASK 1: Loading and combining data files...\n")

    for ticker in TICKER_LIST:
        denoised_path = os.path.join(DENOISED_DIR, f"{ticker}_denoised_S{SENSITIVITY}.csv")
        original_path = os.path.join(ORIGINAL_DIR, f"{ticker}.csv")

        if not os.path.exists(denoised_path):
            print(f"    Skipped {ticker}: Missing denoised file.")
            continue

        if not os.path.exists(original_path):
            print(f"    Skipped {ticker}: Missing original dataset.")
            continue

        try:
            # --- Load original file to get Date column ---
            original_df = pd.read_csv(original_path, index_col=0, parse_dates=True)
            original_df.reset_index(inplace=True)

            # --- Load denoised data ---
            denoised_df = pd.read_csv(denoised_path)

            # --- Merge Date + Close into one clean DataFrame ---
            temp_df = pd.DataFrame({
                'Date': original_df.iloc[:, 0],   # First column after reset is always Date
                ticker: denoised_df['Close']      # Rename 'Close' to ticker
            })

            all_frames.append(temp_df)
            print(f"   {ticker} → Data processed & aligned.")

        except Exception as e:
            print(f"   ERROR processing {ticker}: {e}")

    if not all_frames:
        print("\n FATAL: No valid datasets found. Check your input folders.")
        return

    # ---------------------------------------------------
    # Merge all tickers on 'Date' column
    # ---------------------------------------------------
    merged_df = all_frames[0]
    for next_df in all_frames[1:]:
        merged_df = pd.merge(merged_df, next_df, on='Date', how='inner')

    # --- Clean & standardize Date column ---
    merged_df['Date'] = pd.to_datetime(merged_df['Date'], errors='coerce')
    merged_df.dropna(subset=['Date'], inplace=True)
    merged_df['Date'] = merged_df['Date'].dt.date
    merged_df.set_index('Date', inplace=True)

    # ---------------------------------------------------
    # Save combined data
    # ---------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_path = os.path.join(OUTPUT_DIR, "combined_denoised_prices.csv")
    merged_df.to_csv(combined_path)
    print(f"\n Combined price matrix saved → {combined_path}")
    print(f" Shape: {merged_df.shape}")

    # ---------------------------------------------------
    # TASK 2: Create Graph Structure (Adjacency Matrix)
    # ---------------------------------------------------
    print("\nTASK 2: Creating correlation-based adjacency matrix...\n")

    correlation_matrix = merged_df.corr(method='pearson')
    adjacency_matrix = (correlation_matrix >= CORRELATION_THRESHOLD).astype(int)
    np.fill_diagonal(adjacency_matrix.values, 0)

    adj_path = os.path.join(OUTPUT_DIR, "adjacency_matrix.npy")
    np.save(adj_path, adjacency_matrix.values)

    print("   Correlation matrix calculated.")
    print(f"   Adjacency matrix saved → {adj_path}")

    print("\n" + "=" * 90)
    print(" DATA PREPARATION COMPLETE — READY FOR MODEL TRAINING ")
    print("=" * 90)


if __name__ == "__main__":
    prepare_data()
