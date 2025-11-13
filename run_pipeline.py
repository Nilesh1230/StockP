# ================================================================
# FILE: run_pipeline_verbose.py (FINAL OPTIMIZED VERSION)
# ================================================================
# This version provides a fully explained and optimized implementation
# of the dynamic GAT pipeline for stock correlation and graph creation.
#
# GOAL:
#   For a given target stock (e.g. "RVNL.NS"), we:
#       1. Load all pre-denoised stock price data
#       2. Compute pairwise correlations manually
#       3. Select the most correlated peers
#       4. Combine those series into one master dataset
#       5. Build adjacency matrix for GAT input
#
# OUTPUT FILES:
#   - data/model_ready/combined_denoised_prices.csv
#   - data/model_ready/correlation_matrix.csv
#   - data/model_ready/adjacency_matrix.npy
#
# ================================================================

import os
import argparse
import shutil
import numpy as np
import pandas as pd

# ================================================================
# CONFIGURATION (like a control panel)
# ================================================================
DENOISED_UNIVERSE_DIR = "data/denoised_universe"
DIR_MODEL_READY = "data/model_ready"

DEFAULT_NUM_PEERS = 9
DEFAULT_CORRELATION_THRESHOLD = 0.5  # (kept for backward compatibility)

# ================================================================
# HELPER: Print formatted section headings
# ================================================================
def print_heading(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)

# ================================================================
# FUNCTION: MANUAL CORRELATION CALCULATION
# ================================================================
def manual_correlation(x, y):
    """
    Compute Pearson correlation from scratch (no pandas/numpy shortcut).
    Formula:
        r_xy = Σ[(xi - mean_x) * (yi - mean_y)] / sqrt(Σ(xi - mean_x)^2 * Σ(yi - mean_y)^2)
    """
    n = len(x)
    if n == 0:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    denominator = (denom_x * denom_y) ** 0.5

    return 0.0 if denominator == 0 else numerator / denominator

# ================================================================
# MAIN PIPELINE FUNCTION
# ================================================================
def run_dynamic_pipeline(target_ticker, num_peers, correlation_threshold):

    # ----------------------------------------------------------------
    # STEP 0: Clean start — prepare model output directory
    # ----------------------------------------------------------------
    if os.path.exists(DIR_MODEL_READY):
        shutil.rmtree(DIR_MODEL_READY)
    os.makedirs(DIR_MODEL_READY, exist_ok=True)

    # ----------------------------------------------------------------
    # STEP 1: LOAD ALL PRE-DENOISED STOCK FILES
    # ----------------------------------------------------------------
    print_heading("TASK 1: LOADING PRE-DENOISED STOCK UNIVERSE")

    if not os.path.exists(DENOISED_UNIVERSE_DIR) or not os.listdir(DENOISED_UNIVERSE_DIR):
        print(f"ERROR: Folder '{DENOISED_UNIVERSE_DIR}' not found or empty.")
        print("Please run 'preprocess_universe.py' to create denoised CSVs first.")
        return

    all_denoised_files = [f for f in os.listdir(DENOISED_UNIVERSE_DIR) if f.endswith('.csv')]
    print(f"Total denoised stock files found: {len(all_denoised_files)}")

    df_list = []
    for filename in all_denoised_files:
        ticker_name = filename.replace('.csv', '').replace('_', '.')
        filepath = os.path.join(DENOISED_UNIVERSE_DIR, filename)
        print(f"Reading: {filepath} → {ticker_name}")

        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        if 'Close' not in df.columns:
            print(f"WARNING: 'Close' column missing in {filename}, skipping...")
            continue

        # Rename 'Close' to ticker name
        df.rename(columns={'Close': ticker_name}, inplace=True)
        df_list.append(df[[ticker_name]])

    # Combine all into one master dataframe and interpolate any missing values
    universe_df = pd.concat(df_list, axis=1).interpolate(method='linear').dropna(how='any')
    print(f"\nLoaded {len(universe_df.columns)} stocks successfully.")
    print("Data sample (first 3 rows):")
    print(universe_df.head(3).round(4).to_string())

    # ----------------------------------------------------------------
    # STEP 2: FIND CLOSEST PEERS BASED ON CORRELATION
    # ----------------------------------------------------------------
    print_heading(f"TASK 2: FINDING {num_peers} CLOSEST PEERS FOR TARGET '{target_ticker}'")

    # Add suffix if missing
    if not target_ticker.upper().endswith(".NS"):
        target_ticker += ".NS"

    safe_target = target_ticker.upper()
    if safe_target not in universe_df.columns:
        print(f"ERROR: Target '{safe_target}' not found in universe columns.")
        return

    target_series = universe_df[safe_target].values.tolist()
    print(f"Target series length: {len(target_series)}")
    print(f"Target first 5 values: {target_series[:5]}")

    correlations = {}
    for col in universe_df.columns:
        if col == safe_target:
            continue
        if len(universe_df[col]) < 30:
            continue
        val = manual_correlation(target_series, universe_df[col].values.tolist())
        correlations[col] = val
        print(f"Correlation({safe_target}, {col}) = {val:.4f}")

    sorted_peers = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    top_peers = [k for k, v in sorted_peers[:num_peers]]

    final_group = [safe_target] + top_peers
    print("\nTop correlated peers selected:")
    for rank, name in enumerate(final_group, start=1):
        print(f"  {rank:02d}. {name}")

    # ----------------------------------------------------------------
    # STEP 3: MERGE FINAL GROUP DATA
    # ----------------------------------------------------------------
    print_heading("TASK 3: MERGING FINAL GROUP INTO MASTER CSV")

    final_df = universe_df[final_group].copy()
    final_df.columns = [c.replace('.NS', '') for c in final_df.columns]

    combined_path = os.path.join(DIR_MODEL_READY, "combined_denoised_prices.csv")
    final_df.to_csv(combined_path, index=True)

    print(f"Saved combined master file to: {combined_path}")
    print(f"Shape: {final_df.shape}")
    print("Head (first 5 rows):")
    print(final_df.head(5).round(4).to_string())

    # ----------------------------------------------------------------
    # STEP 4: BUILD MANUAL CORRELATION MATRIX
    # ----------------------------------------------------------------
    print_heading("TASK 4: BUILDING MANUAL CORRELATION & ADJACENCY MATRIX")

    cols = final_df.columns.tolist()
    n = len(cols)
    corr_matrix = np.zeros((n, n))

    print(f"Building {n}x{n} correlation matrix manually...\n")
    for i in range(n):
        for j in range(n):
            xi = final_df[cols[i]].values.tolist()
            yj = final_df[cols[j]].values.tolist()
            corr_matrix[i, j] = manual_correlation(xi, yj)
            if i == j:
                corr_matrix[i, j] = 1.0

    corr_df = pd.DataFrame(corr_matrix, index=cols, columns=cols)
    corr_path = os.path.join(DIR_MODEL_READY, "correlation_matrix.csv")
    corr_df.to_csv(corr_path)
    print(f"Correlation matrix saved to: {corr_path}")
    print("Partial correlation matrix (first 4x4):")
    print(corr_df.iloc[:4, :4].round(3).to_string())

    # ----------------------------------------------------------------
    # ✅ STEP 5: BUILD ADJACENCY MATRIX (Top-K)
    # ----------------------------------------------------------------
    print_heading("TASK 5: BUILDING ADJACENCY MATRIX (Top-K connections)")

    k = 3  # Use top 3 strongest correlations per node
    adjacency = np.zeros_like(corr_matrix, dtype=int)

    for i in range(n):
        sorted_idx = np.argsort(-corr_matrix[i, :])  # descending order
        top_idx = [j for j in sorted_idx if j != i][:k]
        adjacency[i, top_idx] = 1

    adjacency = np.maximum(adjacency, adjacency.T)  # make symmetric
    np.fill_diagonal(adjacency, 0)  # remove self-loops

    adj_df = pd.DataFrame(adjacency, index=cols, columns=cols)
    print("Adjacency Matrix (1=connected, 0=not connected):")
    print(adj_df.iloc[:4, :4].to_string())

    adj_path = os.path.join(DIR_MODEL_READY, "adjacency_matrix.npy")
    np.save(adj_path, adjacency)
    print(f"\nAdjacency matrix saved as binary file: {adj_path}")

    total_edges = int(np.sum(adjacency) / 2)
    print(f"Total edges in undirected graph: {total_edges}")

    # ----------------------------------------------------------------
    # DONE
    # ----------------------------------------------------------------
    print_heading(f"PIPELINE COMPLETED SUCCESSFULLY FOR '{safe_target}'")
    print(f"All files saved in: {DIR_MODEL_READY}")
    print("→ You can now run 'train_model.py' for this group.")

# ================================================================
# COMMAND LINE ENTRY
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully explained and optimized dynamic pipeline for GAT stock grouping.")
    parser.add_argument("target_ticker", type=str, help="Target stock ticker, e.g. RVNL.NS")
    args = parser.parse_args()

    run_dynamic_pipeline(
        target_ticker=args.target_ticker,
        num_peers=DEFAULT_NUM_PEERS,
        correlation_threshold=DEFAULT_CORRELATION_THRESHOLD
    )
