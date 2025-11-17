#!/usr/bin/env python3
"""
run_pipeline.py — build global combined dataset + correlation + adjacency
Usage examples:
  python3 run_pipeline.py --denoised-dir data/denoised_universe --out-dir data/model_ready --top-k 10
  python3 run_pipeline.py --adj-method threshold --threshold 0.55
"""

import os
import argparse
import shutil
import numpy as np
import pandas as pd
from typing import List

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_DENOISED_DIR = "data/denoised_universe"
DEFAULT_OUT_DIR = "data/model_ready"
DEFAULT_TOP_K = 10
DEFAULT_ADJ_METHOD = "topk"   # "topk" or "threshold"
DEFAULT_THRESHOLD = 0.5

# ----------------------------
# Helpers
# ----------------------------
def print_heading(s: str):
    print("\n" + "="*90)
    print(s)
    print("="*90)

def safe_file_to_ticker(fname: str) -> str:
    """
    Convert safe filename like 'ACC_NS.csv' or 'ACC_NSE.csv' into 'ACC.NS' (uses last underscore)
    If pattern unexpected, return base (without .csv)
    """
    base = os.path.basename(fname).replace(".csv", "")
    if "_" in base:
        a, b = base.rsplit("_", 1)
        return f"{a}.{b}"
    return base

def read_denoised_series(filepath: str, ticker: str) -> pd.Series | None:
    """
    Read CSV and return series named by ticker. Expected a 'Close' column.
    Returns None on failure.
    """
    try:
        df = pd.read_csv(filepath, parse_dates=True)
        # detect Date column name
        if "Date" in df.columns:
            df = df.set_index("Date")
        elif df.columns[0].lower() in ("date", "index"):
            df = df.set_index(df.columns[0])
        # prefer 'Close' column; if not present, try 2nd column
        if "Close" in df.columns:
            s = df["Close"].astype(float)
        else:
            if df.shape[1] >= 1:
                s = df.iloc[:, 0].astype(float)
            else:
                print(f"  - {os.path.basename(filepath)}: no usable column")
                return None
        return s.rename(ticker)
    except Exception as e:
        print(f"  - Error reading {filepath}: {e}")
        return None

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(denoised_dir: str, out_dir: str, top_k: int, adj_method: str, threshold: float):
    print_heading("GLOBAL PIPELINE — Building FULL Market Graph")
    if not os.path.exists(denoised_dir):
        raise FileNotFoundError(f"Denoised dir not found: {denoised_dir}")

    # reset out_dir
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(denoised_dir) if f.lower().endswith(".csv")])
    if not files:
        raise SystemExit("No denoised CSV files found. Run preprocess_universe first.")

    series_list: List[pd.Series] = []
    tickers: List[str] = []

    print(f"Found {len(files)} denoised files. Reading...")

    for f in files:
        path = os.path.join(denoised_dir, f)
        t = safe_file_to_ticker(f).upper()
        s = read_denoised_series(path, t)
        if s is None:
            print("  Skipping:", f)
            continue
        # ensure index is datetime and sorted
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
        series_list.append(s)
        tickers.append(t)

    if not series_list:
        raise SystemExit("No valid series loaded.")

    # Align: inner join (common dates). This keeps model simpler and consistent.
    print_heading("ALIGNING ALL STOCKS")
    full_df = pd.concat(series_list, axis=1, join="inner").sort_index()
    print(f"Loaded {len(full_df.columns)} tickers, dates: {full_df.index.min().date()} -> {full_df.index.max().date()}")
    print("Final shape:", full_df.shape)
    # small preview
    print(full_df.head(3))

    # Save combined
    combined_path = os.path.join(out_dir, "combined_denoised_prices.csv")
    full_df.to_csv(combined_path)
    print("Saved combined:", combined_path)

    # Save tickers ordering (important)
    tickers_path = os.path.join(out_dir, "tickers.txt")
    with open(tickers_path, "w") as fh:
        for t in full_df.columns.tolist():
            fh.write(t + "\n")
    print("Saved tickers list:", tickers_path)

    # ------------------------------
    # Compute log-returns for correlation
    # ------------------------------
    print_heading("COMPUTING LOG RETURNS")
    prices = full_df.values.astype(float)
    # compute log returns safely (first row zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log(prices[1:] / prices[:-1])
    lr = np.vstack([np.zeros((1, prices.shape[1])), lr])
    returns_df = pd.DataFrame(lr, index=full_df.index, columns=full_df.columns)
    print(returns_df.head(3))

    # ------------------------------
    # Correlation matrix
    # ------------------------------
    print_heading("BUILDING CORRELATION MATRIX")
    # if there are NaNs in returns_df, fill with 0 (neutral) only for correlation computation
    rr = returns_df.fillna(0).values.T  # shape [N, T]
    corr = np.corrcoef(rr)
    corr = np.clip(corr, -1.0, 1.0)
    corr_df = pd.DataFrame(corr, index=full_df.columns, columns=full_df.columns)
    corr_path = os.path.join(out_dir, "correlation_matrix.csv")
    corr_df.to_csv(corr_path)
    print("Saved:", corr_path)

    # ------------------------------
    # Build adjacency
    # ------------------------------
    print_heading("BUILDING GLOBAL ADJACENCY")
    N = corr.shape[0]
    adj = np.zeros((N, N), dtype=int)

    if adj_method == "threshold":
        print(f"Using threshold method (abs(corr) >= {threshold})")
        adj = (np.abs(corr) >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
    else:
        k = int(top_k)
        print(f"Using top-k method (k={k} per node)")
        for i in range(N):
            row = np.abs(corr[i]).copy()
            row[i] = -1.0
            idx = np.argsort(-row)[:k]
            adj[i, idx] = 1
        # symmetrize
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0)

    # Save adjacency
    adj_npy = os.path.join(out_dir, "adjacency_matrix.npy")
    np.save(adj_npy, adj)
    adj_csv = os.path.join(out_dir, "adjacency_matrix.csv")
    pd.DataFrame(adj, index=full_df.columns, columns=full_df.columns).to_csv(adj_csv)
    print("Saved adjacency (npy + csv):", adj_npy, adj_csv)
    print("Adjacency shape:", adj.shape, "Edges:", int(np.sum(adj)//2))

    print_heading("GLOBAL PIPELINE COMPLETED")
    print("You can now run: python3 trainmodel.py")
    return

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build combined dataset, correlation and adjacency from denoised CSVs.")
    p.add_argument("--denoised-dir", default=DEFAULT_DENOISED_DIR, help="Directory with denoised CSVs")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for model-ready files")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K neighbors per node (for topk method)")
    p.add_argument("--adj-method", choices=["topk","threshold"], default=DEFAULT_ADJ_METHOD, help="Adjacency building method")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Correlation threshold (if using threshold adj_method)")
    args = p.parse_args()

    run_pipeline(args.denoised_dir, args.out_dir, args.top_k, args.adj_method, args.threshold)
