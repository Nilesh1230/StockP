# build_full_graph.py
import os
import numpy as np
import pandas as pd

MODEL_READY = "data/model_ready"
COMBINED = os.path.join(MODEL_READY, "combined_denoised_prices.csv")
K = 5  # top-k neighbors per node

def compute_logreturns(df):
    p = df.values.astype(float)
    lr = np.log(p[1:] / p[:-1])
    lr = np.vstack([np.zeros((1, p.shape[1])), lr])
    return lr

def build_graph(k=K):
    if not os.path.exists(COMBINED):
        raise FileNotFoundError(f"{COMBINED} missing. Run combine_all.py first.")
    df = pd.read_csv(COMBINED, index_col=0, parse_dates=True)
    cols = list(df.columns)
    print("Loaded combined:", df.shape)
    returns = compute_logreturns(df)  # [T, N]
    corr = np.corrcoef(returns.T)
    corr = np.clip(corr, -1.0, 1.0)
    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    corr_path = os.path.join(MODEL_READY, "correlation_matrix.csv")
    corr_df.to_csv(corr_path)
    print("Saved correlation:", corr_path)
    n = corr.shape[0]
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        row = np.abs(corr[i]).copy()
        row[i] = -1.0
        idx_sorted = np.argsort(-row)
        topk = idx_sorted[:k]
        adj[i, topk] = 1
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    adj_path = os.path.join(MODEL_READY, "adjacency_matrix.npy")
    np.save(adj_path, adj)
    print("Saved adjacency:", adj_path)
    return adj, corr_df

if __name__ == "__main__":
    build_graph()
