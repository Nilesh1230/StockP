# debug_predict.py
import os, json, numpy as np, pandas as pd, torch
from datetime import timedelta
import sys

SEQ_LENGTH = 30
MODEL_DIR = "models"
META_DIR = "models/meta"
DENOISED_DIR = "data/denoised_universe"
COMBINED = "data/model_ready/combined_denoised_prices.csv"

def find_meta(t):
    cand = os.path.join(META_DIR, f"{t}.json")
    if os.path.exists(cand): return cand
    # try fuzzy
    for f in os.listdir(META_DIR):
        if t.lower() in f.lower(): return os.path.join(META_DIR, f)
    return None

def find_model(t):
    for f in os.listdir(MODEL_DIR):
        if t.lower() in f.lower() and f.endswith(".pth"): return os.path.join(MODEL_DIR, f)
    return None

def find_denoised(t):
    for f in os.listdir(DENOISED_DIR):
        if t.lower() in f.lower(): return os.path.join(DENOISED_DIR, f)
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_predict.py TICKER")
        sys.exit(1)
    t = sys.argv[1].upper().replace(".NS","")
    print("Ticker:", t)

    meta_f = find_meta(t); model_f = find_model(t); den_f = find_denoised(t)
    print("meta:", meta_f)
    print("model:", model_f)
    print("denoised file:", den_f)
    if not meta_f or not den_f:
        print("Missing files -> stop")
        sys.exit(1)

    meta = json.load(open(meta_f))
    group = meta["group_tickers"]
    mu = np.array(meta["mu"]); sd = np.array(meta["sd"])
    ei = np.array(meta["edge_index"])

    print("\nGroup tickers (len={}):".format(len(group)))
    print(group)
    print("\nmu.shape, sd.shape:", mu.shape, sd.shape)
    # confirm target index (we assume target is first)
    try:
        t_idx = group.index(t + ".NS")
    except ValueError:
        try:
            t_idx = group.index(t)   # maybe stored without .NS
        except ValueError:
            t_idx = 0
    print("Assumed target index in group:", t_idx)

    # load denoised series
    df = pd.read_csv(den_f, parse_dates=["Date"])
    if "Date" in df.columns:
        df = df.set_index("Date")
    print("\nDenoised file rows:", len(df))
    # prefer Close or Denoised_Close
    col = "Denoised_Close" if "Denoised_Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
    prices = df[col].values.astype(float)
    print("Last price (denoised):", prices[-1])
    seq = prices[-SEQ_LENGTH:]
    print("seq shape:", seq.shape)

    # load combined universe slice
    combined = pd.read_csv(COMBINED, index_col=0, parse_dates=True)
    print("\nCombined shape:", combined.shape)
    # ensure group columns exist in combined
    miss = [g for g in group if g not in combined.columns and g.replace(".NS","") not in combined.columns]
    print("Missing in combined:", miss)

    ctx = combined[group].tail(SEQ_LENGTH).values.astype(float)
    print("ctx shape:", ctx.shape)
    # injection
    ctx[:, t_idx] = seq
    print("Injected target into context at index", t_idx)

    # normalization check
    print("\nmu (first 8):", mu[:8])
    print("sd  (first 8):", sd[:8])
    # compute normalized X for target column
    if mu.shape[0] == ctx.shape[1]:
        xnorm = (ctx - mu.reshape(1,-1)) / sd.reshape(1,-1)
        print("xnorm[:, t_idx] sample:", xnorm[:, t_idx][:5])
    else:
        print("mu/sd shape mismatch vs ctx columns:", mu.shape, ctx.shape)

    # historical logreturns of target (recent)
    p = ctx[:, t_idx]
    lr = np.log(p[1:]/p[:-1])
    print("\nRecent log-returns (last 10):", np.round(lr[-10:],6))
    print("LR stats: mean", np.round(lr.mean(),6), "std", np.round(lr.std(),6),
          "min", np.round(lr.min(),6), "max", np.round(lr.max(),6))
    # load model preds raw (if model exists)
    if model_f:
        import torch, importlib
        # load model weights only to inspect dims
        state = torch.load(model_f, map_location="cpu")
        # try to extract last linear bias if present
        print("\nModel state keys sample:", list(state.keys())[:8])
    print("\n--- DEBUG COMPLETE ---")
