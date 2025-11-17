# combine_all.py
import os
import pandas as pd
import numpy as np

DENOISED_DIR = "data/denoised_universe"
OUT_DIR = "data/model_ready"
os.makedirs(OUT_DIR, exist_ok=True)


def safe_to_ticker(fname):
    base = os.path.basename(fname).replace(".csv", "")
    # convert SBIN_NS → SBIN.NS
    if "_" in base:
        parts = base.rsplit("_", 1)
        return (parts[0] + "." + parts[1]).upper()
    return base.upper()


def combine_all():
    files = sorted([f for f in os.listdir(DENOISED_DIR) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSVs in {DENOISED_DIR}. Run preprocess_universe.py first.")

    series = []
    for f in files:
        path = os.path.join(DENOISED_DIR, f)
        df = pd.read_csv(path, parse_dates=["Date"])

        if "Close" not in df.columns:
            print("Skipping", f, "(no Close column)")
            continue

        ticker = safe_to_ticker(f)
        s = df[["Date", "Close"]].rename(columns={"Close": ticker}).set_index("Date")
        series.append(s)
        print("Loaded:", ticker)

    if not series:
        raise RuntimeError("No valid denoised CSVs loaded.")

    # IMPORTANT FIX:
    # UNIVERSAL MODEL MUST USE OUTER JOIN TO KEEP ALL STOCKS
    combined = pd.concat(series, axis=1, join="outer")

    # sort and fill missing values
    combined = combined.sort_index()
    combined = combined.ffill().bfill()

    # DO NOT DROP COLUMNS (SBIN, RVNL, YESBANK get dropped earlier)
    # — universal model needs all stocks
    # (Previous problematic code removed)
    #
    # nan_ratio = combined.isna().mean()
    # drop_cols = nan_ratio[nan_ratio > 0.1].index.tolist()
    # if drop_cols:
    #     combined = combined.drop(columns=drop_cols)

    # Save
    outp = os.path.join(OUT_DIR, "combined_denoised_prices.csv")
    combined.to_csv(outp, index=True)

    print("\nSaved combined matrix:", outp)
    print("Final shape:", combined.shape)
    return combined


if __name__ == "__main__":
    combine_all()
