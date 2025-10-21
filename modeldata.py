
import os
import time
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------
def section(title):
    print("\n" + "═" * 80)
    print(f"📘 {title}")
    print("═" * 80)

def step(title):
    print("\n" + "-" * 70)
    print(f"➡️  {title}")
    print("-" * 70)

def formula(expression, note=None):
    print("\n📌 Formula:")
    print(f"   {expression}")
    if note:
        print(f"   👉 {note}")

def pause(sec=1.2):
    """Narration pause to make console output readable step by step."""
    time.sleep(sec)

# ---------------------------------------------------------------
# Main Data Preparation Pipeline
# ---------------------------------------------------------------
def prepare_data(input_dir, tickers, threshold):
    # STEP 1: Merging All Denoised Price Files
    section("STEP 1: DATA MERGING — Denoised Price Series")
    print("Yahan hum sabhi stocks ke denoised close prices ko ek master table me jodenge.")
    pause(2)

    step("TASK 1: CSV Files ko Read aur Merge karna")
    merged_df = None

    for tkr in tickers:
        file_path = os.path.join(input_dir, f"{tkr}_denoised.csv")
        try:
            df = pd.read_csv(file_path)
            df = df[['Date', 'Close']].rename(columns={'Close': tkr})

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')

            print(f"   {tkr}: file merge ho gayi.")
            pause(0.5)

        except FileNotFoundError:
            print(f"    {tkr} ki file missing hai → skip kar diya.")
            pause(0.5)

    if merged_df is None:
        print("\nERROR: Koi bhi denoised file nahi mili. Pehle denoising script run karo.")
        return

    merged_df.dropna(inplace=True)
    merged_df.set_index('Date', inplace=True)

    out_dir = "data/model_ready"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combined_denoised_prices.csv")
    merged_df.to_csv(out_path)

    print("\n✅ Merging Complete.")
    print(f"📁 Saved at: {out_path}")
    print("🧪 Sample Data:")
    print(merged_df.head())
    pause(3)

    # STEP 2: Build Graph (Correlation + Adjacency)
    section("STEP 2: GRAPH BANANA — Stock Correlations")
    print("Ab hum stocks ke beech correlation calculate karke adjacency matrix banayenge.")
    pause(2)

    # 2.1 Correlation Matrix
    step("STEP 2.1: Pearson Correlation Matrix")
    formula(
        "ρ(i,j) = Cov(ri,rj) / (σi * σj)",
        "Do stocks ke returns ke linear relationship ko measure karta hai."
    )
    correlation_matrix = merged_df.corr(method='pearson')

    print("\n🧾 Correlation Matrix (rounded):")
    print(correlation_matrix.round(2))
    pause(3)

    # 2.2 Adjacency Matrix
    step("STEP 2.2: Adjacency Matrix banani using Threshold")
    print(f"Niyam: agar correlation ≥ {threshold} → edge = 1, warna 0.")
    adjacency_matrix = (correlation_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency_matrix.values, 0)  # no self-loops

    print("\n📊 Adjacency Matrix:")
    print(adjacency_matrix)
    pause(2)

    adj_path = os.path.join(out_dir, "adjacency_matrix.npy")
    np.save(adj_path, adjacency_matrix.values)
    print(f"\n✅ Graph structure saved at: {adj_path}")
    pause(1.5)

    # Finish
    section("🎉 PIPELINE COMPLETE")
    print("✔️ Clean merged price data ready.")
    print("✔️ Correlation matrix generated.")
    print("✔️ Adjacency matrix saved.")
    print("Next Step: GAT + LSTM Model training.")
    print("═" * 80)

# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------
if __name__ == "__main__":
    DEFAULT_TICKERS = [
        'TCS', 'INFY', 'WIPRO', 'HCLTECH',
        'RELIANCE', 'ONGC', 'BPCL', 'GAIL',
        'HDFCBANK', 'ICICIBANK', 'SBIN'
    ]

    parser = argparse.ArgumentParser(
        description="GAT+LSTM Model ke liye Denoised Price Data + Graph Structure Ready Karna"
    )
    parser.add_argument("--input_dir", type=str, default="data/denoised_from_scratch", help="Denoised CSV folder")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of stock tickers")
    parser.add_argument("--threshold", type=float, default=0.7, help="Correlation threshold for edges")

    args = parser.parse_args()
    prepare_data(args.input_dir, args.tickers, args.threshold)
