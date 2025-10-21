# FILE: preprocess_universe.py (Corrected Version)
# FIX: Corrected the typo from os.join.path to os.path.join

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

# =======================================================
#                  CONTROL PANEL
# =======================================================
STOCK_UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 
    'SBIN.NS', 'BHARTIARTL.NS', 'LICI.NS', 'ITC.NS', 'HCLTECH.NS', 'KOTAKBANK.NS', 
    'LT.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'ADANIENT.NS',
    'TATAMOTORS.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'WIPRO.NS', 'ULTRACEMCO.NS', 'ONGC.NS',
    'NESTLEIND.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'NTPC.NS', 'M&M.NS', 'BAJAJFINSV.NS',
    'POWERGRID.NS', 'GAIL.NS', 'ADANIPORTS.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'BPCL.NS',
    'GRASIM.NS', 'DRREDDY.NS', 'CIPLA.NS', 'VEDL.NS', 'IOC.NS', 'EICHERMOT.NS', 'IFCI.NS',
    'UPL.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'APOLLOHOSP.NS', 'BRITANNIA.NS', 'RVNL.NS', 'ZEEL.NS'
]
START_DATE = "2015-01-01"
END_DATE = str(pd.to_datetime('today').date())
SENSITIVITY = 3.0
OUTPUT_DIR = "data/denoised_universe"

# =======================================================
#   HELPER FUNCTIONS (DWT)
# =======================================================
def print_heading(title):
    print("\n" + "="*80)
    print(f"  {title.upper()}")
    print("="*80)

def dwt_from_scratch(signal):
    h0 = np.array([1.0, 1.0]) / np.sqrt(2.0); h1 = np.array([-1.0, 1.0]) / np.sqrt(2.0)
    cA = np.convolve(signal, h0, mode='valid')[::2]; cD = np.convolve(signal, h1, mode='valid')[::2]
    return cA, cD

def idwt_from_scratch(cA, cD):
    min_len = min(len(cA), len(cD)); cA, cD = cA[:min_len], cD[:min_len]
    rec_even = (cA + cD) / np.sqrt(2.0); rec_odd  = (cA - cD) / np.sqrt(2.0)
    recon = np.zeros(2 * min_len, dtype=float); recon[0::2] = rec_even; recon[1::2] = rec_odd
    return recon

# =======================================================
#      MAIN PREPROCESSING WORKFLOW
# =======================================================
def run_universe_preprocessing():
    print_heading("Step 1: Preprocessing Stock Universe")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"  - Starting to download and denoise {len(STOCK_UNIVERSE)} stocks...")
    print(f"  - Denoised files will be saved in: '{OUTPUT_DIR}'")
    
    for i, ticker in enumerate(STOCK_UNIVERSE):
        try:
            print(f"\n  ({i+1}/{len(STOCK_UNIVERSE)}) Processing '{ticker}'...")
            
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if df.empty or 'Close' not in df.columns:
                print(f"    - SKIP: No data found for {ticker}.")
                continue
            
            signal = df['Close'].to_numpy(dtype=float).flatten()
            
            cA, cD = dwt_from_scratch(signal)
            
            if cD.size == 0:
                denoised = signal.copy()
            else:
                sigma = np.median(np.abs(cD)) / 0.6745 if np.median(np.abs(cD)) > 0 else 0
                threshold = sigma * SENSITIVITY
                cD_clean = np.where(np.abs(cD) > threshold, cD, 0)
                denoised = idwt_from_scratch(cA, cD_clean)
            
            if len(denoised) < len(signal): 
                denoised = np.pad(denoised, (0, len(signal)-len(denoised)), 'edge')
            
            denoised_df = pd.DataFrame({'Close': denoised[:len(signal)]}, index=df.index)
            safe_ticker_name = ticker.replace('.', '_')
            
            # --- THIS IS THE CORRECTED LINE ---
            output_path = os.path.join(OUTPUT_DIR, f"{safe_ticker_name}.csv")
            
            denoised_df.to_csv(output_path)
            print(f"    - ✅ Saved denoised data to '{output_path}'")
            
        except Exception as e:
            print(f"    - ❌ ERROR processing {ticker}: {e}")
            
    print_heading("Universe Preprocessing Complete")

if __name__ == "__main__":
    run_universe_preprocessing()