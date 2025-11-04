# FILE: preprocess_universe.py
# AIM: Pre-processes the entire 100-stock universe.
# FINAL FIX: Uses "Returns-Based DWT" (Price -> Returns -> Denoise Returns -> Price)
#          to correctly handle high-cost stocks.

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

# =======================================================
#                  CONTROL PANEL
# =======================================================
STOCK_UNIVERSE = [
    'ACC.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 
    'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AUROPHARMA.NS', 'AXISBANK.NS', 
    'BAJAJFINANCE.NS', 'BAJAJFINSV.NS', 'BAJAJHLDNG.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 
    'BHARATFORG.NS', 'BHARTIARTL.NS', 'BOSCHLTD.NS', 'BPCL.NS', 'BRITANNIA.NS', 
    'BSE.NS', 'CANBK.NS', 'CEATLTD.NS', 'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS', 
    'COLPAL.NS', 'DABUR.NS', 'DIVISLAB.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 
    'GAIL.NS', 'GODREJCP.NS', 'GRASIM.NS', 'HAVELLS.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 
    'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 
    'ICICIBANK.NS', 'ICICIGI.NS', 'ICICIPRULI.NS', 'IDBI.NS', 'IFCI.NS', 'INDIGO.NS', 
    'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS', 'ITC.NS', 'JINDALSTEL.NS', 'JSWSTEEL.NS', 
    'KOTAKBANK.NS', 'LICI.NS', 'LTIM.NS', 'LT.NS', 'LUPIN.NS', 'M&M.NS', 'MARICO.NS', 
    'MARUTI.NS', 'MFSL.NS', 'MOTHERSON.NS', 'MRF.NS', 'MUTHOOTFIN.NS', 'NESTLEIND.NS', 
    'NMDC.NS', 'NTPC.NS', 'ONGC.NS', 'PGINFRA.NS', 'PIDILITIND.NS', 'PNB.NS', 
    'POWERGRID.NS', 'RELIANCE.NS', 'RVNL.NS', 'SAMVARDHANA.NS', 'SBICARD.NS', 
    'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS', 'SUNPHARMA.NS', 
    'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 
    'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'VEDL.NS', 
    'WIPRO.NS', 'ZEEL.NS', 'ZOMATO.NS'
]
START_DATE = "2022-01-01"
END_DATE = str(pd.to_datetime('today').date())
SENSITIVITY = 0.1 # Yeh setting sahi hai
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
    # (FIX) DWT uses CORRELATION, not convolution. This was the bug.
    cA = np.correlate(signal, h0, mode='valid')[::2]
    cD = np.correlate(signal, h1, mode='valid')[::2]
    return cA, cD

def idwt_from_scratch(cA, cD):
    min_len = min(len(cA), len(cD)); cA, cD = cA[:min_len], cD[:min_len]
    rec_even = (cA + cD) / np.sqrt(2.0); rec_odd  = (cA - cD) / np.sqrt(2.0)
    recon = np.zeros(2 * min_len, dtype=float); recon[0::2] = rec_even; recon[1::2] = rec_odd
    return recon

# =======================================================
#      MAIN PREPROCESSING WORKFLOW (FIXED: RETURNS-BASED DWT)
# =======================================================
def run_universe_preprocessing():
    print_heading("Step 1: Preprocessing Stock Universe (Returns-Based DWT)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"  - Starting to download and denoise {len(STOCK_UNIVERSE)} stocks...")
    print(f"  - Using SENSITIVITY = {SENSITIVITY} and START_DATE = {START_DATE}")
    print(f"  - Denoised files will be saved in: '{OUTPUT_DIR}'")
    
    for i, ticker in enumerate(STOCK_UNIVERSE):
        try:
            print(f"\n  ({i+1}/{len(STOCK_UNIVERSE)}) Processing '{ticker}'...")
            
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if df.empty or 'Close' not in df.columns or len(df) < 10:
                print(f"    - SKIP: No data found for {ticker}.")
                continue
            
            # --- (FIX) STEP 1: Get Price and Calculate Returns ---
            price_signal = df['Close'].to_numpy(dtype=float).flatten()
            return_signal = df['Close'].pct_change().fillna(0).to_numpy(dtype=float).flatten()
            
            # --- (FIX) STEP 2: Denoise the "Return" signal ---
            cA, cD = dwt_from_scratch(return_signal) # Yeh ab fixed function use karega
            
            if cD.size == 0:
                denoised_return_signal = return_signal.copy()
            else:
                sigma = np.median(np.abs(cD)) / 0.6745 if np.median(np.abs(cD)) > 0 else 0
                threshold = sigma * SENSITIVITY
                cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD) - threshold)
                denoised_return_signal = idwt_from_scratch(cA, cD_clean)
            
            if len(denoised_return_signal) < len(return_signal): 
                denoised_return_signal = np.pad(denoised_return_signal, (0, len(return_signal)-len(denoised_return_signal)), 'edge')
            
            denoised_return_signal = denoised_return_signal[:len(price_signal)]
            
            # --- (FIX) STEP 3: Reconstruct Price from Denoised Returns ---
            denoised_price_signal = np.zeros_like(price_signal)
            denoised_price_signal[0] = price_signal[0] 
            
            for t in range(1, len(price_signal)):
                denoised_price_signal[t] = denoised_price_signal[t-1] * (1 + denoised_return_signal[t])

            denoised_df = pd.DataFrame({'Close': denoised_price_signal}, index=df.index)
            safe_ticker_name = ticker.replace('.', '_')
            
            output_path = os.path.join(OUTPUT_DIR, f"{safe_ticker_name}.csv")
            
            denoised_df.to_csv(output_path)
            print(f"    - Saved Returns-Denoised data to '{output_path}'")
            
        except Exception as e:
            print(f"    - ERROR processing {ticker}: {e}")
            
    print_heading("Universe Preprocessing Complete (Returns-Based DWT)")

if __name__ == "__main__":
    run_universe_preprocessing()