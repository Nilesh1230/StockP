# ================================================================
# FILE: preprocess_universe_all.py (Final Continuous Version)
# AIM : Preprocesses all 150 stocks (no batching)
# METHOD : Returns-Based DWT using Daubechies-4 (db4)
# ================================================================

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pywt

# =======================================================
#                  CONTROL PANEL
# =======================================================
STOCK_UNIVERSE = [
    # --- Large Cap (100) + Mid/Small Cap (50) Combined ---
    'ACC.NS','ADANIENT.NS','ADANIGREEN.NS','ADANIPORTS.NS','ADANIPOWER.NS',
    'AMBUJACEM.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AUROPHARMA.NS','AXISBANK.NS',
    'BAJAJFINANCE.NS','BAJAJFINSV.NS','BAJAJHLDNG.NS','BANDHANBNK.NS','BANKBARODA.NS',
    'BHARATFORG.NS','BHARTIARTL.NS','BOSCHLTD.NS','BPCL.NS','BRITANNIA.NS',
    'BSE.NS','CANBK.NS','CEATLTD.NS','CHOLAFIN.NS','CIPLA.NS','COALINDIA.NS',
    'COLPAL.NS','DABUR.NS','DIVISLAB.NS','DLF.NS','DRREDDY.NS','EICHERMOT.NS',
    'GAIL.NS','GODREJCP.NS','GRASIM.NS','HAVELLS.NS','HCLTECH.NS','HDFCAMC.NS',
    'HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS',
    'ICICIBANK.NS','ICICIGI.NS','ICICIPRULI.NS','IDBI.NS','IFCI.NS','INDIGO.NS',
    'INDUSINDBK.NS','INFY.NS','IOC.NS','ITC.NS','JINDALSTEL.NS','JSWSTEEL.NS',
    'KOTAKBANK.NS','LICI.NS','LTIM.NS','LT.NS','LUPIN.NS','M&M.NS','MARICO.NS',
    'MARUTI.NS','MFSL.NS','MOTHERSON.NS','MRF.NS','MUTHOOTFIN.NS','NESTLEIND.NS',
    'NMDC.NS','NTPC.NS','ONGC.NS','PGINFRA.NS','PIDILITIND.NS','PNB.NS',
    'POWERGRID.NS','RELIANCE.NS','RVNL.NS','SAMVARDHANA.NS','SBICARD.NS',
    'SBILIFE.NS','SBIN.NS','SHREECEM.NS','SIEMENS.NS','SRF.NS','SUNPHARMA.NS',
    'TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS','TECHM.NS','TITAN.NS',
    'TRENT.NS','TVSMOTOR.NS','UBL.NS','ULTRACEMCO.NS','UPL.NS','VEDL.NS',
    'WIPRO.NS','ZEEL.NS','ETERNAL.NS',
    'AUBANK.NS','ASHOKLEY.NS','BEL.NS','BHEL.NS','CESC.NS','CENTURYTEX.NS',
    'CROMPTON.NS','DEEPAKNTR.NS','EXIDEIND.NS','FEDERALBNK.NS','GMRINFRA.NS',
    'GSPL.NS','HAL.NS','INDHOTEL.NS','INDIACEM.NS','IOB.NS','IRCON.NS','IRFC.NS',
    'ITDC.NS','JKCEMENT.NS','JMFINANCIL.NS','JYOTHYLAB.NS','KAJARIACER.NS',
    'KANSAINER.NS','KARURVYSYA.NS','KEC.NS','NHPC.NS','NLCINDIA.NS','OIL.NS',
    'PATANJALI.NS','PFC.NS','PFIZER.NS','POLYCAB.NS','RADICO.NS','RAJESHEXPO.NS',
    'RBLBANK.NS','SAIL.NS','SJVN.NS','SPARC.NS','SPICEJET.NS','SUNTV.NS',
    'TATACOMM.NS','TATAPOWER.NS','THOMASCOOK.NS','TV18BRDCST.NS','UNIONBANK.NS',
    'VGUARD.NS','VOLTAS.NS','YESBANK.NS','ZYDUSLIFE.NS'
]

START_DATE = "2022-01-01"
END_DATE = str(pd.to_datetime('today').date())
SENSITIVITY = 0.1
OUTPUT_DIR = "data/denoised_universe"

# =======================================================
#   HELPER FUNCTIONS (DB4 DWT)
# =======================================================
def print_heading(title):
    print("\n" + "="*90)
    print(f"  {title.upper()}")
    print("="*90)

def dwt_db4(signal):
    cA, cD = pywt.dwt(signal, 'db4')
    return cA, cD

def idwt_db4(cA, cD):
    return pywt.idwt(cA, cD, 'db4')

# =======================================================
#      MAIN PREPROCESSING WORKFLOW (ALL STOCKS)
# =======================================================
def run_all_preprocessing():
    print_heading("RUNNING DWT DENOISING FOR ALL 150 STOCKS (DB4 RETURNS-BASED)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  - Using sensitivity: {SENSITIVITY}, Start date: {START_DATE}")
    print(f"  - Output folder: {OUTPUT_DIR}\n")

    for i, ticker in enumerate(STOCK_UNIVERSE):
        try:
            print(f"[{i+1}/{len(STOCK_UNIVERSE)}] Downloading '{ticker}' ...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if df.empty or 'Close' not in df.columns or len(df) < 10:
                print(f"    ⚠️ Skipping '{ticker}' — insufficient data.")
                continue

            price_signal = df['Close'].to_numpy(dtype=float).flatten()
            return_signal = df['Close'].pct_change().fillna(0).to_numpy(dtype=float).flatten()

            # --- DWT on Returns ---
            cA, cD = dwt_db4(return_signal)
            sigma = np.median(np.abs(cD)) / 0.6745 if np.median(np.abs(cD)) > 0 else 0
            threshold = sigma * SENSITIVITY
            cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD) - threshold)
            denoised_return_signal = idwt_db4(cA, cD_clean)

            # --- Fix Lengths ---
            if len(denoised_return_signal) < len(return_signal):
                denoised_return_signal = np.pad(
                    denoised_return_signal,
                    (0, len(return_signal) - len(denoised_return_signal)),
                    'edge'
                )
            denoised_return_signal = denoised_return_signal[:len(price_signal)]

            # --- Reconstruct Denoised Price ---
            denoised_price_signal = np.zeros_like(price_signal)
            denoised_price_signal[0] = price_signal[0]
            for t in range(1, len(price_signal)):
                denoised_price_signal[t] = denoised_price_signal[t-1] * (1 + denoised_return_signal[t])

            # --- Save to File ---
            denoised_df = pd.DataFrame({'Date': df.index, 'Close': denoised_price_signal})
            safe_ticker = ticker.replace('.', '_')
            out_path = os.path.join(OUTPUT_DIR, f"{safe_ticker}.csv")
            denoised_df.to_csv(out_path, index=False)
            print(f"    ✅ Saved → {out_path}")

        except Exception as e:
            print(f"    ❌ ERROR processing {ticker}: {e}")
        time.sleep(1)  # slight delay to avoid API throttling

    print_heading("🎯 ALL 150 STOCKS DENOISED SUCCESSFULLY")

# =======================================================
# ENTRY POINT
# =======================================================
if __name__ == "__main__":
    run_all_preprocessing()
