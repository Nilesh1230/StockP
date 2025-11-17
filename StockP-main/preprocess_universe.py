#!/usr/bin/env python3
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 150 STOCK UNIVERSE
# ============================================================
STOCK_UNIVERSE = [
    "ACC.NS","ADANIENT.NS","ADANIGREEN.NS","ADANIPORTS.NS","ADANIPOWER.NS",
    "AMBUJACEM.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AUROPHARMA.NS","AXISBANK.NS",
    "BAJAJFINANCE.NS","BAJAJFINSV.NS","BAJAJHLDNG.NS","BANDHANBNK.NS","BANKBARODA.NS",
    "BHARATFORG.NS","BHARTIARTL.NS","BOSCHLTD.NS","BPCL.NS","BRITANNIA.NS",
    "BSE.NS","CANBK.NS","CEATLTD.NS","CHOLAFIN.NS","CIPLA.NS","COALINDIA.NS",
    "COLPAL.NS","DABUR.NS","DIVISLAB.NS","DLF.NS","DRREDDY.NS","EICHERMOT.NS",
    "GAIL.NS","GODREJCP.NS","GRASIM.NS","HAVELLS.NS","HCLTECH.NS","HDFCAMC.NS",
    "HDFCBANK.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","ICICIGI.NS","ICICIPRULI.NS","IDBI.NS","IFCI.NS","INDIGO.NS",
    "INDUSINDBK.NS","INFY.NS","IOC.NS","ITC.NS","JINDALSTEL.NS","JSWSTEEL.NS",
    "KOTAKBANK.NS","LICI.NS","LTIM.NS","LT.NS","LUPIN.NS","M&M.NS",
    "MARICO.NS","MARUTI.NS","MFSL.NS","MOTHERSON.NS","MRF.NS","MUTHOOTFIN.NS",
    "NESTLEIND.NS","NMDC.NS","NTPC.NS","ONGC.NS","PGINFRA.NS","PIDILITIND.NS",
    "PNB.NS","POWERGRID.NS","RELIANCE.NS","RVNL.NS","SAMVARDHANA.NS","SBICARD.NS",
    "SBILIFE.NS","SBIN.NS","SHREECEM.NS","SIEMENS.NS","SRF.NS","SUNPHARMA.NS",
    "TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS",
    "TRENT.NS","TVSMOTOR.NS","UBL.NS","ULTRACEMCO.NS","UPL.NS","VEDL.NS",
    "WIPRO.NS","ZEEL.NS","ETERNAL.NS","AUBANK.NS","ASHOKLEY.NS","BEL.NS",
    "BHEL.NS","CESC.NS","CENTURYTEX.NS","CROMPTON.NS","DEEPAKNTR.NS","EXIDEIND.NS",
    "FEDERALBNK.NS","GMRINFRA.NS","GSPL.NS","HAL.NS","INDHOTEL.NS","INDIACEM.NS",

    # ---- VALIDATION (15 Stocks) ----
    "IOB.NS","IRCON.NS","IRFC.NS","ITDC.NS","JKCEMENT.NS",
    "JMFINANCIL.NS","JYOTHYLAB.NS","KAJARIACER.NS","KANSAINER.NS","KARURVYSYA.NS",
    "KEC.NS","NHPC.NS","NLCINDIA.NS","OIL.NS","PFC.NS",

    # ---- TEST (15 Stocks) ----
    "PFIZER.NS","POLYCAB.NS","RADICO.NS","RAJESHEXPO.NS","RBLBANK.NS",
    "SAIL.NS","SJVN.NS","SPARC.NS","SPICEJET.NS","SUNTV.NS",
    "TATACOMM.NS","TATAPOWER.NS","THOMASCOOK.NS","VGUARD.NS","VOLTAS.NS"
]



# ============================================================
# SPLIT INTO TRAIN (120) / VAL (15) / TEST (15)
# ============================================================
TRAIN_STOCKS = STOCK_UNIVERSE[:120]
VAL_STOCKS   = STOCK_UNIVERSE[120:135]
TEST_STOCKS  = STOCK_UNIVERSE[135:]

os.makedirs("data/model_ready", exist_ok=True)
np.save("data/model_ready/train_stocks.npy", TRAIN_STOCKS)
np.save("data/model_ready/val_stocks.npy", VAL_STOCKS)
np.save("data/model_ready/test_stocks.npy", TEST_STOCKS)

# ============================================================
# CONFIG
# ============================================================
START_DATE = "2022-01-01"
END_DATE = str(pd.to_datetime("today").date())
SENSITIVITY = 0.1

CSV_DIR = "data/denoised_universe"
PLOT_DIR = "data/denoised_universe_plots"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def print_heading(msg):
    print("\n" + "="*100)
    print("   " + msg)
    print("="*100)

def dwt_haar(x): return pywt.dwt(x, "haar")
def idwt_haar(cA, cD): return pywt.idwt(cA, cD, "haar")

def download_with_retry(ticker, retries=3):
    for _ in range(retries):
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True, timeout=10)
            if df is not None and not df.empty:
                return df
        except:
            time.sleep(1)
    return None

# ============================================================
# MAIN PROCESS
# ============================================================
def preprocess_list(stock_list, tag):

    print_heading(f"PREPROCESSING {tag} ({len(stock_list)} stocks)")

    for i, ticker in enumerate(stock_list):
        print(f"[{i+1}/{len(stock_list)}] {ticker}")

        df = download_with_retry(ticker)
        if df is None or df.empty:
            print("    Skipped: No data.")
            continue

        price = np.array(df["Close"], dtype=float).ravel()

        if len(price) < 10:
            print("    Too little data.")
            continue

        # Log returns
        logret = np.log(price[1:] / price[:-1] + 1e-12)
        logret = np.insert(logret, 0, 0.0)

        # DWT
        cA, cD = dwt_haar(logret)
        sigma = np.median(np.abs(cD)) / 0.6745
        thr = sigma * SENSITIVITY
        cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD) - thr)

        den_lr = idwt_haar(cA, cD_clean)
        den_lr = np.array(den_lr, dtype=float).ravel()

        if len(den_lr) < len(logret):
            den_lr = np.pad(den_lr, (0, len(logret)-len(den_lr)), "edge")
        den_lr = den_lr[:len(logret)]

        # Reconstruct price
        den_price = np.zeros_like(price)
        den_price[0] = price[0]
        for t in range(1, len(price)):
            den_price[t] = den_price[t-1] * np.exp(den_lr[t])

        # Force 1D arrays
        dates = np.array(df.index.strftime("%Y-%m-%d")).ravel()
        den_price = np.array(den_price).ravel()

        if len(dates) != len(den_price):
            print("    Length mismatch. Skipping.")
            continue

        # Save CSV
        safe = ticker.replace(".", "_")
        out_csv = os.path.join(CSV_DIR, f"{tag}_{safe}.csv")

        pd.DataFrame({"Date": dates, "Close": den_price}).to_csv(out_csv, index=False)

        print("   ✔ Saved CSV:", out_csv)

        # Plot
        plt.figure(figsize=(10,4))
        plt.plot(df.index, price, label="Original", alpha=0.5)
        plt.plot(df.index, den_price, label="Denoised", linewidth=2)
        plt.title(f"{ticker} — {tag}")
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, f"{tag}_{safe}.png")
        plt.savefig(plot_path)
        plt.close()

        print("   ✔ Saved Plot:", plot_path)

        time.sleep(0.2)

    print_heading(f"✓ COMPLETED {tag}")


# ============================================================
# RUN ALL
# ============================================================
if __name__ == "__main__":
    preprocess_list(TRAIN_STOCKS, "TRAIN")
    preprocess_list(VAL_STOCKS, "VAL")
    preprocess_list(TEST_STOCKS, "TEST")
