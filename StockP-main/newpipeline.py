import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pywt
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# DIRECTORIES
# -----------------------------------------------------------
DENOISED_DIR = "data/denoised_universe"
MODEL_READY_DIR = "data/model_ready"

os.makedirs(DENOISED_DIR, exist_ok=True)
os.makedirs(MODEL_READY_DIR, exist_ok=True)

# -----------------------------------------------------------
# STOCK UNIVERSE
# -----------------------------------------------------------
STOCK_UNIVERSE = [
    'ACC.NS','ADANIENT.NS','ADANIGREEN.NS','ADANIPORTS.NS','ADANIPOWER.NS',
    'AMBUJACEM.NS','APOLLOHOSP.NS','ASIANPAINT.NS','AUROPHARMA.NS','AXISBANK.NS',
    'BAJAJFINANCE.NS','BAJAJFINSV.NS','BAJAJHLDNG.NS','BANDHANBNK.NS','BANKBARODA.NS',
    'BHARATFORG.NS','BHARTIARTL.NS','BOSCHLTD.NS','BPCL.NS','BRITANNIA.NS',
    'CANBK.NS','CEATLTD.NS','CHOLAFIN.NS','CIPLA.NS','COALINDIA.NS',
    'COLPAL.NS','DABUR.NS','DIVISLAB.NS','DLF.NS','DRREDDY.NS','EICHERMOT.NS',
    'GAIL.NS','GODREJCP.NS','GRASIM.NS','HAVELLS.NS','HCLTECH.NS','HDFCAMC.NS',
    'HDFCBANK.NS','HDFCLIFE.NS','HEROMOTOCO.NS','HINDALCO.NS','HINDUNILVR.NS',
    'ICICIBANK.NS','ICICIGI.NS','ICICIPRULI.NS','IDBI.NS','IFCI.NS','INDIGO.NS',
    'INDUSINDBK.NS','INFY.NS','IOC.NS','ITC.NS','JINDALSTEL.NS','JSWSTEEL.NS',
    'KOTAKBANK.NS','LICI.NS','LTIM.NS','LT.NS','LUPIN.NS','M&M.NS',
    'MARICO.NS','MARUTI.NS','MOTHERSON.NS','MUTHOOTFIN.NS','NESTLEIND.NS',
    'NMDC.NS','NTPC.NS','ONGC.NS','PIDILITIND.NS','PNB.NS','POWERGRID.NS',
    'RELIANCE.NS','RVNL.NS','SBICARD.NS','SBILIFE.NS','SBIN.NS','SIEMENS.NS',
    'SUNPHARMA.NS','TATACONSUM.NS','TATAMOTORS.NS','TATASTEEL.NS','TCS.NS',
    'TECHM.NS','TITAN.NS','TRENT.NS','TVSMOTOR.NS','UBL.NS','ULTRACEMCO.NS',
    'UPL.NS','VEDL.NS','WIPRO.NS','YESBANK.NS','ZYDUSLIFE.NS'
]

# -----------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = str(pd.to_datetime('today').date())
SENSITIVITY = 0.12     # wavelet threshold control
TOP_K = 5              # adjacency neighbors

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def heading(msg):
    print("\n" + "="*80)
    print(msg)
    print("="*80)

def dwt_haar(signal):
    return pywt.dwt(signal, "haar")

def idwt_haar(cA, cD):
    return pywt.idwt(cA, cD, "haar")

def manual_corr(x, y):
    n = len(x)
    mx = sum(x)/n
    my = sum(y)/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx = sum((x[i]-mx)**2 for i in range(n))
    dy = sum((y[i]-my)**2 for i in range(n))
    return 0 if dx*dy == 0 else num / ((dx*dy)**0.5)

# -----------------------------------------------------------
# STEP 1 — DOWNLOAD & DENOISE ALL STOCKS (Log-return DWT)
# -----------------------------------------------------------
heading("STEP 1 — Downloading & Denoising All Stocks (Log-returns)")

all_clean_prices = {}

for i, ticker in enumerate(STOCK_UNIVERSE):
    try:
        print(f"[{i+1}/{len(STOCK_UNIVERSE)}] {ticker} downloading...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

        if df.empty:
            print("  SKIPPED — no data")
            continue

        price = df["Close"].values
        logret = np.log1p(pd.Series(price).pct_change().fillna(0).values)

        # Haar DWT
        cA, cD = dwt_haar(logret)

        med = np.median(np.abs(cD))
        sigma = med/0.6745
        thr = sigma * SENSITIVITY

        cD_clean = np.sign(cD) * np.maximum(0, np.abs(cD)-thr)
        clean_logret = idwt_haar(cA, cD_clean)

        if len(clean_logret) < len(price):
            clean_logret = np.pad(clean_logret,(0,len(price)-len(clean_logret)),'edge')

        clean_price = np.zeros_like(price)
        clean_price[0] = price[0]
        for t in range(1,len(price)):
            clean_price[t] = clean_price[t-1] * np.exp(clean_logret[t])

        all_clean_prices[ticker] = clean_price

        pd.DataFrame({"Date": df.index, "Close": clean_price}).to_csv(
            f"{DENOISED_DIR}/{ticker.replace('.','_')}.csv", index=False)

        print("  ✔ DONE")

    except Exception as e:
        print("  ERROR:", e)

# -----------------------------------------------------------
# STEP 2 — MERGE ALL INTO ONE MASTER CSV
# -----------------------------------------------------------
heading("STEP 2 — Merging All Stocks into Master File")

merged = None
for ticker, data in all_clean_prices.items():
    df = pd.DataFrame(data, columns=[ticker])
    if merged is None:
        merged = df
    else:
        merged = pd.concat([merged, df], axis=1)

merged.to_csv(f"{MODEL_READY_DIR}/combined_denoised_prices.csv", index=False)
print("Saved → model_ready/combined_denoised_prices.csv")

# -----------------------------------------------------------
# STEP 3 — CORRELATION MATRIX (Manual Pearson)
# -----------------------------------------------------------
heading("STEP 3 — Building Correlation Matrix (Manual)")

cols = merged.columns.tolist()
n = len(cols)
corr_matrix = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        corr_matrix[i,j] = manual_corr(merged[cols[i]].values, merged[cols[j]].values)
    corr_matrix[i,i] = 1.0

pd.DataFrame(corr_matrix, index=cols, columns=cols).to_csv(
    f"{MODEL_READY_DIR}/correlation_matrix.csv")

# -----------------------------------------------------------
# STEP 4 — TOP-K ADJACENCY MATRIX
# -----------------------------------------------------------
heading("STEP 4 — Building Global Adjacency Matrix (Top-k)")

adj = np.zeros_like(corr_matrix, dtype=int)

for i in range(n):
    idx = np.argsort(-corr_matrix[i])  # descending
    top = [j for j in idx if j!=i][:TOP_K]
    adj[i, top] = 1

adj = np.maximum(adj, adj.T)
np.fill_diagonal(adj, 0)

np.save(f"{MODEL_READY_DIR}/adjacency_matrix.npy", adj)

print("Saved → adjacency_matrix.npy")
print("TOTAL EDGES:", int(adj.sum()/2))
heading("GLOBAL PREPROCESSING COMPLETE ✔")
