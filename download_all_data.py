# download_all_data.py

import yfinance as yf
import os
import time

# =======================================================
#                  CONTROL PANEL
# =======================================================
# Yahan un sabhi stocks ke ticker symbol daalein jinka data aapko chahiye
# ".NS" National Stock Exchange ke liye hai
TICKER_LIST = [
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS',  # IT Sector
    'RELIANCE.NS', 'ONGC.NS', 'BPCL.NS',         # Energy Sector
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'      # Banking Sector
]

# Data kab se kab tak ka chahiye
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Data save karne ke liye folder
OUTPUT_DIR = "datasets"
# =======================================================

def download_stock_data():
    """
    Downloads historical stock data for a list of tickers using yfinance
    and saves each as a separate CSV file.
    """
    print("="*60)
    print(" DOWNLOADING HISTORICAL DATA FOR ALL STOCKS (USING YFINANCE) ")
    print("="*60)
    
    # Check if the output directory exists, if not, create it
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' created.")
        
    for ticker in TICKER_LIST:
        try:
            print(f"\nFetching data for {ticker}...")
            
            # Download data from Yahoo Finance
            stock_data = yf.download(ticker, start=START_DATE, end=END_DATE)
            
            if not stock_data.empty:
                # Save the data to a CSV file
                # We rename the file to remove '.NS' for simplicity (e.g., 'TCS.NS' -> 'TCS.csv')
                file_name = f"{ticker.replace('.NS', '')}.csv"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                
                stock_data.to_csv(file_path)
                print(f"✅ Data for {ticker} saved successfully to '{file_path}'")
            else:
                print(f"⚠️ Warning: No data returned for {ticker}. It might be delisted or the ticker is wrong.")
                
            # Wait for a moment to avoid sending too many requests too quickly
            time.sleep(1)
            
        except Exception as e:
            print(f" Error fetching data for {ticker}: {e}")

    print("\n" + "="*60)
    print("🎉 All data downloaded successfully!")
    print("="*60)

if __name__ == "__main__":
    download_stock_data()