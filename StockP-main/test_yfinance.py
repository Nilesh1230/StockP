# test_yfinance.py (No API Key needed)

import yfinance as yf
import pandas as pd

# --- CONFIGURATION ---
TICKER_SYMBOL = 'BPCL.NS'  # Yahoo Finance .NS format ka istemal karta hai
# --- END CONFIGURATION ---

try:
    print(f"Yahoo Finance se data fetch kiya ja raha hai (bina API key ke)...")
    
    # Create a ticker object
    tcs_ticker = yf.Ticker(TICKER_SYMBOL)
    
    # Fetch historical data
    # period="1mo" ka matlab hai pichle 1 mahine ka data
    data = tcs_ticker.history(period="1mo")
    
    if not data.empty:
        print(f"\n✅ Successfully fetched DAILY data for {TICKER_SYMBOL}:")
        print("Yeh pichle kuch dino ka data hai (sabse purana sabse upar):")
        # .tail() se sabse recent 5 entries dikhengi
        print(data.tail())
    else:
        print(f"❌ Error: {TICKER_SYMBOL} ke liye koi data nahi mila. Kya ticker symbol sahi hai?")

except Exception as e:
    print(f"❌ Script Error: {e}")