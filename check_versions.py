# check_versions.py

import sys
import pandas as pd
import yfinance as yf

print("\n" + "="*40)
print("   PYTHON ENVIRONMENT CHECK")
print("="*40)
print(f"Python Version: {sys.version}")
print(f"Pandas Version: {pd.__version__}")
print(f"yfinance Version: {yf.__version__}")
print("="*40 + "\n")