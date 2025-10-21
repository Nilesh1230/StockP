# check_shape.py

import yfinance as yf
import numpy as np

TICKER = 'TCS.NS'

print(f"--- Running Diagnostic Test for {TICKER} ---")

try:
    # Download data
    data = yf.download(TICKER, period="1y", progress=False)
    data.dropna(inplace=True)

    # Convert 'Close' column to a NumPy array
    signal_array = data['Close'].to_numpy()

    # --- Print the results ---
    print(f"\nType of the array is: {type(signal_array)}")
    print(f"Shape of the array is: {signal_array.shape}")
    print(f"Number of dimensions: {signal_array.ndim}")

    if signal_array.ndim == 1:
        print("\n✅ SUCCESS: The data is in the correct 1-Dimensional format.")
    else:
        print("\n❌ FAILURE: The data is NOT 1-Dimensional. This is the problem.")

except Exception as e:
    print(f"An error occurred: {e}")

print("\n--- Test Complete ---")