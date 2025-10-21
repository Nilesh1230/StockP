# prepare_model_data_proof.py (Har Kadam ke Proof ke Saath)

import os
import pandas as pd
import numpy as np
import argparse
import time

def prepare_data_with_proof(input_dir, tickers, threshold):
    """
    Model ke liye data taiyar karta hai aur har kadam ka proof dikhata hai.
    """
    print("="*80)
    print(" STEP 1: MODEL KE LIYE DATA TAIYAR KARNA (PROOF KE SAATH) ")
    print("="*80)
    
    # --- Task 1: Sabhi Denoised Data ko Jodna ---
    print("\nTASK 1: Sabhi denoised files ko ek master table mein jodna.")
    print("------------------------------------------------------------------")
    print("Maqsad: Humein ek aisi table chahiye jismein har stock ka clean price ek saath ho.")
    time.sleep(2)
    
    merged_df = None
    
    for ticker in tickers:
        print(f"\n--- {ticker} ki file process ho rahi hai ---")
        file_path = os.path.join(input_dir, f"{ticker}_denoised.csv")
        try:
            df = pd.read_csv(file_path)
            print(f"  - Saboot 1: {ticker} ki file safaltapoorvak load ho gayi.")
            print(f"    File mein {df.shape[0]} rows aur {df.shape[1]} columns hain.")
            print("    Iske shuruaat ke 3 data points yeh hain:")
            print(df.head(3).to_string())
            time.sleep(2)

            df = df[['Date', 'Close']].rename(columns={'Close': ticker})
            
            if merged_df is None:
                merged_df = df
                print(f"  - Saboot 2: Master table banayi gayi aur usmein {ticker} ka data daala gaya.")
            else:
                # Merge karne se pehle, table ka size note karein
                old_shape = merged_df.shape
                merged_df = pd.merge(merged_df, df, on='Date', how='outer')
                print(f"  - Saboot 2: {ticker} ka data master table mein joda gaya.")
                print(f"    Table ka size {old_shape} se badhkar {merged_df.shape} ho gaya hai.")

            print("\n    Ab master table aisi dikh rahi hai:")
            print(merged_df.head(3).to_string())
            time.sleep(3)

        except FileNotFoundError:
            print(f"  - ⚠️ WARNING: {ticker} ki file nahi mili: '{file_path}'. Skip kar rahe hain.")
            continue
            
    if merged_df is None:
        print("\n❌ FATAL ERROR: Ek bhi denoised file nahi mili.")
        return

    merged_df.dropna(inplace=True)
    merged_df.set_index('Date', inplace=True)
    
    output_folder = 'data/model_ready'
    os.makedirs(output_folder, exist_ok=True)
    combined_data_path = os.path.join(output_folder, 'combined_denoised_prices.csv')
    merged_df.to_csv(combined_data_path)
    print(f"\n✅ Task 1 Poora Hua: Final combined data is file mein save ho gaya hai: '{combined_data_path}'.")
    print("Final table ke shuruaat ke 5 rows:")
    print(merged_df.head(5).to_string())

    # ... (Task 2 aage jaari rahega) ...
    # (Task 2 ka code yahan paste karein)

if __name__ == "__main__":
    TICKER_LIST = [
        'TCS', 'INFY', 'WIPRO', 'HCLTECH',
        'RELIANCE', 'ONGC', 'BPCL', 'GAIL',
        'HDFCBANK', 'ICICIBANK', 'SBIN'
    ]
    parser = argparse.ArgumentParser(description="GAT+LSTM model ke liye data taiyar karein.")
    parser.add_argument('--input_dir', type=str, default='data/denoised_from_scratch', help="Denoised CSV files wala folder.")
    parser.add_argument('--tickers', nargs='+', default=TICKER_LIST, help="Process karne wale stock tickers ki list.")
    parser.add_argument('--threshold', type=float, default=0.7, help="Graph connection banane ke liye correlation threshold.")
    args = parser.parse_args()
    
    prepare_data_with_proof(args.input_dir, args.tickers, args.threshold)