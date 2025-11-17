# import numpy as np
# import pandas as pd
# import argparse
# import yfinance as yf
# import matplotlib.pyplot as plt
# import time
# import os

# def correlate_from_scratch(signal, kernel, kernel_name):
#     """
#     1D correlation manually.
#     DWT analysis filters (h0, h1) are used as-is (no flipping).
#     """
#     print(f"\n    [Executing Manual Correlation for '{kernel_name}']")
#     time.sleep(1) 

#     kernel_size = len(kernel)
#     signal_size = len(signal)
#     output_size = signal_size - kernel_size + 1
#     output = np.zeros(output_size)
   
#     for i in range(output_size):
#         window = signal[i : i + kernel_size]
#         multiplied_values = window * kernel 
#         output[i] = np.sum(multiplied_values)
        
#         if i < 2:
#             print(f"\n    - Correlation Step {i+1}:")
#             print(f"      Signal Window : {np.round(window, 4)}")
#             print(f"      Kernel        : {np.round(kernel, 2)}") # Not reversed
#             print(f"      Multiplication: {np.round(multiplied_values, 4)}")
            
#             formula_str = " + ".join([f"({val:.4f})" for val in multiplied_values])
#             print(f"      Sum           : {formula_str} = {output[i]:.4f}")
#             time.sleep(2) 
            
#     return output

# def dwt_haar_manual(signal):
#     """
#     one level of Haar DWT using our manual correlation function.
#     """
#     low_pass_filter = np.array([1, 1]) / np.sqrt(2)
#     high_pass_filter = np.array([-1, 1]) / np.sqrt(2)
#     cA_full = correlate_from_scratch(signal, low_pass_filter, "Trend (Low-Pass)")
#     cD_full = correlate_from_scratch(signal, high_pass_filter, "Noise (High-Pass)")
    
#     print("\n    [Performing Downsampling]")
#     print(f"      cA length before downsampling: {len(cA_full)}")
#     print(f"      cD length before downsampling: {len(cD_full)}")
#     cA = cA_full[::2]
#     cD = cD_full[::2]
#     print(f"      cA length after downsampling: {len(cA)}")
#     print(f"      cD length after downsampling: {len(cD)}")
#     time.sleep(1)
    
#     return cA, cD

# def idwt_haar_manual(cA, cD):
#     min_length = min(len(cA), len(cD))
#     cA, cD = cA[:min_length], cD[:min_length]
#     even_points = (cA + cD) / np.sqrt(2)
#     odd_points  = (cA - cD) / np.sqrt(2)
#     reconstructed_signal = np.zeros(2 * min_length)
#     reconstructed_signal[0::2] = even_points
#     reconstructed_signal[1::2] = odd_points
#     return reconstructed_signal

# # ----------------------------------------------------------------------
# # PART 2: FULL DENOISING PIPELINE WITH VERBOSE OUTPUT (FIXED)
# # ----------------------------------------------------------------------

# def denoise_stock_signal_verbose(ticker_symbol, sensitivity, period):
#     print("="*90)
#     print(f" STEP 1: DOWNLOADING DATA FOR {ticker_symbol} (Period: {period})")
#     print("="*90)
#     time.sleep(1)

#     data = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)
#     data.dropna(inplace=True)

#     if data.empty or len(data) < 10:
#         print(f" No data found for {ticker_symbol}.")
#         return

#     price_signal = data['Close'].to_numpy().flatten()
#     print(f"  - Total data points downloaded: {len(price_signal)}")
#     time.sleep(1)

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 1.5: CALCULATING DAILY RETURNS (pct_change)")
#     print("="*90)
#     print("  - DWT was failing on 'price' (e.g., 3000).")
#     print("  - We will now run the DWT on 'returns' (e.g., 0.01).")
#     time.sleep(1)
    
#     return_signal = pd.Series(price_signal).pct_change().fillna(0).to_numpy()
#     print(f"  - Calculated daily returns. Example: Price {price_signal[0]:.2f} -> {price_signal[1]:.2f} = {return_signal[1]:.4f} ({(return_signal[1]*100):.2f}%)")
#     time.sleep(2)
    
#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 2: HAAR DWT DECOMPOSITION (ON RETURNS)")
#     print("="*90)
#     print("  - Running DWT on RETURN signal...")
#     time.sleep(1)

#     cA, cD = dwt_haar_manual(return_signal)
    
#     print("\n  - DWT process finished.")
#     print("\n Example Calculation for First 2 Coefficients (Recap):")
#     for i in range(min(2, len(return_signal)//2)):
#         r0, r1 = return_signal[2*i], return_signal[2*i+1]
#         print(f"  Pair {i}: Return Signal Values = [{r0:.4f}, {r1:.4f}]")
#         print(f"    cA[{i}] (Trend) = ({r0:.4f} + {r1:.4f}) / 1.414 = {cA[i]:.4f}")
#         print(f"    cD[{i}] (Noise) = (-{r0:.4f} + {r1:.4f}) / 1.414 = {cD[i]:.4f}")
#     time.sleep(2)

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 3: THRESHOLD ESTIMATION (ON RETURN NOISE)")
#     print("="*90)
#     print("  - Calculating threshold on RETURN noise coefficients (cD)...")
#     time.sleep(1)
    
#     median_absolute_deviation = np.median(np.abs(cD))
#     sigma = median_absolute_deviation / 0.6745 if median_absolute_deviation > 0 else 0
#     threshold = sigma * sensitivity
#     print("  - Threshold Calculation :")
#     print(f"    1. Median of absolute noise (|cD|): {median_absolute_deviation:.4f}")
#     print(f"    2. Estimated Noise Level (σ): {median_absolute_deviation:.4f} / 0.6745 = {sigma:.4f}")
#     print(f"    3. Final Threshold (λ) with Sensitivity {sensitivity}: {sigma:.4f} * {sensitivity} = {threshold:.4f}")

#     cD_cleaned = np.sign(cD) * np.maximum(0, np.abs(cD) - threshold)
#     removed_count = np.sum(np.abs(cD) > np.abs(cD_cleaned)) 
#     print(f"\n  - Applied 'Soft Thresholding'. {removed_count} coefficients were 'shrunk'.")
#     time.sleep(2)

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 4: INVERSE HAAR RECONSTRUCTION (DENOISED RETURNS)")
#     print("="*90)
#     time.sleep(1)

#     denoised_return_signal_full = idwt_haar_manual(cA, cD_cleaned)
#     print(f"  - Denoised 'Return' signal reconstructed. Length: {len(denoised_return_signal_full)}")
    
#     if len(denoised_return_signal_full) < len(return_signal):
#         denoised_return_signal = np.pad(denoised_return_signal_full, (0, len(return_signal)-len(denoised_return_signal_full)), 'edge')
#     else:
#         denoised_return_signal = denoised_return_signal_full[:len(return_signal)]

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print("STEP 4.5: RECONSTRUCTING PRICE FROM DENOISED RETURNS")
#     print("="*90)
#     time.sleep(1)
    
#     denoised_price_signal = np.zeros_like(price_signal)
#     denoised_price_signal[0] = price_signal[0] 
    
#     print(f"  - Reconstructing price day-by-day:")
#     print(f"    Day 0: Price = {denoised_price_signal[0]:.2f} (Starting Price)")
    
#     for t in range(1, len(price_signal)):
#         denoised_price_signal[t] = denoised_price_signal[t-1] * (1 + denoised_return_signal[t])
#         if t < 5: 
#             print(f"    Day {t}: Price = {denoised_price_signal[t-1]:.2f} * (1 + {denoised_return_signal[t]:.4f}) = {denoised_price_signal[t]:.2f}")
            
#     print("  - Final Denoised Price Signal is ready.")
#     time.sleep(2)
    
#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print("STEP 5: FINAL COMPARISON TABLE (LAST 7 DAYS)")
#     print("="*90)
#     time.sleep(1)

#     comp_df = pd.DataFrame({
#         'Date': data.index[-7:],
#         'Original_Price': np.round(price_signal[-7:], 2),
#         'Denoised_Price': np.round(denoised_price_signal[-7:], 2)
#     })
#     print(comp_df.to_string(index=False))
#     time.sleep(1)

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 6: PLOTTING ORIGINAL VS DENOISED SIGNAL")
#     print("="*90)
#     time.sleep(1)

#     plt.figure(figsize=(14, 9))
#     plt.plot(data.index, price_signal, label='Original (Noisy) Signal', color='green', alpha=0.5)
#     plt.plot(data.index, denoised_price_signal, label='Denoised (Clean) Signal', color='red', linewidth=2)
#     plt.title(f"{ticker_symbol} - Denoising Analysis (Returns-Based)", fontsize=16)
#     plt.xlabel("Date", fontsize=12)
#     plt.ylabel("Price", fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

#     # ------------------------------------------------------------
#     print("\n" + "="*90)
#     print(" STEP 7: SAVING DENOISED DATA")
#     print("="*90)

#     out_dir = "data/denoised_from_scratch"
#     os.makedirs(out_dir, exist_ok=True)
#     clean_name = ticker_symbol.replace('.NS','')
#     out_path = f"{out_dir}/{clean_name}_denoised.csv"

#     pd.DataFrame({'Date': data.index, 'Denoised_Close': denoised_price_signal}).to_csv(out_path, index=False)
#     print(f" Denoised data saved to: {out_path}")
#     print("\n PROCESS COMPLETE")
#     print("="*90)

# # ------------------------------------------------------------
# # CLI (Command Line Interface) Entry Point
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
#         description="""Manual Haar DWT denoising for stock prices.
    
# Example Usage:
#   python3 run_local_denoising_verbose.py --ticker TCS.NS --period 6mo
#   python3 run_local_denoising_verbose.py --ticker TITAN.NS --sensitivity 0.1 --period 1y
# """)
#     parser.add_argument('--ticker', type=str, required=True, help="Stock ticker (e.g., 'TCS.NS')")
#     parser.add_argument('--sensitivity', type=float, default=0.1, help="Denoising strength. Higher value = more smoothing.") # <-- FINAL FIX
#     parser.add_argument('--period', type=str, default='1y', help="Time period (e.g., '5y', '1y', '6mo', '1mo', '5d')")
#     args = parser.parse_args()

#     denoise_stock_signal_verbose(args.ticker, args.sensitivity, args.period)