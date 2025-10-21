# ================================================================
# FILE: run_local_denoising_verbose_final.py
# AIM : Manual DWT-based Denoising (Super-Detailed for Final Year Project)
# AUTHOR: Nilesh Kumar
# NOTE : Haar Wavelet and Convolution are implemented from scratch.
# ================================================================

import numpy as np
import pandas as pd
import argparse
import yfinance as yf
import matplotlib.pyplot as plt
import time
import os

# ----------------------------------------------------------------------
# PART 1: CORE MATHEMATICAL BUILDING BLOCKS (FROM SCRATCH)
# ----------------------------------------------------------------------

def convolve_from_scratch(signal, kernel, kernel_name):
    """
    This function performs 1D convolution manually, showing every step.
    This is the most fundamental operation in DWT.
    """
    print(f"\n    [Executing Manual Convolution for '{kernel_name}']")
    time.sleep(1)

    kernel_size = len(kernel)
    signal_size = len(signal)
    output_size = signal_size - kernel_size + 1
    output = np.zeros(output_size)
    
    # For convolution, the filter (kernel) is applied in reverse.
    reversed_kernel = kernel[::-1]
    
    # Loop through the signal to apply the kernel at each position.
    for i in range(output_size):
        # Take a small piece of the signal (a "window").
        window = signal[i : i + kernel_size]
        
        # Element-wise multiplication between the signal window and the reversed kernel.
        multiplied_values = window * reversed_kernel
        
        # Sum the results of the multiplication to get the output value for this position.
        output[i] = np.sum(multiplied_values)
        
        # For the first 2 steps, print the entire calculation in detail.
        if i < 2:
            print(f"\n    - Convolution Step {i+1}:")
            print(f"      Signal Window : {np.round(window, 2)}")
            print(f"      Kernel (rev)  : {np.round(reversed_kernel, 2)}")
            print(f"      Multiplication: {np.round(multiplied_values, 2)}")
            
            formula_str = " + ".join([f"({val:.2f})" for val in multiplied_values])
            print(f"      Sum           : {formula_str} = {output[i]:.2f}")
            time.sleep(2)
            
    return output

def dwt_haar_manual(signal):
    """
    Performs one level of Haar DWT using our manual convolution function.
    """
    # Define the Haar Wavelet filters.
    # Low-pass filter captures the "trend" or "approximation".
    low_pass_filter = np.array([1, 1]) / np.sqrt(2)
    # High-pass filter captures the "details" or "noise".
    high_pass_filter = np.array([-1, 1]) / np.sqrt(2)

    # Calculate Approximation Coefficients (cA) using the low-pass filter.
    cA_full = convolve_from_scratch(signal, low_pass_filter, "Trend (Low-Pass)")
    
    # Calculate Detail Coefficients (cD) using the high-pass filter.
    cD_full = convolve_from_scratch(signal, high_pass_filter, "Noise (High-Pass)")
    
    # Downsampling: Keep only every second element to reduce data size.
    print("\n    [Performing Downsampling]")
    print(f"      cA length before downsampling: {len(cA_full)}")
    print(f"      cD length before downsampling: {len(cD_full)}")
    cA = cA_full[::2]
    cD = cD_full[::2]
    print(f"      cA length after downsampling: {len(cA)}")
    print(f"      cD length after downsampling: {len(cD)}")
    time.sleep(1)
    
    return cA, cD

def idwt_haar_manual(cA, cD):
    """
    Performs one level of Inverse Haar DWT to reconstruct the signal.
    """
    min_length = min(len(cA), len(cD))
    cA, cD = cA[:min_length], cD[:min_length]

    # Reconstruct the even and odd indexed points of the original signal.
    even_points = (cA + cD) / np.sqrt(2)
    odd_points  = (cA - cD) / np.sqrt(2)

    # Interleave the even and odd points to get the full signal back.
    reconstructed_signal = np.zeros(2 * min_length)
    reconstructed_signal[0::2] = even_points # Place even points at 0, 2, 4...
    reconstructed_signal[1::2] = odd_points  # Place odd points at 1, 3, 5...
    
    return reconstructed_signal

# ----------------------------------------------------------------------
# PART 2: FULL DENOISING PIPELINE WITH VERBOSE OUTPUT
# ----------------------------------------------------------------------

def denoise_stock_signal_verbose(ticker_symbol, sensitivity, period):
    print("="*90)
    print(f"📌 STEP 1: DOWNLOADING DATA FOR {ticker_symbol} (Period: {period})")
    print("="*90)
    time.sleep(1)

    data = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)
    data.dropna(inplace=True)

    if data.empty:
        print(f"❌ No data found for {ticker_symbol}. Please check the ticker symbol or your internet connection.")
        return

    signal = data['Close'].to_numpy().flatten()
    print(f"  - Total data points downloaded: {len(signal)}")
    time.sleep(1)

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📌 STEP 2: HAAR DWT DECOMPOSITION (MANUAL CALCULATION)")
    print("="*90)
    time.sleep(1)

    cA, cD = dwt_haar_manual(signal)
    
    print("\n  - DWT process finished.")
    print("\n🔸 Example Calculation for First 2 Coefficients (Recap):")
    for i in range(min(2, len(signal)//2)):
        x0, x1 = signal[2*i], signal[2*i+1]
        print(f"  Pair {i}: Signal Values = [{x0:.2f}, {x1:.2f}]")
        print(f"    cA[{i}] (Trend) = ({x0:.2f} + {x1:.2f}) / 1.414 = {cA[i]:.4f}")
        print(f"    cD[{i}] (Noise) = (-{x0:.2f} + {x1:.2f}) / 1.414 = {cD[i]:.4f}")
    time.sleep(2)

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📌 STEP 3: THRESHOLD ESTIMATION & APPLICATION (NOISE REMOVAL)")
    print("="*90)
    time.sleep(1)
    
    # Calculate the threshold value dynamically based on the noise level.
    median_absolute_deviation = np.median(np.abs(cD))
    sigma = median_absolute_deviation / 0.6745
    threshold = sigma * sensitivity
    print("  - Threshold Calculation Steps:")
    print(f"    1. Median of absolute noise (|cD|): {median_absolute_deviation:.4f}")
    print(f"    2. Estimated Noise Level (σ): {median_absolute_deviation:.4f} / 0.6745 = {sigma:.4f}")
    print(f"    3. Final Threshold (λ) with Sensitivity {sensitivity}: {sigma:.4f} * {sensitivity} = {threshold:.4f}")

    # Apply the threshold: if a noise value is smaller than the threshold, set it to 0.
    cD_cleaned = np.where(np.abs(cD) > threshold, cD, 0)
    removed_count = np.sum(cD != cD_cleaned)
    print(f"\n  - Action: {removed_count} noise coefficients were smaller than the threshold and have been set to 0.")
    
    # Show the thresholding rule in action for the first 5 coefficients.
    print("\n🔸 Example of Thresholding Rule:")
    for i in range(min(5, len(cD))):
        val = cD[i]
        decision = "KEPT" if np.abs(val) > threshold else "SET TO ZERO"
        print(f"  - cD[{i}] = {val:.4f}.  |{val:.4f}| > {threshold:.4f}?  Decision: {decision}")
    time.sleep(2)

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📌 STEP 4: INVERSE HAAR RECONSTRUCTION (MANUAL)")
    print("="*90)
    time.sleep(1)

    denoised_signal = idwt_haar_manual(cA, cD_cleaned)

    # Match the length of the reconstructed signal to the original signal.
    if len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal)-len(denoised_signal)), 'edge')
    elif len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]

    print(f"  - Signal reconstructed. Final length: {len(denoised_signal)}")
    
    print("\n🔸 Manual Reconstruction Check (First 2 Pairs):")
    for i in range(min(2, len(cA))):
        a, d_clean = cA[i], cD_cleaned[i]
        x_even = (a + d_clean) / np.sqrt(2)
        x_odd = (a - d_clean) / np.sqrt(2)
        print(f"  Pair {i}: cA={a:.4f}, cD_clean={d_clean:.4f} → Reconstructed Even={x_even:.4f}, Odd={x_odd:.4f}")
    time.sleep(2)

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📊 STEP 5: FINAL COMPARISON TABLE (LAST 7 DAYS)")
    print("="*90)
    time.sleep(1)

    comp_df = pd.DataFrame({
        'Date': data.index[-7:],
        'Original_Price': np.round(signal[-7:], 2),
        'Denoised_Price': np.round(denoised_signal[-7:], 2)
    })
    print(comp_df.to_string(index=False))
    time.sleep(1)

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📈 STEP 6: PLOTTING ORIGINAL VS DENOISED SIGNAL")
    print("="*90)
    time.sleep(1)

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, signal, label='Original (Noisy) Signal', color='blue', alpha=0.5)
    plt.plot(data.index, denoised_signal, label='Denoised (Clean) Signal', color='red', linewidth=2)
    plt.title(f"{ticker_symbol} - Denoising Analysis (From Scratch)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    print("\n" + "="*90)
    print("📁 STEP 7: SAVING DENOISED DATA")
    print("="*90)

    out_dir = "data/denoised_from_scratch"
    os.makedirs(out_dir, exist_ok=True)
    clean_name = ticker_symbol.replace('.NS','')
    out_path = f"{out_dir}/{clean_name}_denoised.csv"

    pd.DataFrame({'Date': data.index, 'Denoised_Close': denoised_signal}).to_csv(out_path, index=False)
    print(f"✅ Denoised data saved to: {out_path}")
    print("\n🎯 PROCESS COMPLETE")
    print("="*90)

# ------------------------------------------------------------
# CLI (Command Line Interface) Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description="""Manual Haar DWT denoising for stock prices with super-detailed mathematical explanations.
    
Example Usage:
  python3 run_local_denoising_verbose.py --ticker TCS.NS --period 6mo
  python3 run_local_denoising_verbose.py --ticker RELIANCE.NS --sensitivity 2.5
""")
    parser.add_argument('--ticker', type=str, required=True, help="Stock ticker (e.g., 'TCS.NS')")
    parser.add_argument('--sensitivity', type=float, default=3.0, help="Denoising strength. Higher value = more smoothing.")
    parser.add_argument('--period', type=str, default='1y', help="Time period (e.g., '5y', '1y', '6mo', '1mo', '5d')")
    args = parser.parse_args()

    denoise_stock_signal_verbose(args.ticker, args.sensitivity, args.period)