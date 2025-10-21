# run_batch_denoising.py (Corrected with More Aggressive Cleaning)

import numpy as np
import pandas as pd
import os
import time

# =======================================================
#                  CONTROL PANEL
# =======================================================
SENSITIVITY = 3.0
INPUT_DIR = "datasets"
OUTPUT_DIR = "data/denoised_single_level"
# =======================================================

# =============================================================================
# HISSAA 1: Core DWT Logic (No changes here)
# =============================================================================

def convolve_from_scratch(signal, kernel):
    kernel_size = len(kernel); output_size = len(signal) - kernel_size + 1
    output = np.zeros(output_size); reversed_kernel = kernel[::-1]
    for i in range(output_size):
        window = signal[i:i+kernel_size]
        output[i] = np.sum(window * reversed_kernel)
    return output

def dwt_from_scratch(signal):
    lp = np.array([1, 1]) / np.sqrt(2); hp = np.array([-1, 1]) / np.sqrt(2)
    cA = convolve_from_scratch(signal, lp)[::2]
    cD = convolve_from_scratch(signal, hp)[::2]
    return cA, cD

def idwt_from_scratch(cA, cD):
    lp = np.array([1, 1])/np.sqrt(2); hp = np.array([1, -1])/np.sqrt(2)
    cA_up = np.zeros(2*len(cA)); cA_up[::2] = cA
    cD_up = np.zeros(2*len(cD)); cD_up[::2] = cD
    recon_A = np.convolve(cA_up, lp, 'full')
    recon_D = np.convolve(cD_up, hp, 'full')
    min_len = min(len(recon_A), len(recon_D))
    return recon_A[:min_len] + recon_D[:min_len]

# =============================================================================
# HISSAA 2: Main Denoising Process (with Aggressive Cleaning)
# =============================================================================

def denoise_single_stock(file_path, sensitivity):
    
    ticker_symbol = os.path.basename(file_path).replace('.csv', '')
    print("\n" + "="*80)
    print(f" DENOISING STARTED FOR: {ticker_symbol} (Sensitivity: {sensitivity})")
    print("="*80)
    
    # Step 1: Load Data
    data = pd.read_csv(file_path)
    
    # === THIS IS THE NEW, MORE AGGRESSIVE CLEANING BLOCK ===
    # 1. Force the 'Close' column to be numeric. Any text or bad data becomes 'NaN'.
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # 2. Drop any rows where the 'Close' column has these NaN errors.
    initial_rows = len(data)
    data.dropna(subset=['Close'], inplace=True)
    cleaned_rows = len(data)
    # =========================================================
    
    original_signal = data['Close'].values
    
    print(f"📂 STEP 1: Data loaded and cleaned.")
    if initial_rows > cleaned_rows:
        print(f"   - Note: Removed {initial_rows - cleaned_rows} bad data rows.")
    print(f"   - Final signal length: {len(original_signal)}")
    
    # This check prevents crashing on empty or very short files
    if len(original_signal) < 2:
        print("   - ⚠️ SKIPPING: Not enough valid data to process.")
        return

    # Step 2: Decompose
    cA, cD = dwt_from_scratch(original_signal)
    print("🔬 STEP 2: Signal decomposed into Trend (cA) and Noise (cD).")

    # Step 3: Thresholding
    sigma = np.median(np.abs(cD)) / 0.6745
    threshold = sigma * sensitivity
    cD_cleaned = np.where(np.abs(cD) > threshold, cD, 0)
    print(f"🧹 STEP 3: Noise cleaned using dynamic threshold: {threshold:.2f}")

    # Step 4: Reconstruct
    denoised_signal = idwt_from_scratch(cA, cD_cleaned)
    print("🏗️ STEP 4: Clean signal rebuilt from cA and cleaned cD.")
    
    # Step 5: Finalize and Save
    if len(denoised_signal) < len(original_signal):
        denoised_signal = np.pad(denoised_signal, (0, len(original_signal) - len(denoised_signal)), 'edge')
    else:
        denoised_signal = denoised_signal[:len(original_signal)]

    output_df = data.copy()
    output_df['Close'] = denoised_signal
    output_filename = f"{ticker_symbol}_denoised_S{sensitivity}.csv"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    output_df.to_csv(output_filepath, index=False)
    print(f"💾 STEP 5: Denoised data saved to '{output_filepath}'")
    
# =============================================================================
# HISSAA 3: Batch Processor (No changes here)
# =============================================================================

def run_batch_process():
    print("Starting batch denoising process for all stocks...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    stock_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not stock_files:
        print(f"❌ Error: No CSV files found in the '{INPUT_DIR}' directory.")
        return
    for file_name in stock_files:
        file_path = os.path.join(INPUT_DIR, file_name)
        denoise_single_stock(file_path, SENSITIVITY)
        time.sleep(1)
    print("\n" + "="*80)
    print("🎉 BATCH PROCESS COMPLETE! All stocks have been denoised. 🎉")
    print("="*80)

if __name__ == "__main__":
    run_batch_process()