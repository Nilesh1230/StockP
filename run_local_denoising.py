# run_local_denoising_verbose.py
# ==============================================
# ==============================================

import numpy as np
import pandas as pd
import argparse
import os
import time

# =============================================================================
# HISSAA 1: Core DWT & Convolution Logic (From Scratch with Formulas)
# =============================================================================

def convolve_from_scratch(signal, kernel, kernel_name="Filter", verbose=False):
    """
    1D Convolution from scratch.
    Prints every calculation (first few steps).
    """
    kernel_size = len(kernel)
    output_size = len(signal) - kernel_size + 1
    output = np.zeros(output_size)
    reversed_kernel = kernel[::-1]

    if verbose:
        print(f"\n[INFO] Starting convolution with '{kernel_name}'...")
        print(f"Kernel (reversed): {np.round(reversed_kernel,2)}")
    
    for i in range(output_size):
        window = signal[i:i+kernel_size]
        multiplied = window * reversed_kernel
        output[i] = np.sum(multiplied)
        
        if verbose and i < 3:  # only first 3 steps to avoid flood
            print(f"\n--- Step {i+1} ---")
            print(f"Signal window  : {np.round(window,2)}")
            print(f"Kernel applied : {np.round(reversed_kernel,2)}")
            print(f"Multiplication : {np.round(multiplied,2)}")
            print(f"Sum (Result)   : {round(output[i],2)}")
            
            # Show explicit formula for first step
            if i==0:
                formula = " + ".join([f"({round(window[j],2)}*{round(reversed_kernel[j],2)})" for j in range(kernel_size)])
                print(f"FORMULA USED -> {formula} = {round(output[i],2)}")
            time.sleep(1)
            
    return output

def dwt_from_scratch(signal, verbose=False):
    """Perform Haar DWT (Approximation + Detail)"""
    lp = np.array([1, 1]) / np.sqrt(2)   # Low-pass
    hp = np.array([-1, 1]) / np.sqrt(2)  # High-pass
    cA = convolve_from_scratch(signal, lp, "Low-Pass (Trend)", verbose)
    cD = convolve_from_scratch(signal, hp, "High-Pass (Noise)", verbose)
    return cA[::2], cD[::2]  # Downsampling by 2

def idwt_from_scratch(cA, cD):
    """Inverse DWT from Haar coefficients"""
    lp = np.array([1, 1])/np.sqrt(2)
    hp = np.array([1, -1])/np.sqrt(2)
    
    cA_up = np.zeros(2*len(cA)); cA_up[::2] = cA
    cD_up = np.zeros(2*len(cD)); cD_up[::2] = cD
    
    recon_A = np.convolve(cA_up, lp, 'full')[:len(cA_up)]
    recon_D = np.convolve(cD_up, hp, 'full')[:len(cD_up)]
    
    return recon_A + recon_D

# =============================================================================
# HISSAA 2: Full Step-by-Step Denoising Process
# =============================================================================

def start_local_process(ticker_symbol):
    print("="*80)
    print(" DWT DENOISING: EXPLANATION")
    print("="*80)
    
    # STEP 1: Load noisy data
    file_path = os.path.join('datasets', f'{ticker_symbol}.csv')
    try:
        data = pd.read_csv(file_path)
        original_signal = data['Close'].values
        print(f"\n📂 STEP 1: LOADING NOISY DATA")
        print(f"File: {file_path}")
        print(f"Signal length: {len(original_signal)}")
        print(f"First 7 noisy points: {np.round(original_signal[:7],2)}")
        time.sleep(2)
    except FileNotFoundError:
        print(f" Error: {file_path} not found")
        return
    
    # STEP 2: DWT decomposition
    print(f"\n🔬 STEP 2: DECOMPOSING THE SIGNAL (Haar DWT)")
    print("Logic: Signal → Trend (cA) + Noise (cD)")
    print("Formula for cA[k]: (x[2k] + x[2k+1])/√2")
    print("Formula for cD[k]: (-x[2k] + x[2k+1])/√2")
    cA, cD = dwt_from_scratch(original_signal, verbose=True)
    
    print("\n[INFO] Downsampled coefficients (first 7):")
    print(f"Trend (cA) : {np.round(cA[:7],2)}")
    print(f"Noise (cD) : {np.round(cD[:7],2)}")
    time.sleep(2)
    
    # STEP 3: Thresholding (Denoising)
    print(f"\n🧹 STEP 3: CLEANING THE NOISE (THRESHOLDING)")
    threshold = sigma * sensitivity
    print(f"Rule: If |cD| < {threshold}, set to 0")
    cD_cleaned = np.where(np.abs(cD) > threshold, cD, 0)
    removed = np.sum(cD != cD_cleaned)
    print(f"Removed small noise coefficients: {removed}")
    time.sleep(2)
    
    # STEP 4: Reconstruct signal
    print(f"\n🏗️ STEP 4: REBUILDING THE CLEAN SIGNAL (Inverse DWT)")
    denoised_signal = idwt_from_scratch(cA, cD_cleaned)
    print(f"First 7 reconstructed points: {np.round(denoised_signal[:7],2)}")
    time.sleep(1)
    
    # STEP 4.5: Match lengths
    print(f"\n📏 STEP 4.5: MATCHING SIGNAL LENGTHS")
    if len(denoised_signal) < len(original_signal):
        diff = len(original_signal) - len(denoised_signal)
        denoised_signal = np.concatenate([denoised_signal, np.repeat(denoised_signal[-1], diff)])
        print(f"Padded {diff} points to match original length.")
    else:
        print("No padding needed.")
    
    # STEP 5: Save denoised data
    print(f"\n💾 STEP 5: SAVING CLEAN DATA")
    output_folder = 'data/denoised_from_scratch'
    os.makedirs(output_folder, exist_ok=True)
    output_df = data.copy()
    output_df['Close'] = denoised_signal
    output_file = os.path.join(output_folder, f"{ticker_symbol}_denoised.csv")
    output_df.to_csv(output_file, index=False)
    print(f" Clean data saved to: {output_file}")
    
    # FINAL TABLE
    print("\n📊 FINAL RESULT: Original vs Denoised (Last 7 days)")
    print("="*80)
    final_df = pd.DataFrame({
        'Original_Noisy_Price': np.round(original_signal[-7:],2),
        'Denoised_Price': np.round(denoised_signal[-7:],2)
    })
    print(final_df)
    print("\n🎉 PROCESS COMPLETE! 🎉")
    print("="*80)

# =============================================================================
# Script Entry
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Locally denoise stock data with full step-by-step formulas.")
    parser.add_argument('--ticker', type=str, required=True, help="Stock ticker symbol (e.g., 'TCS' or 'BPCL')")
    args = parser.parse_args()
    start_local_process(args.ticker)
