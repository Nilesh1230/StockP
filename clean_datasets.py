# clean_datasets.py

import os
import pandas as pd

INPUT_DIR = "datasets"

def clean_all_csv_files():
    """
    Standardizes all CSV files in a directory to ensure they have
    a 'Date' column and proper headers.
    """
    print("="*80)
    print("🧹 STARTING DATA CLEANING AND STANDARDIZATION PROCESS 🧹")
    print(f"Target directory: '{INPUT_DIR}'")
    print("="*80)

    # Find all CSV files in the directory, ignore metadata files
    stock_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv') and 'metadata' not in f]

    if not stock_files:
        print(" No stock CSV files found to clean.")
        return

    for file_name in stock_files:
        file_path = os.path.join(INPUT_DIR, file_name)
        try:
            # Load the CSV, assuming the first column could be the date index
            df = pd.read_csv(file_path, index_col=0)

            # --- The Core Fix ---
            # 1. Reset the index. This turns the date index into a column.
            #    The new column will be named 'index' or 'Date'.
            df.reset_index(inplace=True)

            # 2. Rename the first column to 'Date' regardless of its current name.
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

            # 3. Ensure the 'Date' column is in a clean YYYY-MM-DD format
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            # 4. Save the cleaned file, overwriting the old one
            df.to_csv(file_path, index=False)
            
            print(f"  -  Cleaned and standardized: {file_name}")

        except Exception as e:
            print(f"  -  Error processing {file_name}: {e}")

    print("\n" + "="*80)
    print("CLEANING COMPLETE! All dataset files are now standardized. ")
    print("="*80)


if __name__ == "__main__":
    clean_all_csv_files()