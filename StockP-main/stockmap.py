import pandas as pd, os
cols = pd.read_csv("data/model_ready/combined_denoised_prices.csv", index_col=0).columns.tolist()
# check denoised files exist and ordering
missing=[]
for t in cols:
    safe = t.replace(".","_")
    if not os.path.exists(f"data/denoised_universe/{safe}.csv"):
        missing.append(t)
print("Missing denoised files:", missing)
