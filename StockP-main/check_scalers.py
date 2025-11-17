import json, numpy as np, pandas as pd, os
MR = "data/model_ready"
print("Files:", os.listdir(MR))

cols = pd.read_csv(f"{MR}/combined_denoised_prices.csv", index_col=0).columns.tolist()
print("Num cols:", len(cols))
print("Sample cols:", cols[:10])

price_min = np.load(f"{MR}/price_min.npy")
price_max = np.load(f"{MR}/price_max.npy")
print("price_min shape:", price_min.shape, "price_max shape:", price_max.shape)

# print for some tickers
for t in ["TCS.NS","ACC.NS","IFCI.NS","BEL.NS","SBIN.NS"]:
    if t in cols:
        idx = cols.index(t)
        print(t, "idx", idx, "min", price_min[idx], "max", price_max[idx])
    else:
        print(t, "NOT in columns")
