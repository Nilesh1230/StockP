import numpy as np, pandas as pd, torch, json, os, math
from train_universal import prepare, UniversalGATLSTM  # if in same folder

MR = "data/model_ready"
(X_train,Y_train,X_val,Y_val,X_test,Y_test, edge_index, price_min, price_max, cols) = prepare()

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = UniversalGATLSTM(num_nodes=len(cols)).to(device)
model.load_state_dict(torch.load(os.path.join(MR,"universal_model.pth"), map_location=device))
model.eval()

# build dataloader (no batch)
X_test = X_test.to(device); Y_test = Y_test.to(device)
with torch.no_grad():
    preds = model(X_test, edge_index.to(device)).cpu().numpy()
trues = Y_test.cpu().numpy()

# invert scaling (price)
prng = np.maximum(price_max - price_min, 1e-9)
preds_real = preds * prng.reshape(1,-1) + price_min.reshape(1,-1)
trues_real = trues * prng.reshape(1,-1) + price_min.reshape(1,-1)

# compute metrics per ticker
def rmse(a,b): return math.sqrt(((a-b)**2).mean())
def mae(a,b): return (abs(a-b)).mean()
def mape(a,b): return (abs((a-b)/(b+1e-9))).mean()*100

rows=[]
for i,t in enumerate(cols):
    r = rmse(preds_real[:,i], trues_real[:,i])
    a = mae(preds_real[:,i], trues_real[:,i])
    m = mape(preds_real[:,i], trues_real[:,i])
    rows.append((t, r, a, m))
rows = sorted(rows, key=lambda x: x[2], reverse=True)  # sort by MAE desc
for r in rows[:20]:
    print(r)
