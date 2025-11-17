# run_all_universal.py
import os
os.system("python preprocess_universe.py")
os.system("python combine_all.py")
os.system("python build_full_graph.py")
os.system("python train_universal.py")
print("Universal model built. Use: python predict_universal.py TICKER")
