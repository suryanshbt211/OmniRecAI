import numpy as np
import pandas as pd
import faiss
import joblib

print("Step 1")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

print("Step 2")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

print("Step 3")

items = pd.read_csv(
    "items_realistic.csv"
)

print("Step 4")

features_df = pd.read_parquet(
    "training_features_realistic.parquet"
)

print("Step 5")

ranker = joblib.load(
    "ranker.pkl"
)

print("Step 6")

faiss.normalize_L2(
    item_vectors
)

print("Step 7")

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

print("Step 8")

index.add(
    item_vectors
)

print("Step 9")

print("SUCCESS")