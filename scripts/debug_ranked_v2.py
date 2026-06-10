import numpy as np

print("Step 1")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

print("Step 2")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

print("Step 3")

import pandas as pd

items = pd.read_csv(
    "items_realistic.csv"
)

print("Step 4")

features_df = pd.read_parquet(
    "training_features_realistic.parquet"
)

print("Step 5")

import xgboost as xgb

print("Step 6")

ranker = xgb.XGBClassifier()

print("Step 7")

ranker.load_model(
    "ranker.json"
)

print("Step 8")

import faiss

print("Step 9")

faiss.normalize_L2(
    item_vectors
)

print("Step 10")

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

print("Step 11")

index.add(
    item_vectors
)

print("SUCCESS")