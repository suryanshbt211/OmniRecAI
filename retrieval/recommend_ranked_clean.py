import numpy as np
import pandas as pd
import faiss
import xgboost as xgb

print("Loading assets...")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

items = pd.read_csv(
    "items_realistic.csv"
)

features_df = pd.read_parquet(
    "training_features_realistic.parquet"
)

ranker = xgb.XGBClassifier()
ranker.load_model("ranker.json")

print("Ranker loaded")

faiss.normalize_L2(item_vectors)

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

index.add(item_vectors)

print("FAISS ready")

user_id = 123

user_vector = (
    user_vectors[user_id]
    .reshape(1, -1)
    .astype("float32")
)

faiss.normalize_L2(user_vector)

scores, indices = index.search(
    user_vector,
    10
)

print("\nTop Retrieved Items:\n")

for item_id in indices[0]:

    row = items[
        items["item_id"] == item_id
    ].iloc[0]

    print(
        f"{item_id} | "
        f"{row['category']} | "
        f"Rating={row['rating']}"
    )