import pandas as pd
import numpy as np
import faiss

print("Loading embeddings...")

user_embeddings = np.load(
    "user_embeddings.npy"
).astype("float32")

item_embeddings = np.load(
    "item_embeddings.npy"
).astype("float32")

print("Loading validation data...")

pairs = pd.read_parquet(
    "retrieval_pairs.parquet"
)

pairs = pairs.sample(
    5000,
    random_state=42
)

print("Building FAISS index...")

faiss.normalize_L2(
    item_embeddings
)

index = faiss.IndexFlatIP(
    item_embeddings.shape[1]
)

index.add(
    item_embeddings
)

K_VALUES = [10, 20]

results = {}

for K in K_VALUES:

    hits = 0

    for _, row in pairs.iterrows():

        user_id = int(
            row["user_id"]
        )

        true_item = int(
            row["item_id"]
        )

        user_vector = (
            user_embeddings[user_id]
            .reshape(1, -1)
            .astype("float32")
        )

        faiss.normalize_L2(
            user_vector
        )

        distances, indices = (
            index.search(
                user_vector,
                K
            )
        )

        recommended_items = (
            indices[0]
        )

        if true_item in recommended_items:
            hits += 1

    recall = hits / len(pairs)

    results[K] = recall

print("\nResults")

for k, score in results.items():

    print(
        f"Recall@{k}: "
        f"{score:.4f}"
    )