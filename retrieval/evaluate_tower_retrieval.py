import numpy as np
import pandas as pd
import faiss

print("Loading vectors...")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

pairs = pd.read_parquet(
    "retrieval_pairs_realistic.parquet"
)

pairs = pairs.sample(
    5000,
    random_state=42
)

faiss.normalize_L2(
    item_vectors
)

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

index.add(
    item_vectors
)

for K in [10, 20]:

    hits = 0

    for _, row in pairs.iterrows():

        user_id = int(
            row["user_id"]
        )

        true_item = int(
            row["item_id"]
        )

        user_vector = (
            user_vectors[user_id]
            .reshape(1, -1)
            .astype("float32")
        )

        faiss.normalize_L2(
            user_vector
        )

        _, indices = (
            index.search(
                user_vector,
                K
            )
        )

        if true_item in indices[0]:
            hits += 1

    recall = (
        hits /
        len(pairs)
    )

    print(
        f"Recall@{K}: "
        f"{recall:.4f}"
    )