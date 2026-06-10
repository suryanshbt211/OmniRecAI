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

items = pd.read_csv(
    "items_realistic.csv"
)

print(
    "Building FAISS Index..."
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

print(
    f"Indexed {len(item_vectors)} items"
)

while True:

    user_input = input(
        "\nEnter User ID (q to quit): "
    )

    if user_input.lower() == "q":
        break

    user_id = int(
        user_input
    )

    user_vector = (
        user_vectors[user_id]
        .reshape(1, -1)
        .astype("float32")
    )

    faiss.normalize_L2(
        user_vector
    )

    scores, indices = (
        index.search(
            user_vector,
            10
        )
    )

    print(
        "\nTop Recommendations\n"
    )

    result = (
        items.iloc[
            indices[0]
        ]
        .copy()
    )

    result["score"] = (
        scores[0]
    )

    print(
        result[
            [
                "item_id",
                "category",
                "price",
                "rating",
                "score"
            ]
        ]
    )