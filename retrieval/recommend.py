import numpy as np
import pandas as pd
import faiss

print("Loading embeddings...")

user_embeddings = np.load(
    "feature_user_embeddings.npy"
).astype("float32")

item_embeddings = np.load(
    "feature_item_embeddings.npy"
).astype("float32")

print("Loading items...")

items = pd.read_csv(
    "items_realistic.csv"
)

faiss.normalize_L2(
    item_embeddings
)

index = faiss.IndexFlatIP(
    item_embeddings.shape[1]
)

index.add(
    item_embeddings
)

print(
    f"Indexed {len(item_embeddings)} items"
)

while True:

    user_input = input(
        "\nEnter User ID (or q): "
    )

    if user_input.lower() == "q":
        break

    user_id = int(user_input)

    user_vector = (
        user_embeddings[user_id]
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

    recommendations = (
        items.iloc[
            indices[0]
        ][
            [
                "item_id",
                "category",
                "price",
                "rating"
            ]
        ]
    )

    print(
        recommendations
    )