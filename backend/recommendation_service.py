import numpy as np
import faiss

from backend.repositories import get_item

from backend.cache import (
    get_cached_recommendations,
    cache_recommendations
)

from backend import metrics

print("Loading recommendation assets...")

user_vectors = np.load(
    "models/tower_user_vectors.npy"
).astype("float32")

item_vectors = np.load(
    "models/tower_item_vectors.npy"
).astype("float32")

faiss.normalize_L2(
    item_vectors
)

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

index.add(
    item_vectors
)

print("Recommendation service ready")


def get_recommendations(
    user_id: int,
    top_k: int = 10
):

    cached = get_cached_recommendations(
        user_id
    )

    if cached:

        metrics.cache_hits += 1

        print(
            f"Cache Hit {user_id}"
        )

        return cached

    metrics.cache_misses += 1

    if user_id < 0 or user_id >= len(user_vectors):
        return []

    user_vector = (
        user_vectors[user_id]
        .reshape(1, -1)
        .astype("float32")
    )

    faiss.normalize_L2(
        user_vector
    )

    scores, indices = index.search(
        user_vector,
        top_k
    )

    recommendations = []

    for rank, item_id in enumerate(
        indices[0]
    ):

        item = get_item(
            int(item_id)
        )

        if item is None:
            continue

        recommendations.append(
            {
                "rank": rank + 1,
                "item_id": int(item["item_id"]),
                "category": item["category"],
                "price": float(item["price"]),
                "rating": float(item["rating"]),
                "score": float(
                    scores[0][rank]
                )
            }
        )

    cache_recommendations(
        user_id,
        recommendations
    )

    return recommendations