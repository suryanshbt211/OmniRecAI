import numpy as np
import faiss

print("Loading vectors")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

print(item_vectors.shape)

print("Normalize")

faiss.normalize_L2(
    item_vectors
)

print("Create index")

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

print("Add vectors")

index.add(
    item_vectors
)

print("Done")