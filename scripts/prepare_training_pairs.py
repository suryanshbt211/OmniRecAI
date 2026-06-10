import pandas as pd

print("Loading data...")

df = pd.read_parquet("training_features.parquet")

print("Creating positive samples...")

positive = df[
    df["interaction_type"].isin(
        ["click", "purchase"]
    )
].copy()

positive["target"] = 1

print("Positive samples:", len(positive))

positive = positive[
    [
        "user_id",
        "item_id",
        "target"
    ]
]

positive.to_parquet(
    "retrieval_pairs.parquet",
    index=False
)

print("Saved retrieval_pairs.parquet")
print(positive.shape)