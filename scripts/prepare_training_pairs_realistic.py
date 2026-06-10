import pandas as pd

df = pd.read_parquet(
    "training_features_realistic.parquet"
)

pairs = df[
    df["interaction_type"]
    .isin(
        ["click","purchase"]
    )
][
    [
        "user_id",
        "item_id"
    ]
].copy()

pairs["target"] = 1

pairs.to_parquet(
    "retrieval_pairs_realistic.parquet",
    index=False
)

print(pairs.shape)