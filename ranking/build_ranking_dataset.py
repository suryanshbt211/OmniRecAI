import pandas as pd

print("Loading data...")

df = pd.read_parquet(
    "training_features_realistic.parquet"
)

df["label"] = (
    df["interaction_type"]
    == "purchase"
).astype(int)

features = [

    "age",

    "premium_user",

    "price",

    "rating",

    "user_total_views",
    "user_total_clicks",
    "user_total_purchases",

    "item_total_views",
    "item_total_clicks",
    "item_total_purchases"
]

ranking_df = df[
    [
        "user_id",
        "item_id",
        "label"
    ] + features
]

ranking_df.to_parquet(
    "ranking_dataset.parquet",
    index=False
)

print(
    ranking_df.shape
)

print(
    ranking_df["label"]
    .value_counts()
)