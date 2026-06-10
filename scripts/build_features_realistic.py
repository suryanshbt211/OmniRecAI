import pandas as pd
import numpy as np

users = pd.read_csv(
    "users_realistic.csv"
)

items = pd.read_csv(
    "items_realistic.csv"
)

interactions = pd.read_csv(
    "interactions_realistic.csv"
)

user_views = (
    interactions[
        interactions["interaction_type"]=="view"
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="user_total_views")
)

user_clicks = (
    interactions[
        interactions["interaction_type"]=="click"
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="user_total_clicks")
)

user_purchases = (
    interactions[
        interactions["interaction_type"]=="purchase"
    ]
    .groupby("user_id")
    .size()
    .reset_index(name="user_total_purchases")
)

item_views = (
    interactions[
        interactions["interaction_type"]=="view"
    ]
    .groupby("item_id")
    .size()
    .reset_index(name="item_total_views")
)

item_clicks = (
    interactions[
        interactions["interaction_type"]=="click"
    ]
    .groupby("item_id")
    .size()
    .reset_index(name="item_total_clicks")
)

item_purchases = (
    interactions[
        interactions["interaction_type"]=="purchase"
    ]
    .groupby("item_id")
    .size()
    .reset_index(name="item_total_purchases")
)

dataset = interactions.copy()

dataset = dataset.merge(
    users,
    on="user_id",
    how="left"
)

dataset = dataset.merge(
    items,
    on="item_id",
    how="left"
)

dataset = dataset.merge(
    user_views,
    on="user_id",
    how="left"
)

dataset = dataset.merge(
    user_clicks,
    on="user_id",
    how="left"
)

dataset = dataset.merge(
    user_purchases,
    on="user_id",
    how="left"
)

dataset = dataset.merge(
    item_views,
    on="item_id",
    how="left"
)

dataset = dataset.merge(
    item_clicks,
    on="item_id",
    how="left"
)

dataset = dataset.merge(
    item_purchases,
    on="item_id",
    how="left"
)

dataset.fillna(0, inplace=True)

dataset["label"] = (
    dataset["interaction_type"]
    == "purchase"
).astype(int)

dataset.to_parquet(
    "training_features_realistic.parquet",
    index=False
)

print(dataset.shape)