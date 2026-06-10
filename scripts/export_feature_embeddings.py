import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn

print("Loading data...")

df = pd.read_parquet(
    "training_features_realistic.parquet"
)

encoder = LabelEncoder()

df["favorite_category"] = encoder.fit_transform(
    df["favorite_category"]
)

df["secondary_category"] = encoder.fit_transform(
    df["secondary_category"]
)

df["category"] = encoder.fit_transform(
    df["category"]
)

NUM_USERS = df["user_id"].max() + 1
NUM_ITEMS = df["item_id"].max() + 1
NUM_CATEGORIES = df["category"].nunique()


class FeatureAwareTwoTower(
    nn.Module
):

    def __init__(self):

        super().__init__()

        emb_dim = 32

        self.user_emb = nn.Embedding(
            NUM_USERS,
            emb_dim
        )

        self.item_emb = nn.Embedding(
            NUM_ITEMS,
            emb_dim
        )

        self.cat_emb = nn.Embedding(
            NUM_CATEGORIES,
            8
        )

        self.user_tower = nn.Sequential(
            nn.Linear(
                emb_dim + 8 + 8 + 2,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                64
            )
        )

        self.item_tower = nn.Sequential(
            nn.Linear(
                emb_dim + 8 + 2,
                64
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                64
            )
        )


model = FeatureAwareTwoTower()

model.load_state_dict(
    torch.load(
        "feature_aware_two_tower_v2.pt",
        map_location="cpu"
    )
)

model.eval()

user_embeddings = (
    model.user_emb.weight
    .detach()
    .numpy()
)

item_embeddings = (
    model.item_emb.weight
    .detach()
    .numpy()
)

np.save(
    "feature_user_embeddings.npy",
    user_embeddings
)

np.save(
    "feature_item_embeddings.npy",
    item_embeddings
)

print(
    user_embeddings.shape
)

print(
    item_embeddings.shape
)