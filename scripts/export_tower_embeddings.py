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
    nn.Module):

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

print("Building user vectors...")

users = (
    df[
        [
            "user_id",
            "age",
            "premium_user",
            "favorite_category",
            "secondary_category"
        ]
    ]
    .drop_duplicates("user_id")
)

user_vectors = []

with torch.no_grad():

    for _, row in users.iterrows():

        user_vec = model.user_emb(
            torch.tensor(
                [row["user_id"]],
                dtype=torch.long
            )
        )

        fav = model.cat_emb(
            torch.tensor(
                [row["favorite_category"]],
                dtype=torch.long
            )
        )

        sec = model.cat_emb(
            torch.tensor(
                [row["secondary_category"]],
                dtype=torch.long
            )
        )

        numeric = torch.tensor(
            [
                [
                    row["age"],
                    row["premium_user"]
                ]
            ],
            dtype=torch.float32
        )

        combined = torch.cat(
            [
                user_vec,
                fav,
                sec,
                numeric
            ],
            dim=1
        )

        tower_output = (
            model.user_tower(
                combined
            )
        )

        user_vectors.append(
            tower_output
            .numpy()
            .flatten()
        )

print("Building item vectors...")

items = (
    df[
        [
            "item_id",
            "category",
            "price",
            "rating"
        ]
    ]
    .drop_duplicates("item_id")
)

item_vectors = []

with torch.no_grad():

    for _, row in items.iterrows():

        item_vec = model.item_emb(
            torch.tensor(
                [row["item_id"]],
                dtype=torch.long
            )
        )

        cat = model.cat_emb(
            torch.tensor(
                [row["category"]],
                dtype=torch.long
            )
        )

        numeric = torch.tensor(
            [
                [
                    row["price"],
                    row["rating"]
                ]
            ],
            dtype=torch.float32
        )

        combined = torch.cat(
            [
                item_vec,
                cat,
                numeric
            ],
            dim=1
        )

        tower_output = (
            model.item_tower(
                combined
            )
        )

        item_vectors.append(
            tower_output
            .numpy()
            .flatten()
        )

user_vectors = np.array(
    user_vectors
)

item_vectors = np.array(
    item_vectors
)

np.save(
    "tower_user_vectors.npy",
    user_vectors
)

np.save(
    "tower_item_vectors.npy",
    item_vectors
)

print(
    user_vectors.shape
)

print(
    item_vectors.shape
)