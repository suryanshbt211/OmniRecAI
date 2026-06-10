import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

print("Loading data...")

df = pd.read_parquet(
    "training_features_realistic.parquet"
)

df = df[
    df["interaction_type"]
    .isin(
        ["click", "purchase"]
    )
].copy()

print(df.shape)

cat_encoder = LabelEncoder()

df["favorite_category"] = (
    cat_encoder.fit_transform(
        df["favorite_category"]
    )
)

df["secondary_category"] = (
    cat_encoder.fit_transform(
        df["secondary_category"]
    )
)

df["category"] = (
    cat_encoder.fit_transform(
        df["category"]
    )
)

NUM_USERS = (
    df["user_id"].max() + 1
)

NUM_ITEMS = (
    df["item_id"].max() + 1
)

NUM_CATEGORIES = (
    df["category"].nunique()
)

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


class RetrievalDataset(
    Dataset
):

    def __init__(
        self,
        df
    ):

        self.df = df

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        return {

            "user_id":
                torch.tensor(
                    row["user_id"],
                    dtype=torch.long
                ),

            "item_id":
                torch.tensor(
                    row["item_id"],
                    dtype=torch.long
                ),

            "age":
                torch.tensor(
                    row["age"],
                    dtype=torch.float32
                ),

            "premium":
                torch.tensor(
                    row["premium_user"],
                    dtype=torch.float32
                ),

            "favorite":
                torch.tensor(
                    row["favorite_category"],
                    dtype=torch.long
                ),

            "secondary":
                torch.tensor(
                    row["secondary_category"],
                    dtype=torch.long
                ),

            "category":
                torch.tensor(
                    row["category"],
                    dtype=torch.long
                ),

            "price":
                torch.tensor(
                    row["price"],
                    dtype=torch.float32
                ),

            "rating":
                torch.tensor(
                    row["rating"],
                    dtype=torch.float32
                )
        }


train_loader = DataLoader(
    RetrievalDataset(train_df),
    batch_size=1024,
    shuffle=True
)


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

    def forward(
        self,
        batch
    ):

        user_vec = self.user_emb(
            batch["user_id"]
        )

        fav = self.cat_emb(
            batch["favorite"]
        )

        sec = self.cat_emb(
            batch["secondary"]
        )

        user_num = torch.stack(
            [
                batch["age"],
                batch["premium"]
            ],
            dim=1
        )

        user_input = torch.cat(
            [
                user_vec,
                fav,
                sec,
                user_num
            ],
            dim=1
        )

        user_output = (
            self.user_tower(
                user_input
            )
        )

        item_vec = self.item_emb(
            batch["item_id"]
        )

        item_cat = self.cat_emb(
            batch["category"]
        )

        item_num = torch.stack(
            [
                batch["price"],
                batch["rating"]
            ],
            dim=1
        )

        item_input = torch.cat(
            [
                item_vec,
                item_cat,
                item_num
            ],
            dim=1
        )

        item_output = (
            self.item_tower(
                item_input
            )
        )

        return (
            user_output
            *
            item_output
        ).sum(
            dim=1
        )


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = (
    FeatureAwareTwoTower()
    .to(device)
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(10):

    total_loss = 0

    for batch in train_loader:

        for key in batch:

            batch[key] = (
                batch[key]
                .to(device)
            )

        optimizer.zero_grad()

        pos_scores = (
            model(batch)
        )

        pos_labels = torch.ones(
            len(
                pos_scores
            )
        ).to(device)

        loss = loss_fn(
            pos_scores,
            pos_labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    print(
        f"Epoch {epoch+1}"
        f" Loss={total_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "feature_aware_two_tower.pt"
)

print(
    "Model saved"
)