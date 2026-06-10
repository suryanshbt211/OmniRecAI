import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

print("Loading pairs...")

pairs = pd.read_parquet(
    "retrieval_pairs_realistic.parquet"
)

NUM_USERS = pairs["user_id"].max() + 1
NUM_ITEMS = pairs["item_id"].max() + 1

print("Users:", NUM_USERS)
print("Items:", NUM_ITEMS)

train_df, val_df = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))


class RetrievalDataset(Dataset):

    def __init__(self, df):
        self.users = df["user_id"].values
        self.items = df["item_id"].values

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        return (
            torch.tensor(
                self.users[idx],
                dtype=torch.long
            ),
            torch.tensor(
                self.items[idx],
                dtype=torch.long
            )
        )


train_dataset = RetrievalDataset(
    train_df
)

val_dataset = RetrievalDataset(
    val_df
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False
)


class TwoTowerModel(nn.Module):

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim
        )

    def forward(
        self,
        user_ids,
        item_ids
    ):

        user_vec = self.user_embedding(
            user_ids
        )

        item_vec = self.item_embedding(
            item_ids
        )

        scores = (
            user_vec * item_vec
        ).sum(dim=1)

        return scores


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

model = TwoTowerModel(
    NUM_USERS,
    NUM_ITEMS
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

loss_fn = nn.BCEWithLogitsLoss()


def create_negative_batch(
    batch_users
):

    return torch.randint(
        0,
        NUM_ITEMS,
        (
            len(batch_users),
        )
    )


EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for users, items in train_loader:

        users = users.to(device)
        items = items.to(device)

        positive_labels = torch.ones(
            len(users)
        ).to(device)

        negative_items = create_negative_batch(
            users
        ).to(device)

        negative_labels = torch.zeros(
            len(users)
        ).to(device)

        optimizer.zero_grad()

        positive_scores = model(
            users,
            items
        )

        negative_scores = model(
            users,
            negative_items
        )

        positive_loss = loss_fn(
            positive_scores,
            positive_labels
        )

        negative_loss = loss_fn(
            negative_scores,
            negative_labels
        )

        loss = (
            positive_loss
            +
            negative_loss
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1} "
        f"Loss={total_loss:.4f}"
    )

print(
    "Training Complete"
)

torch.save(
    model.state_dict(),
    "two_tower_model_realistic.pt"
)

user_embeddings = (
    model.user_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

item_embeddings = (
    model.item_embedding.weight
    .detach()
    .cpu()
    .numpy()
)

np.save(
    "user_embeddings_realistic.npy",
    user_embeddings
)

np.save(
    "item_embeddings_realistic.npy",
    item_embeddings
)

print(
    "Saved model and embeddings"
)