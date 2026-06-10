import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N_USERS = 10000
N_ITEMS = 5000
N_INTERACTIONS = 500000

CATEGORIES = [
    "Electronics",
    "Books",
    "Fashion",
    "Sports",
    "Movies",
    "Music",
    "Gaming",
    "Education"
]

print("Generating users...")

users = []

for uid in tqdm(range(N_USERS)):

    fav = random.choice(CATEGORIES)

    secondary = random.choice(
        [x for x in CATEGORIES if x != fav]
    )

    users.append(
        {
            "user_id": uid,
            "age": np.random.randint(18,70),
            "premium_user": np.random.choice(
                [0,1],
                p=[0.7,0.3]
            ),
            "favorite_category": fav,
            "secondary_category": secondary
        }
    )

users_df = pd.DataFrame(users)

print("Generating items...")

items = []

for iid in tqdm(range(N_ITEMS)):

    category = random.choice(
        CATEGORIES
    )

    items.append(
        {
            "item_id": iid,
            "category": category,
            "price": round(
                np.random.uniform(5,500),
                2
            ),
            "rating": round(
                np.random.uniform(1,5),
                2
            )
        }
    )

items_df = pd.DataFrame(items)

print("Building category lookup...")

category_to_items = {}

for category in CATEGORIES:

    category_to_items[category] = (
        items_df[
            items_df["category"] == category
        ]["item_id"]
        .tolist()
    )

print("Generating interactions...")

interaction_types = [
    "view",
    "click",
    "purchase"
]

interaction_probs = [
    0.75,
    0.20,
    0.05
]

start_date = (
    datetime.now()
    - timedelta(days=365)
)

interactions = []

for _ in tqdm(
    range(N_INTERACTIONS)
):

    user_id = np.random.randint(
        0,
        N_USERS
    )

    user = users_df.iloc[user_id]

    r = random.random()

    if r < 0.70:

        category = user[
            "favorite_category"
        ]

    elif r < 0.90:

        category = user[
            "secondary_category"
        ]

    else:

        category = random.choice(
            CATEGORIES
        )

    item_id = random.choice(
        category_to_items[
            category
        ]
    )

    timestamp = (
        start_date
        + timedelta(
            seconds=np.random.randint(
                0,
                365 * 24 * 3600
            )
        )
    )

    interaction = np.random.choice(
        interaction_types,
        p=interaction_probs
    )

    interactions.append(
        {
            "user_id": user_id,
            "item_id": item_id,
            "interaction_type": interaction,
            "timestamp": timestamp
        }
    )

interactions_df = pd.DataFrame(
    interactions
)

print("Saving files...")

users_df.to_csv(
    "users_realistic.csv",
    index=False
)

items_df.to_csv(
    "items_realistic.csv",
    index=False
)

interactions_df.to_csv(
    "interactions_realistic.csv",
    index=False
)

print("Done")

print(users_df.shape)
print(items_df.shape)
print(interactions_df.shape)