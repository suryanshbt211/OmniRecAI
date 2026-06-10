import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
from datetime import datetime, timedelta
import random

fake = Faker()

np.random.seed(42)
random.seed(42)

N_USERS = 10000
N_ITEMS = 5000
N_INTERACTIONS = 500000

print("Generating users...")

users = []

countries = [
    "USA",
    "India",
    "UK",
    "Germany",
    "Canada",
    "Australia"
]

for uid in tqdm(range(N_USERS)):
    users.append(
        {
            "user_id": uid,
            "age": np.random.randint(18, 70),
            "country": random.choice(countries),
            "signup_days_ago": np.random.randint(1, 1000),
            "premium_user": np.random.choice([0, 1], p=[0.7, 0.3]),
        }
    )

users_df = pd.DataFrame(users)

print("Generating items...")

categories = [
    "Electronics",
    "Books",
    "Fashion",
    "Sports",
    "Movies",
    "Music",
    "Gaming",
    "Education"
]

items = []

for iid in tqdm(range(N_ITEMS)):
    items.append(
        {
            "item_id": iid,
            "category": random.choice(categories),
            "price": round(np.random.uniform(5, 500), 2),
            "rating": round(np.random.uniform(1, 5), 2),
        }
    )

items_df = pd.DataFrame(items)

print("Generating interactions...")

interaction_types = [
    "view",
    "click",
    "cart",
    "purchase"
]

weights = [0.7, 0.2, 0.08, 0.02]

start_date = datetime.now() - timedelta(days=365)

interactions = []

for _ in tqdm(range(N_INTERACTIONS)):
    timestamp = start_date + timedelta(
        seconds=np.random.randint(0, 365 * 24 * 3600)
    )

    interactions.append(
        {
            "user_id": np.random.randint(0, N_USERS),
            "item_id": np.random.randint(0, N_ITEMS),
            "interaction_type": np.random.choice(
                interaction_types,
                p=weights
            ),
            "timestamp": timestamp,
        }
    )

interactions_df = pd.DataFrame(interactions)

print("Saving CSV files...")

users_df.to_csv("users.csv", index=False)
items_df.to_csv("items.csv", index=False)
interactions_df.to_csv("interactions.csv", index=False)

print("\nDone!")
print(users_df.shape)
print(items_df.shape)
print(interactions_df.shape)