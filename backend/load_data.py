import pandas as pd
from backend.database import engine

print("Loading data...")

users = pd.read_csv(
    "data/users.csv"
)

items = pd.read_csv(
    "data/items.csv"
)

print(users.shape)
print(items.shape)

users.to_sql(
    "users",
    engine,
    if_exists="replace",
    index=False
)

items.to_sql(
    "items",
    engine,
    if_exists="replace",
    index=False
)

print("Load Complete")