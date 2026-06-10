print("A")

import numpy as np

print("B")

import pandas as pd

print("C")

import faiss

print("D")

import xgboost as xgb

print("E")

print("Loading assets...")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

print("F")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

print("G")

items = pd.read_csv(
    "items_realistic.csv"
)

print("H")

features_df = pd.read_parquet(
    "training_features_realistic.parquet"
)

print("I")

ranker = xgb.XGBClassifier()

print("J")

ranker.load_model(
    "ranker.json"
)

print("K")

print("Loading assets...")

user_vectors = np.load(
    "tower_user_vectors.npy"
).astype("float32")

item_vectors = np.load(
    "tower_item_vectors.npy"
).astype("float32")

items = pd.read_csv(
    "items_realistic.csv"
)

features_df = pd.read_parquet(
    "training_features_realistic.parquet"
)

ranker = xgb.XGBClassifier()

ranker.load_model(
    "ranker.json"
)

faiss.normalize_L2(
    item_vectors
)

index = faiss.IndexFlatIP(
    item_vectors.shape[1]
)

index.add(
    item_vectors
)

print("Ready")

user_id = 123

user_vector = (
    user_vectors[user_id]
    .reshape(1, -1)
    .astype("float32")
)

faiss.normalize_L2(
    user_vector
)

scores, indices = index.search(
    user_vector,
    100
)

print(
    "Retrieved",
    len(indices[0]),
    "candidates"
)

user_rows = features_df[
    features_df["user_id"]
    == user_id
]

user_stats = user_rows.iloc[0]

candidates = []

for item_id in indices[0]:

    item_row = items[
        items["item_id"]
        == item_id
    ]

    if len(item_row) == 0:
        continue

    item_row = item_row.iloc[0]

    feature_vector = {

        "age":
            user_stats["age"],

        "premium_user":
            user_stats["premium_user"],

        "price":
            item_row["price"],

        "rating":
            item_row["rating"],

        "user_total_views":
            user_stats["user_total_views"],

        "user_total_clicks":
            user_stats["user_total_clicks"],

        "user_total_purchases":
            user_stats["user_total_purchases"],

        "item_total_views":
            0,

        "item_total_clicks":
            0,

        "item_total_purchases":
            0
    }

    candidates.append(
        (
            item_id,
            feature_vector
        )
    )

ranking_df = pd.DataFrame(
    [x[1] for x in candidates]
)

print(
    "Ranking",
    len(ranking_df),
    "candidates"
)

ranking_scores = (
    ranker.predict_proba(
        ranking_df
    )[:,1]
)

ranked = []

for idx, score in enumerate(
    ranking_scores
):

    ranked.append(
        (
            candidates[idx][0],
            score
        )
    )

ranked.sort(
    key=lambda x: x[1],
    reverse=True
)

print(
    "\nTop 10 Results\n"
)

for item_id, score in ranked[:10]:

    item_info = items[
        items["item_id"]
        == item_id
    ].iloc[0]

    print(
        item_id,
        item_info["category"],
        round(score,4)
    )