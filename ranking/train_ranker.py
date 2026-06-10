import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score
)

from xgboost import XGBClassifier

print(
    "Loading ranking data..."
)

df = pd.read_parquet(
    "ranking_dataset.parquet"
)

X = df.drop(
    columns=[
        "user_id",
        "item_id",
        "label"
    ]
)

y = df["label"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

print(
    "Training..."
)

model = XGBClassifier(

    n_estimators=200,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

preds = model.predict_proba(
    X_test
)[:,1]

auc = roc_auc_score(
    y_test,
    preds
)

print(
    f"AUC: {auc:.4f}"
)

import joblib

joblib.dump(
    model,
    "ranker.pkl"
)

print(
    "Saved ranker.pkl"
)