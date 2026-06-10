import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("Loading ranking data...")

df = pd.read_parquet(
    "data/ranking_dataset.parquet"
)

print("Dataset Shape:", df.shape)

y = df["label"]

X = df.drop(
    columns=[
        "label",
        "user_id",
        "item_id"
    ]
)

print("Features:", X.columns.tolist())

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)


os.makedirs(
    "mlartifacts",
    exist_ok=True
)
mlflow.set_experiment(
    "omnirec_ranker"
)

with mlflow.start_run():

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        preds
    )

    print(
        f"AUC={auc:.4f}"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "max_depth",
        6
    )

    mlflow.log_param(
        "learning_rate",
        0.05
    )

    mlflow.log_param(
        "train_rows",
        len(X_train)
    )

    mlflow.log_param(
        "test_rows",
        len(X_test)
    )

    mlflow.log_metric(
        "auc",
        float(auc)
    )

    mlflow.xgboost.log_model(
        model,
        artifact_path="ranker"
    )

    model.save_model(
        "models/ranker_mlflow.json"
    )

    print("Model Saved")

print("Training Complete")