import xgboost as xgb

print("Loading")

model = xgb.XGBClassifier()

model.load_model(
    "ranker.json"
)

print("Loaded Successfully")