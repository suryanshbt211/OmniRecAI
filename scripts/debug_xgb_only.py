import xgboost as xgb

print("Before")

model = xgb.XGBClassifier()

print("Created")

model.load_model(
    "ranker.json"
)

print("Loaded")