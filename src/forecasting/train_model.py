import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "../../data/item_sales/processed_sales_features_v2.csv"
MODEL_PATH = "models/random_forest_model.pkl"

features = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
    "rolling_28",
    "promo_flag",
    "promo_depth",
    "stock_ratio",
    "price",
    "weekday",
    "weekend",
    "promo_price_interaction",
    "lag1_stock_interaction"
]

def train_model():

    df = pd.read_csv(DATA_PATH)

    X = df[features]
    y = df["units_sold"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()