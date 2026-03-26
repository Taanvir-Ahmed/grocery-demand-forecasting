import os
import joblib
import pandas as pd

MODEL_PATH = "models/random_forest_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model file not found. Please run 'python src/forecasting/train_model.py' first."
    )

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

def predict_next_day(input_data):

    model = joblib.load(MODEL_PATH)

    df = pd.DataFrame([input_data])

    prediction = model.predict(df[features])[0]

    return prediction


if __name__ == "__main__":

    example_input = {
        "lag_1": 5,
        "lag_3": 4,
        "lag_7": 6,
        "lag_14": 3,
        "lag_28": 5,
        "rolling_7": 4.2,
        "rolling_14": 3.8,
        "rolling_28": 4.5,
        "promo_flag": 0,
        "promo_depth": 0,
        "stock_ratio": 1,
        "price": 2.5,
        "weekday": 2,
        "weekend": 0,
        "promo_price_interaction": 0,
        "lag1_stock_interaction": 5
    }

    pred = predict_next_day(example_input)

    print("Predicted next-day sales:", pred)