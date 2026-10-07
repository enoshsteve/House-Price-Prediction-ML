from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "house_price_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "house_prices.csv"

# Load trained model
model = joblib.load(MODEL_PATH)

# Load dataset
data = pd.read_csv(DATA_PATH)

# Clean dataset
data = data[
    (data["price"] > 0) &
    (data["price"] <= 10000000)
].copy()


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# HOUSE PRICE PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    data_input = request.get_json()

    house_data = pd.DataFrame([{
        "bedrooms": data_input["bedrooms"],
        "bathrooms": data_input["bathrooms"],
        "sqft_living": data_input["sqft_living"],
        "sqft_lot": data_input["sqft_lot"],
        "floors": data_input["floors"],
        "waterfront": data_input["waterfront"],
        "view": data_input["view"],
        "condition": data_input["condition"],
        "sqft_above": data_input["sqft_above"],
        "sqft_basement": data_input["sqft_basement"],
        "yr_built": data_input["yr_built"],
        "yr_renovated": data_input["yr_renovated"],
        "city": data_input["city"],
        "statezip": data_input["statezip"]
    }])

    prediction = model.predict(house_data)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })


# =========================================================
# DATA VISUALIZATION / ANALYTICS
# =========================================================

@app.route("/analytics")
def analytics():

    # -----------------------------------------------------
    # 1. Price Distribution
    # -----------------------------------------------------

    price_hist = pd.cut(
        data["price"],
        bins=10
    ).value_counts().sort_index()

    price_distribution = []

    for interval, count in price_hist.items():

        price_distribution.append({
            "range": f"${interval.left:,.0f} - ${interval.right:,.0f}",
            "count": int(count)
        })


    # -----------------------------------------------------
    # 2. Bedrooms vs Average Price
    # -----------------------------------------------------

    bedrooms_data = (
        data.groupby("bedrooms")["price"]
        .mean()
        .reset_index()
    )

    bedrooms_vs_price = [
        {
            "bedrooms": int(row["bedrooms"]),
            "price": round(float(row["price"]), 2)
        }
        for _, row in bedrooms_data.iterrows()
        if row["bedrooms"] <= 10
    ]


    # -----------------------------------------------------
    # 3. Living Area vs Price
    # -----------------------------------------------------

    # Sample data so the browser doesn't receive thousands
    # of points.
    living_data = data[
        ["sqft_living", "price"]
    ].sample(
        min(500, len(data)),
        random_state=42
    )

    living_area_vs_price = [
        {
            "sqft": int(row["sqft_living"]),
            "price": round(float(row["price"]), 2)
        }
        for _, row in living_data.iterrows()
    ]


    # -----------------------------------------------------
    # 4. City-wise Average Price
    # -----------------------------------------------------

    city_data = (
        data.groupby("city")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    city_average_price = [
        {
            "city": row["city"],
            "price": round(float(row["price"]), 2)
        }
        for _, row in city_data.iterrows()
    ]


    # -----------------------------------------------------
    # 5. Feature Correlation
    # -----------------------------------------------------

    numerical_columns = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated"
    ]

    correlation = (
        data[numerical_columns]
        .corr()["price"]
        .drop("price")
        .sort_values(ascending=False)
    )

    feature_correlation = [
        {
            "feature": feature,
            "correlation": round(float(value), 3)
        }
        for feature, value in correlation.items()
    ]


    # -----------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------

    summary = {
        "total_houses": int(len(data)),
        "average_price": round(float(data["price"].mean()), 2),
        "minimum_price": round(float(data["price"].min()), 2),
        "maximum_price": round(float(data["price"].max()), 2)
    }


    # -----------------------------------------------------
    # Return Everything
    # -----------------------------------------------------

    return jsonify({
        "summary": summary,
        "price_distribution": price_distribution,
        "bedrooms_vs_price": bedrooms_vs_price,
        "living_area_vs_price": living_area_vs_price,
        "city_average_price": city_average_price,
        "feature_correlation": feature_correlation
    })


# =========================================================
# RUN FLASK SERVER
# =========================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)