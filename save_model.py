import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# Load dataset
data = pd.read_csv("dataset/house_prices.csv")

print("Dataset loaded!")

# Remove invalid prices
data = data[
    (data["price"] > 0) &
    (data["price"] <= 10000000)
].copy()

print("Valid houses:", len(data))


# Features
features = [
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
    "yr_renovated",
    "city",
    "statezip"
]

X = data[features]
y = data["price"]


# Categorical and numerical columns
categorical_features = [
    "city",
    "statezip"
]

numerical_features = [
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


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Linear Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# Train using complete cleaned dataset
model.fit(X, y)


# Save model
joblib.dump(model, "model/house_price_model.pkl")

print()
print("======================================")
print("MODEL TRAINING COMPLETED!")
print("Model saved successfully!")
print("Location: model/house_price_model.pkl")
print("======================================")