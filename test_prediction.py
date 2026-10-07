import requests

url = "http://127.0.0.1:5000/predict"

house = {
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 1800,
    "sqft_basement": 0,
    "yr_built": 2000,
    "yr_renovated": 0,
    "city": "Seattle",
    "statezip": "WA 98103"
}

response = requests.post(url, json=house)

print("Prediction Result:")
print(response.json())