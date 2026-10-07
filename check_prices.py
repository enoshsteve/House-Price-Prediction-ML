import pandas as pd

data = pd.read_csv("dataset/house_prices.csv")

print("Above 2M:", (data["price"] > 2000000).sum())
print("Above 5M:", (data["price"] > 5000000).sum())
print("Above 10M:", (data["price"] > 10000000).sum())