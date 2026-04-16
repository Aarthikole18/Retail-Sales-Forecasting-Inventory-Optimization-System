from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import create_features
from src.model import train_model, predict
from src.inventory import calculate_inventory

# Step 1: Load data
df = load_data("data/retail_data.csv")

# Step 2: Clean data
df = clean_data(df)

# Step 3: Feature Engineering
df = create_features(df)

# Step 4: Features & target
X = df[["lag_1", "lag_2", "rolling_mean_3", "day_of_week"]]
y = df["qty_sold"]

# Step 5: Train model
model = train_model(X, y)

# Step 6: Predict
predictions = predict(model, X)

# Step 7: Add predictions
df["predicted_sales"] = predictions

# Step 8: Inventory calculation
safety_stock, reorder_point = calculate_inventory(predictions, y)

# Step 9: Print results
print("\n📊 Forecast Output:")
print(df[["date", "qty_sold", "predicted_sales"]].to_string(index=False))

print("\n📦 Inventory Recommendation:")
print(f"✅ Safety Stock: {round(safety_stock,2)} units")
print(f"✅ Reorder Point: {round(reorder_point,2)} units")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(df["date"], df["qty_sold"], label="Actual Sales")
plt.plot(df["date"], df["predicted_sales"], label="Predicted Sales")

plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast vs Actual")
plt.legend()

plt.savefig("outputs/forecast_plot.png")
plt.show()