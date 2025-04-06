import pandas as pd
import numpy as np

# Generate a large dataset
np.random.seed(42)
num_samples = 100000

data = {
    "distance_km": np.random.uniform(1, 50, num_samples),
    "traffic_factor": np.random.uniform(1, 2, num_samples),
    "weather_factor": np.random.choice(["Clear", "Rainy", "Foggy", "Stormy", "Snowy"], num_samples),
    "demand_supply_factor": np.random.uniform(0.8, 2, num_samples),
    "battery_percent": np.random.randint(5, 101, num_samples),  # Battery between 5–100
}

df = pd.DataFrame(data)

# Fare parameters
base_fare = 5
per_km_rate = 8

# Map weather
weather_mapping = {
    "Clear": 1.0,
    "Rainy": 1.2,
    "Foggy": 1.3,
    "Stormy": 1.5,
    "Snowy": 1.7
}
df["weather_numeric"] = df["weather_factor"].map(weather_mapping)

# Battery multiplier logic
def battery_multiplier(battery):
    if battery < 15:
        return 1.15
    elif battery < 30:
        return 1.10
    elif battery < 50:
        return 1.05
    else:
        return 1.0

df["battery_multiplier"] = df["battery_percent"].apply(battery_multiplier)

# Final fare
df["fare"] = (
    (base_fare + df["distance_km"] * per_km_rate)
    * df["traffic_factor"]
    * df["weather_numeric"]
    * df["demand_supply_factor"]
    * df["battery_multiplier"]
)

# Drop intermediate columns
df.drop(columns=["weather_numeric", "battery_multiplier"], inplace=True)

# Save to CSV
df.to_csv("dataset_battery_surge.csv", index=False)
print("✅ Dataset created with battery surge logic: dataset_battery_surge.csv")
