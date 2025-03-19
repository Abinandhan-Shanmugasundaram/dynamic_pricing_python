import pandas as pd
import numpy as np

# Generate a large dataset with 100,000 samples
np.random.seed(42)
num_samples = 100000

data = {
    "distance_km": np.random.uniform(1, 50, num_samples),
    "traffic_factor": np.random.uniform(1, 2, num_samples),
    "weather_factor": np.random.choice(["Clear", "Rainy", "Foggy", "Stormy", "Snowy"], num_samples),
    "demand_supply_factor": np.random.uniform(0.8, 2, num_samples),
}

df = pd.DataFrame(data)

# Fare calculation
base_fare = 5
per_km_rate = 8
weather_mapping = {"Clear": 1.0, "Rainy": 1.2, "Foggy": 1.3, "Stormy": 1.5, "Snowy": 1.7}
df["weather_numeric"] = df["weather_factor"].map(weather_mapping)

df["fare"] = (
    (base_fare + (df["distance_km"] * per_km_rate))
    * df["traffic_factor"]
    * df["weather_numeric"]
    * df["demand_supply_factor"]
)

df.drop(columns=["weather_numeric"], inplace=True)
df.to_csv("dataset.csv", index=False)

print("✅ Dataset with 100,000 rows created: dataset.csv")
