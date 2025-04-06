import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def plot_mae(mae_scores):
    plt.figure(figsize=(8, 5))
    plt.bar(mae_scores.keys(), mae_scores.values(), color=['blue', 'green', 'red'])
    plt.xlabel("Models")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Model Efficiency (Lower MAE is Better)")
    plt.ylim(min(mae_scores.values()) * 0.9, max(mae_scores.values()) * 1.1)
    plt.xticks(rotation=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("models/mae_comparison.png")
    print("📊 MAE comparison plot saved to models/mae_comparison.png")

def train_models():
    df = pd.read_csv("dataset_battery_surge.csv")

    print("🔍 Checking for missing values before processing...")
    print(df.isnull().sum())

    # Map categorical weather factor
    weather_mapping = {"Clear": 1.0, "Rainy": 1.2, "Foggy": 1.3, "Stormy": 1.5, "Snowy": 1.7}
    df["weather_factor"] = df["weather_factor"].map(weather_mapping).fillna(1.0)

    # Fill missing numerical values
    for col in ["distance_km", "traffic_factor", "demand_supply_factor", "battery_percent", "fare"]:
        df[col] = df[col].fillna(df[col].mean())

    df = df.astype({
        "distance_km": float,
        "traffic_factor": float,
        "weather_factor": float,
        "demand_supply_factor": float,
        "battery_percent": float,
        "fare": float
    })

    # Features and target
    X = df[["distance_km", "traffic_factor", "weather_factor", "demand_supply_factor", "battery_percent"]]
    y = df["fare"]

    # Double-check for any lingering nulls
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        print("❌ ERROR: NaN values detected after processing! Fixing...")
        X = X.fillna(0)
        y = y.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale input features
    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_test_scaled = input_scaler.transform(X_test)
    joblib.dump(input_scaler, "models/input_scaler.pkl")
    print("✅ input_scaler saved to models/input_scaler.pkl")

    # Scale output fare for MLP
    fare_scaler = StandardScaler()
    y_train_scaled = fare_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    joblib.dump(fare_scaler, "models/fare_scaler.pkl")
    print("✅ fare_scaler saved to models/fare_scaler.pkl")

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100),
        "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(64, 32, 16), max_iter=1000, random_state=42),
    }

    mae_scores = {}

    for name, model in models.items():
        print(f"🚀 Training {name} model...")
        
        if name == "NeuralNetwork":
            model.fit(X_train_scaled, y_train_scaled)
            predictions_scaled = model.predict(X_test_scaled)
            predictions = fare_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        else:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mae_scores[name] = mae

        joblib.dump(model, f"models/{name}.pkl")
        print(f"✅ Saved {name} model to models/{name}.pkl")

    best_model_name = min(mae_scores, key=mae_scores.get)
    print(f"\n✅ Best Model: {best_model_name} with MAE: {mae_scores[best_model_name]:.2f}")

    with open("models/best_model.txt", "w") as f:
        f.write(best_model_name)

    plot_mae(mae_scores)
    return best_model_name

if __name__ == "__main__":
    train_models()
