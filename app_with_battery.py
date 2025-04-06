from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import threading
import joblib
import os
from trainModel_with_battery import train_models

# 📌 Initialize Flask App
app = Flask(__name__)
CORS(app)

# 📁 Define BASE and MODEL directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(BASE_DIR, "dataset_battery_surge.csv")

# ✅ Load the Best Model
def load_best_model():
    try:
        best_model_path = os.path.join(MODEL_DIR, "best_model.txt")
        with open(best_model_path, "r") as f:
            best_model_name = f.read().strip()
        return joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}.pkl"))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# ✅ /predict Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        required_fields = ["distance_km", "traffic_factor", "weather_factor", "demand_supply_factor", "battery_percent"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # 🔄 Weather Mapping
        weather_mapping = {
            "clear": 1.0,
            "rainy": 1.2,
            "foggy": 1.3,
            "stormy": 1.5,
            "snowy": 1.7
        }

        weather_input = data["weather_factor"]
        if isinstance(weather_input, str):
            weather_value = weather_mapping.get(weather_input.lower())
            if weather_value is None:
                return jsonify({"error": f"Invalid weather factor: {weather_input}"}), 400
        elif isinstance(weather_input, (int, float)):
            weather_value = float(weather_input)
            if weather_value not in weather_mapping.values():
                return jsonify({"error": f"Invalid numeric weather factor: {weather_value}"}), 400
        else:
            return jsonify({"error": "Invalid format for weather factor"}), 400

        # 📌 Load Scalers
        scaler_path = os.path.join(MODEL_DIR, "input_scaler.pkl")
        fare_scaler_path = os.path.join(MODEL_DIR, "fare_scaler.pkl")

        if not os.path.exists(scaler_path):
            return jsonify({"error": "Missing input_scaler.pkl file"}), 500
        if not os.path.exists(fare_scaler_path):
            return jsonify({"error": "Missing fare_scaler.pkl file"}), 500

        scaler = joblib.load(scaler_path)
        fare_scaler = joblib.load(fare_scaler_path)

        # 📌 Normalize Input
        input_features = [[
            data["distance_km"],
            data["traffic_factor"],
            weather_value,
            data["demand_supply_factor"],
            data["battery_percent"]  # ✅ NEW field
        ]]
        input_data = scaler.transform(input_features)

        # 📌 Load Model & Predict
        model = load_best_model()
        if model is None:
            return jsonify({"error": "No trained model available"}), 500

        predicted_fare = model.predict(input_data)[0]
        predicted_fare = float(fare_scaler.inverse_transform([[predicted_fare]])[0][0])

        # 📌 Append to Dataset
        new_data = pd.DataFrame([{
            "distance_km": data["distance_km"],
            "traffic_factor": data["traffic_factor"],
            "weather_factor": weather_value,
            "demand_supply_factor": data["demand_supply_factor"],
            "battery_percent": data["battery_percent"],  # ✅ NEW field
            "fare": predicted_fare
        }])
        new_data.to_csv(DATASET_PATH, mode="a", header=False, index=False)

        # 🔁 Retrain Model in Background
        threading.Thread(target=train_models).start()

        return jsonify({
            "predicted_fare": predicted_fare,
            "message": "Data added! Model retraining in background.",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ✅ Run Flask Server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
