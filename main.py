from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Helper function to extract dynamic features
def extract_features(data):
    timestamp = pd.to_datetime(data["timestamp"])
    timestamp_numeric = timestamp.timestamp()
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek

    # Assume no prior change for real-time data (or it could come dynamically)
    soil_moisture_change = 0
    watering_event = 1 if soil_moisture_change > 10 else 0

    # Ensure consistent feature order
    feature_names = ["temperature", "humidity", "timestamp_numeric", 
                     "soil_moisture_change", "hour", "dayofweek", "watering_event"]

    # Return a DataFrame with correct feature names
    features = pd.DataFrame([[data["temperature"], data["humidity"], timestamp_numeric, 
                              soil_moisture_change, hour, dayofweek, watering_event]],
                            columns=feature_names)

    return features

# Function to dynamically load model and scaler for the specific plant
def load_model_and_scaler(plant_name):
    model_path = os.path.join("model", plant_name, "soil_moisture_model.pkl")
    scaler_path = os.path.join("model", plant_name, "soil_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for plant: {plant_name}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Input validation
        if "plant_name" not in data or "temperature" not in data or "humidity" not in data or "timestamp" not in data:
            return jsonify({"error": "Missing plant_name, temperature, humidity, or timestamp"}), 400

        plant_name = data["plant_name"]

        # Load model and scaler for the specific plant
        model, scaler = load_model_and_scaler(plant_name)

        # Extract features and maintain feature names
        features = extract_features(data)

        # Scale input using the fitted scaler
        features_scaled = scaler.transform(features)

        # Predict soil moisture using the Random Forest model
        predicted_scaled = model.predict(features_scaled)[0]

        # Inverse transform to get the original soil moisture
        original_features = np.zeros((1, 7))  # Placeholder array
        original_features[0, -1] = predicted_scaled  # Set predicted value
        predicted_soil_moisture = scaler.inverse_transform(original_features)[0][-1]

        return jsonify({
            "predicted_soil_moisture": round(float(predicted_soil_moisture), 2),
            "timestamp": data["timestamp"]
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
