from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)


# Function to dynamically load model and scaler for the specific plant
def load_model_components(plant_name):
    model_path = os.path.join("model", plant_name, "best_soil_substitute_model.pkl")
    scaler_path = os.path.join("model", plant_name, "best_soil_substitute_scaler.pkl")
    feature_info_path = os.path.join("model", plant_name, "model_feature_info.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_info_path]):
        raise FileNotFoundError(f"Missing model components for plant: {plant_name}")

    model = joblib.load(model_path)  # Added this line
    scaler = joblib.load(scaler_path)
    feature_info = joblib.load(feature_info_path)
    
    # Return the loaded components
    return model, scaler, feature_info['feature_columns']

def preprocess_input(data, feature_columns):
    timestamp = pd.to_datetime(data["timestamp"])
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month
    day_of_year = timestamp.dayofyear

    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Convert string inputs to float
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])

    temp_humidity_interaction = temperature * humidity
    temp_squared = temperature ** 2
    humidity_squared = humidity ** 2

    # For real-time prediction, use provided values or training averages as fallbacks
    # Ideally, you should maintain a sliding window of recent readings
    temp_rolling_3 = float(data.get('temp_rolling_3', temperature))
    humidity_rolling_3 = float(data.get('humidity_rolling_3', humidity))
    temp_rolling_24 = float(data.get('temp_rolling_24', temperature))
    humidity_rolling_24 = float(data.get('humidity_rolling_24', humidity))
    temp_lag_1 = float(data.get('temp_lag_1', temperature))
    humidity_lag_1 = float(data.get('humidity_lag_1', humidity))


    # Assemble input dictionary
    input_dict = {
        'humidity': humidity,
        'temperature': temperature,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month': month,
        'day_of_year': day_of_year,
        'temp_humidity_interaction': temp_humidity_interaction,
        'temp_squared': temp_squared,
        'humidity_squared': humidity_squared,
        'temp_rolling_3': temp_rolling_3,
        'humidity_rolling_3': humidity_rolling_3,
        'temp_rolling_24': temp_rolling_24,
        'humidity_rolling_24': humidity_rolling_24,
        'temp_lag_1': temp_lag_1,
        'humidity_lag_1': humidity_lag_1
    }

    # Ensure correct feature order
    feature_values = [input_dict[feat] for feat in feature_columns]
    return pd.DataFrame([feature_values], columns=feature_columns)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not all(k in data for k in ["plant_name", "temperature", "humidity", "timestamp"]):
            return jsonify({"error": "Missing required keys"}), 400

        plant_name = data["plant_name"]
        model, scaler, feature_columns = load_model_components(plant_name)

        features = preprocess_input(data, feature_columns)
        features_scaled = scaler.transform(features)
        
        # Convert back to DataFrame with proper column names to avoid warning
        features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)
        
        prediction = model.predict(features_scaled_df)[0]

        return jsonify({
            "predicted_soil_moisture": round(float(prediction), 2),
            "timestamp": data["timestamp"]
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)