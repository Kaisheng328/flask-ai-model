from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
# Load the trained model and scaler
model = joblib.load("soil_moisture_model.pkl")  # Load XGBoost model
scaler = joblib.load("soil_scaler.pkl")  # Load MinMaxScaler

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "temperature" not in data or "humidity" not in data or "timestamp" not in data:
        return jsonify({"error": "Missing temperature, humidity, or timestamp"}), 400

    try:
        # Convert timestamp to numerical format (Unix timestamp)
        timestamp_numeric = pd.to_datetime(data["timestamp"]).timestamp()

        # Prepare input for the model
        features = np.array([[data["temperature"], data["humidity"], timestamp_numeric]])

        # Predict soil moisture (scaled value)
        predicted_scaled = model.predict(features)[0]

        # Convert back to original soil moisture range
        predicted_soil_moisture = scaler.inverse_transform(np.array([[predicted_scaled]]))[0][0]

        return jsonify({
           "predicted_soil_moisture": round(float(predicted_soil_moisture), 2),

            "timestamp": data["timestamp"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
