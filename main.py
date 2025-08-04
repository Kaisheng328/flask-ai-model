from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import os
from datetime import datetime
import traceback
from werkzeug.utils import secure_filename
import tempfile
import shutil

app = Flask(__name__)

# Configuration
MODEL_BASE_DIR = "model"
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_model_directory(plant_name):
    """Create directory structure for plant model"""
    plant_dir = os.path.join(MODEL_BASE_DIR, plant_name)
    os.makedirs(plant_dir, exist_ok=True)
    return plant_dir

def preprocess_data_for_training(df):
    """Enhanced data preprocessing with feature engineering for training"""
    try:
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # Create cyclical features for better time representation
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Feature Engineering
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['temp_squared'] = df['temperature'] ** 2
        df['humidity_squared'] = df['humidity'] ** 2

        # Rolling averages (if data is chronologically ordered)
        df = df.sort_values('timestamp')
        df['temp_rolling_3'] = df['temperature'].rolling(window=3, min_periods=1).mean()
        df['humidity_rolling_3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
        df['temp_rolling_24'] = df['temperature'].rolling(window=24, min_periods=1).mean()
        df['humidity_rolling_24'] = df['humidity'].rolling(window=24, min_periods=1).mean()

        # Lag features
        df['temp_lag_1'] = df['temperature'].shift(1)
        df['humidity_lag_1'] = df['humidity'].shift(1)

        # Drop rows with NaN values created by lag features
        df = df.dropna()

        return df
    except Exception as e:
        raise Exception(f"Error in data preprocessing: {str(e)}")

def train_models(df, plant_name, training_start=None):
    """Train and evaluate multiple models"""
    try:
        # Feature columns
        feature_columns = [
            'humidity', 'temperature',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'month', 'day_of_year',
            'temp_humidity_interaction', 'temp_squared', 'humidity_squared',
            'temp_rolling_3', 'humidity_rolling_3',
            'temp_rolling_24', 'humidity_rolling_24',
            'temp_lag_1', 'humidity_lag_1'
        ]

        features = df[feature_columns]
        target = df['soil_moisture']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, shuffle=True
        )

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter tuning for Random Forest
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=3,  # Reduced for faster training
            scoring='r2',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        best_rf = rf_grid.best_estimator_

        # Hyperparameter tuning for Gradient Boosting
        gb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        gb_grid = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            gb_param_grid,
            cv=3,  # Reduced for faster training
            scoring='r2',
            n_jobs=-1
        )
        gb_grid.fit(X_train, y_train)
        best_gb = gb_grid.best_estimator_

        # Final model evaluation
        models_final = {
            'Tuned Random Forest': best_rf,
            'Tuned Gradient Boosting': best_gb
        }

        best_model = None
        best_score = -np.inf
        best_name = ""
        results = {}

        for name, model in models_final.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            }

            if r2 > best_score:
                best_score = r2
                best_model = model
                best_name = name

        # Save the best model
        plant_dir = create_model_directory(plant_name)
        
        model_path = os.path.join(plant_dir, "best_soil_substitute_model.pkl")
        scaler_path = os.path.join(plant_dir, "best_soil_substitute_scaler.pkl")
        feature_info_path = os.path.join(plant_dir, "model_feature_info.pkl")
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Get current timestamp
        current_time = datetime.now()
        if training_start:
            training_end = datetime.now()
            training_duration = training_end - training_start
            training_time = str(training_duration)
            training_duration_seconds = training_duration.total_seconds()
        else:
            training_time = None
            training_duration_seconds = None
        
        feature_info = {
            'feature_columns': feature_columns,
            'scaler_needed': True,
            'plant_name': plant_name,
            'created_at': current_time.isoformat(),
            'created_timestamp': current_time.timestamp(),
            'created_readable': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_date': current_time.isoformat(),  # Keep for backward compatibility
            'model_type': best_name,
            'performance': results[best_name],
            'training_duration': training_time,
            'training_duration_seconds': training_duration_seconds,
            'data_points_used': len(df)
        }
        joblib.dump(feature_info, feature_info_path)

        return {
            'success': True,
            'best_model': best_name,
            'r2_score': best_score,
            'rmse': results[best_name]['RMSE'],
            'mae': results[best_name]['MAE'],
            'model_path': plant_dir,
            'all_results': results,
            'created_at': current_time.isoformat(),
            'created_readable': current_time.strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        raise Exception(f"Error in model training: {str(e)}")

# Function to dynamically load model and scaler for the specific plant
def load_model_components(plant_name):
    model_path = os.path.join(MODEL_BASE_DIR, plant_name, "best_soil_substitute_model.pkl")
    scaler_path = os.path.join(MODEL_BASE_DIR, plant_name, "best_soil_substitute_scaler.pkl")
    feature_info_path = os.path.join(MODEL_BASE_DIR, plant_name, "model_feature_info.pkl")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, feature_info_path]):
        raise FileNotFoundError(f"Missing model components for plant: {plant_name}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_info = joblib.load(feature_info_path)
    
    return model, scaler, feature_info['feature_columns']

def preprocess_input_for_prediction(data, feature_columns):
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

# =================== TRAINING ENDPOINTS ===================

@app.route('/train', methods=['POST'])
def train_model():
    """Train model endpoint"""
    try:
        # Check if plant_name is provided
        if 'plant_name' not in request.form:
            return jsonify({'error': 'Plant name is required'}), 400
        
        plant_name = request.form['plant_name'].strip()
        if not plant_name:
            return jsonify({'error': 'Plant name cannot be empty'}), 400

        # Check if CSV file is provided
        if 'csv_file' not in request.files:
            return jsonify({'error': 'CSV file is required'}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400

        # Create temporary file to save uploaded CSV
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.csv', delete=False) as temp_file:
            file.save(temp_file.name)
            temp_filename = temp_file.name

        try:
            # Read CSV data
            df = pd.read_csv(temp_filename)
            
            # Validate required columns
            required_columns = ['timestamp', 'temperature', 'humidity', 'soil_moisture']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Missing required columns: {missing_columns}'
                }), 400

            # Check if we have enough data
            if len(df) < 50:
                return jsonify({
                    'error': 'Insufficient data for training. At least 50 records are required'
                }), 400

            # Preprocess data
            df_processed = preprocess_data_for_training(df)
            
            if len(df_processed) < 30:
                return jsonify({
                    'error': 'Insufficient data after preprocessing. More historical data needed'
                }), 400

            # Train models
            training_start = datetime.now()
            results = train_models(df_processed, plant_name, training_start)
            training_end = datetime.now()
            
            training_duration = training_end - training_start
            training_time = str(training_duration)
            
            # Add timing and metadata to results
            results['training_time'] = training_time
            results['training_duration_seconds'] = training_duration.total_seconds()
            results['created_at'] = training_start.isoformat()
            results['completed_at'] = training_end.isoformat()
            results['created_readable'] = training_start.strftime('%Y-%m-%d %H:%M:%S')
            results['data_points'] = len(df_processed)
            results['original_data_points'] = len(df)
            results['message'] = f'Model trained successfully for {plant_name}'

            return jsonify(results), 200

        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

    except Exception as e:
        return jsonify({
            'error': 'Training failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

# =================== PREDICTION ENDPOINTS ===================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        if not all(k in data for k in ["plant_name", "temperature", "humidity", "timestamp"]):
            return jsonify({"error": "Missing required keys"}), 400

        plant_name = data["plant_name"]
        
        # Load model components for the specific plant
        try:
            model, scaler, feature_columns = load_model_components(plant_name)
        except FileNotFoundError as e:
            return jsonify({
                "error": f"Model not found for plant '{plant_name}'. Please train the model first.",
                "details": str(e)
            }), 404

        # Preprocess input data
        features = preprocess_input_for_prediction(data, feature_columns)
        features_scaled = scaler.transform(features)
        
        # Convert back to DataFrame with proper column names to avoid warning
        features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(features_scaled_df)[0]

        return jsonify({
            "plant_name": plant_name,
            "predicted_soil_moisture": round(float(prediction), 2),
            "timestamp": data["timestamp"],
            "input_data": {
                "temperature": data["temperature"],
                "humidity": data["humidity"]
            }
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except ValueError as e:
        return jsonify({
            "error": "Invalid input data",
            "details": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction endpoint for multiple data points"""
    try:
        data = request.json
        
        if "plant_name" not in data or "predictions" not in data:
            return jsonify({"error": "Missing required keys: plant_name, predictions"}), 400
        
        plant_name = data["plant_name"]
        predictions_data = data["predictions"]
        
        if not isinstance(predictions_data, list):
            return jsonify({"error": "predictions must be a list"}), 400
        
        # Load model components
        try:
            model, scaler, feature_columns = load_model_components(plant_name)
        except FileNotFoundError as e:
            return jsonify({
                "error": f"Model not found for plant '{plant_name}'. Please train the model first.",
                "details": str(e)
            }), 404
        
        results = []
        
        for i, pred_data in enumerate(predictions_data):
            try:
                # Validate required fields for each prediction
                if not all(k in pred_data for k in ["temperature", "humidity", "timestamp"]):
                    results.append({
                        "index": i,
                        "error": "Missing required keys: temperature, humidity, timestamp"
                    })
                    continue
                
                # Preprocess and predict
                features = preprocess_input_for_prediction(pred_data, feature_columns)
                features_scaled = scaler.transform(features)
                features_scaled_df = pd.DataFrame(features_scaled, columns=feature_columns)
                prediction = model.predict(features_scaled_df)[0]
                
                results.append({
                    "index": i,
                    "predicted_soil_moisture": round(float(prediction), 2),
                    "timestamp": pred_data["timestamp"],
                    "input_data": {
                        "temperature": pred_data["temperature"],
                        "humidity": pred_data["humidity"]
                    }
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "plant_name": plant_name,
            "predictions": results,
            "total_predictions": len(predictions_data),
            "successful_predictions": len([r for r in results if "error" not in r])
        })
        
    except Exception as e:
        return jsonify({
            "error": "Batch prediction failed",
            "details": str(e)
        }), 500

# =================== MODEL MANAGEMENT ENDPOINTS ===================

@app.route('/model/status/<plant_name>', methods=['GET'])
def get_model_status(plant_name):
    """Check if model exists for a plant"""
    try:
        plant_dir = os.path.join(MODEL_BASE_DIR, plant_name)
        
        required_files = [
            "best_soil_substitute_model.pkl",
            "best_soil_substitute_scaler.pkl", 
            "model_feature_info.pkl"
        ]
        
        files_exist = all(
            os.path.exists(os.path.join(plant_dir, filename)) 
            for filename in required_files
        )
        
        if not files_exist:
            return jsonify({
                'exists': False,
                'plant_name': plant_name,
                'message': 'Model not found'
            }), 404
        
        # Load model info
        feature_info_path = os.path.join(plant_dir, "model_feature_info.pkl")
        feature_info = joblib.load(feature_info_path)
        
        return jsonify({
            'exists': True,
            'plant_name': plant_name,
            'model_info': feature_info,
            'model_path': plant_dir
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to check model status',
            'details': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List all available trained models"""
    try:
        if not os.path.exists(MODEL_BASE_DIR):
            return jsonify({'models': []}), 200
        
        models = []
        for plant_dir in os.listdir(MODEL_BASE_DIR):
            plant_path = os.path.join(MODEL_BASE_DIR, plant_dir)
            if os.path.isdir(plant_path):
                feature_info_path = os.path.join(plant_path, "model_feature_info.pkl")
                if os.path.exists(feature_info_path):
                    try:
                        feature_info = joblib.load(feature_info_path)
                        created_at = feature_info.get('created_at')
                        created_readable = feature_info.get('created_readable')
                        
                        # Handle old models without created_at
                        if not created_at and 'training_date' in feature_info:
                            try:
                                # Try to parse old training_date
                                old_date = datetime.fromisoformat(feature_info['training_date'].replace('Z', '+00:00'))
                                created_at = old_date.isoformat()
                                created_readable = old_date.strftime('%Y-%m-%d %H:%M:%S')
                            except:
                                created_at = feature_info.get('training_date', 'Unknown')
                                created_readable = 'Unknown'
                        
                        models.append({
                            'plant_name': plant_dir,
                            'created_at': created_at,
                            'created_readable': created_readable,
                            'training_date': feature_info.get('training_date'),  # Keep for compatibility
                            'model_type': feature_info.get('model_type'),
                            'performance': feature_info.get('performance', {}),
                            'data_points_used': feature_info.get('data_points_used'),
                            'training_duration': feature_info.get('training_duration')
                        })
                    except:
                        # Skip if can't load model info, but still add basic info
                        models.append({
                            'plant_name': plant_dir,
                            'created_at': 'Unknown',
                            'created_readable': 'Unknown',
                            'training_date': 'Unknown',
                            'model_type': 'Unknown',
                            'performance': {}
                        })
        
        # Sort models by creation time (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return jsonify({
            'models': models,
            'count': len(models)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list models',
            'details': str(e)
        }), 500

@app.route('/models/sorted', methods=['GET'])
def list_models_sorted():
    """List all available trained models with sorting options"""
    try:
        sort_by = request.args.get('sort_by', 'created_at')  # created_at, plant_name, performance
        order = request.args.get('order', 'desc')  # asc, desc
        
        if not os.path.exists(MODEL_BASE_DIR):
            return jsonify({'models': []}), 200
        
        models = []
        for plant_dir in os.listdir(MODEL_BASE_DIR):
            plant_path = os.path.join(MODEL_BASE_DIR, plant_dir)
            if os.path.isdir(plant_path):
                feature_info_path = os.path.join(plant_path, "model_feature_info.pkl")
                if os.path.exists(feature_info_path):
                    try:
                        feature_info = joblib.load(feature_info_path)
                        created_at = feature_info.get('created_at')
                        created_readable = feature_info.get('created_readable')
                        created_timestamp = feature_info.get('created_timestamp', 0)
                        
                        # Handle old models without created_at
                        if not created_at and 'training_date' in feature_info:
                            try:
                                old_date = datetime.fromisoformat(feature_info['training_date'].replace('Z', '+00:00'))
                                created_at = old_date.isoformat()
                                created_readable = old_date.strftime('%Y-%m-%d %H:%M:%S')
                                created_timestamp = old_date.timestamp()
                            except:
                                created_at = feature_info.get('training_date', 'Unknown')
                                created_readable = 'Unknown'
                                created_timestamp = 0
                        
                        performance = feature_info.get('performance', {})
                        r2_score = performance.get('R²', 0) if performance else 0
                        
                        models.append({
                            'plant_name': plant_dir,
                            'created_at': created_at,
                            'created_readable': created_readable,
                            'created_timestamp': created_timestamp,
                            'training_date': feature_info.get('training_date'),
                            'model_type': feature_info.get('model_type'),
                            'performance': performance,
                            'r2_score': r2_score,
                            'data_points_used': feature_info.get('data_points_used'),
                            'training_duration': feature_info.get('training_duration')
                        })
                    except:
                        models.append({
                            'plant_name': plant_dir,
                            'created_at': 'Unknown',
                            'created_readable': 'Unknown',
                            'created_timestamp': 0,
                            'training_date': 'Unknown',
                            'model_type': 'Unknown',
                            'performance': {},
                            'r2_score': 0
                        })
        
        # Sort models based on parameters
        reverse_order = order.lower() == 'desc'
        
        if sort_by == 'created_at':
            models.sort(key=lambda x: x.get('created_timestamp', 0), reverse=reverse_order)
        elif sort_by == 'plant_name':
            models.sort(key=lambda x: x.get('plant_name', '').lower(), reverse=reverse_order)
        elif sort_by == 'performance':
            models.sort(key=lambda x: x.get('r2_score', 0), reverse=reverse_order)
        elif sort_by == 'data_points':
            models.sort(key=lambda x: x.get('data_points_used', 0), reverse=reverse_order)
        
        return jsonify({
            'models': models,
            'count': len(models),
            'sorted_by': sort_by,
            'order': order
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to list models',
            'details': str(e)
        }), 500

@app.route('/models/recent', methods=['GET'])
def list_recent_models():
    """Get recently trained models (last 10)"""
    try:
        limit = int(request.args.get('limit', 10))
        
        if not os.path.exists(MODEL_BASE_DIR):
            return jsonify({'models': []}), 200
        
        models = []
        for plant_dir in os.listdir(MODEL_BASE_DIR):
            plant_path = os.path.join(MODEL_BASE_DIR, plant_dir)
            if os.path.isdir(plant_path):
                feature_info_path = os.path.join(plant_path, "model_feature_info.pkl")
                if os.path.exists(feature_info_path):
                    try:
                        feature_info = joblib.load(feature_info_path)
                        created_timestamp = feature_info.get('created_timestamp', 0)
                        
                        # Handle old models
                        if not created_timestamp and 'training_date' in feature_info:
                            try:
                                old_date = datetime.fromisoformat(feature_info['training_date'].replace('Z', '+00:00'))
                                created_timestamp = old_date.timestamp()
                            except:
                                created_timestamp = 0
                        
                        models.append({
                            'plant_name': plant_dir,
                            'created_at': feature_info.get('created_at'),
                            'created_readable': feature_info.get('created_readable'),
                            'created_timestamp': created_timestamp,
                            'model_type': feature_info.get('model_type'),
                            'performance': feature_info.get('performance', {}),
                            'data_points_used': feature_info.get('data_points_used')
                        })
                    except:
                        continue
        
        # Sort by creation time (newest first) and limit
        models.sort(key=lambda x: x.get('created_timestamp', 0), reverse=True)
        recent_models = models[:limit]
        
        return jsonify({
            'models': recent_models,
            'count': len(recent_models),
            'total_models': len(models),
            'limit': limit
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get recent models',
            'details': str(e)
        }), 500

@app.route('/models/stats', methods=['GET'])
def get_models_stats():
    """Get statistics about all trained models"""
    try:
        if not os.path.exists(MODEL_BASE_DIR):
            return jsonify({
                'total_models': 0,
                'model_types': {},
                'avg_performance': 0,
                'best_model': None,
                'oldest_model': None,
                'newest_model': None
            }), 200
        
        models = []
        for plant_dir in os.listdir(MODEL_BASE_DIR):
            plant_path = os.path.join(MODEL_BASE_DIR, plant_dir)
            if os.path.isdir(plant_path):
                feature_info_path = os.path.join(plant_path, "model_feature_info.pkl")
                if os.path.exists(feature_info_path):
                    try:
                        feature_info = joblib.load(feature_info_path)
                        created_timestamp = feature_info.get('created_timestamp', 0)
                        
                        if not created_timestamp and 'training_date' in feature_info:
                            try:
                                old_date = datetime.fromisoformat(feature_info['training_date'].replace('Z', '+00:00'))
                                created_timestamp = old_date.timestamp()
                            except:
                                created_timestamp = 0
                        
                        performance = feature_info.get('performance', {})
                        r2_score = performance.get('R²', 0) if performance else 0
                        
                        models.append({
                            'plant_name': plant_dir,
                            'created_timestamp': created_timestamp,
                            'created_readable': feature_info.get('created_readable'),
                            'model_type': feature_info.get('model_type'),
                            'r2_score': r2_score,
                            'performance': performance
                        })
                    except:
                        continue
        
        if not models:
            return jsonify({
                'total_models': 0,
                'model_types': {},
                'avg_performance': 0,
                'best_model': None,
                'oldest_model': None,
                'newest_model': None
            }), 200
        
        # Calculate statistics
        total_models = len(models)
        model_types = {}
        r2_scores = []
        
        for model in models:
            model_type = model['model_type']
            if model_type:
                model_types[model_type] = model_types.get(model_type, 0) + 1
            
            if model['r2_score'] > 0:
                r2_scores.append(model['r2_score'])
        
        avg_performance = sum(r2_scores) / len(r2_scores) if r2_scores else 0
        
        # Find best, oldest, newest models
        best_model = max(models, key=lambda x: x['r2_score']) if models else None
        oldest_model = min(models, key=lambda x: x['created_timestamp']) if models else None
        newest_model = max(models, key=lambda x: x['created_timestamp']) if models else None
        
        return jsonify({
            'total_models': total_models,
            'model_types': model_types,
            'avg_performance': round(avg_performance, 3),
            'best_model': {
                'plant_name': best_model['plant_name'],
                'r2_score': best_model['r2_score'],
                'model_type': best_model['model_type'],
                'created_readable': best_model['created_readable']
            } if best_model else None,
            'oldest_model': {
                'plant_name': oldest_model['plant_name'],
                'created_readable': oldest_model['created_readable']
            } if oldest_model else None,
            'newest_model': {
                'plant_name': newest_model['plant_name'],
                'created_readable': newest_model['created_readable']
            } if newest_model else None
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get model statistics',
            'details': str(e)
        }), 500
def delete_model(plant_name):
    """Delete a trained model"""
    try:
        plant_dir = os.path.join(MODEL_BASE_DIR, plant_name)
        
        if not os.path.exists(plant_dir):
            return jsonify({
                'error': 'Model not found'
            }), 404
        
        # Remove the entire plant directory
        shutil.rmtree(plant_dir)
        
        return jsonify({
            'message': f'Model for {plant_name} deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete model',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ml-combined-service',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == "__main__":
    # Create model base directory if it doesn't exist
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=8080, debug=True)