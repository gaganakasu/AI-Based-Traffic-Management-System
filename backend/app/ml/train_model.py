import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .data_generator import generate_synthetic_data
except ImportError:
    from data_generator import generate_synthetic_data

def train_traffic_model(data_path="data/traffic_training_data.csv", model_dir="models"):
    """
    Loads dataset, trains RandomForestRegressor, evaluates performance, and serializes artifacts.
    """
    print("Checking if training data exists...")
    # Generate data if not present
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Generating synthetic dataset...")
        os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else ".", exist_ok=True)
        generate_synthetic_data(data_path, num_records=2000)
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows.")
    
    # Feature columns and Target column
    # We use numerical metrics of traffic to predict optimal green time (in seconds)
    features = ["vehicle_count", "traffic_density", "queue_length", "is_peak_hour"]
    target = "optimal_green_time"
    
    X = df[features]
    y = df[target]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Evaluation Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.3f} seconds")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.3f} seconds")
    print(f"  R-squared (R2): {r2:.3f}")
    
    # Save artifacts
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "traffic_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    metrics_path = os.path.join(model_dir, "model_metrics.json")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    metrics = {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "features": features,
        "training_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    # Adjust paths relative to root execution
    # Determine base directory path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_file = os.path.join(base_dir, "data", "traffic_training_data.csv")
    models_dir = os.path.join(base_dir, "models")
    train_traffic_model(data_file, models_dir)
