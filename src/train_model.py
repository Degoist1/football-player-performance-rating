"""
Model Training Module: Football Player Performance Rating System
Academic Mini Project - Experiment 12

This script:
1. Loads preprocessed training and testing data
2. Trains a Linear Regression model (Baseline)
3. Trains a Random Forest Regressor (Ensemble model)
4. Evaluates both models using MAE, MSE, RMSE, and R2 score
5. Compares both models side-by-side
6. Saves the trained model bundle into `models/player_rating_model.joblib`
"""

import os
import sys
import warnings
# Suppress benign macOS Accelerate BLAS matmul status flag warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_preprocess_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def evaluate_predictions(y_true, y_pred):
    """Calculates MAE, MSE, RMSE, and R2 metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

def train_and_evaluate(dataset_path="dataset/football_players.csv"):
    print("=" * 68)
    print(" STEP 4: TRAINING & EVALUATING MACHINE LEARNING MODELS")
    print("=" * 68)
    
    # 1. Load preprocessed train and test data
    X_train, X_test, y_train, y_test, preprocessor, df = load_and_preprocess_data(dataset_path)
    
    print(f"\n1. Dataset Loaded Successfully:")
    print(f"   • Training Samples : {X_train.shape[0]} (80%)")
    print(f"   • Testing Samples  : {X_test.shape[0]} (20%)")
    print(f"   • Features Count   : {X_train.shape[1]}")
    
    # 2. Train Model 1: Linear Regression (Baseline)
    print("\n2. Training Model 1: Linear Regression (Baseline)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = evaluate_predictions(y_test, lr_preds)
    print("   ✓ Linear Regression training complete.")
    
    # 3. Train Model 2: Random Forest Regressor (Ensemble Comparison)
    print("\n3. Training Model 2: Random Forest Regressor (100 Trees)...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_preds)
    print("   ✓ Random Forest Regressor training complete.")
    
    # 4. Display Evaluation Metrics Comparison Table
    print("\n" + "=" * 68)
    print("              MODEL PERFORMANCE COMPARISON ON TEST DATA")
    print("=" * 68)
    print(f"{'Evaluation Metric':<18} | {'Linear Regression':<22} | {'Random Forest':<20}")
    print("-" * 68)
    print(f"{'MAE (Rating pts)':<18} | {lr_metrics['MAE']:<22.4f} | {rf_metrics['MAE']:<20.4f}")
    print(f"{'MSE':<18} | {lr_metrics['MSE']:<22.4f} | {rf_metrics['MSE']:<20.4f}")
    print(f"{'RMSE (Rating pts)':<18} | {lr_metrics['RMSE']:<22.4f} | {rf_metrics['RMSE']:<20.4f}")
    print(f"{'R² Score (0 to 1)':<18} | {lr_metrics['R2']:<22.4f} | {rf_metrics['R2']:<20.4f}")
    print("=" * 68)
    
    # Determine the primary deployment model
    best_name = "Linear Regression" if lr_metrics["R2"] >= rf_metrics["R2"] else "Random Forest Regressor"
    best_model = lr_model if best_name == "Linear Regression" else rf_model
    best_metrics = lr_metrics if best_name == "Linear Regression" else rf_metrics
    
    print(f"\n>> Selected Best Model: {best_name}")
    print(f"   Accuracy / Explanation of Variance (R²): {best_metrics['R2']*100:.2f}%")
    print(f"   Average Prediction Error (MAE): ±{best_metrics['MAE']:.2f} rating points out of 10.0")
    
    # 5. Save Complete Model Bundle
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "preprocessor": preprocessor,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "lr_metrics": lr_metrics,
        "rf_metrics": rf_metrics,
        "y_test": y_test.values,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds
    }
    
    model_save_path = os.path.join("models", "player_rating_model.joblib")
    joblib.dump(model_bundle, model_save_path)
    print(f"\n>> Model bundle successfully saved to: {model_save_path}")
    print("=" * 68)
    
    return model_bundle

if __name__ == "__main__":
    train_and_evaluate()
