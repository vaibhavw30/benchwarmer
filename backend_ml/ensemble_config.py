"""
Ensemble Model Configuration

This module provides centralized configuration management for the ensemble model.
Modify these settings to tune ensemble behavior without retraining.
"""

import os
import json
import pandas as pd


DEFAULT_WEIGHTS = {
    "xgb_weight": 0.5,
    "ridge_weight": 0.5,
    "description": "Default 50/50 ensemble"
}


def load_ensemble_weights(weights_path="ensemble_weights.json"):
    """
    Load ensemble weights from file or environment variable.

    Priority:
    1. Environment variable (ENSEMBLE_WEIGHTS) - for production override
    2. Weights file (from training) - ensemble_weights.json
    3. Default weights - 50/50 split

    Args:
        weights_path (str): Path to the weights JSON file

    Returns:
        tuple: (xgb_weight, ridge_weight)
    """

    # Priority 1: Environment variable (for production override)
    env_weights = os.getenv("ENSEMBLE_WEIGHTS")
    if env_weights:
        try:
            weights = json.loads(env_weights)
            xgb_weight = weights.get('xgb_weight', 0.5)
            ridge_weight = weights.get('ridge_weight', 0.5)
            print(f"✓ Loaded ensemble weights from environment: XGB={xgb_weight:.1f}, Ridge={ridge_weight:.1f}")
            return xgb_weight, ridge_weight
        except json.JSONDecodeError:
            print(f"⚠️  Invalid ENSEMBLE_WEIGHTS format in environment, falling back...")

    # Priority 2: Weights file (from training)
    try:
        with open(weights_path, 'r') as f:
            weights = json.load(f)
        xgb_weight = weights.get('xgb_weight', 0.5)
        ridge_weight = weights.get('ridge_weight', 0.5)
        print(f"✓ Loaded ensemble weights from {weights_path}: XGB={xgb_weight:.1f}, Ridge={ridge_weight:.1f}")
        return xgb_weight, ridge_weight
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load weights file: {e}")

    # Priority 3: Default weights
    print(f"⚠️  Using default ensemble weights: 50/50")
    return DEFAULT_WEIGHTS['xgb_weight'], DEFAULT_WEIGHTS['ridge_weight']


def save_ensemble_weights(xgb_weight, ridge_weight, test_accuracy=None, weights_path="ensemble_weights.json"):
    """
    Save ensemble weights to file.

    Args:
        xgb_weight (float): Weight for XGBoost predictions (0-1)
        ridge_weight (float): Weight for Ridge predictions (0-1)
        test_accuracy (float, optional): Test set accuracy with these weights
        weights_path (str): Path to save the weights JSON file
    """

    config = {
        "xgb_weight": float(xgb_weight),
        "ridge_weight": float(ridge_weight),
        "test_accuracy": float(test_accuracy) if test_accuracy else None,
        "updated_at": str(pd.Timestamp.now())
    }

    with open(weights_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved ensemble weights to {weights_path}: XGB={xgb_weight:.1f}, Ridge={ridge_weight:.1f}")

    if test_accuracy:
        print(f"   Test Accuracy: {test_accuracy:.2%}")


def get_model_info(weights_path="ensemble_weights.json"):
    """
    Get information about the current ensemble configuration.

    Args:
        weights_path (str): Path to the weights JSON file

    Returns:
        dict: Configuration info including weights, accuracy, and last update time
    """

    try:
        with open(weights_path, 'r') as f:
            config = json.load(f)
        return config
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "xgb_weight": DEFAULT_WEIGHTS['xgb_weight'],
            "ridge_weight": DEFAULT_WEIGHTS['ridge_weight'],
            "test_accuracy": None,
            "updated_at": None
        }


# Model paths (used across train/predict/backtest)
MODEL_PATH = "xgboost_nba_model.pkl"
RIDGE_MODEL_PATH = "ridge_nba_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"


# Feature list (must match exactly across all modules)
FEATURES = [
    # --- ELO ---
    'ELO_H', 'ELO_A',

    # --- MISMATCHES ---
    'REB_MISMATCH', 'TOV_MISMATCH', 'SHOOTING_GAP',

    # --- HOME ---
    'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H',
    'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',

    # --- AWAY ---
    'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
    'FATIGUE_SCORE_A', 'MOMENTUM_A'
]


if __name__ == "__main__":
    # Demo: Display current configuration
    print("\n" + "="*60)
    print("ENSEMBLE MODEL CONFIGURATION")
    print("="*60)

    xgb_w, ridge_w = load_ensemble_weights()
    info = get_model_info()

    print(f"\nCurrent Weights:")
    print(f"  XGBoost:  {xgb_w:.1%}")
    print(f"  Ridge:    {ridge_w:.1%}")

    if info.get('test_accuracy'):
        print(f"\nTest Accuracy: {info['test_accuracy']:.2%}")

    if info.get('updated_at'):
        print(f"Last Updated: {info['updated_at']}")

    print(f"\nFeature Count: {len(FEATURES)}")
    print(f"Model Artifacts:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {RIDGE_MODEL_PATH}")
    print(f"  - {SCALER_PATH}")
    print(f"  - {ENSEMBLE_WEIGHTS_PATH}")

    print("\n" + "="*60)
