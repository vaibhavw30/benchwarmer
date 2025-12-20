# backend_ml/train_model.py

import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_engine import build_training_dataset # Import from our Step 1 file

# CONFIG
MODEL_PATH = "xgboost_nba_model.pkl"

def train_and_save_model():
    # 1. Get Data
    df = build_training_dataset()
    
    # 2. Define Features (X) and Target (y)
    # These must match exactly what we will generate in predict.py later!
    features = [
        'PTS_ROLLING_H', 'FG_PCT_ROLLING_H', 'DAYS_REST_H',
        'PTS_ROLLING_A', 'FG_PCT_ROLLING_A', 'DAYS_REST_A'
    ]
    target = 'HOME_WIN'
    
    X = df[features]
    y = df[target]
    
    # 3. Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train XGBoost
    print("🧠 Training XGBoost Model...")
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"🎯 Model Accuracy: {acc:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 6. Save
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()