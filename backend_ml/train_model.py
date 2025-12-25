import pandas as pd
import joblib
import os
import sys
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = "xgboost_nba_model.pkl"

def train_and_optimize_model():
    """
    Trains and optimizes the NBA prediction model with comprehensive error handling.
    """
    try:
        from data_engine import build_training_dataset
    except ImportError as e:
        print(f"❌ Error importing data_engine: {e}")
        return False

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    try:
        if os.path.exists("nba_training_cache.csv"):
            print("📂 Loading data from cache...")
            df = pd.read_csv("nba_training_cache.csv")
            print(f"   Loaded {len(df)} games from cache")
        else:
            print("📡 No cache found, fetching fresh data...")
            df = build_training_dataset()

        if df.empty:
            print("❌ Error: Dataset is empty!")
            return False

    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return False

    # ============================================================================
    # 2. VALIDATE FEATURES
    # ============================================================================
    # UPDATED FEATURES LIST (FOUR FACTORS)
    features = [
        'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H', 
        'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',
        
        'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
        'FATIGUE_SCORE_A', 'MOMENTUM_A'
    ]
    target = 'HOME_WIN'

    missing_features = [f for f in features + [target] if f not in df.columns]
    if missing_features:
        print(f"❌ Missing required columns: {missing_features}")
        return False

    # ============================================================================
    # 3. PREPARE TRAIN/TEST SPLIT
    # ============================================================================
    try:
        df = df.sort_values('GAME_DATE_H')
        split_index = int(len(df) * 0.85)

        train_df = df.iloc[:split_index]
        test_df = df.iloc[split_index:]

        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        print(f"\n📊 Dataset Summary:")
        print(f"   Training: {len(X_train)} games")
        print(f"   Testing: {len(X_test)} games")

    except Exception as e:
        print(f"❌ Error preparing train/test split: {e}")
        return False

    # ============================================================================
    # 4. TRAIN MODEL
    # ============================================================================
    try:
        print(f"\n🧠 Training XGBoost Model...")

        param_grid = {
            'n_estimators': [150, 250],
            'learning_rate': [0.03, 0.05],
            'max_depth': [3, 4],
            'scale_pos_weight': [0.85, 0.9],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }

        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        tscv = TimeSeriesSplit(n_splits=3)

        print("🔎 Grid Search with Cross-Validation...")
        grid_search = GridSearchCV(xgb, param_grid, cv=tscv, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"\n🌟 Best Parameters: {grid_search.best_params_}")

    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

    # ============================================================================
    # 5. EVALUATE MODEL
    # ============================================================================
    try:
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        print(f"\n🎯 Final Model Accuracy: {acc:.2%}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Away Win', 'Home Win']))

        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\n🔍 Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['Feature']:25} {row['Importance']:.4f}")

    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        return False

    # ============================================================================
    # 6. SAVE MODEL
    # ============================================================================
    try:
        joblib.dump(best_model, MODEL_PATH)
        print(f"\n💾 Model saved to: {MODEL_PATH}")
        return True

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

if __name__ == "__main__":
    success = train_and_optimize_model()
    sys.exit(0 if success else 1)