import pandas as pd
import joblib
import os
import sys
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV

MODEL_PATH = "xgboost_nba_model.pkl"
RIDGE_MODEL_PATH = "ridge_nba_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"

def save_artifacts(xgb_model, ridge_model, scaler, weights_config, output_dir="."):
    """Write the 4 model artifacts to output_dir (default: cwd, as always)."""
    joblib.dump(xgb_model, os.path.join(output_dir, MODEL_PATH))
    joblib.dump(ridge_model, os.path.join(output_dir, RIDGE_MODEL_PATH))
    joblib.dump(scaler, os.path.join(output_dir, SCALER_PATH))
    with open(os.path.join(output_dir, ENSEMBLE_WEIGHTS_PATH), 'w') as f:
        json.dump(weights_config, f, indent=2)

def build_weights_config(xgb_weight, ridge_weight, test_accuracy, test_brier, train_date=None):
    """Assemble the ensemble_weights.json payload.

    test_brier is the held-out ensemble Brier score; it is the baseline the
    nightly drift gate compares against (see scheduled_retrain.measure_drift).
    """
    return {
        "xgb_weight": xgb_weight,
        "ridge_weight": ridge_weight,
        "test_accuracy": float(test_accuracy),
        "test_brier": float(test_brier),
        "train_date": str(train_date if train_date is not None else pd.Timestamp.now()),
    }

def train_and_optimize_model(output_dir="."):
    try:
        from data_engine import load_or_build_training_dataset
    except ImportError as e:
        print(f"❌ Error importing data_engine: {e}")
        return False

    # 1. LOAD DATA
    try:
        df = load_or_build_training_dataset()

        if df.empty:
            print("❌ Error: Dataset is empty!")
            return False

    except Exception as e:
        print(f"❌ Unexpected error loading data: {e}")
        return False

    # 2. VALIDATE FEATURES
    features = [
        # --- ELO ---
        'ELO_H', 'ELO_A',
        
        # --- NEW MISMATCHES ---
        'REB_MISMATCH', 'TOV_MISMATCH', 'SHOOTING_GAP',
        
        # --- HOME ---
        'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H', 
        'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',
        
        # --- AWAY ---
        'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
        'FATIGUE_SCORE_A', 'MOMENTUM_A'
    ]
    target = 'HOME_WIN'

    missing_features = [f for f in features + [target] if f not in df.columns]
    if missing_features:
        print(f"❌ Missing required columns: {missing_features}")
        print("💡 Solution: Delete 'nba_training_cache.csv' and run again.")
        return False

    df[features] = df[features].fillna(df[features].mean())

    # 3. SPLIT
    df = df.sort_values('GAME_DATE_H')
    split_index = int(len(df) * 0.85)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"\n📊 Training on {len(X_train)} games...")

    # 3B. SCALE FEATURES FOR RIDGE
    # XGBoost doesn't need scaling, but Ridge does
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("📏 Scaled features for Ridge (mean=0, std=1)")

    # 4. TRAIN
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

    print("🔎 Grid Search Tuning...")
    grid_search = GridSearchCV(xgb, param_grid, cv=tscv, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"🌟 Best Params: {grid_search.best_params_}")

    # 4B. TRAIN RIDGE REGRESSION
    print("\n🔎 Training Ridge Regression with Cross-Validation...")

    # RidgeClassifierCV automatically tunes alpha via CV
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    ridge = RidgeClassifierCV(
        alphas=alphas,
        cv=tscv,  # Use same TimeSeriesSplit as XGBoost
        scoring='accuracy'
    )
    ridge.fit(X_train_scaled, y_train)

    print(f"🌟 Best Ridge Alpha: {ridge.alpha_}")

    # 5. EVALUATE BOTH MODELS
    print("\n" + "="*80)
    print("📊 MODEL EVALUATION")
    print("="*80)

    best_model = grid_search.best_estimator_

    # XGBoost Evaluation
    xgb_preds = best_model.predict(X_test)
    xgb_probs = best_model.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_auc = roc_auc_score(y_test, xgb_probs[:, 1])
    xgb_logloss = log_loss(y_test, xgb_probs)

    print(f"\n🤖 XGBoost Performance:")
    print(f"   Accuracy: {xgb_acc:.2%}")
    print(f"   ROC-AUC: {xgb_auc:.4f}")
    print(f"   Log Loss: {xgb_logloss:.4f}")

    # Ridge Evaluation
    ridge_preds = ridge.predict(X_test_scaled)
    # Ridge doesn't have predict_proba, use decision_function + sigmoid
    ridge_decision = ridge.decision_function(X_test_scaled)
    ridge_probs = 1 / (1 + np.exp(-ridge_decision))  # Sigmoid transform
    ridge_probs_2d = np.column_stack([1 - ridge_probs, ridge_probs])

    ridge_acc = accuracy_score(y_test, ridge_preds)
    ridge_auc = roc_auc_score(y_test, ridge_probs)
    ridge_logloss = log_loss(y_test, ridge_probs_2d)

    print(f"\n📏 Ridge Regression Performance:")
    print(f"   Accuracy: {ridge_acc:.2%}")
    print(f"   ROC-AUC: {ridge_auc:.4f}")
    print(f"   Log Loss: {ridge_logloss:.4f}")

    # 5B. ENSEMBLE EVALUATION
    print(f"\n🎯 Ensemble Performance:")

    # Test different weight combinations
    best_ensemble_acc = 0
    best_weights = (0.5, 0.5)

    for xgb_weight in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ridge_weight = 1 - xgb_weight

        # Weighted average of probabilities
        ensemble_probs = (xgb_weight * xgb_probs[:, 1] +
                          ridge_weight * ridge_probs)
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        ensemble_acc = accuracy_score(y_test, ensemble_preds)

        if ensemble_acc > best_ensemble_acc:
            best_ensemble_acc = ensemble_acc
            best_weights = (xgb_weight, ridge_weight)

        print(f"   XGB:{xgb_weight:.1f} Ridge:{ridge_weight:.1f} -> Acc: {ensemble_acc:.2%}")

    # Evaluate best ensemble
    xgb_w, ridge_w = best_weights
    ensemble_probs_best = (xgb_w * xgb_probs[:, 1] + ridge_w * ridge_probs)
    ensemble_probs_2d = np.column_stack([1 - ensemble_probs_best, ensemble_probs_best])
    ensemble_preds_best = (ensemble_probs_best > 0.5).astype(int)

    ensemble_auc = roc_auc_score(y_test, ensemble_probs_best)
    ensemble_logloss = log_loss(y_test, ensemble_probs_2d)
    ensemble_brier = brier_score_loss(y_test, ensemble_probs_best)

    print(f"\n✨ BEST ENSEMBLE (XGB:{xgb_w:.1f}, Ridge:{ridge_w:.1f}):")
    print(f"   Accuracy: {best_ensemble_acc:.2%}")
    print(f"   ROC-AUC: {ensemble_auc:.4f}")
    print(f"   Log Loss: {ensemble_logloss:.4f}")

    # 5C. DETAILED CLASSIFICATION REPORTS
    print("\n" + "="*80)
    print("📋 CLASSIFICATION REPORTS")
    print("="*80)

    print("\n🤖 XGBoost:")
    print(classification_report(y_test, xgb_preds,
                              target_names=['Away Win', 'Home Win']))

    print("\n📏 Ridge Regression:")
    print(classification_report(y_test, ridge_preds,
                              target_names=['Away Win', 'Home Win']))

    print("\n✨ Ensemble:")
    print(classification_report(y_test, ensemble_preds_best,
                              target_names=['Away Win', 'Home Win']))

    # 5D. FEATURE IMPORTANCE COMPARISON
    print("\n" + "="*80)
    print("🔍 FEATURE IMPORTANCE")
    print("="*80)

    # XGBoost Feature Importance (Gain)
    xgb_importance = pd.DataFrame({
        'Feature': features,
        'XGB_Importance': best_model.feature_importances_
    }).sort_values('XGB_Importance', ascending=False)

    # Ridge Coefficients (absolute value)
    ridge_importance = pd.DataFrame({
        'Feature': features,
        'Ridge_Coef': np.abs(ridge.coef_[0])
    }).sort_values('Ridge_Coef', ascending=False)

    # Merge and display
    importance_df = pd.merge(xgb_importance, ridge_importance, on='Feature')
    importance_df['Avg_Importance'] = (
        importance_df['XGB_Importance'] / importance_df['XGB_Importance'].sum() +
        importance_df['Ridge_Coef'] / importance_df['Ridge_Coef'].sum()
    ) / 2

    print("\n📊 Top 10 Features (Combined Importance):")
    top_10 = importance_df.sort_values('Avg_Importance', ascending=False).head(10)
    for idx, row in top_10.iterrows():
        print(f"   {row['Feature']:25} XGB:{row['XGB_Importance']:.4f}  Ridge:{row['Ridge_Coef']:.4f}")

    # 6. SAVE ALL MODELS
    weights_config = build_weights_config(xgb_w, ridge_w, best_ensemble_acc, ensemble_brier)
    save_artifacts(best_model, ridge, scaler, weights_config, output_dir)

    print(f"\n💾 Models Saved:")
    print(f"   XGBoost:  {MODEL_PATH}")
    print(f"   Ridge:    {RIDGE_MODEL_PATH}")
    print(f"   Scaler:   {SCALER_PATH}")
    print(f"   Weights:  {ENSEMBLE_WEIGHTS_PATH}")

    return True

if __name__ == "__main__":
    success = train_and_optimize_model()
    sys.exit(0 if success else 1)