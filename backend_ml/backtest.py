import pandas as pd
import joblib
import json
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report
from data_engine import load_or_build_training_dataset

MODEL_PATH = "xgboost_nba_model.pkl"
RIDGE_MODEL_PATH = "ridge_nba_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"

def run_backtest(days_back=7):
    print("⏳ Loading Data for Backtest...")
    
    # 1. Load Data (rebuilds automatically if the cache is stale; set
    # FORCE_REFRESH=1 to always pull fresh data regardless of age)
    df = load_or_build_training_dataset()


    print("🧠 Loading Models...")
    xgb_model = joblib.load(MODEL_PATH)
    ridge_model = joblib.load(RIDGE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load weights
    try:
        with open(ENSEMBLE_WEIGHTS_PATH, 'r') as f:
            weights_config = json.load(f)
        xgb_weight = weights_config['xgb_weight']
        ridge_weight = weights_config['ridge_weight']
        print(f"📊 Loaded ensemble weights: XGB={xgb_weight:.1f}, Ridge={ridge_weight:.1f}")
    except:
        xgb_weight, ridge_weight = 0.5, 0.5
        print(f"⚠️  Using default ensemble weights: 50/50")
    
    # 2. Filter for "Recent Past"
    # Convert dates
    df['GAME_DATE_H'] = pd.to_datetime(df['GAME_DATE_H'])
    
    # Get range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Filter
    recent_games = df[(df['GAME_DATE_H'] >= start_date) & (df['GAME_DATE_H'] < end_date)].copy()
    
    if recent_games.empty:
        print(f"❌ No games found in the last {days_back} days.")
        print("💡 Hint: Delete 'nba_training_cache.csv' and run 'data_engine.py' to fetch fresh stats.")
        return

    print(f"\n🧪 Backtesting on {len(recent_games)} games from {start_date.date()} to {end_date.date()}...\n")
    
    # 3. Prepare Features (Must match train_model.py EXACTLY)
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
    
    X = recent_games[features]
    y_actual = recent_games[target]
    X_scaled = scaler.transform(X)

    # 4. PREDICT WITH ALL MODELS
    # XGBoost
    xgb_preds = xgb_model.predict(X)
    xgb_probs = xgb_model.predict_proba(X)[:, 1]

    # Ridge
    ridge_preds = ridge_model.predict(X_scaled)
    ridge_decision = ridge_model.decision_function(X_scaled)
    ridge_probs = 1 / (1 + np.exp(-ridge_decision))

    # Ensemble
    ensemble_probs = xgb_weight * xgb_probs + ridge_weight * ridge_probs
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    # 5. SCORE ALL THREE
    recent_games['XGB_PRED'] = xgb_preds
    recent_games['XGB_PROB'] = xgb_probs
    recent_games['XGB_CORRECT'] = (xgb_preds == recent_games['HOME_WIN'])

    recent_games['RIDGE_PRED'] = ridge_preds
    recent_games['RIDGE_PROB'] = ridge_probs
    recent_games['RIDGE_CORRECT'] = (ridge_preds == recent_games['HOME_WIN'])

    recent_games['ENSEMBLE_PRED'] = ensemble_preds
    recent_games['ENSEMBLE_PROB'] = ensemble_probs
    recent_games['ENSEMBLE_CORRECT'] = (ensemble_preds == recent_games['HOME_WIN'])

    # Calculate accuracies
    xgb_acc = accuracy_score(y_actual, xgb_preds)
    ridge_acc = accuracy_score(y_actual, ridge_preds)
    ensemble_acc = accuracy_score(y_actual, ensemble_preds)
    
    # 6. DISPLAY COMPARATIVE RESULTS
    print(f"\n📊 BACKTEST RESULTS (Last {days_back} Days)")
    print(f"========================================")
    print(f"🤖 XGBoost:  {xgb_acc:.1%} ({recent_games['XGB_CORRECT'].sum()}/{len(recent_games)})")
    print(f"📏 Ridge:    {ridge_acc:.1%} ({recent_games['RIDGE_CORRECT'].sum()}/{len(recent_games)})")
    print(f"✨ Ensemble: {ensemble_acc:.1%} ({recent_games['ENSEMBLE_CORRECT'].sum()}/{len(recent_games)})")
    print(f"========================================\n")

    # 7. PER-GAME BREAKDOWN
    recent_games['ACTUAL_WINNER'] = np.where(recent_games['HOME_WIN'] == 1, 'Home', 'Away')

    for _, game in recent_games.iterrows():
        result_icon = "✅" if game['ENSEMBLE_CORRECT'] else "❌"

        print(f"{result_icon} Date: {game['GAME_DATE_H'].date()} | Game ID: {game['GAME_ID']}")
        print(f"   Actual: {'Home' if game['HOME_WIN'] else 'Away'} Win")
        print(f"   🤖 XGB:      {'Home' if game['XGB_PRED'] else 'Away'} ({game['XGB_PROB']:.1%}) {'✓' if game['XGB_CORRECT'] else '✗'}")
        print(f"   📏 Ridge:    {'Home' if game['RIDGE_PRED'] else 'Away'} ({game['RIDGE_PROB']:.1%}) {'✓' if game['RIDGE_CORRECT'] else '✗'}")
        print(f"   ✨ Ensemble: {'Home' if game['ENSEMBLE_PRED'] else 'Away'} ({game['ENSEMBLE_PROB']:.1%}) {'✓' if game['ENSEMBLE_CORRECT'] else '✗'}")

        # Highlight ensemble rescues
        if game['ENSEMBLE_CORRECT'] and not game['XGB_CORRECT']:
            print(f"   🎯 Ensemble rescued XGBoost mistake!")
        elif game['ENSEMBLE_CORRECT'] and not game['RIDGE_CORRECT']:
            print(f"   🎯 Ensemble rescued Ridge mistake!")

        print("-" * 50)

if __name__ == "__main__":
    run_backtest(7)