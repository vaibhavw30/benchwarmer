"""
NBA Model Backtesting Script

Tests the trained model against historical data to evaluate:
- Accuracy over time
- Performance by confidence level
- ROI if betting against Vegas spreads
- Home/Away prediction accuracy
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from data_engine import TEAM_ALTITUDES

MODEL_PATH = "xgboost_nba_model.pkl"
CACHE_PATH = "nba_training_cache.csv"

def backtest_model():
    """
    Backtests the model on the test set from the training data.
    """
    print("\n" + "="*80)
    print("🔬 NBA MODEL BACKTESTING")
    print("="*80 + "\n")

    # Load model
    print("📦 Loading model...")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"   ✓ Model loaded\n")
    except FileNotFoundError:
        print("❌ Model not found. Run train_model.py first.")
        return False

    # Load data
    print("📊 Loading historical data...")
    try:
        df = pd.read_csv(CACHE_PATH)
        print(f"   ✓ Loaded {len(df):,} games\n")
    except FileNotFoundError:
        print("❌ Cache not found. Run train_model.py first.")
        return False

    # Features (must match train_model.py)
    features = [
        'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H',
        'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',
        'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
        'FATIGUE_SCORE_A', 'MOMENTUM_A'
    ]

    # Use last 15% as backtest set (similar to test set)
    df = df.sort_values('GAME_DATE_H')
    backtest_start_idx = int(len(df) * 0.85)
    backtest_df = df.iloc[backtest_start_idx:].copy()

    print(f"📅 Backtest Period:")
    print(f"   Date Range: {backtest_df['GAME_DATE_H'].min()} to {backtest_df['GAME_DATE_H'].max()}")
    print(f"   Games: {len(backtest_df):,}\n")

    # Make predictions
    print("🔮 Generating predictions...")
    X_backtest = backtest_df[features]
    y_true = backtest_df['HOME_WIN']

    # Get probabilities
    y_pred_proba = model.predict_proba(X_backtest)[:, 1]  # Home win probability
    y_pred = (y_pred_proba > 0.5).astype(int)

    backtest_df['PRED_HOME_WIN'] = y_pred
    backtest_df['PRED_PROB'] = y_pred_proba

    # Overall Accuracy
    accuracy = (y_pred == y_true).mean()
    print(f"\n{'='*80}")
    print(f"📊 OVERALL PERFORMANCE")
    print(f"{'='*80}")
    print(f"Total Games: {len(backtest_df):,}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct Predictions: {(y_pred == y_true).sum():,}")
    print(f"Incorrect Predictions: {(y_pred != y_true).sum():,}")

    # Accuracy by Confidence Level
    print(f"\n{'='*80}")
    print(f"📈 ACCURACY BY CONFIDENCE LEVEL")
    print(f"{'='*80}")

    confidence_levels = [
        ('All Predictions', 0.0, 1.0),
        ('Low Confidence (50-55%)', 0.50, 0.55),
        ('Medium Confidence (55-60%)', 0.55, 0.60),
        ('High Confidence (60-65%)', 0.60, 0.65),
        ('Very High Confidence (65%+)', 0.65, 1.0),
    ]

    for label, min_conf, max_conf in confidence_levels:
        # Consider both home and away predictions
        home_conf_mask = (y_pred_proba >= min_conf) & (y_pred_proba <= max_conf)
        away_conf_mask = ((1 - y_pred_proba) >= min_conf) & ((1 - y_pred_proba) <= max_conf)
        mask = home_conf_mask | away_conf_mask

        if mask.sum() > 0:
            conf_accuracy = (y_pred[mask] == y_true[mask]).mean()
            print(f"{label:30} {mask.sum():5} games → {conf_accuracy:.2%} accuracy")

    # Home vs Away Performance
    print(f"\n{'='*80}")
    print(f"🏠 HOME VS AWAY PREDICTIONS")
    print(f"{'='*80}")

    home_pred_mask = y_pred == 1
    away_pred_mask = y_pred == 0

    if home_pred_mask.sum() > 0:
        home_accuracy = (y_pred[home_pred_mask] == y_true[home_pred_mask]).mean()
        print(f"Predicted Home Wins: {home_pred_mask.sum():,} games → {home_accuracy:.2%} accuracy")

    if away_pred_mask.sum() > 0:
        away_accuracy = (y_pred[away_pred_mask] == y_true[away_pred_mask]).mean()
        print(f"Predicted Away Wins: {away_pred_mask.sum():,} games → {away_accuracy:.2%} accuracy")

    # Monthly Performance
    print(f"\n{'='*80}")
    print(f"📅 PERFORMANCE BY MONTH")
    print(f"{'='*80}")

    backtest_df['GAME_MONTH'] = pd.to_datetime(backtest_df['GAME_DATE_H']).dt.to_period('M')
    monthly_stats = []

    for month in backtest_df['GAME_MONTH'].unique():
        month_mask = backtest_df['GAME_MONTH'] == month
        month_df = backtest_df[month_mask]

        if len(month_df) > 0:
            month_accuracy = (month_df['PRED_HOME_WIN'] == month_df['HOME_WIN']).mean()
            monthly_stats.append({
                'Month': str(month),
                'Games': len(month_df),
                'Accuracy': month_accuracy
            })

    monthly_df = pd.DataFrame(monthly_stats).sort_values('Month')
    for _, row in monthly_df.iterrows():
        print(f"{row['Month']:10} {row['Games']:4} games → {row['Accuracy']:.2%}")

    # Edge Cases Analysis
    print(f"\n{'='*80}")
    print(f"🔍 EDGE CASES ANALYSIS")
    print(f"{'='*80}")

    # Altitude games
    altitude_mask = backtest_df['HOME_ALTITUDE'] > 0
    if altitude_mask.sum() > 0:
        altitude_acc = (backtest_df.loc[altitude_mask, 'PRED_HOME_WIN'] == backtest_df.loc[altitude_mask, 'HOME_WIN']).mean()
        print(f"High Altitude Games (DEN/UTA): {altitude_mask.sum():,} games → {altitude_acc:.2%}")

    # Fatigue games
    fatigue_mask = (backtest_df['FATIGUE_SCORE_A'] > 0.4)
    if fatigue_mask.sum() > 0:
        fatigue_acc = (backtest_df.loc[fatigue_mask, 'PRED_HOME_WIN'] == backtest_df.loc[fatigue_mask, 'HOME_WIN']).mean()
        print(f"Heavy Away Fatigue: {fatigue_mask.sum():,} games → {fatigue_acc:.2%}")

    # Betting Simulation (simple)
    print(f"\n{'='*80}")
    print(f"💰 BETTING SIMULATION")
    print(f"{'='*80}")

    # Simulate betting $100 on each high-confidence prediction (>60%)
    bet_amount = 100
    high_conf_mask = ((y_pred_proba >= 0.60) | (y_pred_proba <= 0.40))

    if high_conf_mask.sum() > 0:
        bets = backtest_df[high_conf_mask].copy()
        total_bets = len(bets)
        wins = (bets['PRED_HOME_WIN'] == bets['HOME_WIN']).sum()
        losses = total_bets - wins

        # Simple calculation (assuming -110 odds)
        profit_per_win = bet_amount * 0.91  # Win $91 on $100 bet at -110
        loss_per_loss = bet_amount

        total_profit = (wins * profit_per_win) - (losses * loss_per_loss)
        roi = (total_profit / (total_bets * bet_amount)) * 100

        print(f"High Confidence Bets (>60% or <40%):")
        print(f"   Total Bets: {total_bets:,}")
        print(f"   Wins: {wins:,} ({wins/total_bets*100:.1f}%)")
        print(f"   Losses: {losses:,} ({losses/total_bets*100:.1f}%)")
        print(f"   Total Profit/Loss: ${total_profit:,.2f}")
        print(f"   ROI: {roi:.2f}%")
        print(f"\n   Note: This assumes -110 odds on all bets (not realistic)")

    # Save detailed results
    output_file = 'backtest_results.csv'
    backtest_df[['GAME_DATE_H', 'PTS_H', 'PTS_A', 'HOME_WIN', 'PRED_HOME_WIN', 'PRED_PROB']].to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    print(f"\n{'='*80}\n")
    return True

if __name__ == "__main__":
    import sys
    success = backtest_model()
    sys.exit(0 if success else 1)
