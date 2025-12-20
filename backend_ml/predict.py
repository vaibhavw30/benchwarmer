# backend_ml/predict.py

import pandas as pd
import joblib
import datetime
from data_engine import fetch_season_data, calculate_advanced_features, initialize_supabase
from nba_api.stats.endpoints import scoreboardv2

# CONFIG
MODEL_PATH = "xgboost_nba_model.pkl"

def get_todays_matchups():
    """
    Uses NBA API to find out who is playing TODAY.
    """
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    # Note: scoreboardV2 is finicky with dates, using current system date
    board = scoreboardv2.ScoreboardV2(game_date=today, day_offset=0)
    games = board.game_header.get_data_frame()
    
    matchups = []
    if not games.empty:
        for _, game in games.iterrows():
            matchups.append({
                'GAME_ID': game['GAME_ID'],
                'HOME_TEAM_ID': game['HOME_TEAM_ID'],
                'AWAY_TEAM_ID': game['VISITOR_TEAM_ID'],
                'GAME_DATE': today
            })
    return pd.DataFrame(matchups)

def get_latest_team_stats():
    """
    We need the MOST RECENT rolling stats for every team to predict tonight's game.
    We fetch the whole season, calculate rolling stats, and take the last row per team.
    """
    df = fetch_season_data('2023-24') # Fetch current season
    df = calculate_advanced_features(df) # Calculate rolling averages
    
    # Group by team and take the last available game's rolling stats
    latest_stats = df.groupby('TEAM_ID').last().reset_index()
    return latest_stats

def predict_tonight():
    # 1. Load Model
    print("Loading Model...")
    try:
        model = joblib.load(MODEL_PATH)
    except:
        print("❌ Model not found! Run train_model.py first.")
        return

    # 2. Get Today's Games
    todays_games = get_todays_matchups()
    if todays_games.empty:
        print("No games scheduled for today.")
        return

    # 3. Get Team Stats (The Context)
    print("Fetching latest team stats...")
    latest_stats = get_latest_team_stats()
    
    predictions_to_upload = []

    print(f"🔮 Predicting {len(todays_games)} games...")
    
    for _, game in todays_games.iterrows():
        # Find Home Team Stats
        home_stats = latest_stats[latest_stats['TEAM_ID'] == game['HOME_TEAM_ID']]
        # Find Away Team Stats
        away_stats = latest_stats[latest_stats['TEAM_ID'] == game['AWAY_TEAM_ID']]
        
        if home_stats.empty or away_stats.empty:
            print(f"Skipping game {game['GAME_ID']} (Missing stats)")
            continue
            
        # Construct Feature Row (Must match train_model.py order EXACTLY)
        features = pd.DataFrame([{
            'PTS_ROLLING_H': home_stats['PTS_ROLLING'].values[0],
            'FG_PCT_ROLLING_H': home_stats['FG_PCT_ROLLING'].values[0],
            'DAYS_REST_H': 1, # Approximating rest as 1 for now (can be improved)
            'PTS_ROLLING_A': away_stats['PTS_ROLLING'].values[0],
            'FG_PCT_ROLLING_A': away_stats['FG_PCT_ROLLING'].values[0],
            'DAYS_REST_A': 1
        }])
        
        # Predict
        prob_home_win = model.predict_proba(features)[0][1] # Probability of "1" (Home Win)
        winner = "Home" if prob_home_win > 0.5 else "Away"
        
        # Prepare for Supabase
        predictions_to_upload.append({
            "game_id": game['GAME_ID'],
            "date": game['GAME_DATE'],
            "home_team_id": int(game['HOME_TEAM_ID']),
            "away_team_id": int(game['AWAY_TEAM_ID']),
            "home_rolling_pts": float(features['PTS_ROLLING_H'].values[0]),
            "away_rolling_pts": float(features['PTS_ROLLING_A'].values[0]),
            "winner": winner,
            "win_probability": float(prob_home_win)
        })
        
        print(f"Game {game['GAME_ID']}: Home Win Prob {prob_home_win:.2%}")

    # 4. Upload to Supabase
    if predictions_to_upload:
        supabase = initialize_supabase()
        supabase.table('games').upsert(predictions_to_upload).execute()
        print("✅ Predictions uploaded to Supabase!")

if __name__ == "__main__":
    predict_tonight()