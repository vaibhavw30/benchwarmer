import pandas as pd
import joblib
import datetime
import numpy as np
import os
import sys
import requests
from dotenv import load_dotenv
from data_engine import (
    fetch_season_data,
    calculate_four_factors,
    calculate_rolling_features,
    initialize_supabase,
    TEAM_ALTITUDES
)
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

load_dotenv()
MODEL_PATH = "xgboost_nba_model.pkl"
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

# Team ID to Name mapping (NBA API)
nba_teams = teams.get_teams()
TEAM_NAMES = {team['id']: team['full_name'] for team in nba_teams}

# Mapping Odds API Names to NBA IDs
ODDS_TEAM_MAPPING = {
    'Atlanta Hawks': 1610612737, 'Boston Celtics': 1610612738, 'Brooklyn Nets': 1610612751,
    'Charlotte Hornets': 1610612766, 'Chicago Bulls': 1610612741, 'Cleveland Cavaliers': 1610612739,
    'Dallas Mavericks': 1610612742, 'Denver Nuggets': 1610612743, 'Detroit Pistons': 1610612765,
    'Golden State Warriors': 1610612744, 'Houston Rockets': 1610612745, 'Indiana Pacers': 1610612754,
    'Los Angeles Clippers': 1610612746, 'Los Angeles Lakers': 1610612747, 'Memphis Grizzlies': 1610612763,
    'Miami Heat': 1610612748, 'Milwaukee Bucks': 1610612749, 'Minnesota Timberwolves': 1610612750,
    'New Orleans Pelicans': 1610612740, 'New York Knicks': 1610612752, 'Oklahoma City Thunder': 1610612760,
    'Orlando Magic': 1610612753, 'Philadelphia 76ers': 1610612755, 'Phoenix Suns': 1610612756,
    'Portland Trail Blazers': 1610612757, 'Sacramento Kings': 1610612758, 'San Antonio Spurs': 1610612759,
    'Toronto Raptors': 1610612761, 'Utah Jazz': 1610612762, 'Washington Wizards': 1610612764
}

def fetch_live_odds():
    """Fetch live spreads from The-Odds-API"""
    if not ODDS_API_KEY:
        print("⚠️  No Odds API Key found in .env (Skipping Vegas comparison)")
        return {}
    
    print("💰 Fetching Live Odds from Vegas...")
    try:
        url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
        params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
        res = requests.get(url, params=params)
        
        if res.status_code != 200:
            print(f"⚠️  Odds API Error: {res.status_code}")
            return {}

        data = res.json()
        odds_map = {}
        for game in data:
            home_team = game['home_team']
            home_id = ODDS_TEAM_MAPPING.get(home_team)
            if home_id and game['bookmakers']:
                # Simplification: Take first bookmaker
                outcomes = game['bookmakers'][0]['markets'][0]['outcomes']
                for outcome in outcomes:
                    if outcome['name'] == home_team:
                        odds_map[home_id] = outcome['price'] # American Odds
        return odds_map
    except Exception as e:
        print(f"❌ Error fetching odds: {e}")
        return {}

def american_to_prob(american_odds):
    """Converts American Odds (-150) to Probability (0.60)"""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return (-american_odds) / (-american_odds + 100)

def get_matchups_for_date(date_str=None, day_offset=0):
    try:
        if date_str is None:
            target_date = datetime.datetime.now()
        else:
            target_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')

        target_date = target_date + datetime.timedelta(days=day_offset)
        date_formatted = target_date.strftime('%Y-%m-%d')

        print(f"📅 Checking games for: {date_formatted}")

        board = scoreboardv2.ScoreboardV2(game_date=date_formatted, day_offset=0)
        games = board.game_header.get_data_frame()

        matchups = []
        if not games.empty:
            for _, game in games.iterrows():
                matchups.append({
                    'GAME_ID': game['GAME_ID'],
                    'HOME_TEAM_ID': game['HOME_TEAM_ID'],
                    'AWAY_TEAM_ID': game['VISITOR_TEAM_ID'],
                    'GAME_DATE': date_formatted,
                    'HOME_TEAM_NAME': TEAM_NAMES.get(game['HOME_TEAM_ID'], 'Unknown'),
                    'AWAY_TEAM_NAME': TEAM_NAMES.get(game['VISITOR_TEAM_ID'], 'Unknown')
                })

        return pd.DataFrame(matchups), date_formatted
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        return pd.DataFrame(), None

def get_latest_team_stats():
    df = fetch_season_data('2024-25')
    if df.empty: return pd.DataFrame()
    
    df = calculate_four_factors(df) # Updated
    
    df['WIN_PCT'] = df.groupby(['TEAM_ID', 'SEASON_ID'])['WIN'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    
    df = calculate_rolling_features(df)
    
    return df.groupby('TEAM_ID').tail(1)

def predict_games(day_offset=0):
    print("\n" + "="*80)
    print("🏀 NBA GAME PREDICTIONS (Four Factors + Odds)")
    print("="*80 + "\n")

    # Load model
    print("📦 Loading model...")
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at {MODEL_PATH}")
            return False
        model = joblib.load(MODEL_PATH)
        print(f"   ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Get games
    games_df, game_date = get_matchups_for_date(day_offset=day_offset)
    if games_df.empty:
        if day_offset == 0:
            print("\n🔍 Checking tomorrow's schedule...")
            games_df, game_date = get_matchups_for_date(day_offset=1)
            if games_df.empty: return False
    
    # Get Odds
    vegas_odds = fetch_live_odds()

    # Get latest team stats
    print("\n📊 Fetching latest team statistics...")
    try:
        latest_stats = get_latest_team_stats()
        if latest_stats.empty: return False
        print(f"   ✓ Stats fetched for {len(latest_stats)} teams\n")
    except Exception as e:
        print(f"❌ Error fetching stats: {e}")
        return False

    # Make predictions
    print("="*80)
    predictions_to_upload = []

    for _, game in games_df.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['AWAY_TEAM_ID']
        home_name = game['HOME_TEAM_NAME']
        away_name = game['AWAY_TEAM_NAME']

        home_stats = latest_stats[latest_stats['TEAM_ID'] == home_id]
        away_stats = latest_stats[latest_stats['TEAM_ID'] == away_id]

        if home_stats.empty or away_stats.empty:
            continue

        try:
            game_date_dt = pd.to_datetime(game['GAME_DATE'])

            # --- HOME TEAM FEATURES ---
            last_date_h = pd.to_datetime(home_stats['GAME_DATE'].values[0])
            rest_h = min((game_date_dt - last_date_h).days, 7)
            fatigue_flag_h = 1 if rest_h <= 1 else 0
            win_pct_h = home_stats['WIN_PCT'].values[0] if not np.isnan(home_stats['WIN_PCT'].values[0]) else 0.5
            fatigue_score_h = fatigue_flag_h * (1 - win_pct_h)
            momentum_h = home_stats['OFF_RATING_EWMA'].values[0] / home_stats['OFF_RATING_SEASON'].values[0]

            # --- AWAY TEAM FEATURES ---
            last_date_a = pd.to_datetime(away_stats['GAME_DATE'].values[0])
            rest_a = min((game_date_dt - last_date_a).days, 7)
            fatigue_flag_a = 1 if rest_a <= 1 else 0
            win_pct_a = away_stats['WIN_PCT'].values[0] if not np.isnan(away_stats['WIN_PCT'].values[0]) else 0.5
            fatigue_score_a = fatigue_flag_a * (1 - win_pct_a)
            momentum_a = away_stats['OFF_RATING_EWMA'].values[0] / away_stats['OFF_RATING_SEASON'].values[0]

            # --- DYNAMIC REBOUND PCT ---
            orb_pct_h = home_stats['OREB_EWMA'].values[0] / (home_stats['OREB_EWMA'].values[0] + away_stats['DREB_EWMA'].values[0])
            orb_pct_a = away_stats['OREB_EWMA'].values[0] / (away_stats['OREB_EWMA'].values[0] + home_stats['DREB_EWMA'].values[0])

            altitude = TEAM_ALTITUDES.get(home_id, 0)

            # Build feature vector (Must match train_model.py)
            features = pd.DataFrame([{
                'EFG_PCT_EWMA_H': home_stats['EFG_PCT_EWMA'].values[0],
                'TOV_PCT_EWMA_H': home_stats['TOV_PCT_EWMA'].values[0],
                'ORB_PCT_EWMA_H': orb_pct_h,
                'FT_RATE_EWMA_H': home_stats['FT_RATE_EWMA'].values[0],
                'FATIGUE_SCORE_H': fatigue_score_h, 'MOMENTUM_H': momentum_h, 'HOME_ALTITUDE': altitude,

                'EFG_PCT_EWMA_A': away_stats['EFG_PCT_EWMA'].values[0],
                'TOV_PCT_EWMA_A': away_stats['TOV_PCT_EWMA'].values[0],
                'ORB_PCT_EWMA_A': orb_pct_a,
                'FT_RATE_EWMA_A': away_stats['FT_RATE_EWMA'].values[0],
                'FATIGUE_SCORE_A': fatigue_score_a, 'MOMENTUM_A': momentum_a
            }])

            # Predict
            probs = model.predict_proba(features)[0]
            prob_home = probs[1]
            winner_name = home_name if prob_home > 0.5 else away_name
            confidence = max(prob_home, 1-prob_home)
            
            # Compare with Vegas
            vegas_msg = ""
            if home_id in vegas_odds:
                implied_prob = american_to_prob(vegas_odds[home_id])
                diff = prob_home - implied_prob
                if diff > 0.10: vegas_msg = "🔥 VALUE BET (Model > Vegas)"
                elif diff < -0.10: vegas_msg = "⚠️ FADE MODEL"
                else: vegas_msg = f"(Vegas agrees: {implied_prob:.1%})"

            # Display
            print(f"\n🏟️  {home_name} vs {away_name}")
            print(f"   └─ Prediction: {winner_name} ({confidence*100:.1f}%) {vegas_msg}")
            if fatigue_score_a > 0.4: print(f"      🚨 AWAY FATIGUE DETECTED")

            predictions_to_upload.append({
                "game_id": game['GAME_ID'],
                "date": game['GAME_DATE'],
                "home_team_id": int(home_id),
                "away_team_id": int(away_id),
                "home_win_probability": float(prob_home),
                "away_win_probability": float(1 - prob_home),
                "predicted_winner": "Home" if prob_home > 0.5 else "Away",
                "confidence_score": float(confidence),
                # Model features for transparency
                "home_efg_pct": float(features['EFG_PCT_EWMA_H'].values[0]),
                "away_efg_pct": float(features['EFG_PCT_EWMA_A'].values[0]),
                "home_fatigue_score": float(fatigue_score_h),
                "away_fatigue_score": float(fatigue_score_a),
                "home_momentum": float(momentum_h),
                "away_momentum": float(momentum_a),
                "altitude_advantage": int(altitude),
                "model_version": "four_factors_v1"
            })

        except Exception as e:
            print(f"❌ Error predicting {home_name} vs {away_name}: {e}")
            continue

    if predictions_to_upload:
        try:
            supabase = initialize_supabase()
            if supabase:
                print("\n📤 Uploading to Supabase...")

                # First, upsert games
                games_data = []
                for pred in predictions_to_upload:
                    games_data.append({
                        "game_id": pred["game_id"],
                        "game_date": pred["date"],
                        "season": "2024-25",
                        "home_team_id": pred["home_team_id"],
                        "away_team_id": pred["away_team_id"],
                        "status": "upcoming"
                    })

                result = supabase.table('games').upsert(games_data, on_conflict='game_id').execute()
                print(f"   ✓ Upserted {len(games_data)} game(s)")

                # Then, upsert predictions
                result = supabase.table('game_predictions').upsert(predictions_to_upload, on_conflict='game_id').execute()
                print(f"   ✓ Upserted {len(predictions_to_upload)} prediction(s)")
                print("   ✅ Upload successful!")
        except Exception as e:
            print(f"   ⚠️  Upload failed: {e}")
            import traceback
            traceback.print_exc()

    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    success = predict_games()
    sys.exit(0 if success else 1)