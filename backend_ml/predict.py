import pandas as pd
import joblib
import datetime
import numpy as np
import os
import json
import requests
import pytz
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
from data_engine import (
    fetch_season_data,
    calculate_four_factors,
    calculate_rolling_features,
    structure_data_for_model,
    initialize_supabase,
    TEAM_ALTITUDES
)
from player_impact_engine import calculate_injury_impact
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

load_dotenv()
load_dotenv(Path(__file__).parent / '.env.local', override=True)

# Try to import Elo bias corrections if they exist
try:
    from elo_corrections import ELO_CORRECTIONS
    print(f"📊 Loaded {len(ELO_CORRECTIONS)} Elo bias corrections")
except ImportError:
    ELO_CORRECTIONS = {}

MODEL_PATH = "xgboost_nba_model.pkl"
RIDGE_MODEL_PATH = "ridge_nba_model.pkl"
SCALER_PATH = "feature_scaler.pkl"
ENSEMBLE_WEIGHTS_PATH = "ensemble_weights.json"
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4") 
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

nba_teams = teams.get_teams()
TEAM_NAMES = {team['id']: team['full_name'] for team in nba_teams}
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
    if not ODDS_API_KEY: return {}
    try:
        url = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
        params = {'apiKey': ODDS_API_KEY, 'regions': 'us', 'markets': 'h2h', 'oddsFormat': 'american'}
        res = requests.get(url, params=params)
        if res.status_code != 200: return {}
        
        odds_map = {}
        for game in res.json():
            home_id = ODDS_TEAM_MAPPING.get(game['home_team'])
            if home_id and game['bookmakers']:
                odds_map[home_id] = game['bookmakers'][0]['markets'][0]['outcomes'][0]['price'] # simplified
        return odds_map
    except: return {}

def generate_prediction_explanation(home_team, away_team, prob_home, features, injury_penalties=None):
    """Generate detailed number-based explanation for prediction"""
    winner = home_team if prob_home > 0.5 else away_team
    confidence = max(prob_home, 1 - prob_home)

    # Extract features
    home_elo = features.get('home_elo', 1500)
    away_elo = features.get('away_elo', 1500)
    reb_mismatch = features.get('reb_mismatch', 0)
    shoot_gap = features.get('shooting_gap', 0)
    tov_mismatch = features.get('tov_mismatch', 0)
    home_fatigue = features.get('home_fatigue_score', 0)
    away_fatigue = features.get('away_fatigue_score', 0)
    home_momentum = features.get('home_momentum', 1.0)
    away_momentum = features.get('away_momentum', 1.0)

    # Extract injury penalties if provided
    home_injury = injury_penalties.get('home', 0) if injury_penalties else 0
    away_injury = injury_penalties.get('away', 0) if injury_penalties else 0

    # Build explanation
    # Note: Model weighs factors roughly as: Elo (35%), Shooting (25%), Turnovers (15%), Rebounding (10%), Other (15%)
    factors = []

    # 1. Elo difference (most important factor)
    elo_diff = abs(home_elo - away_elo)
    if elo_diff > 20:
        if home_elo > away_elo:
            factors.append(f"{home_team} has a {elo_diff:.0f}-point Elo advantage ({home_elo:.0f} vs {away_elo:.0f})")
        else:
            factors.append(f"{away_team} has a {elo_diff:.0f}-point Elo advantage ({away_elo:.0f} vs {home_elo:.0f})")

    # 1b. Injury impact (critical context for Elo)
    if home_injury < -30 or away_injury < -30:
        injury_notes = []
        if home_injury < -50:
            injury_notes.append(f"{home_team} is severely weakened by injuries ({abs(home_injury):.0f} Elo hit)")
        elif home_injury < -30:
            injury_notes.append(f"{home_team} is missing key players ({abs(home_injury):.0f} Elo penalty)")

        if away_injury < -50:
            injury_notes.append(f"{away_team} is severely weakened by injuries ({abs(away_injury):.0f} Elo hit)")
        elif away_injury < -30:
            injury_notes.append(f"{away_team} is missing key players ({abs(away_injury):.0f} Elo penalty)")

        # Add injury notes to factors (they're critical context)
        factors.extend(injury_notes)

    # 2. Shooting efficiency
    if abs(shoot_gap) > 0.01:  # 1% difference
        if shoot_gap > 0:
            factors.append(f"{home_team} shoots {shoot_gap:.1%} better ({abs(shoot_gap)*100:.1f} percentage points)")
        else:
            factors.append(f"{away_team} has a {abs(shoot_gap):.1%} shooting advantage")

    # 3. Rebounding
    if abs(reb_mismatch) > 0.5:
        if reb_mismatch > 0:
            factors.append(f"{home_team} has a +{reb_mismatch:.1f} rebounding edge")
        else:
            factors.append(f"{away_team} dominates the boards (+{abs(reb_mismatch):.1f} advantage)")

    # 4. Turnovers
    if abs(tov_mismatch) > 0.01:
        if tov_mismatch > 0:
            factors.append(f"{away_team} turns it over {tov_mismatch:.1%} more frequently")
        else:
            factors.append(f"{home_team} has ball security issues ({abs(tov_mismatch):.1%} higher TOV rate)")

    # 5. Fatigue
    fatigue_diff = abs(home_fatigue - away_fatigue)
    if fatigue_diff > 0.1:
        if home_fatigue > away_fatigue:
            factors.append(f"{home_team} may be fatigued (fatigue score: {home_fatigue:.2f})")
        else:
            factors.append(f"{away_team} is playing on short rest (fatigue: {away_fatigue:.2f})")

    # 6. Momentum
    momentum_diff = abs(home_momentum - away_momentum)
    if momentum_diff > 0.05:
        hot_team = home_team if home_momentum > away_momentum else away_team
        hot_pct = (max(home_momentum, away_momentum) - 1) * 100
        if hot_pct > 0:
            factors.append(f"{hot_team} is playing {hot_pct:.1f}% above their season average recently")
        else:
            cold_team = home_team if home_momentum < away_momentum else away_team
            cold_pct = (1 - min(home_momentum, away_momentum)) * 100
            factors.append(f"{cold_team} is struggling ({cold_pct:.1f}% below season average)")

    # Check if Elo alone would suggest a different probability
    # Standard Elo win probability formula: 1 / (1 + 10^((away_elo - home_elo)/400))
    elo_home_prob = 1 / (1 + 10**((away_elo - home_elo)/400))
    elo_expected_winner = home_team if elo_home_prob > 0.5 else away_team
    elo_expected_conf = max(elo_home_prob, 1 - elo_home_prob)

    # Build final explanation
    if confidence < 0.55:
        intro = f"This is a toss-up with {winner} having a razor-thin {confidence*100:.1f}% edge. "
    elif confidence < 0.65:
        intro = f"{winner} is favored at {confidence*100:.1f}%. "
    else:
        intro = f"{winner} is heavily favored ({confidence*100:.1f}% win probability). "

    # Add note if other factors are significantly offsetting Elo
    if elo_diff > 100:  # Only for large Elo gaps
        if winner != elo_expected_winner:
            intro += f"⚠️ (Upset alert: Elo alone favors {elo_expected_winner} at {elo_expected_conf*100:.1f}%) "
        elif abs(confidence - elo_expected_conf) > 0.10:
            if confidence < elo_expected_conf:
                intro += f"📉 (Other factors drag this down from Elo's {elo_expected_conf*100:.1f}%) "
            else:
                intro += f"📈 (Other factors boost this from Elo's {elo_expected_conf*100:.1f}%) "

    # Identify if loser has any dominant advantage
    loser = away_team if prob_home > 0.5 else home_team
    loser_advantages = [f for f in factors if f.startswith(loser)]

    # Combine top factors
    if len(factors) == 0:
        explanation = intro + "Both teams are evenly matched across all metrics."
    elif len(factors) <= 3:
        explanation = intro + ", ".join(factors[:-1]) + (", and " if len(factors) > 1 else "") + factors[-1] + "."
    else:
        # Show all factors if there are competing advantages
        explanation = intro + ", ".join(factors[:3]) + ", and " + factors[3] + "."

    # Add clarification if the predicted loser has significant advantages
    if loser_advantages and len(factors) > 1:
        # Count winner's advantages
        winner_advantages = [f for f in factors if not f.startswith(loser)]
        if len(winner_advantages) >= 1:
            # Identify what the loser is good at
            loser_strength = ""
            if any("board" in f.lower() or "rebound" in f.lower() for f in loser_advantages):
                loser_strength = "rebounding"
            elif any("shoot" in f.lower() for f in loser_advantages):
                loser_strength = "shooting"
            elif any("turnover" in f.lower() for f in loser_advantages):
                loser_strength = "ball control"

            if loser_strength:
                if loser_strength == "rebounding":
                    explanation += f" Despite {loser}'s {loser_strength} advantage, the model weighs Elo rating and shooting efficiency (~60% combined) more heavily than rebounding (~10%), giving {winner} the edge."
                else:
                    explanation += f" Despite {loser}'s {loser_strength} advantage, {winner}'s combination of advantages in higher-weighted factors (Elo, shooting) outweigh it."

    # Add injury context note if significant injuries but not already mentioned
    if (home_injury < -20 or away_injury < -20) and not any('injur' in f.lower() for f in factors):
        injury_context = []
        if home_injury < -20:
            injury_context.append(f"{home_team} ({abs(home_injury):.0f} Elo)")
        if away_injury < -20:
            injury_context.append(f"{away_team} ({abs(away_injury):.0f} Elo)")
        if injury_context:
            explanation += f" 🏥 Injury adjustments: {', '.join(injury_context)}."

    # Try Azure OpenAI for enhanced explanation (optional)
    if AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT:
        try:
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            prompt = f"""Rewrite this NBA prediction explanation to be more engaging while keeping all the numbers and facts:

{explanation}

Keep it 2-3 sentences maximum. Maintain all specific numbers, percentages, injury mentions, and any "Despite..." clauses explaining feature importance."""

            res = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.7
            )
            ai_explanation = res.choices[0].message.content.strip()
            if ai_explanation and len(ai_explanation) > 20:
                return ai_explanation
        except Exception:
            pass  # Fall back to rule-based explanation

    return explanation

def get_matchups_for_date(day_offset=0):
    try:
        et_tz = pytz.timezone('America/New_York')
        target_date = datetime.datetime.now(et_tz) + datetime.timedelta(days=day_offset)
        date_formatted = target_date.strftime('%Y-%m-%d')
        print(f"📅 Checking games for: {date_formatted}")

        board = scoreboardv2.ScoreboardV2(game_date=date_formatted, day_offset=0)
        games = board.game_header.get_data_frame()
        
        matchups = []
        if not games.empty:
            for _, game in games.iterrows():
                matchups.append({
                    'GAME_ID': game['GAME_ID'],
                    'GAME_DATE': date_formatted,
                    'HOME_TEAM_ID': game['HOME_TEAM_ID'],
                    'AWAY_TEAM_ID': game['VISITOR_TEAM_ID'],
                    'HOME_TEAM_NAME': TEAM_NAMES.get(game['HOME_TEAM_ID'], 'Unknown'),
                    'AWAY_TEAM_NAME': TEAM_NAMES.get(game['VISITOR_TEAM_ID'], 'Unknown')
                })
        return pd.DataFrame(matchups), date_formatted
    except: return pd.DataFrame(), None

def get_latest_team_stats():
    df = fetch_season_data('2024-25')
    if df.empty: return {}
    
    df = calculate_four_factors(df)
    df = calculate_rolling_features(df)
    df_structured = structure_data_for_model(df) # This gets us the Elo
    
    # Extract latest stats
    # We need Raw EWMA stats to calculate mismatches dynamically
    latest_home = df_structured.sort_values('GAME_DATE_H').groupby('TEAM_ID_H').tail(1)
    latest_away = df_structured.sort_values('GAME_DATE_H').groupby('TEAM_ID_A').tail(1)

    # Build stats map using hybrid approach:
    # - Get Elo from structured data (df_structured)
    # - Get EWMA stats from rolling features (df)
    stats_map = {}
    # Fill from rolling df
    latest_rolling = df.sort_values('GAME_DATE').groupby('TEAM_ID').tail(1)
    for _, row in latest_rolling.iterrows():
        stats_map[row['TEAM_ID']] = {
            'EFG_PCT_EWMA': row['EFG_PCT_EWMA'],
            'TOV_PCT_EWMA': row['TOV_PCT_EWMA'],
            'OREB_EWMA': row['OREB_EWMA'],
            'DREB_EWMA': row['DREB_EWMA'],
            'FT_RATE_EWMA': row['FT_RATE_EWMA'],
            'WIN_PCT': row['WIN_PCT'],
            'OFF_RATING_EWMA': row['OFF_RATING_EWMA'],
            'SEASON_OFF_RATING': row['OFF_RATING_SEASON']
        }
        
    # Fill Elo from structured
    for _, row in latest_home.iterrows():
        tid = row['TEAM_ID_H']
        if tid in stats_map: stats_map[tid]['ELO'] = row['ELO_H']
        
    for _, row in latest_away.iterrows():
        tid = row['TEAM_ID_A']
        if tid in stats_map: 
            # prefer the latest elo
            if 'ELO' not in stats_map[tid]: stats_map[tid]['ELO'] = row['ELO_A']
            
    return stats_map

def update_recent_game_results():
    """Update game results from last 3 days before making predictions"""
    try:
        from update_game_results import update_completed_games
        print("🔄 Updating recent game results...")
        update_completed_games(days_back=3)
    except Exception as e:
        print(f"⚠️  Could not update game results: {e}")
        # Continue anyway - not critical for predictions


def predict_games(day_offset=0):
    print("\n" + "="*80)
    print("🏀 NBA PREDICTIONS (Elo + Mismatches + Injury Impact)")
    print("="*80 + "\n")

    # Update game results first
    update_recent_game_results()

    # Load all models
    if not os.path.exists(MODEL_PATH):
        print(f"❌ XGBoost model not found: {MODEL_PATH}"); return False
    if not os.path.exists(RIDGE_MODEL_PATH):
        print(f"❌ Ridge model not found: {RIDGE_MODEL_PATH}"); return False
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Scaler not found: {SCALER_PATH}"); return False

    xgb_model = joblib.load(MODEL_PATH)
    ridge_model = joblib.load(RIDGE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load ensemble weights (with fallback)
    try:
        with open(ENSEMBLE_WEIGHTS_PATH, 'r') as f:
            weights_config = json.load(f)
        xgb_weight = weights_config['xgb_weight']
        ridge_weight = weights_config['ridge_weight']
        print(f"📊 Loaded ensemble weights: XGB={xgb_weight:.1f}, Ridge={ridge_weight:.1f}")
    except:
        xgb_weight, ridge_weight = 0.5, 0.5
        print(f"⚠️  Using default ensemble weights: 50/50")

    games_df, _ = get_matchups_for_date(day_offset)
    if games_df.empty: games_df, _ = get_matchups_for_date(day_offset+1)
    if games_df.empty: return False

    stats = get_latest_team_stats()
    preds = []

    for _, game in games_df.iterrows():
        hid, aid = game['HOME_TEAM_ID'], game['AWAY_TEAM_ID']
        if hid not in stats or aid not in stats: continue
        
        h_stats, a_stats = stats[hid], stats[aid]

        # Get base Elo ratings
        base_home_elo = h_stats.get('ELO', 1500)
        base_away_elo = a_stats.get('ELO', 1500)

        # Apply bias corrections (if any)
        bias_correction_home = ELO_CORRECTIONS.get(hid, 0)
        bias_correction_away = ELO_CORRECTIONS.get(aid, 0)
        base_home_elo += bias_correction_home
        base_away_elo += bias_correction_away

        # Calculate injury impact and adjust Elo
        injury_penalties = calculate_injury_impact(hid, aid)
        adjusted_home_elo = base_home_elo + injury_penalties[hid]
        adjusted_away_elo = base_away_elo + injury_penalties[aid]

        # Calc Mismatches
        reb_mismatch = h_stats['OREB_EWMA'] - a_stats['DREB_EWMA']
        tov_mismatch = a_stats['TOV_PCT_EWMA'] - h_stats['TOV_PCT_EWMA']
        shoot_gap = h_stats['EFG_PCT_EWMA'] - a_stats['EFG_PCT_EWMA']

        # Calc Standard
        altitude = TEAM_ALTITUDES.get(hid, 0)
        fatigue_h = 0.5 * (1 - h_stats['WIN_PCT']) # Simplified for prediction speed
        fatigue_a = 0.5 * (1 - a_stats['WIN_PCT'])

        features = pd.DataFrame([{
            'ELO_H': adjusted_home_elo, 'ELO_A': adjusted_away_elo,
            'REB_MISMATCH': reb_mismatch, 'TOV_MISMATCH': tov_mismatch, 'SHOOTING_GAP': shoot_gap,
            'EFG_PCT_EWMA_H': h_stats['EFG_PCT_EWMA'], 'TOV_PCT_EWMA_H': h_stats['TOV_PCT_EWMA'],
            'ORB_PCT_EWMA_H': h_stats['OREB_EWMA'] / (h_stats['OREB_EWMA'] + a_stats['DREB_EWMA']),
            'FT_RATE_EWMA_H': h_stats['FT_RATE_EWMA'], 'FATIGUE_SCORE_H': fatigue_h,
            'MOMENTUM_H': h_stats['OFF_RATING_EWMA']/h_stats['SEASON_OFF_RATING'], 'HOME_ALTITUDE': altitude,
            'EFG_PCT_EWMA_A': a_stats['EFG_PCT_EWMA'], 'TOV_PCT_EWMA_A': a_stats['TOV_PCT_EWMA'],
            'ORB_PCT_EWMA_A': a_stats['OREB_EWMA'] / (a_stats['OREB_EWMA'] + h_stats['DREB_EWMA']),
            'FT_RATE_EWMA_A': a_stats['FT_RATE_EWMA'], 'FATIGUE_SCORE_A': fatigue_a,
            'MOMENTUM_A': a_stats['OFF_RATING_EWMA']/a_stats['SEASON_OFF_RATING']
        }])

        # Scale features for Ridge
        features_scaled = scaler.transform(features)

        # Get predictions from both models
        xgb_probs = xgb_model.predict_proba(features)[0]
        xgb_prob_home = xgb_probs[1]

        # Ridge uses decision_function + sigmoid
        ridge_decision = ridge_model.decision_function(features_scaled)[0]
        ridge_prob_home = 1 / (1 + np.exp(-ridge_decision))

        # Ensemble weighted average
        ensemble_prob_home = (xgb_weight * xgb_prob_home +
                              ridge_weight * ridge_prob_home)

        # Determine winner from ensemble
        winner = "Home" if ensemble_prob_home > 0.5 else "Away"
        confidence = max(ensemble_prob_home, 1 - ensemble_prob_home)

        # Check model agreement
        xgb_winner = "Home" if xgb_prob_home > 0.5 else "Away"
        ridge_winner = "Home" if ridge_prob_home > 0.5 else "Away"
        models_agree = (xgb_winner == ridge_winner)
        
        # Explanation with all features (using adjusted Elo)
        expl = generate_prediction_explanation(
            game['HOME_TEAM_NAME'],
            game['AWAY_TEAM_NAME'],
            ensemble_prob_home,
            {
                'home_elo': adjusted_home_elo,
                'away_elo': adjusted_away_elo,
                'reb_mismatch': reb_mismatch,
                'tov_mismatch': tov_mismatch,
                'shooting_gap': shoot_gap,
                'home_fatigue_score': fatigue_h,
                'away_fatigue_score': fatigue_a,
                'home_momentum': h_stats['OFF_RATING_EWMA'] / h_stats['SEASON_OFF_RATING'],
                'away_momentum': a_stats['OFF_RATING_EWMA'] / a_stats['SEASON_OFF_RATING']
            },
            injury_penalties={'home': injury_penalties[hid], 'away': injury_penalties[aid]}
        )

        # Display prediction with injury alerts
        print(f"🏟️ {game['HOME_TEAM_NAME']} vs {game['AWAY_TEAM_NAME']} -> {winner} ({confidence:.1%})")

        # Show model disagreement
        if not models_agree:
            print(f"   ⚠️  Models disagree: XGB={xgb_winner}({xgb_prob_home:.1%}), Ridge={ridge_winner}({ridge_prob_home:.1%})")

        # Show injury impact if significant
        if injury_penalties[hid] < -30 or injury_penalties[aid] < -30:
            print(f"   🏥 Injury Impact:")
            if injury_penalties[hid] < -30:
                print(f"      Home: {base_home_elo:.0f} → {adjusted_home_elo:.0f} ({injury_penalties[hid]:.0f} Elo)")
            if injury_penalties[aid] < -30:
                print(f"      Away: {base_away_elo:.0f} → {adjusted_away_elo:.0f} ({injury_penalties[aid]:.0f} Elo)")

        print(f"   📝 {expl}")

        preds.append({
            "game_id": game['GAME_ID'], "date": game['GAME_DATE'],
            "home_team_id": int(hid), "away_team_id": int(aid),

            # Ensemble predictions (primary)
            "home_win_probability": float(ensemble_prob_home),
            "away_win_probability": float(1 - ensemble_prob_home),
            "predicted_winner": winner,
            "confidence_score": float(confidence),

            # Individual model predictions (for analysis)
            "xgb_home_prob": float(xgb_prob_home),
            "ridge_home_prob": float(ridge_prob_home),
            "models_agree": bool(models_agree),

            # Existing fields (unchanged)
            "home_elo": int(adjusted_home_elo),
            "away_elo": int(adjusted_away_elo),
            "home_injury_penalty": float(injury_penalties[hid]),
            "away_injury_penalty": float(injury_penalties[aid]),
            "reb_mismatch": float(reb_mismatch),
            "tov_mismatch": float(tov_mismatch),
            "shooting_gap": float(shoot_gap),
            "explanation": expl
        })

    if preds:
        try:
            supabase = initialize_supabase()
            if supabase:
                # Dedupe and upload logic matches previous
                games_up = {p['game_id']: {"game_id": p['game_id'], "game_date": p['date'], "home_team_id": p['home_team_id'], "away_team_id": p['away_team_id'], "season": "2024-25"} for p in preds}
                supabase.table('games').upsert(list(games_up.values()), on_conflict='game_id').execute()

                preds_up = {p['game_id']: p for p in preds}

                # Try uploading with ensemble fields
                try:
                    supabase.table('game_predictions').upsert(list(preds_up.values()), on_conflict='game_id').execute()
                    print("✅ Upload successful!")
                except Exception as e:
                    # If ensemble fields not in database, retry without them
                    if 'models_agree' in str(e) or 'xgb_home_prob' in str(e) or 'ridge_home_prob' in str(e):
                        print("⚠️  Database schema doesn't have ensemble columns yet. Uploading without them...")
                        print("   Run 'python3 migrate_db.py' to see migration instructions.")

                        # Remove ensemble-specific fields
                        preds_legacy = []
                        for p in preds_up.values():
                            p_legacy = p.copy()
                            p_legacy.pop('xgb_home_prob', None)
                            p_legacy.pop('ridge_home_prob', None)
                            p_legacy.pop('models_agree', None)
                            preds_legacy.append(p_legacy)

                        supabase.table('game_predictions').upsert(preds_legacy, on_conflict='game_id').execute()
                        print("✅ Upload successful (legacy mode - ensemble details not saved)")
                    else:
                        raise  # Re-raise if it's a different error
        except Exception as e:
            print(f"⚠️ Upload failed: {e}")
    
    return True

if __name__ == "__main__":
    predict_games()