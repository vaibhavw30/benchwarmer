import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamelog
from supabase import create_client
import os
from pathlib import Path
from dotenv import load_dotenv

# Load from both .env files
load_dotenv()  # Load root .env
load_dotenv(Path(__file__).parent / '.env.local', override=True)  # Load local secrets

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

TEAM_ALTITUDES = {
    1610612743: 5280, # Denver
    1610612762: 4226, # Utah
    1610612756: 1117, # Phoenix
    1610612760: 1200, # OKC
}

def initialize_supabase():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

def fetch_season_data(season):
    print(f"🏀 Fetching {season}...")
    try:
        log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season').get_data_frames()[0]
        time.sleep(1)
        return log
    except Exception as e:
        print(f"Error fetching {season}: {e}")
        return pd.DataFrame()

def fetch_all_history():
    seasons = [
        '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
        '2020-21', '2021-22', '2022-23', '2023-24'
    ]
    frames = []
    for s in seasons:
        frames.append(fetch_season_data(s))
    return pd.concat(frames, ignore_index=True)

def calculate_four_factors(df):
    """
    Calculates the 'Four Factors' + Possession Count.
    """
    # 1. Possessions (Standard Formula)
    df['POSS'] = df['FGA'] + 0.44*df['FTA'] - df['OREB'] + df['TOV']
    
    # 2. Effective Field Goal % (Shooting Efficiency)
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    
    # 3. Turnover % (Ball Security)
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44*df['FTA'] + df['TOV'])
    
    # 4. Free Throw Rate (Aggression)
    df['FT_RATE'] = df['FTA'] / df['FGA']
    
    # 5. Offensive Rating (Points Efficiency)
    df['OFF_RATING'] = (df['PTS'] / df['POSS']) * 100
    
    # Win Indicator for Scaling
    df['WIN'] = df['WL'].map({'W': 1, 'L': 0})
    
    return df

def calculate_rolling_features(df):
    print("🔄 Calculating Four Factors EWMA...")
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    
    # We roll the Four Factors components. 
    # Note: We roll OREB and DREB raw numbers so we can calc percentages dynamically later.
    features_to_roll = ['EFG_PCT', 'TOV_PCT', 'FT_RATE', 'OFF_RATING', 'OREB', 'DREB'] 
    
    # 1. EWMA (Recent Form)
    df_rolled = df.groupby('TEAM_ID')[features_to_roll].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    df_rolled.columns = [f"{col}_EWMA" for col in df_rolled.columns]
    
    # 2. Season Averages
    df_season = df.groupby(['TEAM_ID', 'SEASON_ID'])[features_to_roll].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df_season.columns = [f"{col}_SEASON" for col in df_season.columns]
    
    # 3. Win Pct (for Fatigue Scaling)
    df['WIN_PCT'] = df.groupby(['TEAM_ID', 'SEASON_ID'])['WIN'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df = pd.concat([df, df_rolled, df_season], axis=1)
    
    # 4. Rest Days
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days.fillna(3)
    df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)
    
    return df

def structure_data_for_model(df):
    print("🏗️ Structuring final dataset...")
    
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    merged = pd.merge(home_df, away_df, on='GAME_ID', suffixes=('_H', '_A'))
    
    # --- CALCULATE REBOUND % (Factor #4) ---
    # Must be done AFTER merge because it requires Opponent stats
    # ORB% = Home_ORB / (Home_ORB + Away_DRB)
    
    # We use the EWMA versions to predict future performance
    merged['ORB_PCT_EWMA_H'] = merged['OREB_EWMA_H'] / (merged['OREB_EWMA_H'] + merged['DREB_EWMA_A'])
    merged['ORB_PCT_EWMA_A'] = merged['OREB_EWMA_A'] / (merged['OREB_EWMA_A'] + merged['DREB_EWMA_H'])
    
    # Fill potential /0 errors
    merged = merged.fillna(0)

    merged['HOME_ALTITUDE'] = merged['TEAM_ID_H'].map(TEAM_ALTITUDES).fillna(0)
    
    # Fatigue Calculation
    merged['FATIGUE_SCORE_H'] = np.where(merged['DAYS_REST_H'] <= 1, 1, 0) * (1 - merged['WIN_PCT_H'])
    merged['FATIGUE_SCORE_A'] = np.where(merged['DAYS_REST_A'] <= 1, 1, 0) * (1 - merged['WIN_PCT_A'])

    # Star Impact (Momentum)
    merged['MOMENTUM_H'] = merged['OFF_RATING_EWMA_H'] / merged['OFF_RATING_SEASON_H']
    merged['MOMENTUM_A'] = merged['OFF_RATING_EWMA_A'] / merged['OFF_RATING_SEASON_A']

    cols = [
        'GAME_ID', 'GAME_DATE_H', 'PTS_H', 'PTS_A',
        
        # HOME FOUR FACTORS + CONTEXT
        'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H', 
        'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',
        
        # AWAY FOUR FACTORS + CONTEXT
        'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
        'FATIGUE_SCORE_A', 'MOMENTUM_A'
    ]
    
    # Remove early season noise where EWMA is 0
    return merged[cols].replace([np.inf, -np.inf], 0).dropna()

def build_training_dataset():
    full_data = fetch_all_history()
    if full_data.empty: return pd.DataFrame()

    step1 = calculate_four_factors(full_data)
    step2 = calculate_rolling_features(step1)
    final_df = structure_data_for_model(step2)
    
    final_df['HOME_WIN'] = (final_df['PTS_H'] > final_df['PTS_A']).astype(int)
    
    final_df.to_csv("nba_training_cache.csv", index=False)
    print(f"✅ Four Factors Dataset Ready: {len(final_df)} games.")
    return final_df