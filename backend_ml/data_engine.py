# backend_ml/data_engine.py

import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamelog
from supabase import create_client
import os
from dotenv import load_dotenv

# Load Environment Variables from BOTH root .env and local .env.local
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = Path(__file__).resolve().parent

# Load root .env first (for SUPABASE_URL)
load_dotenv(dotenv_path=root_dir / '.env')
# Then load local .env.local (for SUPABASE_SERVICE_KEY) - this overrides if conflicts
load_dotenv(dotenv_path=backend_dir / '.env.local', override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # From .env.local

def initialize_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================================
# 1. FETCH RAW DATA
# ============================================================================

def fetch_season_data(season='2023-24'):
    """
    Fetches raw game logs (Points, Assists, etc) for a whole season.
    """
    print(f"🏀 Fetching raw data for {season}...")
    try:
        # Request regular season data
        log = leaguegamelog.LeagueGameLog(
            season=season, 
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        return log
    except Exception as e:
        print(f"Error fetching season {season}: {e}")
        return pd.DataFrame()

# ============================================================================
# 2. FEATURE ENGINEERING (The "Holistic" Logic)
# ============================================================================

def calculate_advanced_features(df):
    """
    Turns raw scores into predictive 'Rolling Averages'.
    Logic: We want to know a team's form *entering* the game.
    """
    print("⚙️ Calculating rolling stats (Holistic Data)...")
    
    # Sort by date so "previous game" logic works
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    
    # Define what stats define "Team Strength"
    features_to_roll = ['PTS', 'FG_PCT', 'AST', 'REB', 'PLUS_MINUS']
    
    # Calculate Last 5 Games Average (shift(1) ensures we don't include TONIGHT'S stats in the prediction)
    df_rolled = df.groupby('TEAM_ID')[features_to_roll].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    # Rename columns to identify them as rolling stats
    df_rolled.columns = [f"{col}_ROLLING" for col in df_rolled.columns]
    
    # Combine
    df = pd.concat([df, df_rolled], axis=1)
    
    # Calculate Rest Days
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days.fillna(3)
    df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7) # Cap at 7 days
    
    return df

def structure_data_for_model(df):
    """
    Merges the two rows per game (Home & Away) into one row for the model.
    """
    # Identify Home vs Away
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    # Merge on Game ID
    merged = pd.merge(home_df, away_df, on='GAME_ID', suffixes=('_H', '_A'))
    
    # Select final columns for Training
    # We keep the RESULT (PTS_H, PTS_A) and the FEATURES (Rolling stats)
    cols = [
        'GAME_ID', 'GAME_DATE_H', 
        'TEAM_NAME_H', 'PTS_H', 'PTS_ROLLING_H', 'FG_PCT_ROLLING_H', 'DAYS_REST_H',
        'TEAM_NAME_A', 'PTS_A', 'PTS_ROLLING_A', 'FG_PCT_ROLLING_A', 'DAYS_REST_A'
    ]
    
    # If rolling stats are missing (start of season), drop them
    return merged[cols].dropna()

# ============================================================================
# 3. MASTER FUNCTION (Run this to build dataset)
# ============================================================================

def build_training_dataset():
    """
    Fetches 2 years of data, processes it, and returns a clean DataFrame.
    """
    # Fetch last season and this season
    df_23 = fetch_season_data('2022-23')
    df_24 = fetch_season_data('2023-24')
    
    full_data = pd.concat([df_23, df_24])
    
    # Process
    processed_df = calculate_advanced_features(full_data)
    training_data = structure_data_for_model(processed_df)
    
    # Create Target: Did Home Team Win? (1 = Yes, 0 = No)
    training_data['HOME_WIN'] = (training_data['PTS_H'] > training_data['PTS_A']).astype(int)
    
    print(f"✅ Dataset Ready: {len(training_data)} games.")
    return training_data

if __name__ == "__main__":
    # Test run
    df = build_training_dataset()
    print(df.head())
    # Optional: Save to CSV to inspect
    df.to_csv('nba_final_training_data.csv', index=False)