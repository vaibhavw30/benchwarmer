import pandas as pd
import numpy as np
import time
from nba_api.stats.endpoints import leaguegamelog
from supabase import create_client
import os
from pathlib import Path
from dotenv import load_dotenv
from elo_engine import generate_elo_features

load_dotenv()
load_dotenv(Path(__file__).parent / '.env.local', override=True)

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
        '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
    ]
    frames = []
    for s in seasons:
        frames.append(fetch_season_data(s))
    return pd.concat(frames, ignore_index=True)

def calculate_four_factors(df):
    df['POSS'] = df['FGA'] + 0.44*df['FTA'] - df['OREB'] + df['TOV']
    df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44*df['FTA'] + df['TOV'])
    df['FT_RATE'] = df['FTA'] / df['FGA']
    df['OFF_RATING'] = (df['PTS'] / df['POSS']) * 100
    df['WIN'] = df['WL'].map({'W': 1, 'L': 0})
    return df

def calculate_rolling_features(df):
    print("🔄 Calculating Four Factors EWMA...")
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE')
    
    # We MUST roll 'DREB' to calculate Rebound Mismatch later
    features_to_roll = ['EFG_PCT', 'TOV_PCT', 'FT_RATE', 'OFF_RATING', 'OREB', 'DREB'] 
    
    df_rolled = df.groupby('TEAM_ID')[features_to_roll].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=1).mean()
    )
    df_rolled.columns = [f"{col}_EWMA" for col in df_rolled.columns]
    
    df_season = df.groupby(['TEAM_ID', 'SEASON_ID'])[features_to_roll].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df_season.columns = [f"{col}_SEASON" for col in df_season.columns]
    
    df['WIN_PCT'] = df.groupby(['TEAM_ID', 'SEASON_ID'])['WIN'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df = pd.concat([df, df_rolled, df_season], axis=1)
    
    df['PREV_GAME_DATE'] = df.groupby('TEAM_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days.fillna(3)
    df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)
    
    return df

def structure_data_for_model(df):
    print("🏗️ Structuring final dataset (Elo + Mismatches)...")
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)
    home_df = df[df['IS_HOME'] == 1].copy()
    away_df = df[df['IS_HOME'] == 0].copy()
    
    merged = pd.merge(home_df, away_df, on='GAME_ID', suffixes=('_H', '_A'))
    
    # 1. Elo
    merged = generate_elo_features(merged)

    # 2. Style Mismatches
    merged['REB_MISMATCH'] = merged['OREB_EWMA_H'] - merged['DREB_EWMA_A']
    merged['TOV_MISMATCH'] = merged['TOV_PCT_EWMA_A'] - merged['TOV_PCT_EWMA_H']
    merged['SHOOTING_GAP'] = merged['EFG_PCT_EWMA_H'] - merged['EFG_PCT_EWMA_A']

    # 3. Standard Features
    merged['ORB_PCT_EWMA_H'] = merged['OREB_EWMA_H'] / (merged['OREB_EWMA_H'] + merged['DREB_EWMA_A'])
    merged['ORB_PCT_EWMA_A'] = merged['OREB_EWMA_A'] / (merged['OREB_EWMA_A'] + merged['DREB_EWMA_H'])
    merged = merged.fillna(0)

    merged['HOME_ALTITUDE'] = merged['TEAM_ID_H'].map(TEAM_ALTITUDES).fillna(0)
    
    merged['FATIGUE_SCORE_H'] = np.where(merged['DAYS_REST_H'] <= 1, 1, 0) * (1 - merged['WIN_PCT_H'])
    merged['FATIGUE_SCORE_A'] = np.where(merged['DAYS_REST_A'] <= 1, 1, 0) * (1 - merged['WIN_PCT_A'])

    merged['MOMENTUM_H'] = merged['OFF_RATING_EWMA_H'] / merged['OFF_RATING_SEASON_H']
    merged['MOMENTUM_A'] = merged['OFF_RATING_EWMA_A'] / merged['OFF_RATING_SEASON_A']

    cols = [
        'GAME_ID', 'GAME_DATE_H', 'PTS_H', 'PTS_A',
        'TEAM_ID_H', 'TEAM_ID_A',  # <--- FIXED: Added these back!
        
        # New Features
        'ELO_H', 'ELO_A', 
        'REB_MISMATCH', 'TOV_MISMATCH', 'SHOOTING_GAP',
        
        # Standard Features
        'EFG_PCT_EWMA_H', 'TOV_PCT_EWMA_H', 'ORB_PCT_EWMA_H', 'FT_RATE_EWMA_H', 
        'FATIGUE_SCORE_H', 'MOMENTUM_H', 'HOME_ALTITUDE',
        'EFG_PCT_EWMA_A', 'TOV_PCT_EWMA_A', 'ORB_PCT_EWMA_A', 'FT_RATE_EWMA_A',
        'FATIGUE_SCORE_A', 'MOMENTUM_A'
    ]
    
    return merged[cols].replace([np.inf, -np.inf], 0).dropna()

def build_training_dataset():
    full_data = fetch_all_history()
    if full_data.empty: return pd.DataFrame()

    step1 = calculate_four_factors(full_data)
    step2 = calculate_rolling_features(step1)
    final_df = structure_data_for_model(step2)
    
    final_df['HOME_WIN'] = (final_df['PTS_H'] > final_df['PTS_A']).astype(int)

    tmp_cache_path = "nba_training_cache.csv.tmp"
    final_df.to_csv(tmp_cache_path, index=False)
    os.rename(tmp_cache_path, "nba_training_cache.csv")
    print(f"✅ Full Dataset Ready: {len(final_df)} games.")
    return final_df

def load_or_build_training_dataset(cache_path="nba_training_cache.csv", max_age_days=3):
    """Read the training cache if it's fresh enough, otherwise rebuild it.

    Freshness is judged by the newest GAME_DATE_H in the cache (not file
    mtime) so a cache that was merely re-saved without new games still
    counts as stale. Set FORCE_REFRESH=1 to always rebuild.
    """
    force_refresh = os.getenv("FORCE_REFRESH") == "1"

    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            cache_date = pd.to_datetime(df['GAME_DATE_H']).max()
            age_days = (pd.Timestamp.now() - cache_date).days
            if age_days <= max_age_days:
                print(f"📂 Loading data from cache ({age_days}d old, newest game {cache_date.date()})...")
                return df
            print(f"🔄 Cache newest game is {age_days}d old (>{max_age_days}d threshold), rebuilding...")
        except Exception as e:
            print(f"⚠️ Cache unreadable ({e!r}), rebuilding...")
    elif force_refresh:
        print("🔄 FORCE_REFRESH=1 set, rebuilding training dataset...")
    else:
        print("📡 No cache found, fetching fresh data...")

    return build_training_dataset()
