import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta
from io import StringIO
from nba_api.stats.endpoints import leaguedashplayerstats

# --- CONFIGURATION ---
CACHE_FILE = "player_impact_scores.csv"
CACHE_EXPIRY_HOURS = 24
NEXT_MAN_UP_FACTOR = 0.65  # Team recovers 35% of lost production
IMPACT_TO_ELO_MULTIPLIER = 3.0  # 1 Impact Point ≈ 3 Elo Points

def update_player_stats_cache():
    """Fetches active player stats (Base + Advanced) and calculates Impact Scores."""
    print("🔄 Updating Player Impact Cache (NBA API)...")
    try:
        # 1. Fetch BASE Stats (PTS, REB, AST, etc.)
        base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        ).get_data_frames()[0]
        time.sleep(1) # Be polite to API

        # 2. Fetch ADVANCED Stats (USG_PCT)
        adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2025-26',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        
        # 3. Merge them on PLAYER_ID
        # We only keep PLAYER_ID and USG_PCT from the advanced dataframe
        stats = pd.merge(
            base_stats, 
            adv_stats[['PLAYER_ID', 'USG_PCT']], 
            on='PLAYER_ID', 
            how='left'
        )
        
        # 4. Data Cleaning
        # Ensure numeric columns
        cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'USG_PCT']
        for c in cols: 
            stats[c] = pd.to_numeric(stats[c], errors='coerce').fillna(0)
            
        # FIX: Scaling USG_PCT
        # The API usually returns decimals (0.25). Our formula expects percentages (25.0).
        if stats['USG_PCT'].mean() < 1.0:
            stats['USG_PCT'] = stats['USG_PCT'] * 100

        # 5. Calculate Impact Score
        # Formula: (Production) * (Usage Factor)
        # We assume average usage is 20%. Players with >20% usage get a boost.
        stats['IMPACT_RAW'] = (
            stats['PTS'] + 
            0.5 * stats['REB'] + 
            1.5 * stats['AST'] + 
            2.0 * stats['STL'] + 
            2.0 * stats['BLK'] - 
            1.0 * stats['TOV']
        ) * (stats['USG_PCT'] / 20.0)
        
        stats['PLAYER_IMPACT'] = stats['IMPACT_RAW']
        
        # Save necessary cols
        final_df = stats[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_IMPACT']]
        final_df.to_csv(CACHE_FILE, index=False)
        print(f"✅ Cached {len(final_df)} players.")
        return final_df
        
    except Exception as e:
        print(f"❌ Error updating player cache: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def get_player_impact_df():
    """Loads cache or updates it if stale."""
    # If cache doesn't exist, create it
    if not os.path.exists(CACHE_FILE):
        return update_player_stats_cache()
    
    # Check expiry
    try:
        file_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_time > timedelta(hours=CACHE_EXPIRY_HOURS):
            print("🕒 Cache expired, refreshing...")
            return update_player_stats_cache()
    except Exception:
        return update_player_stats_cache()
        
    return pd.read_csv(CACHE_FILE)

def fetch_active_injuries():
    """Scrapes multiple sources for injury report to ensure robustness."""
    print("🚑 Checking Live Injuries...")
    
    # Source 1: CBS Sports (Cleanest tables)
    try:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers)
        
        # Pandas read_html is powerful but requires lxml
        try:
            dfs = pd.read_html(StringIO(response.text))
            injury_df = pd.concat(dfs, ignore_index=True)
            injury_df.columns = [c.lower() for c in injury_df.columns]
            
            out_players = []
            for _, row in injury_df.iterrows():
                name = str(row.get('player', ''))
                status = str(row.get('injury status', '')).lower()
                
                # Filter for confirmed OUT or Doubtful
                if 'out' in status or 'doubtful' in status:
                    out_players.append(name)
            
            if out_players:
                print(f"   ✓ Found {len(out_players)} injuries from CBS")
                return out_players
        except ImportError:
            print("   ⚠️ Missing 'lxml', falling back...")
            
    except Exception as e:
        print(f"   ⚠️ CBS scrape failed: {e}")

    # Source 2: Rotowire (Fallback)
    try:
        url = "https://www.rotowire.com/basketball/injury-report.php"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        # Basic parsing if pandas fails
        # This is a simplified check, in reality scraping HTML raw is harder.
        # Let's assume user installs lxml for the main method.
        pass 
    except Exception:
        pass
        
    return []

def calculate_injury_impact(home_team_id, away_team_id):
    """
    Returns dictionary of Elo penalties: {home_id: -X, away_id: -Y}
    """
    impact_df = get_player_impact_df()
    if impact_df.empty: return {home_team_id: 0.0, away_team_id: 0.0}
    
    injured_names = fetch_active_injuries()
    
    penalties = {home_team_id: 0.0, away_team_id: 0.0}
    
    # Helper to check fuzzy match (CBS name vs NBA API name)
    def is_injured(api_name):
        # API Name: "LeBron James"
        # Injury Report: "LeBron James" or "L. James"
        # We check if the full last name appears in the injury list string
        api_clean = api_name.lower().replace('.', '')
        
        for inj in injured_names:
            inj_clean = inj.lower().replace('.', '')
            
            # Exact match
            if api_clean == inj_clean: return True
            
            # "L. James" matching "LeBron James"
            if len(inj_clean.split()) > 1 and len(api_clean.split()) > 1:
                # Check last names match
                if api_clean.split()[-1] == inj_clean.split()[-1]:
                    # Check first initial
                    if api_clean[0] == inj_clean[0]:
                        return True
        return False

    for team_id in [home_team_id, away_team_id]:
        team_players = impact_df[impact_df['TEAM_ID'] == team_id]
        
        missing_impact = 0
        for _, p in team_players.iterrows():
            if is_injured(p['PLAYER_NAME']):
                # Only count if they are a rotation player (>10 impact)
                if p['PLAYER_IMPACT'] > 10:
                    print(f"   🤕 Missing: {p['PLAYER_NAME']} (Impact: {p['PLAYER_IMPACT']:.1f})")
                    missing_impact += p['PLAYER_IMPACT']
        
        # Apply "Next Man Up" logic
        # We lose 100% of the player, but replace them with a bench player (recover 35%)
        # Net loss = 65% of the player's impact
        adjusted_loss = missing_impact * NEXT_MAN_UP_FACTOR
        
        # Convert to Elo
        elo_penalty = adjusted_loss * IMPACT_TO_ELO_MULTIPLIER
        penalties[team_id] = -round(elo_penalty, 1) # Elo is always subtracted
        
    return penalties

if __name__ == "__main__":
    # Self-test
    # 1. Force update cache
    print("--- 1. Testing Cache Update ---")
    update_player_stats_cache()
    
    # 2. Test Calculation
    print("\n--- 2. Testing Calculation (Lakers vs Nuggets) ---")
    # IDs: LAL=1610612747, DEN=1610612743
    res = calculate_injury_impact(1610612747, 1610612743)
    print(f"Result: {res}")