import pandas as pd
import numpy as np

# Elo Constants (FiveThirtyEight Standard)
K_FACTOR = 20
HOME_ADVANTAGE = 100
BASE_ELO = 1500

def get_win_prob(elo_a, elo_b):
    """Calculates win probability for Team A vs Team B"""
    diff = elo_a - elo_b
    return 1 / (1 + 10 ** (-diff / 400))

def update_elo(winner_elo, loser_elo, margin_of_victory):
    """
    Updates Elo ratings after a game.
    Includes 'Margin of Victory' multiplier (blowing out teams rewards you more).
    """
    # Expected Result
    expected_win = get_win_prob(winner_elo, loser_elo)
    
    # Margin Multiplier (Auto-correlation adjustment)
    elo_diff = winner_elo - loser_elo
    mov_mult = np.log(margin_of_victory + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))
    
    shift = K_FACTOR * mov_mult * (1 - expected_win)
    return winner_elo + shift, loser_elo - shift

def generate_elo_features(df):
    print("⚡ Running Elo Simulation (10 seasons)...")
    
    # Initialize Elos
    team_elos = {team_id: BASE_ELO for team_id in df['TEAM_ID_H'].unique()}
    # Also catch away teams if they somehow didn't appear as home teams
    for team_id in df['TEAM_ID_A'].unique():
        if team_id not in team_elos:
            team_elos[team_id] = BASE_ELO
    
    elo_h_list = []
    elo_a_list = []
    prob_h_list = []
    
    # Sort chronologically to simulate history correctly
    df = df.sort_values('GAME_DATE_H').reset_index(drop=True)
    
    for idx, row in df.iterrows():
        home_id = row['TEAM_ID_H']
        away_id = row['TEAM_ID_A']
        margin = abs(row['PTS_H'] - row['PTS_A'])
        
        cur_elo_h = team_elos.get(home_id, BASE_ELO)
        cur_elo_a = team_elos.get(away_id, BASE_ELO)
        
        # Store Pre-Game Elo (What the model sees)
        elo_h_list.append(cur_elo_h)
        elo_a_list.append(cur_elo_a)
        
        # Add Home Court Advantage for the probability calc
        win_prob_h = get_win_prob(cur_elo_h + HOME_ADVANTAGE, cur_elo_a)
        prob_h_list.append(win_prob_h)
        
        # Update Elos Post-Game
        if row['PTS_H'] > row['PTS_A']: # Home Win
            new_h, new_a = update_elo(cur_elo_h, cur_elo_a, margin)
        else: # Away Win
            new_a, new_h = update_elo(cur_elo_a, cur_elo_h, margin)
            
        team_elos[home_id] = new_h
        team_elos[away_id] = new_a

    df['ELO_H'] = elo_h_list
    df['ELO_A'] = elo_a_list
    df['ELO_PROB_H'] = prob_h_list
    
    return df