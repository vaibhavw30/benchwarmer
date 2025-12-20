# ML Backend Implementation Guide - Step by Step

This guide walks you through implementing the complete ML pipeline from scratch.

---

## 📋 Overview: The ML Pipeline

```
1. DATA COLLECTION → 2. FEATURE ENGINEERING → 3. MODEL TRAINING → 4. PREDICTIONS
   (data_engine.py)     (data_engine.py)        (train_model.py)    (predict.py)
```

---

# PHASE 1: Data Collection (data_engine.py)

## Step 1.1: Choose Your Data Source

You need historical NBA game data. Here are your options:

### Option A: Ball Don't Lie API (Recommended - FREE, no key)

**Pros:** Free, no API key, easy to use
**Cons:** Limited data, might be slower

```python
# Test it first:
import requests

response = requests.get('https://www.balldontlie.io/api/v1/games?seasons[]=2024&per_page=5')
print(response.json())
```

### Option B: NBA Stats API (Unofficial)

**Pros:** Official data, comprehensive
**Cons:** Requires headers, might block scrapers

```python
headers = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://stats.nba.com/'
}
url = 'https://stats.nba.com/stats/scoreboardV2?GameDate=2024-12-17&LeagueID=00&DayOffset=0'
```

### Option C: ESPN Hidden API

**Pros:** Good data, reliable
**Cons:** Undocumented, might change

```python
url = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
```

**Decision:** Start with Ball Don't Lie API for simplicity.

---

## Step 1.2: Implement `fetch_nba_schedule()`

**File:** `backend_ml/data_engine.py`

**Goal:** Get all games for a date range and return as structured data

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_nba_schedule(start_date, end_date):
    """
    Fetch NBA game schedule from Ball Don't Lie API.

    Returns:
        DataFrame with columns: game_id, home_team_id, away_team_id,
                                game_date, home_team_name, away_team_name
    """

    # Step 1: Convert dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Step 2: Extract season (e.g., 2024 for 2024-25 season)
    season = start.year if start.month >= 10 else start.year - 1

    # Step 3: Fetch games from API
    all_games = []
    page = 1

    while True:
        url = f'https://www.balldontlie.io/api/v1/games?seasons[]={season}&per_page=100&page={page}'
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        games = data.get('data', [])

        if not games:
            break

        # Step 4: Filter games by date range
        for game in games:
            game_date = datetime.strptime(game['date'][:10], '%Y-%m-%d')

            if start <= game_date <= end:
                all_games.append({
                    'game_id': str(game['id']),
                    'home_team_id': game['home_team']['id'],
                    'home_team_name': game['home_team']['full_name'],
                    'away_team_id': game['visitor_team']['id'],
                    'away_team_name': game['visitor_team']['full_name'],
                    'game_date': game['date'],
                    'home_score': game['home_team_score'],
                    'away_score': game['visitor_team_score'],
                    'season': f"{season}-{str(season+1)[-2:]}"
                })

        # Check if there are more pages
        if page >= data['meta']['total_pages']:
            break

        page += 1

    # Step 5: Convert to DataFrame
    df = pd.DataFrame(all_games)
    print(f"✅ Fetched {len(df)} games from {start_date} to {end_date}")

    return df
```

**Test it:**

```python
# In backend_ml/
from data_engine import fetch_nba_schedule

# Get games from last week
games = fetch_nba_schedule('2024-12-10', '2024-12-17')
print(games.head())
```

---

## Step 1.3: Implement `fetch_team_stats()`

**Goal:** Get current season stats for a team

```python
def fetch_team_stats(team_id, season='2024-25'):
    """
    Fetch team statistics from Ball Don't Lie API.

    Note: Ball Don't Lie doesn't have team stats, so we'll calculate them
    from game data. For production, use NBA Stats API.
    """

    # Step 1: Fetch all games for this team this season
    season_year = int(season.split('-')[0])
    url = f'https://www.balldontlie.io/api/v1/games?seasons[]={season_year}&team_ids[]={team_id}&per_page=100'

    response = requests.get(url)
    if response.status_code != 200:
        return None

    games = response.json().get('data', [])

    if not games:
        return None

    # Step 2: Calculate statistics
    total_games = len(games)
    wins = 0
    total_points = 0
    total_points_allowed = 0

    for game in games:
        is_home = game['home_team']['id'] == team_id

        if is_home:
            team_score = game['home_team_score']
            opp_score = game['visitor_team_score']
        else:
            team_score = game['visitor_team_score']
            opp_score = game['home_team_score']

        # Only count finished games
        if team_score and opp_score:
            total_points += team_score
            total_points_allowed += opp_score

            if team_score > opp_score:
                wins += 1

    losses = total_games - wins

    # Step 3: Calculate metrics
    stats = {
        'team_id': team_id,
        'games_played': total_games,
        'wins': wins,
        'losses': losses,
        'win_percentage': wins / total_games if total_games > 0 else 0,
        'points_per_game': total_points / total_games if total_games > 0 else 0,
        'points_allowed_per_game': total_points_allowed / total_games if total_games > 0 else 0,
        'season': season
    }

    stats['net_rating'] = stats['points_per_game'] - stats['points_allowed_per_game']

    return stats
```

**Test it:**

```python
# Lakers team_id is typically 14
lakers_stats = fetch_team_stats(14, '2024-25')
print(lakers_stats)
```

---

## Step 1.4: Store Data in Supabase

**Goal:** Save fetched data to database

```python
from config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from supabase import create_client

def initialize_supabase():
    """Create Supabase client"""
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def store_games_in_supabase(games_df):
    """
    Store games in Supabase games table.

    Args:
        games_df: DataFrame from fetch_nba_schedule()
    """
    supabase = initialize_supabase()

    # First, we need to map team_ids to our teams table
    # For now, let's assume teams are already in the database

    for _, game in games_df.iterrows():
        # Get team UUIDs from our teams table
        home_team = supabase.table('teams').select('id').eq('team_id', game['home_team_id']).execute()
        away_team = supabase.table('teams').select('id').eq('team_id', game['away_team_id']).execute()

        if not home_team.data or not away_team.data:
            print(f"⚠️ Teams not found for game {game['game_id']}, skipping")
            continue

        game_data = {
            'game_id': game['game_id'],
            'home_team_id': home_team.data[0]['id'],
            'away_team_id': away_team.data[0]['id'],
            'game_date': game['game_date'],
            'season': game['season'],
            'status': 'completed' if game['home_score'] else 'upcoming',
            'home_team_score': game['home_score'],
            'away_team_score': game['away_score']
        }

        # Upsert (insert or update if exists)
        result = supabase.table('games').upsert(game_data, on_conflict='game_id').execute()

    print(f"✅ Stored {len(games_df)} games in Supabase")
```

**Test it:**

```python
games = fetch_nba_schedule('2024-12-10', '2024-12-17')
store_games_in_supabase(games)
```

---

# PHASE 2: Feature Engineering

## Step 2.1: Understand What Features You Need

For NBA predictions, you need:

### Basic Features:
- Team win/loss record
- Points per game (offense)
- Points allowed per game (defense)
- Home/away record

### Advanced Features:
- Recent form (last 5/10 games win %)
- Head-to-head record
- Days of rest
- Back-to-back games
- Home court advantage

### Context Features:
- Season timing (early vs late season)
- Playoff implications
- Winning/losing streak

---

## Step 2.2: Implement `calculate_advanced_features()`

```python
def calculate_game_features(game_row, all_games_df):
    """
    Calculate features for ONE game for training.

    Args:
        game_row: Single game from DataFrame
        all_games_df: All historical games (for context)

    Returns:
        dict: Features for this game
    """

    game_date = pd.to_datetime(game_row['game_date'])
    home_team_id = game_row['home_team_id']
    away_team_id = game_row['away_team_id']

    # Filter to games BEFORE this game (no future data leakage!)
    historical = all_games_df[pd.to_datetime(all_games_df['game_date']) < game_date]

    # --- HOME TEAM FEATURES ---
    home_games = historical[
        (historical['home_team_id'] == home_team_id) |
        (historical['away_team_id'] == home_team_id)
    ]

    # Last 10 games for home team
    home_recent = home_games.tail(10)
    home_wins_last_10 = 0
    home_ppg_last_10 = []

    for _, g in home_recent.iterrows():
        is_home = g['home_team_id'] == home_team_id
        team_score = g['home_score'] if is_home else g['away_score']
        opp_score = g['away_score'] if is_home else g['home_score']

        if team_score and opp_score:
            home_ppg_last_10.append(team_score)
            if team_score > opp_score:
                home_wins_last_10 += 1

    # --- AWAY TEAM FEATURES (same logic) ---
    away_games = historical[
        (historical['home_team_id'] == away_team_id) |
        (historical['away_team_id'] == away_team_id)
    ]

    away_recent = away_games.tail(10)
    away_wins_last_10 = 0
    away_ppg_last_10 = []

    for _, g in away_recent.iterrows():
        is_home = g['home_team_id'] == away_team_id
        team_score = g['home_score'] if is_home else g['away_score']
        opp_score = g['away_score'] if is_home else g['home_score']

        if team_score and opp_score:
            away_ppg_last_10.append(team_score)
            if team_score > opp_score:
                away_wins_last_10 += 1

    # --- COMPILE FEATURES ---
    features = {
        'home_wins_last_10': home_wins_last_10,
        'away_wins_last_10': away_wins_last_10,
        'home_ppg_last_10': np.mean(home_ppg_last_10) if home_ppg_last_10 else 0,
        'away_ppg_last_10': np.mean(away_ppg_last_10) if away_ppg_last_10 else 0,
        'home_games_played': len(home_games),
        'away_games_played': len(away_games),
        # Add more features here...
    }

    return features
```

---

# PHASE 3: Model Training (train_model.py)

## Step 3.1: Prepare Training Data

```python
def prepare_training_dataset():
    """
    Load historical games and create feature matrix + labels.

    Returns:
        X: Feature matrix (DataFrame)
        y: Labels (1 if home team won, 0 if away won)
    """

    # Step 1: Fetch 2-3 seasons of historical games
    print("Fetching historical data...")
    games_2022 = fetch_nba_schedule('2022-10-01', '2023-06-30')
    games_2023 = fetch_nba_schedule('2023-10-01', '2024-06-30')

    all_games = pd.concat([games_2022, games_2023], ignore_index=True)

    # Step 2: Filter to only completed games (with scores)
    completed = all_games[all_games['home_score'].notna()].copy()
    print(f"Found {len(completed)} completed games")

    # Step 3: Calculate features for each game
    features_list = []
    labels_list = []

    for idx, game in completed.iterrows():
        # Calculate features
        features = calculate_game_features(game, all_games)

        # Create label: 1 if home team won, 0 if away won
        home_won = 1 if game['home_score'] > game['away_score'] else 0

        features_list.append(features)
        labels_list.append(home_won)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(completed)} games...")

    # Step 4: Convert to DataFrame and Series
    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list, name='home_won')

    print(f"\n✅ Dataset ready:")
    print(f"   Features: {X.shape}")
    print(f"   Labels: {y.shape}")
    print(f"   Home team win rate: {y.mean():.1%}")

    return X, y
```

---

## Step 3.2: Train XGBoost Model

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train_baseline_model():
    """Train a baseline XGBoost model."""

    # Step 1: Load data
    X, y = prepare_training_dataset()

    # Step 2: Temporal split (most recent 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"\nTrain set: {len(X_train)} games")
    print(f"Test set: {len(X_test)} games")

    # Step 3: Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Step 4: Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n🎯 Model Performance:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   ROC-AUC: {roc_auc:.3f}")

    # Step 5: Save model
    import joblib
    joblib.dump(model, './models/xgboost_model_v1.pkl')
    print("\n✅ Model saved to ./models/xgboost_model_v1.pkl")

    return model
```

---

# PHASE 4: Generate Predictions (predict.py)

## Step 4.1: Load Model and Predict

```python
import joblib

def predict_todays_games():
    """Generate predictions for today's games."""

    # Step 1: Load trained model
    model = joblib.load('./models/xgboost_model_v1.pkl')

    # Step 2: Fetch today's games
    from datetime import date
    today = date.today().strftime('%Y-%m-%d')
    todays_games = fetch_nba_schedule(today, today)

    if todays_games.empty:
        print("No games scheduled for today")
        return

    # Step 3: Get historical data for feature calculation
    # (fetch recent games to calculate team stats)

    # Step 4: Calculate features for each game
    predictions = []

    for _, game in todays_games.iterrows():
        # Calculate same features as training
        features = calculate_game_features(game, historical_games)

        # Convert to DataFrame with same columns as training
        X = pd.DataFrame([features])

        # Predict
        home_win_prob = model.predict_proba(X)[0, 1]
        away_win_prob = 1 - home_win_prob

        predictions.append({
            'game_id': game['game_id'],
            'home_team': game['home_team_name'],
            'away_team': game['away_team_name'],
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'confidence': abs(home_win_prob - 0.5) * 2
        })

    # Step 5: Store in Supabase
    store_predictions_in_supabase(predictions)

    return predictions
```

---

## 📌 Summary: Your Implementation Checklist

### Week 1: Data Foundation
- [ ] Implement `fetch_nba_schedule()`
- [ ] Implement `fetch_team_stats()`
- [ ] Test fetching 1 week of games
- [ ] Store games in Supabase
- [ ] Verify data in Supabase dashboard

### Week 2: Historical Data
- [ ] Fetch 2-3 seasons of historical games
- [ ] Implement `calculate_game_features()`
- [ ] Create training dataset
- [ ] Validate: ~2000-3000 games with features

### Week 3: Model Training
- [ ] Implement `prepare_training_dataset()`
- [ ] Train baseline XGBoost model
- [ ] Evaluate: Target >60% accuracy
- [ ] Save model to disk
- [ ] Generate feature importance plot

### Week 4: Predictions & Integration
- [ ] Implement `predict_todays_games()`
- [ ] Test predictions on recent games
- [ ] Store predictions in Supabase
- [ ] Verify frontend shows real predictions
- [ ] Set up daily automated predictions

---

**Start with Phase 1, Step 1.2 tomorrow. Get `fetch_nba_schedule()` working first!**
