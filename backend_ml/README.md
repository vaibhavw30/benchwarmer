# NBA Holistic Predictor - Backend ML

This directory contains the machine learning backend for predicting NBA game outcomes using XGBoost.

## Setup

### 1. Install Python Dependencies

```bash
cd backend_ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in this directory:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_key

# NBA Data API (if using paid service)
NBA_API_KEY=your_api_key_here
```

### 3. Module Overview

```
backend_ml/
├── data_engine.py    # Data collection and feature engineering
├── train_model.py    # Model training and evaluation
├── predict.py        # Generate predictions for games
└── requirements.txt  # Python dependencies
```

## Usage Workflow

### Step 1: Collect Historical Data

```python
from data_engine import update_daily_data, initialize_supabase

# Initialize Supabase connection
supabase = initialize_supabase()

# Fetch and store historical data
update_daily_data()
```

### Step 2: Train the Model

```python
from train_model import train_full_pipeline

# Train model with hyperparameter tuning
results = train_full_pipeline(
    start_season='2018-19',
    end_season='2023-24',
    tune_hyperparameters=True
)
```

This will:
- Load historical game data
- Engineer features
- Train XGBoost model
- Evaluate performance
- Save model to `./models/`

### Step 3: Generate Predictions

```python
from predict import predict_todays_games

# Generate predictions for today's games
predictions = predict_todays_games()

# Predictions are automatically stored in Supabase
for pred in predictions:
    print(f"{pred['away_team']} @ {pred['home_team']}")
    print(f"Win probability: {pred['home_win_probability']:.1%}")
```

### Step 4: Schedule Daily Updates

Set up a cron job or cloud scheduler to run predictions daily:

```bash
# Run every morning at 8 AM
0 8 * * * cd /path/to/backend_ml && python predict.py
```

## Implementation Checklist

### Data Collection (`data_engine.py`)
- [ ] Implement `fetch_nba_schedule()` - Get game schedules
- [ ] Implement `fetch_team_stats()` - Get team statistics
- [ ] Implement `fetch_player_stats()` - Get player data
- [ ] Implement `calculate_advanced_features()` - Feature engineering
- [ ] Implement `store_game_data()` - Save to Supabase
- [ ] Set up daily data updates

### Model Training (`train_model.py`)
- [ ] Implement `load_training_data()` - Load historical data
- [ ] Implement `prepare_features_and_labels()` - Process data
- [ ] Implement `train_baseline_model()` - Train initial model
- [ ] Implement `hyperparameter_tuning()` - Optimize model
- [ ] Implement `evaluate_model()` - Assess performance
- [ ] Implement `save_model()` - Persist trained model

### Predictions (`predict.py`)
- [ ] Implement `load_production_model()` - Load trained model
- [ ] Implement `get_game_features()` - Prepare features for prediction
- [ ] Implement `predict_game()` - Generate single prediction
- [ ] Implement `predict_todays_games()` - Daily prediction pipeline
- [ ] Implement `log_prediction_performance()` - Track accuracy

## Key Features to Engineer

When implementing `calculate_advanced_features()`, include:

### Team Performance Metrics
- Points per game (offense)
- Points allowed per game (defense)
- Offensive/Defensive rating
- Net rating
- Pace (possessions per game)

### Recent Form
- Last 5/10/20 game performance
- Winning/losing streaks
- Home vs away splits
- Performance vs playoff teams

### Matchup Specific
- Head-to-head record
- Rest days differential
- Back-to-back game indicator
- Travel distance

### Player Impact
- Key player availability
- Injury impact score
- Roster depth
- Star player efficiency

## Model Performance Goals

Target metrics for a production-ready model:

- **Accuracy**: > 65% (better than random 50%)
- **ROC-AUC**: > 0.70
- **Calibration**: Predictions should be well-calibrated (70% predictions are correct 70% of the time)
- **Confidence**: High-confidence predictions (>75%) should have >70% accuracy

## Data Sources

Consider these data sources:

1. **Official NBA Stats API**: stats.nba.com
2. **Ball Don't Lie API**: Free NBA stats API
3. **ESPN API**: Game schedules and scores
4. **Sports Reference**: Historical data (web scraping)
5. **Odds API**: Betting lines for validation

## Next Steps

1. Start with `data_engine.py` - Get data flowing into Supabase
2. Build up historical dataset (at least 2-3 seasons)
3. Experiment with features in `train_model.py`
4. Iterate on model performance
5. Deploy `predict.py` for daily predictions
6. Build API layer to serve predictions to frontend

## Notes

- Always use temporal train/test splits (no future data in training)
- Monitor prediction performance and retrain periodically
- Consider ensemble methods if single model plateaus
- Track feature importance to understand model decisions
