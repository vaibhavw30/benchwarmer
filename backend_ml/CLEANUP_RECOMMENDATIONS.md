# File Cleanup Recommendations

After implementing the ensemble model, here are files that may no longer be needed:

## Files Safe to Remove

### 1. `config.py` (❌ HAS SYNTAX ERROR)
- **Issue**: Lines 13-14 have a syntax error (`load_dotenv` split across lines)
- **Status**: Not imported by any main scripts (predict.py, train_model.py load dotenv directly)
- **Recommendation**: **DELETE** - Not needed, ensemble_config.py handles model config
- **Command**: `rm config.py`

### 2. `check_db.py` (Utility Script)
- **Purpose**: Simple database check utility
- **Usage**: One-off debugging
- **Recommendation**: **KEEP** or move to a `scripts/` folder for utilities
- **Why**: Useful for quick database checks

### 3. `backfill_and_test.py` (One-off Script)
- **Purpose**: Backfill historical games and test
- **Usage**: Probably used once during initial setup
- **Recommendation**: **DELETE** if backfill is complete
- **Alternative**: Archive to `scripts/archive/` if you want to keep it
- **Command**: `rm backfill_and_test.py` or `mkdir -p scripts/archive && mv backfill_and_test.py scripts/archive/`

## Files to Keep

### Core Model Files (DO NOT DELETE)
- ✅ `train_model.py` - Training pipeline
- ✅ `predict.py` - Prediction pipeline
- ✅ `backtest.py` - Backtesting
- ✅ `data_engine.py` - Data processing
- ✅ `elo_engine.py` - Elo ratings
- ✅ `player_impact_engine.py` - Injury impact
- ✅ `ensemble_config.py` - Ensemble configuration
- ✅ `migrate_db.py` - Database migration helper
- ✅ `update_game_results.py` - Game result updates

### Model Artifacts (DO NOT DELETE)
- ✅ `xgboost_nba_model.pkl` - XGBoost model
- ✅ `ridge_nba_model.pkl` - Ridge model
- ✅ `feature_scaler.pkl` - Feature scaler
- ✅ `ensemble_weights.json` - Optimal weights

### Data Files (DO NOT DELETE)
- ✅ `nba_training_cache.csv` - Training data cache
- ✅ `player_impact_scores.csv` - Player impact data

### Analysis/Testing Files (Optional - Keep for Future Use)
- ⚠️ `model_bias_analyzer.py` - Useful for analyzing model biases
- ⚠️ `test_player_impact.py` - Useful for testing player impact calculations

### Documentation
- ✅ `README.md`
- ✅ `supabase_schema.sql`

## Recommended Cleanup Commands

```bash
# Navigate to backend_ml directory
cd /Users/vaibhav.wudaru/nba-holistic-predictor/backend_ml

# Remove config.py (has syntax error, not used)
rm config.py

# Optional: Remove backfill script if no longer needed
rm backfill_and_test.py

# Optional: Create scripts folder for utilities
mkdir -p scripts/utilities
mv check_db.py scripts/utilities/
```

## After Cleanup

You should have these core files:
```
backend_ml/
├── train_model.py           # Core: Training
├── predict.py               # Core: Predictions
├── backtest.py              # Core: Backtesting
├── data_engine.py           # Core: Data processing
├── elo_engine.py            # Core: Elo ratings
├── player_impact_engine.py  # Core: Injury impact
├── update_game_results.py   # Core: Game updates
├── ensemble_config.py       # Core: Configuration
├── migrate_db.py            # Utility: Migration
├── xgboost_nba_model.pkl    # Model artifact
├── ridge_nba_model.pkl      # Model artifact
├── feature_scaler.pkl       # Model artifact
├── ensemble_weights.json    # Model artifact
├── nba_training_cache.csv   # Data cache
├── player_impact_scores.csv # Data cache
└── README.md                # Documentation
```

## Summary

- **Delete**: config.py (broken, unused)
- **Optional Delete**: backfill_and_test.py (one-off script)
- **Keep Everything Else**: Core functionality + useful utilities
