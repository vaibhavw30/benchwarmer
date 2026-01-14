# NBA Holistic Predictor - Model Parameters & Features

## Overview
This document explains all the features and parameters used by the XGBoost prediction model to forecast NBA game outcomes.

---

## 📊 Model Input Features (13 Total)

### **Four Factors** (Dean Oliver's Basketball Analytics)

The Four Factors are statistically proven to be the most important determinants of basketball success:

#### 1. **Effective Field Goal Percentage (EFG%)**
- **What it is**: Shooting efficiency adjusted for 3-pointers being worth more
- **Formula**: `(FG + 0.5 × 3P) / FGA`
- **Why it matters**: The single most important factor in winning games
- **Home Feature**: `EFG_PCT_EWMA_H`
- **Away Feature**: `EFG_PCT_EWMA_A`
- **Range**: 0.45 - 0.60 (45% - 60%)

#### 2. **Turnover Percentage (TOV%)**
- **What it is**: How often a team turns the ball over per 100 possessions
- **Formula**: `TOV / (FGA + 0.44 × FTA + TOV)`
- **Why it matters**: Fewer turnovers = more scoring opportunities
- **Home Feature**: `TOV_PCT_EWMA_H`
- **Away Feature**: `TOV_PCT_EWMA_A`
- **Range**: 0.10 - 0.18 (10% - 18%, lower is better)

#### 3. **Offensive Rebound Percentage (ORB%)**
- **What it is**: Percentage of available offensive rebounds grabbed
- **Formula**: `Team ORB / (Team ORB + Opponent DRB)`
- **Why it matters**: Second-chance points and extended possessions
- **Home Feature**: `ORB_PCT_EWMA_H`
- **Away Feature**: `ORB_PCT_EWMA_A`
- **Range**: 0.20 - 0.35 (20% - 35%)
- **Note**: Dynamically calculated based on both teams' rebounding

#### 4. **Free Throw Rate (FT Rate)**
- **What it is**: How often a team gets to the free-throw line
- **Formula**: `FTA / FGA`
- **Why it matters**: Free throws are the most efficient points in basketball
- **Home Feature**: `FT_RATE_EWMA_H`
- **Away Feature**: `FT_RATE_EWMA_A`
- **Range**: 0.15 - 0.30 (15% - 30%)

---

### **Fatigue & Rest**

#### 5. **Fatigue Score (Home)**
- **What it is**: Measure of how tired the home team is
- **Formula**: `1 if (days since last game ≤ 1) else 0 × (1 - Win%)`
- **Why it matters**: Back-to-back games significantly impact performance
- **Feature**: `FATIGUE_SCORE_H`
- **Range**: 0.0 - 1.0 (higher = more tired)
- **Impact**: Teams on back-to-backs lose ~10% more often

#### 6. **Fatigue Score (Away)**
- **What it is**: Measure of how tired the away team is
- **Formula**: `1 if (days since last game ≤ 1) else 0 × (1 - Win%)`
- **Feature**: `FATIGUE_SCORE_A`
- **Range**: 0.0 - 1.0
- **Note**: Away teams are more impacted by fatigue due to travel

---

### **Momentum & Form**

#### 7. **Momentum (Home)**
- **What it is**: Recent performance vs season average
- **Formula**: `Recent Offensive Rating (EWMA) / Season Avg Offensive Rating`
- **Why it matters**: Teams on hot/cold streaks continue that trend
- **Feature**: `MOMENTUM_H`
- **Range**: 0.85 - 1.15
  - `> 1.0` = Playing above season average
  - `< 1.0` = Playing below season average
  - `1.05` = Playing 5% better than usual

#### 8. **Momentum (Away)**
- **What it is**: Recent performance vs season average
- **Formula**: Same as home
- **Feature**: `MOMENTUM_A`
- **Range**: 0.85 - 1.15

---

### **Home Court Advantage**

#### 9. **Altitude Advantage**
- **What it is**: Elevation of home arena (feet above sea level)
- **Why it matters**: Higher altitude = less oxygen = visiting team fatigue
- **Feature**: `HOME_ALTITUDE`
- **Range**: 0 - 5,280 feet
- **Key Values**:
  - Denver Nuggets: **5,280 ft** (mile high)
  - Utah Jazz: **4,226 ft**
  - Phoenix Suns: **1,086 ft**
  - Most teams: **0-500 ft**
- **Impact**: Denver has significant home court advantage due to altitude

---

## 📈 How Features Are Calculated

### **EWMA (Exponentially Weighted Moving Average)**
- Most features use EWMA instead of simple averages
- **Why**: Recent games matter more than games from months ago
- **Alpha (decay factor)**: 0.1
- **Formula**: `EWMA[t] = alpha × value[t] + (1 - alpha) × EWMA[t-1]`
- **Effect**: Last 10 games weighted heavily, older games fade

### **Dynamic Calculations**
- **ORB%**: Calculated for each matchup based on both teams' rebounding stats
- **Fatigue**: Recalculated for each game based on days of rest
- **Momentum**: Updated after every game

---

## 🎯 Model Output

### **Predictions**
- **Home Win Probability**: 0.0 - 1.0 (0% - 100%)
- **Away Win Probability**: `1 - Home Win Probability`
- **Confidence Score**: `max(Home%, Away%)`
- **Predicted Winner**: "Home" or "Away"

### **Additional Context**
- **Vegas Odds Comparison**: Model vs betting markets
- **AI Explanation**: GPT-5.2 generated reasoning
- **Value Bets**: When model disagrees with Vegas by >10%

---

## 🧠 Model Details

### **Algorithm**: XGBoost (Gradient Boosted Decision Trees)

### **Hyperparameters** (Optimized via Grid Search)
- `learning_rate`: 0.05
- `max_depth`: 3
- `n_estimators`: 150
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `scale_pos_weight`: 0.9

### **Training Data**
- **Dataset**: 10,732 NBA games (past 3 seasons)
- **Split**: 85% train, 15% test
- **Accuracy**: **61.55%**
- **Features Used**: 13 (listed above)

### **Feature Importance** (Top 5)
1. `EFG_PCT_EWMA_H` - 12.3%
2. `EFG_PCT_EWMA_A` - 11.9%
3. `FATIGUE_SCORE_A` - 11.3%
4. `FATIGUE_SCORE_H` - 8.2%
5. `HOME_ALTITUDE` - 7.4%

---

## 📊 Data Storage (Supabase)

### **game_predictions Table Columns**

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | VARCHAR(20) | NBA API game ID |
| `date` | DATE | Game date |
| `home_team_id` | INTEGER | Home team NBA ID |
| `away_team_id` | INTEGER | Away team NBA ID |
| `home_win_probability` | DECIMAL(5,4) | Model prediction (0-1) |
| `away_win_probability` | DECIMAL(5,4) | 1 - home probability |
| `predicted_winner` | VARCHAR(10) | "Home" or "Away" |
| `confidence_score` | DECIMAL(5,4) | Max probability |
| `home_efg_pct` | DECIMAL(5,4) | Home team EFG% |
| `away_efg_pct` | DECIMAL(5,4) | Away team EFG% |
| `home_fatigue_score` | DECIMAL(5,4) | Home fatigue 0-1 |
| `away_fatigue_score` | DECIMAL(5,4) | Away fatigue 0-1 |
| `home_momentum` | DECIMAL(5,4) | Home recent form |
| `away_momentum` | DECIMAL(5,4) | Away recent form |
| `altitude_advantage` | INTEGER | Feet above sea level |
| `explanation` | TEXT | GPT-5.2 generated reasoning |
| `model_version` | VARCHAR(20) | "four_factors_v1" |

---

## 🔄 Prediction Workflow

1. **Fetch Today's Games** (NBA API)
   - Get matchups for current date (Eastern Time)

2. **Get Latest Team Stats** (Last 3 Seasons)
   - Calculate Four Factors
   - Calculate EWMA rolling averages
   - Determine fatigue scores

3. **Build Feature Vector** (13 features per game)
   - Home team: EFG%, TOV%, ORB%, FT%, Fatigue, Momentum
   - Away team: EFG%, TOV%, ORB%, FT%, Fatigue, Momentum
   - Home advantage: Altitude

4. **Model Prediction**
   - XGBoost outputs probability
   - Calculate confidence score

5. **Generate AI Explanation** (Azure OpenAI GPT-5.2)
   - Analyze key factors
   - Generate 2-3 sentence casual explanation
   - Fallback to rule-based if API unavailable

6. **Compare with Vegas Odds** (The Odds API)
   - Fetch live betting lines
   - Calculate implied probability
   - Flag value bets (>10% difference)

7. **Save to Supabase**
   - Upsert game record
   - Upsert prediction with all features
   - Store AI explanation

8. **Display on Frontend**
   - Filter games (last 2 days)
   - Sort by date (newest first)
   - Show predictions with explanations

---

## 🎓 References

### **Four Factors**
- Dean Oliver, "Basketball on Paper" (2004)
- Proven to account for ~90% of team success

### **Altitude Impact**
- Studies show 2-3% performance decrease at high altitude
- Denver Nuggets: 60%+ home win rate historically

### **EWMA**
- Better than simple moving average for time-series predictions
- Alpha = 0.1 balances recent form with stability

### **XGBoost**
- Industry-standard gradient boosting algorithm
- Handles non-linear relationships well
- Resistant to overfitting with proper tuning

---

## 🚀 Future Improvements

### Potential Features to Add
- **Injuries**: Key player availability
- **Head-to-Head**: Historical matchup records
- **Home/Away Splits**: Team performance by location
- **Pace**: Game tempo impact
- **Defensive Rating**: Four Factors only cover offense heavily
- **Travel Distance**: Cross-country games impact away teams
- **Schedule**: 3-in-4 nights, 4-in-5, etc.

### Model Enhancements
- **Ensemble**: Combine XGBoost with Neural Network
- **Real-time**: Update during games
- **Player-level**: Individual impact models
- **Bayesian**: Incorporate prior beliefs and uncertainty

---

*Last Updated: December 26, 2025*
*Model Version: four_factors_v1*
*Accuracy: 61.55% on test set*
