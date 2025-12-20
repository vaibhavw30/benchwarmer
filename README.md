# NBA Holistic Prediction Platform

An AI-powered platform that predicts NBA game outcomes using machine learning (XGBoost) and provides real-time predictions through a modern web interface.

## Project Overview

This platform combines:
- **Machine Learning**: XGBoost models trained on historical NBA data
- **Modern Frontend**: React + Vite + Tailwind CSS
- **Cloud Database**: Supabase for data storage and real-time updates
- **Comprehensive Features**: Team stats, player data, and advanced metrics

## Project Structure

```
nba-holistic-predictor/
│
├── frontend_web/              # React frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Navbar.jsx     # Navigation bar
│   │   │   └── GameCard.jsx   # Game prediction card
│   │   ├── App.jsx            # Main dashboard
│   │   ├── main.jsx           # React entry point
│   │   ├── index.css          # Global styles (Tailwind)
│   │   └── supabaseClient.js  # Supabase connection
│   ├── public/                # Static assets
│   ├── index.html             # HTML entry point
│   ├── package.json           # Dependencies
│   ├── vite.config.js         # Vite configuration
│   └── tailwind.config.js     # Tailwind configuration
│
├── backend_ml/                # Python ML backend
│   ├── data_engine.py         # Data collection & feature engineering
│   ├── train_model.py         # Model training & evaluation
│   ├── predict.py             # Generate predictions
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Backend documentation
│
├── SUPABASE_SCHEMA.sql        # Database schema
└── README.md                  # This file
```

## Tech Stack

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first styling
- **Lucide React**: Modern icon library
- **Supabase JS Client**: Database integration

### Backend
- **Python 3.10+**: Core language
- **XGBoost**: Machine learning model
- **Pandas & NumPy**: Data processing
- **Scikit-learn**: Model evaluation
- **Supabase**: Cloud PostgreSQL database

### Infrastructure
- **Supabase**: Backend-as-a-Service (database, auth, storage)
- **Vercel/Netlify**: Frontend hosting (recommended)
- **Cloud scheduler**: Daily prediction updates

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- Supabase account (free tier available)
- NBA data API access (optional, see data sources)

### 1. Database Setup

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Go to SQL Editor in your Supabase dashboard
3. Copy the contents of `SUPABASE_SCHEMA.sql`
4. Execute the SQL to create tables
5. Note your project URL and anon key

### 2. Frontend Setup

```bash
cd frontend_web

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Add your Supabase credentials to .env
# VITE_SUPABASE_URL=your_supabase_url
# VITE_SUPABASE_ANON_KEY=your_anon_key

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. Backend Setup

```bash
cd backend_ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cat > .env << EOF
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
NBA_API_KEY=your_api_key_if_needed
EOF
```

## Implementation Roadmap

### Phase 1: Data Collection ✅ (Scaffolding Complete)
- [x] Set up project structure
- [ ] Implement NBA data fetching
- [ ] Store historical game data in Supabase
- [ ] Collect team and player statistics
- [ ] Engineer prediction features

### Phase 2: Model Development
- [ ] Load and prepare training data
- [ ] Train baseline XGBoost model
- [ ] Perform hyperparameter tuning
- [ ] Evaluate model performance
- [ ] Achieve >65% accuracy target

### Phase 3: Prediction Pipeline
- [ ] Load trained model for inference
- [ ] Fetch today's games and data
- [ ] Generate predictions
- [ ] Store predictions in Supabase
- [ ] Set up daily automated predictions

### Phase 4: Frontend Integration
- [ ] Connect frontend to Supabase
- [ ] Display real predictions (replace mock data)
- [ ] Implement working filters
- [ ] Add prediction details modal
- [ ] Show prediction confidence levels

### Phase 5: Advanced Features
- [ ] Real-time game updates
- [ ] Historical prediction tracking
- [ ] Model performance dashboard
- [ ] Betting odds comparison
- [ ] User accounts and favorites
- [ ] Mobile responsive design

## Usage

### Training a Model

```python
# In backend_ml/
from train_model import train_full_pipeline

# Train model with historical data
results = train_full_pipeline(
    start_season='2018-19',
    end_season='2023-24',
    tune_hyperparameters=True
)
```

### Generating Predictions

```python
# In backend_ml/
from predict import predict_todays_games

# Generate predictions for today's games
predictions = predict_todays_games()
```

### Viewing Predictions

1. Start the frontend: `npm run dev` in `frontend_web/`
2. Open `http://localhost:3000`
3. View today's predictions on the dashboard

## Key Features (Planned)

- **AI Win Probabilities**: Machine learning predictions for every game
- **Confidence Scores**: Know which predictions are most reliable
- **Team Analytics**: View detailed team statistics and trends
- **Historical Performance**: Track prediction accuracy over time
- **Real-time Updates**: Live game status and score updates
- **Advanced Filters**: Search by team, date, or confidence level

## Data Sources

Recommended NBA data sources:

1. **NBA Stats API** (stats.nba.com): Official NBA statistics
2. **Ball Don't Lie API**: Free NBA data API
3. **ESPN API**: Game schedules and scores
4. **Basketball Reference**: Historical data
5. **The Odds API**: Betting lines for comparison

## Model Features

The prediction model considers:

### Team Performance
- Offensive/defensive ratings
- Points per game
- Field goal percentages
- Recent form (last 5/10 games)

### Context
- Home court advantage
- Days of rest
- Back-to-back games
- Travel distance

### Matchup History
- Head-to-head record
- Recent meetings
- Playoff implications

### Player Impact
- Key player availability
- Injury status
- Roster depth

## Performance Goals

Target metrics:
- **Accuracy**: >65% (baseline: 50% random)
- **High Confidence Accuracy**: >70% for predictions with 75%+ confidence
- **Calibration**: Well-calibrated probabilities
- **ROC-AUC**: >0.70

## Development Workflow

1. **Collect Data**: Use `data_engine.py` to fetch and store NBA data
2. **Train Model**: Use `train_model.py` to train and evaluate models
3. **Generate Predictions**: Use `predict.py` for daily predictions
4. **View Results**: Frontend displays predictions from Supabase

## Environment Variables

### Frontend (.env in frontend_web/)
```bash
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### Backend (.env in backend_ml/)
```bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_key
NBA_API_KEY=your_nba_api_key
```

## Deployment

### Frontend
Deploy to Vercel or Netlify:

```bash
cd frontend_web
npm run build
# Deploy dist/ folder
```

### Backend
Options:
1. **Cloud Functions**: Deploy prediction pipeline to AWS Lambda or Google Cloud Functions
2. **Scheduled Jobs**: Use GitHub Actions or cloud scheduler for daily predictions
3. **API Service**: Deploy FastAPI wrapper for real-time predictions

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

MIT License - feel free to use this for learning or personal projects.

## Next Steps

1. **Implement Data Collection**: Start with `backend_ml/data_engine.py`
2. **Gather Historical Data**: Collect 2-3 seasons of game data
3. **Train Initial Model**: Use `backend_ml/train_model.py`
4. **Test Predictions**: Generate predictions with `backend_ml/predict.py`
5. **Connect Frontend**: Replace mock data with real Supabase queries
6. **Iterate**: Improve features, model, and UI based on performance

## Support

For questions or issues:
1. Check the README files in each directory
2. Review the detailed comments in the code
3. Consult the Supabase documentation
4. Check XGBoost and scikit-learn docs for ML questions

---

**Built with:** React, Python, XGBoost, Supabase, Tailwind CSS

**Status:** 🚧 In Development - Scaffolding Complete
