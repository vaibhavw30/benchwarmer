# NBA Holistic Predictor - Project Structure

## Visual Folder Structure

```
nba-holistic-predictor/
│
├── frontend_web/                          # React Frontend Application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx                 # 🎨 Top navigation bar with branding
│   │   │   └── GameCard.jsx               # 🎴 Individual game prediction card
│   │   │
│   │   ├── App.jsx                        # 📊 Main dashboard with game grid
│   │   ├── main.jsx                       # ⚛️  React application entry point
│   │   ├── index.css                      # 🎨 Global styles (Tailwind directives)
│   │   └── supabaseClient.js              # 🔌 Supabase database connection
│   │
│   ├── public/                            # Static assets (images, icons)
│   │
│   ├── index.html                         # 📄 HTML entry point
│   ├── package.json                       # 📦 NPM dependencies
│   ├── vite.config.js                     # ⚙️  Vite build configuration
│   ├── tailwind.config.js                 # 🎨 Tailwind CSS configuration
│   ├── postcss.config.js                  # 🎨 PostCSS configuration
│   ├── .env.example                       # 🔑 Environment variables template
│   └── .gitignore                         # 🚫 Git ignore rules
│
├── backend_ml/                            # Python ML Backend
│   ├── data_engine.py                     # 📊 Data collection & feature engineering
│   │                                      #    - Fetch NBA schedules
│   │                                      #    - Collect team/player stats
│   │                                      #    - Calculate advanced features
│   │                                      #    - Store data in Supabase
│   │
│   ├── train_model.py                     # 🤖 Model training & evaluation
│   │                                      #    - Load historical data
│   │                                      #    - Train XGBoost model
│   │                                      #    - Hyperparameter tuning
│   │                                      #    - Evaluate performance
│   │                                      #    - Save trained model
│   │
│   ├── predict.py                         # 🔮 Generate predictions
│   │                                      #    - Load trained model
│   │                                      #    - Fetch today's games
│   │                                      #    - Generate win probabilities
│   │                                      #    - Store predictions in Supabase
│   │
│   ├── requirements.txt                   # 📦 Python dependencies
│   ├── README.md                          # 📖 Backend documentation
│   ├── .env.example                       # 🔑 Environment variables template
│   └── .gitignore                         # 🚫 Git ignore rules
│
├── SUPABASE_SCHEMA.sql                    # 🗄️  Database schema definition
│                                          #    - teams table
│                                          #    - games table
│                                          #    - team_stats table
│                                          #    - player_stats table
│                                          #    - game_predictions table
│                                          #    - prediction_performance table
│
├── PROJECT_STRUCTURE.md                   # 📋 This file
└── README.md                              # 📖 Main project documentation
```

## File Purposes

### Frontend Files

| File | Purpose | Status |
|------|---------|--------|
| `App.jsx` | Main dashboard with game predictions grid, filters, and search | ✅ Wireframe complete |
| `Navbar.jsx` | Top navigation with app branding and logo | ✅ Wireframe complete |
| `GameCard.jsx` | Displays single game prediction with teams, probabilities, and confidence | ✅ Wireframe complete |
| `supabaseClient.js` | Supabase connection and helper functions | ✅ Ready for credentials |
| `index.css` | Tailwind directives and global dark mode styles | ✅ Complete |
| `main.jsx` | React entry point | ✅ Complete |
| `package.json` | NPM dependencies (React, Vite, Tailwind, Supabase) | ✅ Complete |
| `vite.config.js` | Vite dev server and build configuration | ✅ Complete |
| `tailwind.config.js` | Tailwind with NBA-themed dark colors | ✅ Complete |

### Backend Files

| File | Purpose | Status |
|------|---------|--------|
| `data_engine.py` | Data fetching, feature engineering, Supabase storage | 📝 Skeleton with detailed comments |
| `train_model.py` | XGBoost model training, tuning, and evaluation | 📝 Skeleton with detailed comments |
| `predict.py` | Load model, generate predictions for games | 📝 Skeleton with detailed comments |
| `requirements.txt` | Python packages (xgboost, pandas, sklearn, supabase) | ✅ Complete |
| `README.md` | Backend setup and implementation guide | ✅ Complete |

### Database Files

| File | Purpose | Status |
|------|---------|--------|
| `SUPABASE_SCHEMA.sql` | Complete PostgreSQL schema with tables, indexes, and sample queries | ✅ Ready to execute |

## Component Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    App.jsx (Dashboard)                    │  │
│  │  ┌────────────┐  ┌──────────────────────────────────┐    │  │
│  │  │ Navbar.jsx │  │   Game Predictions Grid          │    │  │
│  │  └────────────┘  │                                  │    │  │
│  │                  │  ┌──────────┐  ┌──────────┐      │    │  │
│  │  ┌──────────┐   │  │GameCard  │  │GameCard  │ ...  │    │  │
│  │  │ Filters  │   │  └──────────┘  └──────────┘      │    │  │
│  │  └──────────┘   └──────────────────────────────────┘    │  │
│  └──────────────────────────┬───────────────────────────────┘  │
└─────────────────────────────┼──────────────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  supabaseClient.js   │
                   └──────────┬───────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        SUPABASE DATABASE                        │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐ │
│  │  teams   │  │  games   │  │team_stats  │  │game_predictions││
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                   ┌──────────┴───────────┐
                   │   Backend Python     │
                   │                      │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────┐           │
│  │data_engine.py │→ │train_model.py│→ │predict.py │           │
│  │               │  │              │  │           │           │
│  │ Fetch NBA data│  │ Train XGBoost│  │ Generate  │           │
│  │ Calculate     │  │ Optimize     │  │ predictions│          │
│  │ features      │  │ Evaluate     │  │ Store in  │           │
│  │ Store in DB   │  │ Save model   │  │ Supabase  │           │
│  └───────────────┘  └──────────────┘  └───────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Phase (One-time / Periodic)

```
1. Historical NBA Games
   ↓
2. data_engine.py
   - Fetch game data from NBA APIs
   - Collect team/player statistics
   - Calculate features (ratings, form, matchups)
   - Store in Supabase
   ↓
3. train_model.py
   - Load historical data from Supabase
   - Prepare features and labels
   - Train XGBoost model
   - Evaluate performance
   - Save model to disk
   ↓
4. Trained Model (saved as .pkl file)
```

### Prediction Phase (Daily)

```
1. Upcoming Games Schedule
   ↓
2. predict.py
   - Load trained model
   - Fetch today's games
   - Collect current team/player stats
   - Calculate features (same as training)
   - Generate win probabilities
   ↓
3. Store predictions in Supabase
   (game_predictions table)
   ↓
4. Frontend queries Supabase
   ↓
5. User sees predictions on dashboard
```

## Technology Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND STACK                         │
├─────────────────────────────────────────────────────────────┤
│ React 18          │ UI Framework                           │
│ Vite              │ Build tool & dev server                │
│ Tailwind CSS      │ Utility-first styling                  │
│ Lucide React      │ Icon library                           │
│ @supabase/supabase-js │ Database client                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      BACKEND STACK                          │
├─────────────────────────────────────────────────────────────┤
│ Python 3.10+      │ Core language                          │
│ XGBoost           │ ML algorithm for predictions           │
│ Pandas            │ Data manipulation                      │
│ NumPy             │ Numerical computing                    │
│ Scikit-learn      │ ML utilities (split, metrics)          │
│ Supabase Python   │ Database client                        │
│ Joblib            │ Model serialization                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    DATABASE & HOSTING                       │
├─────────────────────────────────────────────────────────────┤
│ Supabase          │ PostgreSQL database + auth + storage   │
│ Vercel/Netlify    │ Frontend hosting (recommended)         │
│ Cloud Functions   │ Backend hosting (AWS Lambda, etc.)     │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Order

### ✅ Phase 1: Scaffolding (COMPLETE)
- [x] Project structure created
- [x] Frontend wireframe with mock data
- [x] Backend skeleton with detailed comments
- [x] Database schema designed
- [x] Documentation written

### 📋 Phase 2: Data Pipeline (NEXT)
- [ ] Implement `fetch_nba_schedule()` in data_engine.py
- [ ] Implement `fetch_team_stats()` in data_engine.py
- [ ] Implement `calculate_advanced_features()` in data_engine.py
- [ ] Populate Supabase with historical data
- [ ] Verify data quality and completeness

### 🤖 Phase 3: Model Training
- [ ] Implement `load_training_data()` in train_model.py
- [ ] Implement `prepare_features_and_labels()` in train_model.py
- [ ] Train baseline XGBoost model
- [ ] Perform hyperparameter tuning
- [ ] Evaluate model (target: >65% accuracy)

### 🔮 Phase 4: Predictions
- [ ] Implement `load_production_model()` in predict.py
- [ ] Implement `get_game_features()` in predict.py
- [ ] Generate and store predictions
- [ ] Set up daily automated runs

### 🎨 Phase 5: Frontend Integration
- [ ] Replace mock data with Supabase queries
- [ ] Implement working filters
- [ ] Add loading states
- [ ] Polish UI/UX
- [ ] Deploy to production

## File Statistics

| Category | File Count | Lines of Code (approx) |
|----------|------------|------------------------|
| Frontend | 12 files | ~800 lines |
| Backend | 7 files | ~1000 lines (with comments) |
| Database | 1 file | ~400 lines |
| Docs | 3 files | ~600 lines |
| **Total** | **23 files** | **~2800 lines** |

## Quick Start Commands

```bash
# Frontend
cd frontend_web
npm install
npm run dev

# Backend
cd backend_ml
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Database
# 1. Go to supabase.com and create project
# 2. Run SUPABASE_SCHEMA.sql in SQL editor
```

---

**Last Updated**: 2025-12-17
**Project Status**: Scaffolding Complete ✅
