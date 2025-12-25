-- ==============================================================================
-- NBA Holistic Predictor - Supabase Database Schema
-- ==============================================================================
-- Run this SQL in your Supabase SQL Editor to create the required tables
-- Dashboard: https://supabase.com/dashboard/project/YOUR_PROJECT/editor

-- ==============================================================================
-- 1. TEAMS TABLE
-- ==============================================================================
-- Stores NBA team information
CREATE TABLE IF NOT EXISTS teams (
    id BIGSERIAL PRIMARY KEY,
    team_id INTEGER UNIQUE NOT NULL, -- NBA API team ID
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    city VARCHAR(100),
    conference VARCHAR(20),
    division VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_teams_team_id ON teams(team_id);

-- ==============================================================================
-- 2. GAMES TABLE
-- ==============================================================================
-- Stores NBA game information
CREATE TABLE IF NOT EXISTS games (
    id BIGSERIAL PRIMARY KEY,
    game_id VARCHAR(20) UNIQUE NOT NULL, -- NBA API game ID (e.g., "0022300123")
    game_date DATE NOT NULL,
    season VARCHAR(10) NOT NULL, -- e.g., "2024-25"
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),
    status VARCHAR(20) DEFAULT 'upcoming', -- 'upcoming', 'live', 'completed'
    home_score INTEGER,
    away_score INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_games_game_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_home_team ON games(home_team_id);
CREATE INDEX IF NOT EXISTS idx_games_away_team ON games(away_team_id);

-- ==============================================================================
-- 3. GAME_PREDICTIONS TABLE
-- ==============================================================================
-- Stores AI model predictions for games
CREATE TABLE IF NOT EXISTS game_predictions (
    id BIGSERIAL PRIMARY KEY,
    game_id VARCHAR(20) NOT NULL UNIQUE REFERENCES games(game_id) ON DELETE CASCADE,
    home_team_id INTEGER REFERENCES teams(team_id),
    away_team_id INTEGER REFERENCES teams(team_id),

    -- Prediction probabilities
    home_win_probability DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    away_win_probability DECIMAL(5,4) NOT NULL,

    -- Predicted winner
    predicted_winner VARCHAR(10) NOT NULL, -- 'Home' or 'Away'
    confidence_score DECIMAL(5,4), -- Max of home/away probability

    -- Model features (for transparency)
    home_efg_pct DECIMAL(5,4),
    away_efg_pct DECIMAL(5,4),
    home_fatigue_score DECIMAL(5,4),
    away_fatigue_score DECIMAL(5,4),
    home_momentum DECIMAL(5,4),
    away_momentum DECIMAL(5,4),
    altitude_advantage INTEGER,

    -- Vegas comparison (optional)
    vegas_home_odds INTEGER,
    vegas_spread DECIMAL(4,1),
    model_edge DECIMAL(5,4), -- Difference from Vegas implied probability

    -- Metadata
    model_version VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON game_predictions(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON game_predictions(created_at DESC);

-- ==============================================================================
-- 4. TEAM_STATS TABLE (Optional - for historical tracking)
-- ==============================================================================
CREATE TABLE IF NOT EXISTS team_stats (
    id BIGSERIAL PRIMARY KEY,
    team_id INTEGER REFERENCES teams(team_id),
    date DATE NOT NULL,
    season VARCHAR(10) NOT NULL,

    -- Four Factors
    efg_pct DECIMAL(5,4),
    tov_pct DECIMAL(5,4),
    orb_pct DECIMAL(5,4),
    ft_rate DECIMAL(5,4),

    -- Additional stats
    off_rating DECIMAL(6,2),
    def_rating DECIMAL(6,2),
    pace DECIMAL(5,2),

    -- Rolling averages
    efg_pct_ewma DECIMAL(5,4),
    off_rating_ewma DECIMAL(6,2),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(team_id, date)
);

CREATE INDEX IF NOT EXISTS idx_team_stats_team_date ON team_stats(team_id, date DESC);

-- ==============================================================================
-- 5. PREDICTION_RESULTS TABLE (Optional - for backtesting)
-- ==============================================================================
-- Tracks accuracy of predictions after games complete
CREATE TABLE IF NOT EXISTS prediction_results (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES game_predictions(id) ON DELETE CASCADE,
    game_id VARCHAR(20) REFERENCES games(game_id),

    -- Actual result
    actual_winner VARCHAR(10), -- 'Home' or 'Away'
    was_correct BOOLEAN,

    -- For ROI tracking
    bet_placed BOOLEAN DEFAULT FALSE,
    bet_amount DECIMAL(10,2),
    bet_profit_loss DECIMAL(10,2),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_was_correct ON prediction_results(was_correct);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON prediction_results(created_at DESC);

-- ==============================================================================
-- 6. ROW LEVEL SECURITY (RLS) - Optional but recommended
-- ==============================================================================
-- Enable RLS for security
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE games ENABLE ROW LEVEL SECURITY;
ALTER TABLE game_predictions ENABLE ROW LEVEL SECURITY;

-- Allow public read access (anyone can view)
CREATE POLICY "Public read access on teams" ON teams FOR SELECT USING (true);
CREATE POLICY "Public read access on games" ON games FOR SELECT USING (true);
CREATE POLICY "Public read access on predictions" ON game_predictions FOR SELECT USING (true);

-- Only authenticated users (service role) can insert/update
CREATE POLICY "Service role can insert teams" ON teams FOR INSERT WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role can update teams" ON teams FOR UPDATE USING (auth.role() = 'service_role');
CREATE POLICY "Service role can insert games" ON games FOR ALL WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service role can insert predictions" ON game_predictions FOR ALL WITH CHECK (auth.role() = 'service_role');

-- ==============================================================================
-- 7. SAMPLE DATA - NBA Teams (Run this to populate teams)
-- ==============================================================================
INSERT INTO teams (team_id, name, abbreviation, city, conference, division) VALUES
(1610612737, 'Atlanta Hawks', 'ATL', 'Atlanta', 'Eastern', 'Southeast'),
(1610612738, 'Boston Celtics', 'BOS', 'Boston', 'Eastern', 'Atlantic'),
(1610612751, 'Brooklyn Nets', 'BKN', 'Brooklyn', 'Eastern', 'Atlantic'),
(1610612766, 'Charlotte Hornets', 'CHA', 'Charlotte', 'Eastern', 'Southeast'),
(1610612741, 'Chicago Bulls', 'CHI', 'Chicago', 'Eastern', 'Central'),
(1610612739, 'Cleveland Cavaliers', 'CLE', 'Cleveland', 'Eastern', 'Central'),
(1610612742, 'Dallas Mavericks', 'DAL', 'Dallas', 'Western', 'Southwest'),
(1610612743, 'Denver Nuggets', 'DEN', 'Denver', 'Western', 'Northwest'),
(1610612765, 'Detroit Pistons', 'DET', 'Detroit', 'Eastern', 'Central'),
(1610612744, 'Golden State Warriors', 'GSW', 'Golden State', 'Western', 'Pacific'),
(1610612745, 'Houston Rockets', 'HOU', 'Houston', 'Western', 'Southwest'),
(1610612754, 'Indiana Pacers', 'IND', 'Indiana', 'Eastern', 'Central'),
(1610612746, 'Los Angeles Clippers', 'LAC', 'Los Angeles', 'Western', 'Pacific'),
(1610612747, 'Los Angeles Lakers', 'LAL', 'Los Angeles', 'Western', 'Pacific'),
(1610612763, 'Memphis Grizzlies', 'MEM', 'Memphis', 'Western', 'Southwest'),
(1610612748, 'Miami Heat', 'MIA', 'Miami', 'Eastern', 'Southeast'),
(1610612749, 'Milwaukee Bucks', 'MIL', 'Milwaukee', 'Eastern', 'Central'),
(1610612750, 'Minnesota Timberwolves', 'MIN', 'Minnesota', 'Western', 'Northwest'),
(1610612740, 'New Orleans Pelicans', 'NOP', 'New Orleans', 'Western', 'Southwest'),
(1610612752, 'New York Knicks', 'NYK', 'New York', 'Eastern', 'Atlantic'),
(1610612760, 'Oklahoma City Thunder', 'OKC', 'Oklahoma City', 'Western', 'Northwest'),
(1610612753, 'Orlando Magic', 'ORL', 'Orlando', 'Eastern', 'Southeast'),
(1610612755, 'Philadelphia 76ers', 'PHI', 'Philadelphia', 'Eastern', 'Atlantic'),
(1610612756, 'Phoenix Suns', 'PHX', 'Phoenix', 'Western', 'Pacific'),
(1610612757, 'Portland Trail Blazers', 'POR', 'Portland', 'Western', 'Northwest'),
(1610612758, 'Sacramento Kings', 'SAC', 'Sacramento', 'Western', 'Pacific'),
(1610612759, 'San Antonio Spurs', 'SAS', 'San Antonio', 'Western', 'Southwest'),
(1610612761, 'Toronto Raptors', 'TOR', 'Toronto', 'Eastern', 'Atlantic'),
(1610612762, 'Utah Jazz', 'UTA', 'Utah', 'Western', 'Northwest'),
(1610612764, 'Washington Wizards', 'WAS', 'Washington', 'Eastern', 'Southeast')
ON CONFLICT (team_id) DO NOTHING;

-- ==============================================================================
-- DONE! Your database is ready for the NBA Predictor
-- ==============================================================================
-- Next steps:
-- 1. Run this SQL in Supabase SQL Editor
-- 2. Copy your SUPABASE_URL and SUPABASE_SERVICE_KEY to backend_ml/.env.local
-- 3. Run predict.py to start uploading predictions
-- 4. Frontend will automatically fetch and display them
