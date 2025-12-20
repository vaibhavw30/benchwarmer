-- ============================================================================
-- NBA HOLISTIC PREDICTOR - SUPABASE DATABASE SCHEMA
-- ============================================================================
--
-- This file contains the SQL schema for the NBA prediction platform.
-- Run these commands in your Supabase SQL editor to set up the database.
--
-- SETUP INSTRUCTIONS:
-- 1. Go to your Supabase project dashboard
-- 2. Navigate to SQL Editor
-- 3. Create a new query
-- 4. Copy and paste this entire file
-- 5. Execute the query
--
-- ============================================================================

-- Enable UUID extension for generating unique IDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- TABLE: teams
-- Stores information about NBA teams
-- ============================================================================
CREATE TABLE IF NOT EXISTS teams (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id INTEGER UNIQUE NOT NULL,  -- NBA's official team ID
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    city VARCHAR(100),
    conference VARCHAR(20),  -- 'Eastern' or 'Western'
    division VARCHAR(20),    -- 'Atlantic', 'Central', etc.
    established_year INTEGER,
    logo_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster lookups
CREATE INDEX idx_teams_team_id ON teams(team_id);
CREATE INDEX idx_teams_abbreviation ON teams(abbreviation);

-- ============================================================================
-- TABLE: games
-- Stores information about NBA games (past and upcoming)
-- ============================================================================
CREATE TABLE IF NOT EXISTS games (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id VARCHAR(50) UNIQUE NOT NULL,  -- NBA's official game ID
    home_team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    away_team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    game_date TIMESTAMPTZ NOT NULL,
    season VARCHAR(20),  -- e.g., '2024-25'
    status VARCHAR(20) DEFAULT 'upcoming',  -- 'upcoming', 'live', 'completed'
    venue VARCHAR(200),

    -- Final scores (NULL if game hasn't finished)
    home_team_score INTEGER,
    away_team_score INTEGER,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX idx_games_game_date ON games(game_date);
CREATE INDEX idx_games_home_team ON games(home_team_id);
CREATE INDEX idx_games_away_team ON games(away_team_id);
CREATE INDEX idx_games_status ON games(status);
CREATE INDEX idx_games_season ON games(season);

-- ============================================================================
-- TABLE: team_stats
-- Stores daily/periodic team statistics
-- ============================================================================
CREATE TABLE IF NOT EXISTS team_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    season VARCHAR(20),

    -- Basic Stats
    games_played INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    win_percentage DECIMAL(5, 3),

    -- Offensive Stats
    points_per_game DECIMAL(5, 1),
    field_goal_percentage DECIMAL(5, 3),
    three_point_percentage DECIMAL(5, 3),
    free_throw_percentage DECIMAL(5, 3),
    assists_per_game DECIMAL(5, 1),
    turnovers_per_game DECIMAL(5, 1),

    -- Defensive Stats
    points_allowed_per_game DECIMAL(5, 1),
    rebounds_per_game DECIMAL(5, 1),
    steals_per_game DECIMAL(5, 1),
    blocks_per_game DECIMAL(5, 1),

    -- Advanced Stats
    offensive_rating DECIMAL(6, 2),
    defensive_rating DECIMAL(6, 2),
    net_rating DECIMAL(6, 2),
    pace DECIMAL(5, 2),
    true_shooting_percentage DECIMAL(5, 3),
    effective_field_goal_percentage DECIMAL(5, 3),

    -- Recent Form (last 10 games)
    last_10_wins INTEGER,
    last_10_losses INTEGER,

    -- Home/Away Splits
    home_wins INTEGER DEFAULT 0,
    home_losses INTEGER DEFAULT 0,
    away_wins INTEGER DEFAULT 0,
    away_losses INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure one record per team per date
    UNIQUE(team_id, date)
);

-- Create indexes
CREATE INDEX idx_team_stats_team_date ON team_stats(team_id, date DESC);
CREATE INDEX idx_team_stats_season ON team_stats(season);

-- ============================================================================
-- TABLE: player_stats
-- Stores player statistics (optional - for more advanced predictions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id INTEGER NOT NULL,  -- NBA's official player ID
    team_id UUID REFERENCES teams(id) ON DELETE CASCADE,
    player_name VARCHAR(100),
    date DATE NOT NULL,
    season VARCHAR(20),

    -- Status
    is_active BOOLEAN DEFAULT true,
    injury_status VARCHAR(50),  -- 'healthy', 'questionable', 'out', etc.

    -- Basic Stats
    games_played INTEGER DEFAULT 0,
    minutes_per_game DECIMAL(5, 2),
    points_per_game DECIMAL(5, 1),
    rebounds_per_game DECIMAL(5, 1),
    assists_per_game DECIMAL(5, 1),
    steals_per_game DECIMAL(5, 1),
    blocks_per_game DECIMAL(5, 1),

    -- Shooting Stats
    field_goal_percentage DECIMAL(5, 3),
    three_point_percentage DECIMAL(5, 3),
    free_throw_percentage DECIMAL(5, 3),

    -- Advanced Stats
    player_efficiency_rating DECIMAL(6, 2),
    usage_rate DECIMAL(5, 3),
    true_shooting_percentage DECIMAL(5, 3),

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(player_id, team_id, date)
);

-- Create indexes
CREATE INDEX idx_player_stats_team_date ON player_stats(team_id, date DESC);
CREATE INDEX idx_player_stats_player ON player_stats(player_id);

-- ============================================================================
-- TABLE: game_predictions
-- Stores AI-generated predictions for games
-- ============================================================================
CREATE TABLE IF NOT EXISTS game_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    game_id UUID REFERENCES games(id) ON DELETE CASCADE,

    -- Predictions
    home_win_probability DECIMAL(5, 3) NOT NULL,  -- 0.000 to 1.000
    away_win_probability DECIMAL(5, 3) NOT NULL,
    predicted_winner VARCHAR(10),  -- 'home' or 'away'
    confidence_score DECIMAL(5, 3),  -- 0.000 to 1.000

    -- Optional: Predicted scores
    predicted_home_score INTEGER,
    predicted_away_score INTEGER,
    predicted_point_spread DECIMAL(5, 1),

    -- Model Information
    model_version VARCHAR(50),
    prediction_timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Features used (stored as JSON for flexibility)
    features_used JSONB,

    -- Actual outcome (filled after game completes)
    actual_winner VARCHAR(10),  -- 'home' or 'away'
    prediction_correct BOOLEAN,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- One prediction per game per model version
    UNIQUE(game_id, model_version)
);

-- Create indexes
CREATE INDEX idx_predictions_game ON game_predictions(game_id);
CREATE INDEX idx_predictions_timestamp ON game_predictions(prediction_timestamp DESC);
CREATE INDEX idx_predictions_confidence ON game_predictions(confidence_score DESC);

-- ============================================================================
-- TABLE: prediction_performance
-- Tracks model performance over time
-- ============================================================================
CREATE TABLE IF NOT EXISTS prediction_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,

    -- Performance Metrics
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    accuracy DECIMAL(5, 3),

    -- Confidence-based accuracy
    high_confidence_predictions INTEGER DEFAULT 0,  -- confidence > 0.7
    high_confidence_correct INTEGER DEFAULT 0,
    high_confidence_accuracy DECIMAL(5, 3),

    -- Time-based metrics
    last_7_days_accuracy DECIMAL(5, 3),
    last_30_days_accuracy DECIMAL(5, 3),
    season_to_date_accuracy DECIMAL(5, 3),

    -- Advanced metrics
    roc_auc_score DECIMAL(5, 3),
    log_loss DECIMAL(6, 4),
    brier_score DECIMAL(5, 4),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(model_version, evaluation_date)
);

-- Create indexes
CREATE INDEX idx_performance_model_date ON prediction_performance(model_version, evaluation_date DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update_updated_at trigger to all tables
CREATE TRIGGER update_teams_updated_at BEFORE UPDATE ON teams
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_games_updated_at BEFORE UPDATE ON games
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_team_stats_updated_at BEFORE UPDATE ON team_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_player_stats_updated_at BEFORE UPDATE ON player_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON game_predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) - OPTIONAL
-- ============================================================================
-- Uncomment these if you want to enable Row Level Security
-- This is useful if you have multiple users/organizations

-- ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE games ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE team_stats ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE player_stats ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE game_predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE prediction_performance ENABLE ROW LEVEL SECURITY;

-- Example policy: Allow public read access to predictions
-- CREATE POLICY "Public predictions read access" ON game_predictions
--     FOR SELECT USING (true);

-- ============================================================================
-- SAMPLE DATA INSERTION (FOR TESTING)
-- ============================================================================
-- You can uncomment and run these to add sample teams

-- INSERT INTO teams (team_id, name, abbreviation, city, conference, division) VALUES
-- (1610612747, 'Los Angeles Lakers', 'LAL', 'Los Angeles', 'Western', 'Pacific'),
-- (1610612744, 'Golden State Warriors', 'GSW', 'San Francisco', 'Western', 'Pacific'),
-- (1610612738, 'Boston Celtics', 'BOS', 'Boston', 'Eastern', 'Atlantic'),
-- (1610612748, 'Miami Heat', 'MIA', 'Miami', 'Eastern', 'Southeast'),
-- (1610612749, 'Milwaukee Bucks', 'MIL', 'Milwaukee', 'Eastern', 'Central'),
-- (1610612755, 'Philadelphia 76ers', 'PHI', 'Philadelphia', 'Eastern', 'Atlantic'),
-- (1610612756, 'Phoenix Suns', 'PHX', 'Phoenix', 'Western', 'Pacific'),
-- (1610612742, 'Dallas Mavericks', 'DAL', 'Dallas', 'Western', 'Southwest'),
-- (1610612743, 'Denver Nuggets', 'DEN', 'Denver', 'Western', 'Northwest'),
-- (1610612746, 'Los Angeles Clippers', 'LAC', 'Los Angeles', 'Western', 'Pacific');

-- ============================================================================
-- USEFUL QUERIES FOR YOUR APPLICATION
-- ============================================================================

-- Get today's games with predictions
-- SELECT
--     g.game_id,
--     ht.name as home_team,
--     at.name as away_team,
--     g.game_date,
--     gp.home_win_probability,
--     gp.away_win_probability,
--     gp.confidence_score
-- FROM games g
-- JOIN teams ht ON g.home_team_id = ht.id
-- JOIN teams at ON g.away_team_id = at.id
-- LEFT JOIN game_predictions gp ON g.id = gp.game_id
-- WHERE DATE(g.game_date) = CURRENT_DATE
-- ORDER BY g.game_date;

-- Get team's recent performance
-- SELECT * FROM team_stats
-- WHERE team_id = (SELECT id FROM teams WHERE abbreviation = 'LAL')
-- ORDER BY date DESC
-- LIMIT 1;

-- Get prediction accuracy for a model version
-- SELECT
--     model_version,
--     accuracy,
--     high_confidence_accuracy,
--     total_predictions
-- FROM prediction_performance
-- WHERE model_version = 'v1.0.0'
-- ORDER BY evaluation_date DESC
-- LIMIT 1;

-- ============================================================================
-- NOTES
-- ============================================================================
-- 1. After running this schema, populate the teams table with all 30 NBA teams
-- 2. Set up your backend Python scripts to regularly update team_stats
-- 3. Use the data_engine.py module to fetch and store game data
-- 4. The frontend will query game_predictions to display results
-- 5. Consider setting up database backups in Supabase dashboard
-- 6. Monitor database performance and add indexes as needed
-- ============================================================================
