import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import GameCard from './components/GameCard';
import { Search, Filter, Calendar, RefreshCw } from 'lucide-react';
import { supabase } from './supabaseClient';

/**
 * MOCK DATA
 * This data simulates what will eventually come from Supabase
 * Replace this with real API calls once backend is ready
 */
const App = () => {
  const [games, setGames] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all'); // 'all', 'upcoming', 'live', 'completed'
  const [minConfidence, setMinConfidence] = useState(0); // Confidence threshold (0-50)
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch game predictions from Supabase
   *
   * This queries the game_predictions table and joins with:
   * - games table (to get game details)
   * - teams table (to get team names)
   */
  const fetchGames = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Get date 2 days ago to show recent and upcoming games
      const twoDaysAgo = new Date();
      twoDaysAgo.setDate(twoDaysAgo.getDate() - 2);
      const filterDate = twoDaysAgo.toISOString().split('T')[0];

      // Query Supabase - fetch predictions with game and team data
      // Filter to show only recent/upcoming games and sort by game date
      const { data, error } = await supabase
        .from('game_predictions')
        .select(`
          *,
          games!game_predictions_game_id_fkey (
            game_id,
            game_date,
            status,
            home_team:home_team_id (name, abbreviation),
            away_team:away_team_id (name, abbreviation)
          )
        `)
        .gte('date', filterDate)
        .order('date', { ascending: false })
        .limit(50);

      if (error) {
        console.error('Supabase error:', error);
        setError('Failed to load predictions. Please check your connection.');
        setGames([]);
      } else if (data && data.length > 0) {
        // Transform Supabase data to match component format
        const today = new Date();
        today.setHours(0, 0, 0, 0); // Reset to midnight for date comparison

        const formattedGames = data.map(prediction => {
          const game = prediction.games;

          let dateStr = 'TBD';
          let gameDate = null;

          if (game?.game_date) {
            // Parse date as local time, not UTC, to avoid timezone shift
            const dateParts = game.game_date.split('-');
            gameDate = new Date(dateParts[0], dateParts[1] - 1, dateParts[2]);
            dateStr = gameDate.toLocaleString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
              weekday: 'short'
            });
          }

          // Determine game status based on date
          let gameStatus = game?.status || 'upcoming';
          if (gameDate) {
            const gameDateMidnight = new Date(gameDate);
            gameDateMidnight.setHours(0, 0, 0, 0);

            if (gameDateMidnight < today) {
              // Past games - check if we have actual status, otherwise mark as completed
              gameStatus = game?.status === 'live' ? 'live' : 'completed';
            } else if (gameDateMidnight.getTime() === today.getTime()) {
              // Today's games - mark as upcoming (or actual status if available)
              gameStatus = game?.status || 'upcoming';
            } else {
              // Future games
              gameStatus = 'upcoming';
            }
          }

          return {
            id: prediction.id,
            homeTeam: game?.home_team?.name || 'Unknown',
            awayTeam: game?.away_team?.name || 'Unknown',
            date: dateStr,
            homeProbability: Math.round((prediction.home_win_probability || 0) * 100),
            awayProbability: Math.round((prediction.away_win_probability || 0) * 100),
            status: gameStatus,
            explanation: prediction.explanation || null
          };
        });

        setGames(formattedGames);
        console.log(`✅ Loaded ${formattedGames.length} prediction(s) from Supabase`);
      } else {
        console.log('ℹ️ No predictions found in database');
        setGames([]);
      }
    } catch (err) {
      console.error('Exception fetching games:', err);
      setError('An unexpected error occurred.');
      setGames([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGames();
  }, []);

  /**
   * Filter games based on search term, status, and confidence threshold
   */
  const filteredGames = games.filter(game => {
    const matchesSearch =
      game.homeTeam.toLowerCase().includes(searchTerm.toLowerCase()) ||
      game.awayTeam.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = filterStatus === 'all' || game.status === filterStatus;

    // Calculate confidence (spread between probabilities)
    const confidence = Math.abs(game.homeProbability - game.awayProbability);
    const matchesConfidence = confidence >= minConfidence;

    return matchesSearch && matchesStatus && matchesConfidence;
  });

  return (
    <div className="min-h-screen bg-nba-darker">
      <Navbar />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">
            Recent & Upcoming Games
          </h2>
          <p className="text-gray-400">
            AI-powered win probability predictions with GPT-5.2 explanations
          </p>
        </div>

        {/* Filter Bar */}
        <div className="bg-nba-dark border border-gray-800 rounded-lg p-4 mb-6">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search Input */}
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-500" />
                <input
                  type="text"
                  placeholder="Search teams..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-nba-darker border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-nba-blue transition-colors"
                />
              </div>
            </div>

            {/* Status Filter - TODO: Make functional */}
            <div className="flex gap-2">
              <button
                onClick={() => setFilterStatus('all')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${filterStatus === 'all'
                  ? 'bg-nba-blue text-white'
                  : 'bg-nba-darker text-gray-400 hover:text-white'
                  }`}
              >
                All Games
              </button>
              <button
                onClick={() => setFilterStatus('upcoming')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${filterStatus === 'upcoming'
                  ? 'bg-nba-blue text-white'
                  : 'bg-nba-darker text-gray-400 hover:text-white'
                  }`}
              >
                Upcoming
              </button>
              <button
                onClick={() => setFilterStatus('live')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${filterStatus === 'live'
                  ? 'bg-red-500 text-white'
                  : 'bg-nba-darker text-gray-400 hover:text-white'
                  }`}
              >
                Live
              </button>
            </div>

            {/* Refresh Button */}
            <button
              onClick={fetchGames}
              disabled={isLoading}
              className="px-4 py-2 bg-nba-darker border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:border-nba-blue transition-colors flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              <span className="hidden md:inline">Refresh</span>
            </button>
          </div>

          {/* Confidence Filter Slider */}
          <div className="mt-4 pt-4 border-t border-gray-800">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm text-gray-400 flex items-center gap-2">
                <Filter className="w-4 h-4" />
                Min Confidence (Spread)
              </label>
              <span className="text-sm font-medium text-nba-blue">
                {minConfidence}%+
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              step="5"
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #1d4ed8 0%, #1d4ed8 ${(minConfidence / 50) * 100}%, #374151 ${(minConfidence / 50) * 100}%, #374151 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>All Games</span>
              <span>High Confidence Only</span>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded relative mb-6" role="alert">
            <strong className="font-bold">Error: </strong>
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {/* Games Grid */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="text-center">
              <RefreshCw className="w-12 h-12 text-nba-blue animate-spin mx-auto mb-4" />
              <p className="text-gray-400">Loading predictions...</p>
            </div>
          </div>
        ) : filteredGames.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredGames.map((game) => (
              <GameCard key={game.id} game={game} />
            ))}
          </div>
        ) : (
          <div className="text-center py-20">
            <Filter className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400 text-lg">No games found</p>
            <p className="text-gray-600 text-sm mt-2">
              Try adjusting your filters or search term
            </p>
          </div>
        )}

        {/* TODO: Add pagination or infinite scroll */}
      </main>
    </div>
  );
}

export default App;
