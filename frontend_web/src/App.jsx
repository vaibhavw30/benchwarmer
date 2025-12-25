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
      // Query Supabase - fetch predictions with game and team data
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
        .order('created_at', { ascending: false })
        .limit(20);

      if (error) {
        console.error('Supabase error:', error);
        setError('Failed to load predictions. Please check your connection.');
        setGames([]);
      } else if (data && data.length > 0) {
        // Transform Supabase data to match component format
        const formattedGames = data.map(prediction => {
          const game = prediction.games;

          let dateStr = 'TBD';
          if (game?.game_date) {
            dateStr = new Date(game.game_date).toLocaleString('en-US', {
              month: 'short',
              day: 'numeric',
              year: 'numeric',
              hour: 'numeric',
              minute: '2-digit'
            });
          }

          return {
            id: prediction.id,
            homeTeam: game?.home_team?.name || 'Unknown',
            awayTeam: game?.away_team?.name || 'Unknown',
            date: dateStr,
            homeProbability: Math.round((prediction.home_win_probability || 0) * 100),
            awayProbability: Math.round((prediction.away_win_probability || 0) * 100),
            status: game?.status || 'upcoming'
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
   * Filter games based on search term and status
   * TODO: Implement more advanced filtering (date range, team, etc.)
   */
  const filteredGames = games.filter(game => {
    const matchesSearch =
      game.homeTeam.toLowerCase().includes(searchTerm.toLowerCase()) ||
      game.awayTeam.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus = filterStatus === 'all' || game.status === filterStatus;

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-nba-darker">
      <Navbar />

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-white mb-2">
            Today's Predictions
          </h2>
          <p className="text-gray-400">
            AI-powered win probability predictions for upcoming NBA games
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

          {/* TODO: Add more filter options */}
          {/* - Date range picker */}
          {/* - Team selector */}
          {/* - Confidence threshold slider */}
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
