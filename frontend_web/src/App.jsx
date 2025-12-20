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
const mockGames = [
  {
    id: '1',
    homeTeam: 'Los Angeles Lakers',
    awayTeam: 'Golden State Warriors',
    date: 'Dec 20, 2025 - 7:30 PM',
    homeProbability: 58,
    awayProbability: 42,
    status: 'upcoming'
  },
  {
    id: '2',
    homeTeam: 'Boston Celtics',
    awayTeam: 'Miami Heat',
    date: 'Dec 20, 2025 - 7:00 PM',
    homeProbability: 65,
    awayProbability: 35,
    status: 'upcoming'
  },
  {
    id: '3',
    homeTeam: 'Milwaukee Bucks',
    awayTeam: 'Philadelphia 76ers',
    date: 'Dec 20, 2025 - 8:00 PM',
    homeProbability: 52,
    awayProbability: 48,
    status: 'live'
  },
  {
    id: '4',
    homeTeam: 'Phoenix Suns',
    awayTeam: 'Dallas Mavericks',
    date: 'Dec 20, 2025 - 9:00 PM',
    homeProbability: 61,
    awayProbability: 39,
    status: 'upcoming'
  },
  {
    id: '5',
    homeTeam: 'Denver Nuggets',
    awayTeam: 'Los Angeles Clippers',
    date: 'Dec 19, 2025 - 8:00 PM',
    homeProbability: 55,
    awayProbability: 45,
    status: 'completed'
  },
  {
    id: '6',
    homeTeam: 'Brooklyn Nets',
    awayTeam: 'Chicago Bulls',
    date: 'Dec 20, 2025 - 7:30 PM',
    homeProbability: 49,
    awayProbability: 51,
    status: 'upcoming'
  }
];

/**
 * App Component - Main Dashboard
 *
 * This is the main dashboard that displays:
 * 1. A filter bar for searching and filtering games
 * 2. A grid of game prediction cards
 *
 * TODO - Future Implementation Steps:
 * - Replace mockGames with real data from Supabase
 * - Implement actual filter functionality (by date, team, status)
 * - Add real-time updates for live games
 * - Add pagination or infinite scroll for large datasets
 * - Add loading states and error handling
 */
function App() {
  const [games, setGames] = useState(mockGames);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all'); // 'all', 'upcoming', 'live', 'completed'
  const [isLoading, setIsLoading] = useState(false);

  /**
   * Fetch game predictions from Supabase
   *
   * This queries the game_predictions table and joins with:
   * - games table (to get game details)
   * - teams table (to get team names)
   */
  const fetchGames = async () => {
    setIsLoading(true);

    try {
      // Query Supabase with nested joins
      const { data, error } = await supabase
        .from('game_predictions')
        .select(`
          id,
          home_win_probability,
          away_win_probability,
          confidence_score,
          predicted_winner,
          game:games!game_id (
            id,
            game_id,
            game_date,
            status,
            home_team:teams!games_home_team_id_fkey (
              name,
              abbreviation
            ),
            away_team:teams!games_away_team_id_fkey (
              name,
              abbreviation
            )
          )
        `)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Supabase error:', error);
        // Fallback to mock data on error
        setGames(mockGames);
      } else if (data && data.length > 0) {
        // Transform Supabase data to match our component format
        const formattedGames = data.map(prediction => ({
          id: prediction.id,
          homeTeam: prediction.game.home_team.name,
          awayTeam: prediction.game.away_team.name,
          date: new Date(prediction.game.game_date).toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            year: 'numeric',
            hour: 'numeric',
            minute: '2-digit'
          }),
          homeProbability: Math.round(prediction.home_win_probability * 100),
          awayProbability: Math.round(prediction.away_win_probability * 100),
          status: prediction.game.status
        }));

        setGames(formattedGames);
        console.log(`✅ Loaded ${formattedGames.length} game(s) from Supabase`);
      } else {
        // No data in database yet, use mock data
        console.warn('⚠️ No predictions found in database, using mock data');
        setGames(mockGames);
      }
    } catch (err) {
      console.error('Exception fetching games:', err);
      // Fallback to mock data on exception
      setGames(mockGames);
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
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filterStatus === 'all'
                    ? 'bg-nba-blue text-white'
                    : 'bg-nba-darker text-gray-400 hover:text-white'
                }`}
              >
                All Games
              </button>
              <button
                onClick={() => setFilterStatus('upcoming')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filterStatus === 'upcoming'
                    ? 'bg-nba-blue text-white'
                    : 'bg-nba-darker text-gray-400 hover:text-white'
                }`}
              >
                Upcoming
              </button>
              <button
                onClick={() => setFilterStatus('live')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filterStatus === 'live'
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
