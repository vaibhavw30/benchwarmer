import React from 'react';
import { Calendar, TrendingUp, Trophy } from 'lucide-react';

/**
 * GameCard Component
 *
 * Displays a single NBA game with:
 * - Home and Away team names
 * - Game date and time
 * - AI-predicted win probability for each team
 * - Visual indicators showing which team is favored
 *
 * Props:
 * @param {Object} game - Game data object
 * @param {string} game.id - Unique game identifier
 * @param {string} game.homeTeam - Home team name
 * @param {string} game.awayTeam - Away team name
 * @param {string} game.date - Game date (formatted string)
 * @param {number} game.homeProbability - AI prediction for home team (0-100)
 * @param {number} game.awayProbability - AI prediction for away team (0-100)
 * @param {string} game.status - Game status: 'upcoming', 'live', 'completed'
 */
const GameCard = ({ game }) => {
  const {
    homeTeam,
    awayTeam,
    date,
    homeProbability,
    awayProbability,
    status = 'upcoming'
  } = game;

  // Determine which team is favored
  const homeIsFavored = homeProbability > awayProbability;
  const awayIsFavored = awayProbability > homeProbability;

  return (
    <div className="bg-nba-dark border border-gray-800 rounded-lg p-6 hover:border-nba-blue transition-all duration-300 hover:shadow-lg hover:shadow-nba-blue/20">
      {/* Game Status Badge */}
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <Calendar className="w-4 h-4" />
          <span>{date}</span>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${
          status === 'live' ? 'bg-red-500/20 text-red-400' :
          status === 'completed' ? 'bg-gray-500/20 text-gray-400' :
          'bg-blue-500/20 text-blue-400'
        }`}>
          {status.toUpperCase()}
        </span>
      </div>

      {/* Teams Matchup */}
      <div className="space-y-3">
        {/* Away Team */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex items-center justify-center">
              <span className="text-lg font-bold">{awayTeam.substring(0, 3).toUpperCase()}</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">{awayTeam}</h3>
              <p className="text-xs text-gray-500">Away</p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-2">
              {awayIsFavored && <Trophy className="w-4 h-4 text-nba-orange" />}
              <span className={`text-2xl font-bold ${
                awayIsFavored ? 'text-nba-orange' : 'text-gray-400'
              }`}>
                {awayProbability}%
              </span>
            </div>
            <p className="text-xs text-gray-500">Win Probability</p>
          </div>
        </div>

        {/* Divider */}
        <div className="border-t border-gray-800"></div>

        {/* Home Team */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-nba-blue to-nba-purple rounded-lg flex items-center justify-center">
              <span className="text-lg font-bold">{homeTeam.substring(0, 3).toUpperCase()}</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">{homeTeam}</h3>
              <p className="text-xs text-gray-500">Home</p>
            </div>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-2">
              {homeIsFavored && <Trophy className="w-4 h-4 text-nba-orange" />}
              <span className={`text-2xl font-bold ${
                homeIsFavored ? 'text-nba-orange' : 'text-gray-400'
              }`}>
                {homeProbability}%
              </span>
            </div>
            <p className="text-xs text-gray-500">Win Probability</p>
          </div>
        </div>
      </div>

      {/* AI Confidence Indicator */}
      <div className="mt-4 pt-4 border-t border-gray-800">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center gap-1">
            <TrendingUp className="w-3 h-3" />
            <span>AI Confidence</span>
          </div>
          <span className="text-nba-blue font-medium">
            {Math.abs(homeProbability - awayProbability)}% spread
          </span>
        </div>
      </div>

      {/* TODO: Add click handler to show detailed prediction breakdown */}
      {/* TODO: Add favorite/bookmark functionality */}
    </div>
  );
};

export default GameCard;
