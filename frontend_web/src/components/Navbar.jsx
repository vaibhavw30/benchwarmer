import React from 'react';
import { Activity, TrendingUp } from 'lucide-react';

/**
 * Navbar Component
 *
 * A simple navigation bar displaying the app branding and logo.
 * Uses Lucide React icons and Tailwind CSS for styling.
 */
const Navbar = () => {
  return (
    <nav className="bg-nba-darker border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-nba-blue to-nba-purple rounded-lg">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">
                NBA Holistic Predictor
              </h1>
              <p className="text-xs text-gray-400 flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                AI-Powered Game Predictions
              </p>
            </div>
          </div>

          {/* Future: Add navigation items here */}
          <div className="flex items-center space-x-4">
            {/* TODO: Add user menu, settings, or additional navigation */}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
