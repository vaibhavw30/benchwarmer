import { createClient } from '@supabase/supabase-js';

/**
 * Supabase Client Configuration
 *
 * This file sets up the connection to your Supabase backend.
 *
 * SETUP INSTRUCTIONS:
 * 1. Create a Supabase project at https://supabase.com
 * 2. Copy your project URL and anon key from the Supabase dashboard
 * 3. Create a .env file in the PROJECT ROOT (not in frontend_web) with:
 *    SUPABASE_URL=your_supabase_project_url
 *    SUPABASE_ANON_KEY=your_supabase_anon_key
 * 4. Vite will automatically load these from the root .env file
 *
 * Note: Vite requires the VITE_ prefix for frontend environment variables,
 * but our .env file uses plain names. We reference both for flexibility.
 *
 * IMPORTANT: Never commit your .env file to version control!
 */

// Supabase project credentials from environment variables
// Try both VITE_ prefixed and plain names for flexibility
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || import.meta.env.SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || import.meta.env.SUPABASE_ANON_KEY || '';

// Validate that environment variables are set
if (!supabaseUrl || !supabaseAnonKey) {
  console.warn(
    '⚠️ Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in the root .env file.'
  );
}

// Create and export the Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/**
 * USAGE EXAMPLES:
 *
 * // Fetch all game predictions
 * const { data, error } = await supabase
 *   .from('game_predictions')
 *   .select('*')
 *   .order('game_date', { ascending: true });
 *
 * // Fetch predictions for a specific date
 * const { data, error } = await supabase
 *   .from('game_predictions')
 *   .select('*')
 *   .eq('game_date', '2025-12-20');
 *
 * // Real-time subscription to updates
 * const subscription = supabase
 *   .channel('game_predictions')
 *   .on('postgres_changes', {
 *     event: '*',
 *     schema: 'public',
 *     table: 'game_predictions'
 *   }, (payload) => {
 *     console.log('Change received!', payload);
 *   })
 *   .subscribe();
 *
 * // Fetch historical team statistics
 * const { data, error } = await supabase
 *   .from('team_stats')
 *   .select('*')
 *   .eq('team_name', 'Los Angeles Lakers')
 *   .order('date', { ascending: false })
 *   .limit(10);
 */

// Helper function to fetch today's game predictions
export const fetchTodaysPredictions = async () => {
  try {
    const today = new Date().toISOString().split('T')[0];

    const { data, error } = await supabase
      .from('game_predictions')
      .select(`
        *,
        home_team:teams!game_predictions_home_team_id_fkey(*),
        away_team:teams!game_predictions_away_team_id_fkey(*)
      `)
      .gte('game_date', today)
      .order('game_date', { ascending: true });

    if (error) throw error;
    return { data, error: null };
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return { data: null, error };
  }
};

// Helper function to fetch predictions by team
export const fetchPredictionsByTeam = async (teamName) => {
  try {
    const { data, error } = await supabase
      .from('game_predictions')
      .select(`
        *,
        home_team:teams!game_predictions_home_team_id_fkey(*),
        away_team:teams!game_predictions_away_team_id_fkey(*)
      `)
      .or(`home_team.name.ilike.%${teamName}%,away_team.name.ilike.%${teamName}%`)
      .order('game_date', { ascending: false })
      .limit(20);

    if (error) throw error;
    return { data, error: null };
  } catch (error) {
    console.error('Error fetching team predictions:', error);
    return { data: null, error };
  }
};

// Helper function to fetch team statistics
export const fetchTeamStats = async (teamId) => {
  try {
    const { data, error } = await supabase
      .from('team_stats')
      .select('*')
      .eq('team_id', teamId)
      .order('date', { ascending: false })
      .limit(1);

    if (error) throw error;
    return { data: data?.[0] || null, error: null };
  } catch (error) {
    console.error('Error fetching team stats:', error);
    return { data: null, error };
  }
};

export default supabase;
