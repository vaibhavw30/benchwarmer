"""
Automatic Game Results Updater

Fetches completed game scores from NBA API and updates Supabase.
Run this daily (or after games) to keep results current for bias analysis.
"""

import os
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2
from data_engine import initialize_supabase
import time


def update_completed_games(days_back=3):
    """
    Update game results for the last N days

    Args:
        days_back (int): How many days to look back for games
    """
    print(f"\n🏀 Updating game results for last {days_back} days...")

    try:
        supabase = initialize_supabase()
        if not supabase:
            print("❌ Could not connect to Supabase")
            return False

        updated_count = 0
        error_count = 0

        for days_ago in range(days_back):
            date = datetime.now() - timedelta(days=days_ago)
            date_str = date.strftime('%Y-%m-%d')

            print(f"\n📅 Checking {date_str}...")

            try:
                # Fetch scoreboard for this date
                board = scoreboardv2.ScoreboardV2(game_date=date_str, day_offset=0)
                games = board.game_header.get_data_frame()

                if games.empty:
                    print(f"   ℹ️  No games found")
                    continue

                print(f"   Found {len(games)} games")

                for _, game in games.iterrows():
                    game_id = game['GAME_ID']
                    game_status = game['GAME_STATUS_TEXT']

                    # Map NBA API status to our status
                    if 'Final' in game_status:
                        status = 'completed'
                        home_score = int(game.get('PTS_HOME', 0) or 0)
                        away_score = int(game.get('PTS_AWAY', 0) or 0)

                        # Update in Supabase
                        try:
                            result = supabase.table('games').update({
                                'home_score': home_score,
                                'away_score': away_score,
                                'status': status
                            }).eq('game_id', game_id).execute()

                            # Check if any row was updated
                            if result.data:
                                print(f"      ✓ Updated {game_id}: {away_score}-{home_score} (Final)")
                                updated_count += 1
                            else:
                                # Game might not exist in our DB yet
                                print(f"      ⚠️  Game {game_id} not found in database")

                        except Exception as e:
                            print(f"      ❌ Error updating {game_id}: {e}")
                            error_count += 1

                    elif 'PM' in game_status or 'AM' in game_status:
                        # Game hasn't started yet
                        continue
                    else:
                        # Game is in progress or postponed
                        status = 'in_progress' if game_status != 'PPD' else 'postponed'

                        try:
                            supabase.table('games').update({
                                'status': status
                            }).eq('game_id', game_id).execute()
                        except:
                            pass

                # Be nice to NBA API
                time.sleep(0.5)

            except Exception as e:
                print(f"   ❌ Error fetching games for {date_str}: {e}")
                error_count += 1

        print("\n" + "="*60)
        print(f"✅ Update Complete!")
        print(f"   Updated: {updated_count} games")
        if error_count > 0:
            print(f"   Errors: {error_count}")
        print("="*60)

        return True

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🏀 NBA Game Results Updater")
    print("="*60)

    # Update last 3 days by default
    success = update_completed_games(days_back=3)

    if success:
        print("\n💡 Next Steps:")
        print("   1. Run this script daily (or set up as cron job)")
        print("   2. After a week of data, run: python model_bias_analyzer.py")
        print("   3. Review bias report and apply corrections")
        print("\n📅 Recommended: Add to cron:")
        print("   0 6 * * * cd /path/to/backend_ml && venv/bin/python update_game_results.py")
