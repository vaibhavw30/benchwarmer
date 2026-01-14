"""
Model Bias Analyzer

Analyzes historical predictions to identify systematic biases against/for specific teams.
Helps detect if the model consistently over/under-estimates certain teams.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_engine import initialize_supabase
from nba_api.stats.static import teams
import matplotlib.pyplot as plt
import seaborn as sns

# Team ID to Name mapping
nba_teams = teams.get_teams()
TEAM_ID_TO_NAME = {team['id']: team['full_name'] for team in nba_teams}
TEAM_ID_TO_ABBR = {team['id']: team['abbreviation'] for team in nba_teams}


def fetch_historical_predictions(days_back=30):
    """
    Fetch historical predictions and actual outcomes from Supabase

    Args:
        days_back (int): Number of days to look back

    Returns:
        pd.DataFrame: Predictions with actual outcomes
    """
    print(f"\n📊 Fetching predictions from last {days_back} days...")

    try:
        supabase = initialize_supabase()
        if not supabase:
            print("❌ Could not connect to Supabase")
            return pd.DataFrame()

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Fetch predictions
        response = supabase.table('game_predictions') \
            .select('*, games!inner(*)') \
            .gte('date', start_date.strftime('%Y-%m-%d')) \
            .lte('date', end_date.strftime('%Y-%m-%d')) \
            .execute()

        if not response.data:
            print("⚠️  No historical predictions found")
            return pd.DataFrame()

        # Convert to DataFrame
        predictions_df = pd.DataFrame(response.data)
        print(f"   ✓ Found {len(predictions_df)} predictions")

        # Extract game info from nested structure
        if 'games' in predictions_df.columns:
            predictions_df['actual_home_score'] = predictions_df['games'].apply(
                lambda x: x.get('home_score') if isinstance(x, dict) else None
            )
            predictions_df['actual_away_score'] = predictions_df['games'].apply(
                lambda x: x.get('away_score') if isinstance(x, dict) else None
            )
            predictions_df['game_status'] = predictions_df['games'].apply(
                lambda x: x.get('status') if isinstance(x, dict) else 'upcoming'
            )

        # Filter only completed games
        completed = predictions_df[
            (predictions_df['actual_home_score'].notna()) &
            (predictions_df['actual_away_score'].notna())
        ].copy()

        print(f"   ✓ {len(completed)} completed games with outcomes")

        if completed.empty:
            print("⚠️  No completed games found - need actual results to analyze bias")
            return pd.DataFrame()

        # Calculate actual winner
        completed['actual_winner'] = completed.apply(
            lambda row: 'Home' if row['actual_home_score'] > row['actual_away_score'] else 'Away',
            axis=1
        )

        # Calculate prediction accuracy
        completed['prediction_correct'] = completed['predicted_winner'] == completed['actual_winner']

        return completed

    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def analyze_team_bias(predictions_df):
    """
    Analyze bias for each team

    Args:
        predictions_df (pd.DataFrame): Historical predictions with outcomes

    Returns:
        pd.DataFrame: Team-level bias analysis
    """
    if predictions_df.empty:
        return pd.DataFrame()

    print("\n🔍 Analyzing team-level bias...")

    team_stats = []

    # Get unique teams
    all_team_ids = set(predictions_df['home_team_id'].unique()) | set(predictions_df['away_team_id'].unique())

    for team_id in all_team_ids:
        team_name = TEAM_ID_TO_NAME.get(team_id, f"Team {team_id}")
        team_abbr = TEAM_ID_TO_ABBR.get(team_id, "???")

        # Get games where this team played
        home_games = predictions_df[predictions_df['home_team_id'] == team_id].copy()
        away_games = predictions_df[predictions_df['away_team_id'] == team_id].copy()

        # Calculate stats for home games
        home_predicted_wins = len(home_games[home_games['predicted_winner'] == 'Home'])
        home_actual_wins = len(home_games[home_games['actual_winner'] == 'Home'])
        home_prediction_accuracy = home_games['prediction_correct'].mean() if len(home_games) > 0 else 0

        # Calculate stats for away games
        away_predicted_wins = len(away_games[away_games['predicted_winner'] == 'Away'])
        away_actual_wins = len(away_games[away_games['actual_winner'] == 'Away'])
        away_prediction_accuracy = away_games['prediction_correct'].mean() if len(away_games) > 0 else 0

        # Overall stats
        total_games = len(home_games) + len(away_games)
        if total_games == 0:
            continue

        predicted_wins = home_predicted_wins + away_predicted_wins
        actual_wins = home_actual_wins + away_actual_wins

        # Calculate bias: negative means model underestimates team
        win_bias = predicted_wins - actual_wins
        win_rate_bias = (predicted_wins / total_games) - (actual_wins / total_games) if total_games > 0 else 0

        # Prediction accuracy
        all_games = pd.concat([home_games, away_games])
        overall_accuracy = all_games['prediction_correct'].mean() if len(all_games) > 0 else 0

        # Calculate average predicted vs actual performance
        home_games['predicted_home_win_prob'] = home_games['home_win_probability']
        away_games['predicted_away_win_prob'] = away_games['away_win_probability']

        avg_predicted_prob = (
            home_games['predicted_home_win_prob'].mean() * len(home_games) +
            away_games['predicted_away_win_prob'].mean() * len(away_games)
        ) / total_games if total_games > 0 else 0

        actual_win_rate = actual_wins / total_games if total_games > 0 else 0
        probability_bias = avg_predicted_prob - actual_win_rate

        team_stats.append({
            'team_id': team_id,
            'team_name': team_name,
            'team_abbr': team_abbr,
            'total_games': total_games,
            'predicted_wins': predicted_wins,
            'actual_wins': actual_wins,
            'win_bias': win_bias,
            'win_rate_bias': win_rate_bias,
            'avg_predicted_prob': avg_predicted_prob,
            'actual_win_rate': actual_win_rate,
            'probability_bias': probability_bias,
            'accuracy': overall_accuracy,
            'home_games': len(home_games),
            'away_games': len(away_games)
        })

    team_bias_df = pd.DataFrame(team_stats)

    # Sort by absolute win bias (most biased teams first)
    team_bias_df = team_bias_df.sort_values('win_bias', ascending=True)

    return team_bias_df


def print_bias_report(team_bias_df, min_games=5):
    """
    Print a comprehensive bias report

    Args:
        team_bias_df (pd.DataFrame): Team bias analysis
        min_games (int): Minimum games to include in report
    """
    if team_bias_df.empty:
        print("⚠️  No data to analyze")
        return

    # Filter teams with enough games
    significant_teams = team_bias_df[team_bias_df['total_games'] >= min_games].copy()

    print("\n" + "="*80)
    print("📊 MODEL BIAS ANALYSIS REPORT")
    print("="*80)

    # Overall stats
    total_predictions = team_bias_df['total_games'].sum()
    overall_accuracy = (team_bias_df['accuracy'] * team_bias_df['total_games']).sum() / total_predictions

    print(f"\n📈 Overall Statistics:")
    print(f"   Total Predictions: {total_predictions}")
    print(f"   Overall Accuracy: {overall_accuracy:.1%}")

    # Most underestimated teams (model predicts them to lose more than they do)
    print(f"\n🔻 MOST UNDERESTIMATED TEAMS (Model thinks they're worse than they are):")
    print("   " + "-"*76)
    underestimated = significant_teams.nsmallest(5, 'win_bias')

    for _, row in underestimated.iterrows():
        print(f"   {row['team_abbr']:3s} {row['team_name']:25s} | Games: {row['total_games']:2.0f} | "
              f"Predicted: {row['predicted_wins']:2.0f} wins | Actual: {row['actual_wins']:2.0f} wins | "
              f"Bias: {row['win_bias']:+.1f} | Accuracy: {row['accuracy']:.1%}")

    # Most overestimated teams (model predicts them to win more than they do)
    print(f"\n🔺 MOST OVERESTIMATED TEAMS (Model thinks they're better than they are):")
    print("   " + "-"*76)
    overestimated = significant_teams.nlargest(5, 'win_bias')

    for _, row in overestimated.iterrows():
        print(f"   {row['team_abbr']:3s} {row['team_name']:25s} | Games: {row['total_games']:2.0f} | "
              f"Predicted: {row['predicted_wins']:2.0f} wins | Actual: {row['actual_wins']:2.0f} wins | "
              f"Bias: {row['win_bias']:+.1f} | Accuracy: {row['accuracy']:.1%}")

    # Worst prediction accuracy
    print(f"\n⚠️  LOWEST PREDICTION ACCURACY:")
    print("   " + "-"*76)
    worst_accuracy = significant_teams.nsmallest(5, 'accuracy')

    for _, row in worst_accuracy.iterrows():
        print(f"   {row['team_abbr']:3s} {row['team_name']:25s} | Games: {row['total_games']:2.0f} | "
              f"Accuracy: {row['accuracy']:.1%} | "
              f"Win Bias: {row['win_bias']:+.1f} | "
              f"Prob Bias: {row['probability_bias']:+.1%}")

    print("\n" + "="*80)


def generate_elo_corrections(team_bias_df, min_games=5, correction_factor=20):
    """
    Generate Elo corrections to fix systematic biases

    Args:
        team_bias_df (pd.DataFrame): Team bias analysis
        min_games (int): Minimum games to consider for corrections
        correction_factor (float): How much to adjust Elo per win bias

    Returns:
        dict: Team ID -> Elo adjustment
    """
    print(f"\n🔧 Generating Elo Corrections...")
    print(f"   Correction Factor: {correction_factor} Elo per win bias")
    print(f"   Min Games Threshold: {min_games}")

    corrections = {}

    significant_teams = team_bias_df[team_bias_df['total_games'] >= min_games].copy()

    for _, row in significant_teams.iterrows():
        team_id = row['team_id']
        win_bias = row['win_bias']

        # Negative win bias = model underestimates = need positive Elo boost
        # Positive win bias = model overestimates = need negative Elo reduction
        elo_adjustment = -win_bias * correction_factor

        # Only apply corrections for significant biases
        if abs(elo_adjustment) > 10:
            corrections[team_id] = round(elo_adjustment, 1)

    if corrections:
        print(f"\n   Recommended Elo Adjustments for {len(corrections)} teams:")
        for team_id, adjustment in sorted(corrections.items(), key=lambda x: x[1], reverse=True):
            team_abbr = TEAM_ID_TO_ABBR.get(team_id, "???")
            team_name = TEAM_ID_TO_NAME.get(team_id, "Unknown")
            bias_type = "BOOST" if adjustment > 0 else "REDUCE"
            print(f"      {team_abbr:3s} {team_name:25s}: {adjustment:+5.1f} Elo ({bias_type})")
    else:
        print("   ✅ No significant biases detected - model is well calibrated!")

    return corrections


def save_corrections_to_file(corrections, filename='elo_corrections.py'):
    """Save corrections as a Python dict for easy import"""
    if not corrections:
        print("\n⚠️  No corrections to save")
        return

    with open(filename, 'w') as f:
        f.write("# Elo corrections based on historical bias analysis\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("ELO_CORRECTIONS = {\n")
        for team_id, adjustment in sorted(corrections.items()):
            team_abbr = TEAM_ID_TO_ABBR.get(team_id, "???")
            f.write(f"    {team_id}: {adjustment:+.1f},  # {team_abbr}\n")
        f.write("}\n")

    print(f"\n💾 Corrections saved to: {filename}")
    print(f"   To use: Import this dict and add to Elo ratings in predict.py")


def plot_bias_visualization(team_bias_df, min_games=5):
    """Create visualization of team biases"""
    try:
        significant_teams = team_bias_df[team_bias_df['total_games'] >= min_games].copy()

        if len(significant_teams) < 3:
            print("⚠️  Not enough data for visualization")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Win Bias
        ax1 = axes[0]
        significant_teams_sorted = significant_teams.sort_values('win_bias')
        colors = ['red' if x < 0 else 'green' for x in significant_teams_sorted['win_bias']]

        ax1.barh(significant_teams_sorted['team_abbr'], significant_teams_sorted['win_bias'], color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Win Bias (Predicted Wins - Actual Wins)')
        ax1.set_ylabel('Team')
        ax1.set_title('Model Win Bias by Team\n(Negative = Underestimated, Positive = Overestimated)')
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Prediction Accuracy
        ax2 = axes[1]
        significant_teams_sorted_acc = significant_teams.sort_values('accuracy')
        colors_acc = ['red' if x < 0.5 else 'green' for x in significant_teams_sorted_acc['accuracy']]

        ax2.barh(significant_teams_sorted_acc['team_abbr'], significant_teams_sorted_acc['accuracy'], color=colors_acc, alpha=0.7)
        ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='50% (Random)')
        ax2.set_xlabel('Prediction Accuracy')
        ax2.set_ylabel('Team')
        ax2.set_title('Prediction Accuracy by Team')
        ax2.set_xlim(0, 1)
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_bias_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: model_bias_analysis.png")

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def run_bias_analysis(days_back=30, min_games=5, generate_corrections=True):
    """
    Run complete bias analysis

    Args:
        days_back (int): Days of history to analyze
        min_games (int): Minimum games threshold
        generate_corrections (bool): Whether to generate Elo corrections
    """
    # Fetch data
    predictions_df = fetch_historical_predictions(days_back)

    if predictions_df.empty:
        print("\n❌ Cannot analyze bias without historical data")
        print("\n💡 Make sure you have:")
        print("   1. Been running predictions for several days")
        print("   2. Games have completed and results are in Supabase")
        print("   3. The 'games' table has 'home_score' and 'away_score' columns")
        return None, None

    # Analyze bias
    team_bias_df = analyze_team_bias(predictions_df)

    # Print report
    print_bias_report(team_bias_df, min_games)

    # Generate corrections
    corrections = None
    if generate_corrections:
        corrections = generate_elo_corrections(team_bias_df, min_games)
        if corrections:
            save_corrections_to_file(corrections)

    # Create visualization
    plot_bias_visualization(team_bias_df, min_games)

    return team_bias_df, corrections


if __name__ == "__main__":
    print("🏀 NBA Model Bias Analyzer")
    print("="*80)

    # Run analysis for last 30 days
    team_bias_df, corrections = run_bias_analysis(
        days_back=30,
        min_games=3,  # Lower threshold for testing
        generate_corrections=True
    )

    if team_bias_df is not None and not team_bias_df.empty:
        print("\n✅ Analysis complete!")
        print(f"\n💡 Next Steps:")
        print("   1. Review the bias report above")
        print("   2. Check if 76ers appear in 'MOST UNDERESTIMATED' section")
        print("   3. Import elo_corrections.py in predict.py to apply fixes")
        print("   4. Re-run analysis after a week to verify corrections work")
    else:
        print("\n⏳ Not enough historical data yet")
        print("   Run predictions for a few more days and try again")
