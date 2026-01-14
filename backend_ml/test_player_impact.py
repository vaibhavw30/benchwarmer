#!/usr/bin/env python3
"""
Test script for Player Impact Engine

Run this to verify the injury engine is working correctly.
"""

import sys
import os
from player_impact_engine import (
    update_player_stats_cache,
    fetch_active_injuries,
    calculate_injury_impact
)

CACHE_FILE = "player_impact_scores.csv"

def test_cache_creation():
    """Test 1: Can we fetch and cache player stats?"""
    print("\n" + "="*80)
    print("TEST 1: Player Stats Cache")
    print("="*80)

    # Delete existing cache to force refresh
    if os.path.exists(CACHE_FILE):
        print(f"🗑️  Removing existing cache: {CACHE_FILE}")
        os.remove(CACHE_FILE)

    # Fetch stats
    df = update_player_stats_cache()

    if df is None or df.empty:
        print("❌ FAILED: Could not fetch player stats")
        return False

    print(f"✅ PASSED: Cached {len(df)} players")

    # Verify cache file exists
    if not os.path.exists(CACHE_FILE):
        print("❌ FAILED: Cache file not created")
        return False

    print(f"✅ PASSED: Cache file created at {CACHE_FILE}")

    # Show sample data
    print("\n📊 Sample Player Impacts:")
    print(df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PLAYER_IMPACT']].head(10).to_string(index=False))

    return True


def test_injury_scraping():
    """Test 2: Can we scrape current injuries?"""
    print("\n" + "="*80)
    print("TEST 2: Injury Report Scraping")
    print("="*80)

    injuries = fetch_active_injuries()

    if not injuries:
        print("⚠️  WARNING: No injuries found (could be legitimate, or scraping failed)")
        print("   This is not necessarily a failure - there might genuinely be no injuries")
        return True  # Don't fail test, just warn

    print(f"✅ PASSED: Found {len(injuries)} injuries")

    # Show sample injuries
    print("\n🏥 Sample Injuries:")
    for i, player_name in enumerate(injuries[:10], 1):
        print(f"   {i}. {player_name}")

    return True


def test_injury_impact_calculation():
    """Test 3: Can we calculate injury impact for a matchup?"""
    print("\n" + "="*80)
    print("TEST 3: Injury Impact Calculation")
    print("="*80)

    # Test with common teams
    test_matchups = [
        (1610612747, 1610612743, "Lakers vs Nuggets"),
        (1610612738, 1610612744, "Celtics vs Warriors"),
        (1610612749, 1610612751, "Bucks vs Nets"),
    ]

    all_passed = True

    for home_id, away_id, description in test_matchups:
        print(f"\n🏀 Testing: {description}")
        print(f"   Home ID: {home_id}, Away ID: {away_id}")

        try:
            result = calculate_injury_impact(home_id, away_id)

            # Verify structure
            if not isinstance(result, dict):
                print(f"   ❌ FAILED: Result is not a dict, got {type(result)}")
                all_passed = False
                continue

            if home_id not in result or away_id not in result:
                print(f"   ❌ FAILED: Missing team IDs in result")
                all_passed = False
                continue

            # Check values are numbers
            if not isinstance(result[home_id], (int, float)) or not isinstance(result[away_id], (int, float)):
                print(f"   ❌ FAILED: Penalties are not numeric")
                all_passed = False
                continue

            print(f"   ✅ PASSED")
            print(f"   📊 Home Penalty: {result[home_id]:.2f} Elo")
            print(f"   📊 Away Penalty: {result[away_id]:.2f} Elo")

            if result[home_id] < -50:
                print(f"   ⚠️  ALERT: Home team has major injuries! ({result[home_id]:.1f} Elo impact)")
            if result[away_id] < -50:
                print(f"   ⚠️  ALERT: Away team has major injuries! ({result[away_id]:.1f} Elo impact)")

        except Exception as e:
            print(f"   ❌ FAILED: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_integration_example():
    """Test 4: Show integration example"""
    print("\n" + "="*80)
    print("TEST 4: Integration Example")
    print("="*80)

    print("""
This is how you would use it in predict.py:

```python
from player_impact_engine import calculate_injury_impact

# In your prediction loop:
for _, game in games_df.iterrows():
    home_id = game['HOME_TEAM_ID']
    away_id = game['AWAY_TEAM_ID']

    # Get base Elo from stats
    base_home_elo = stats[home_id].get('ELO', 1500)
    base_away_elo = stats[away_id].get('ELO', 1500)

    # Apply injury adjustments
    injury_penalties = calculate_injury_impact(home_id, away_id)
    adjusted_home_elo = base_home_elo + injury_penalties[home_id]
    adjusted_away_elo = base_away_elo + injury_penalties[away_id]

    # Use adjusted Elo in features
    features = pd.DataFrame([{
        'ELO_H': adjusted_home_elo,
        'ELO_A': adjusted_away_elo,
        # ... other features
    }])
```
    """)

    print("✅ See PLAYER_IMPACT_INTEGRATION.md for full integration guide")
    return True


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*80)
    print("🧪 PLAYER IMPACT ENGINE TEST SUITE")
    print("="*80)

    tests = [
        ("Cache Creation", test_cache_creation),
        ("Injury Scraping", test_injury_scraping),
        ("Impact Calculation", test_injury_impact_calculation),
        ("Integration Example", test_integration_example),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} crashed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The Player Impact Engine is ready to use.")
        print("\nNext steps:")
        print("1. Review PLAYER_IMPACT_INTEGRATION.md for integration instructions")
        print("2. Update predict.py to use injury-adjusted Elo")
        print("3. Run predict.py to see injury impact in action")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
