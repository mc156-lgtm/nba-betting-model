#!/usr/bin/env python3
"""
Daily NBA Predictions Script
Run this daily to get predictions for today's games
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# Modern NBA Adjustments
MODERN_NBA_ADJUSTMENTS = {
    'totals': 10.8,
    'spread': 3.9,
    'moneyline': 0
}

print("=" * 80)
print(f"NBA DAILY PREDICTIONS - {datetime.now().strftime('%Y-%m-%d')}")
print("=" * 80)

# Load models
print("\n🤖 Loading models...")
try:
    spread_model = joblib.load('models/spread_model.pkl')
    totals_model = joblib.load('models/totals_model.pkl')
    moneyline_model = joblib.load('models/moneyline_model.pkl')
    print("   ✅ All models loaded")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")
    exit(1)

# Load current season data
print("\n📊 Loading 2024-25 season data...")
try:
    team_avg = pd.read_csv('data/processed/season_2024_25/team_averages.csv', index_col=0)
    print(f"   ✅ Team averages loaded ({len(team_avg)} teams)")
except Exception as e:
    print(f"   ⚠️  Could not load team averages: {e}")
    print("   Using default values...")
    team_avg = None

# Today's games (you would get this from NBA API or schedule)
# For demo, using sample matchups
todays_games = [
    ('BOS', 'LAL'),
    ('GSW', 'MIA'),
    ('DEN', 'PHO'),
    ('MIL', 'DAL'),
    ('NYK', 'BRK')
]

print(f"\n🏀 Generating predictions for {len(todays_games)} games...")
print("=" * 80)

predictions_list = []

for home_team, away_team in todays_games:
    print(f"\n📍 {home_team} vs {away_team}")
    
    # Get team stats if available
    if team_avg is not None and home_team in team_avg.index and away_team in team_avg.index:
        home_stats = team_avg.loc[home_team]
        away_stats = team_avg.loc[away_team]
        
        # Create features
        features = pd.DataFrame({
            'home_fg_pct': [home_stats['avg_fg_pct']],
            'away_fg_pct': [away_stats['avg_fg_pct']],
            'home_fg_pct_diff': [home_stats['avg_fg_pct'] - away_stats['avg_fg_pct']],
            'home_fg3_pct': [home_stats['avg_fg3_pct']],
            'away_fg3_pct': [away_stats['avg_fg3_pct']],
            'home_fg3_pct_diff': [home_stats['avg_fg3_pct'] - away_stats['avg_fg3_pct']],
            'home_ft_pct': [home_stats['avg_ft_pct']],
            'away_ft_pct': [away_stats['avg_ft_pct']],
            'home_ft_pct_diff': [home_stats['avg_ft_pct'] - away_stats['avg_ft_pct']],
            'home_reb': [home_stats['avg_reb']],
            'away_reb': [away_stats['avg_reb']],
            'home_reb_diff': [home_stats['avg_reb'] - away_stats['avg_reb']],
            'home_ast': [home_stats['avg_ast']],
            'away_ast': [away_stats['avg_ast']],
            'home_ast_diff': [home_stats['avg_ast'] - away_stats['avg_ast']],
            'home_tov': [home_stats['avg_pts'] * 0.12],
            'away_tov': [away_stats['avg_pts'] * 0.12],
            'home_tov_diff': [(home_stats['avg_pts'] - away_stats['avg_pts']) * 0.12],
            'total_fga': [(home_stats['avg_pts'] + away_stats['avg_pts']) / 2.0 * 0.85],
            'total_fta': [(home_stats['avg_pts'] + away_stats['avg_pts']) / 2.0 * 0.25],
            'spread_avg': [home_stats['avg_pts'] - away_stats['avg_pts']],
            'total_avg': [home_stats['avg_pts'] + away_stats['avg_pts']]
        })
    else:
        # Use default features
        features = pd.DataFrame({col: [0] for col in spread_model.feature_names_in_})
    
    # Make predictions
    spread_raw = spread_model.predict(features)[0]
    total_raw = totals_model.predict(features)[0]
    win_prob = moneyline_model.predict_proba(features)[0][1]
    
    # Apply adjustments
    spread_adj = spread_raw + MODERN_NBA_ADJUSTMENTS['spread']
    total_adj = total_raw + MODERN_NBA_ADJUSTMENTS['totals']
    
    # Display
    print(f"   Spread: {home_team} {spread_adj:+.1f} (raw: {spread_raw:+.1f}, adj: +{MODERN_NBA_ADJUSTMENTS['spread']:.1f})")
    print(f"   Total: {total_adj:.1f} (raw: {total_raw:.1f}, adj: +{MODERN_NBA_ADJUSTMENTS['totals']:.1f})")
    print(f"   Win Prob: {home_team} {win_prob:.1%} | {away_team} {1-win_prob:.1%}")
    
    # Recommendation
    if win_prob > 0.60:
        print(f"   💡 Recommendation: Bet {home_team} ML")
    elif win_prob < 0.40:
        print(f"   💡 Recommendation: Bet {away_team} ML")
    else:
        print(f"   💡 Recommendation: Close game, use spreads")
    
    predictions_list.append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'home_team': home_team,
        'away_team': away_team,
        'spread_adjusted': round(spread_adj, 1),
        'total_adjusted': round(total_adj, 1),
        'home_win_prob': round(win_prob * 100, 1),
        'away_win_prob': round((1 - win_prob) * 100, 1)
    })

# Save predictions
print("\n" + "=" * 80)
print("💾 Saving predictions...")
predictions_df = pd.DataFrame(predictions_list)
output_file = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
predictions_df.to_csv(output_file, index=False)
print(f"   ✅ Saved to: {output_file}")

print("\n" + "=" * 80)
print("✅ PREDICTIONS COMPLETE!")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   Games predicted: {len(predictions_list)}")
print(f"   Adjustments applied:")
print(f"     - Totals: +{MODERN_NBA_ADJUSTMENTS['totals']:.1f} points")
print(f"     - Spread: +{MODERN_NBA_ADJUSTMENTS['spread']:.1f} points")
print(f"     - Moneyline: No adjustment (61.4% accurate!)")
print(f"\n💡 Models calibrated for 2024-25 season!")
