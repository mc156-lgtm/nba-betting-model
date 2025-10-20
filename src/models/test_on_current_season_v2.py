#!/usr/bin/env python3
"""
Test existing models on 2024-25 season data (CORRECTED VERSION)
Uses proper feature format that models were trained on
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TESTING MODELS ON 2024-25 SEASON (CORRECTED)")
print("=" * 80)

# Load 2024-25 season data
print("\n📊 Loading 2024-25 season data...")
games_df = pd.read_csv('../../data/processed/season_2024_25/games_2024_25.csv')
team_games = pd.read_csv('../../data/processed/season_2024_25/team_game_stats.csv')
team_avg = pd.read_csv('../../data/processed/season_2024_25/team_averages.csv', index_col=0)

print(f"   Games: {len(games_df):,}")
print(f"   Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")

# Load trained models
print("\n🤖 Loading trained models...")
spread_model = joblib.load('../../models/spread_model.pkl')
totals_model = joblib.load('../../models/totals_model.pkl')
moneyline_model = joblib.load('../../models/moneyline_model.pkl')
print("   ✅ All models loaded")

# Required features (from model training)
required_features = ['home_fg_pct', 'away_fg_pct', 'home_fg_pct_diff', 
                     'home_fg3_pct', 'away_fg3_pct', 'home_fg3_pct_diff',
                     'home_ft_pct', 'away_ft_pct', 'home_ft_pct_diff',
                     'home_reb', 'away_reb', 'home_reb_diff',
                     'home_ast', 'away_ast', 'home_ast_diff',
                     'home_tov', 'away_tov', 'home_tov_diff',
                     'total_fga', 'total_fta', 'spread_avg', 'total_avg']

# Prepare features using team averages
print("\n🔧 Preparing features with correct format...")

features_list = []

for idx, game in games_df.iterrows():
    home_team = game['team_home']
    away_team = game['team_away']
    
    if home_team in team_avg.index and away_team in team_avg.index:
        home_stats = team_avg.loc[home_team]
        away_stats = team_avg.loc[away_team]
        
        # Create features matching training format
        features = {
            # Shooting percentages
            'home_fg_pct': home_stats['avg_fg_pct'],
            'away_fg_pct': away_stats['avg_fg_pct'],
            'home_fg_pct_diff': home_stats['avg_fg_pct'] - away_stats['avg_fg_pct'],
            
            'home_fg3_pct': home_stats['avg_fg3_pct'],
            'away_fg3_pct': away_stats['avg_fg3_pct'],
            'home_fg3_pct_diff': home_stats['avg_fg3_pct'] - away_stats['avg_fg3_pct'],
            
            'home_ft_pct': home_stats['avg_ft_pct'],
            'away_ft_pct': away_stats['avg_ft_pct'],
            'home_ft_pct_diff': home_stats['avg_ft_pct'] - away_stats['avg_ft_pct'],
            
            # Box score stats
            'home_reb': home_stats['avg_reb'],
            'away_reb': away_stats['avg_reb'],
            'home_reb_diff': home_stats['avg_reb'] - away_stats['avg_reb'],
            
            'home_ast': home_stats['avg_ast'],
            'away_ast': away_stats['avg_ast'],
            'home_ast_diff': home_stats['avg_ast'] - away_stats['avg_ast'],
            
            # Turnovers (estimate from other stats)
            'home_tov': home_stats['avg_pts'] * 0.12,  # Estimate ~12% of possessions
            'away_tov': away_stats['avg_pts'] * 0.12,
            'home_tov_diff': (home_stats['avg_pts'] - away_stats['avg_pts']) * 0.12,
            
            # Pace indicators (estimates)
            'total_fga': (home_stats['avg_pts'] + away_stats['avg_pts']) / 2.0 * 0.85,  # Estimate FGA
            'total_fta': (home_stats['avg_pts'] + away_stats['avg_pts']) / 2.0 * 0.25,  # Estimate FTA
            
            # Spread and total averages
            'spread_avg': home_stats['avg_pts'] - away_stats['avg_pts'],
            'total_avg': home_stats['avg_pts'] + away_stats['avg_pts'],
            
            # Actual results
            'actual_total': game['total_points'],
            'actual_spread': game['point_diff'],
            'actual_home_win': game['home_win']
        }
        features_list.append(features)

features_df = pd.DataFrame(features_list)
print(f"   Features created for {len(features_df):,} games")

# Prepare X (features) for predictions
X = features_df[required_features].fillna(0)

# ============================================================================
# TEST TOTALS MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TESTING TOTALS MODEL")
print("=" * 80)

predicted_totals = totals_model.predict(X)
actual_totals = features_df['actual_total'].values

# Calculate metrics
mae = mean_absolute_error(actual_totals, predicted_totals)

# Calculate over/under accuracy
avg_total = actual_totals.mean()
predicted_over = predicted_totals > avg_total
actual_over = actual_totals > avg_total
over_under_acc = accuracy_score(actual_over, predicted_over)

print(f"\n📊 Results:")
print(f"   MAE: {mae:.2f} points")
print(f"   Average actual total: {actual_totals.mean():.1f}")
print(f"   Average predicted total: {predicted_totals.mean():.1f}")
print(f"   Prediction bias: {predicted_totals.mean() - actual_totals.mean():+.1f} points")
print(f"   Over/Under accuracy: {over_under_acc:.1%}")

# Show distribution
print(f"\n📈 Prediction Distribution:")
print(f"   Min prediction: {predicted_totals.min():.1f}")
print(f"   Max prediction: {predicted_totals.max():.1f}")
print(f"   Std deviation: {predicted_totals.std():.1f}")

# ============================================================================
# TEST SPREAD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TESTING SPREAD MODEL")
print("=" * 80)

predicted_spreads = spread_model.predict(X)
actual_spreads = features_df['actual_spread'].values

# Calculate metrics
mae = mean_absolute_error(actual_spreads, predicted_spreads)

# Calculate cover accuracy
predicted_home_cover = predicted_spreads > 0
actual_home_cover = actual_spreads > 0
cover_acc = accuracy_score(actual_home_cover, predicted_home_cover)

print(f"\n📊 Results:")
print(f"   MAE: {mae:.2f} points")
print(f"   Average actual spread: {actual_spreads.mean():.1f}")
print(f"   Average predicted spread: {predicted_spreads.mean():.1f}")
print(f"   Prediction bias: {predicted_spreads.mean() - actual_spreads.mean():+.1f} points")
print(f"   Cover accuracy: {cover_acc:.1%}")

print(f"\n📈 Prediction Distribution:")
print(f"   Min prediction: {predicted_spreads.min():.1f}")
print(f"   Max prediction: {predicted_spreads.max():.1f}")
print(f"   Std deviation: {predicted_spreads.std():.1f}")

# ============================================================================
# TEST MONEYLINE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🎯 TESTING MONEYLINE MODEL")
print("=" * 80)

predicted_home_wins = moneyline_model.predict(X)
actual_home_wins = features_df['actual_home_win'].values

# Calculate metrics
accuracy = accuracy_score(actual_home_wins, predicted_home_wins)

# Get probabilities
probas = moneyline_model.predict_proba(X)[:, 1]

# High confidence predictions (>70%)
high_conf_mask = (probas > 0.7) | (probas < 0.3)
if high_conf_mask.sum() > 0:
    high_conf_acc = accuracy_score(
        actual_home_wins[high_conf_mask],
        predicted_home_wins[high_conf_mask]
    )
    high_conf_count = high_conf_mask.sum()
else:
    high_conf_acc = 0
    high_conf_count = 0

print(f"\n📊 Results:")
print(f"   Overall accuracy: {accuracy:.1%}")
print(f"   Home wins predicted: {predicted_home_wins.sum()} / {len(predicted_home_wins)}")
print(f"   Actual home wins: {actual_home_wins.sum()} / {len(actual_home_wins)}")
print(f"   Average win probability: {probas.mean():.1%}")
print(f"\n   High confidence predictions (>70%): {high_conf_count}")
print(f"   High confidence accuracy: {high_conf_acc:.1%}")

# ============================================================================
# COMPARISON: OLD NBA vs NEW NBA
# ============================================================================
print("\n" + "=" * 80)
print("📈 COMPARISON: TRAINING DATA vs 2024-25 SEASON")
print("=" * 80)

print(f"\n🏀 Scoring:")
print(f"   Training data (2006-2018): ~200 points avg")
print(f"   2024-25 season: {games_df['total_points'].mean():.1f} points avg")
print(f"   Difference: +{games_df['total_points'].mean() - 200:.1f} points (+{(games_df['total_points'].mean() - 200) / 200 * 100:.1f}%)")

print(f"\n🏟️  Home Advantage:")
print(f"   Training data (2006-2018): ~62% home win rate")
print(f"   2024-25 season: {games_df['home_win'].mean():.1%} home win rate")
print(f"   Difference: {(games_df['home_win'].mean() - 0.62) * 100:+.1f} percentage points")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS & ADJUSTMENTS")
print("=" * 80)

total_bias = predicted_totals.mean() - actual_totals.mean()
spread_bias = predicted_spreads.mean() - actual_spreads.mean()

print(f"\n🎯 Totals Predictions:")
if abs(total_bias) > 5:
    print(f"   ⚠️  Model underpredicts by {abs(total_bias):.1f} points")
    print(f"   💡 ADJUSTMENT: Add {-total_bias:.1f} points to all totals predictions")
    print(f"   📊 Adjusted MAE would be: {mean_absolute_error(actual_totals, predicted_totals - total_bias):.2f}")
else:
    print(f"   ✅ Model is well-calibrated (bias: {total_bias:+.1f} points)")

print(f"\n🎯 Spread Predictions:")
if abs(spread_bias) > 2:
    print(f"   ⚠️  Model has bias of {spread_bias:+.1f} points")
    print(f"   💡 ADJUSTMENT: Add {-spread_bias:.1f} points to all spread predictions")
    print(f"   📊 Adjusted MAE would be: {mean_absolute_error(actual_spreads, predicted_spreads - spread_bias):.2f}")
else:
    print(f"   ✅ Model is well-calibrated (bias: {spread_bias:+.1f} points)")

print(f"\n🎯 Moneyline Predictions:")
if accuracy > 0.55:
    print(f"   ✅ Model is performing well ({accuracy:.1%} accuracy)")
    print(f"   💡 Profitable betting threshold: >52.4% accuracy")
else:
    print(f"   ⚠️  Model accuracy is below profitable threshold ({accuracy:.1%})")
    print(f"   💡 Use high-confidence predictions only (>70%)")

print("\n" + "=" * 80)
print("✅ TESTING COMPLETE!")
print("=" * 80)

# Save results
results = {
    'totals_mae': mae,
    'totals_bias': total_bias,
    'spread_mae': mean_absolute_error(actual_spreads, predicted_spreads),
    'spread_bias': spread_bias,
    'moneyline_accuracy': accuracy,
    'high_conf_accuracy': high_conf_acc,
    'games_tested': len(features_df)
}

import json
with open('../../data/processed/season_2024_25/model_performance.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: model_performance.json")

