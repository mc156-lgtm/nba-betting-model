"""
Restructure game data to have one row per game with home/away scores
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("RESTRUCTURING GAME DATA")
print("=" * 80)

# Load the processed data
data_file = Path('../../data/processed/hallmark_games_with_odds.csv')
df = pd.read_csv(data_file)

print(f"\n📂 Loaded {len(df):,} records")
print(f"Columns: {list(df.columns)}")

# Separate home and away games
print("\n🔧 Separating home and away games...")
home_games = df[df['is_home'] == 't'].copy()
away_games = df[df['is_home'] == 'f'].copy()

print(f"  Home games: {len(home_games):,}")
print(f"  Away games: {len(away_games):,}")

# Rename columns for home team
home_games = home_games.rename(columns={
    'team_id': 'home_team_id',
    'pts': 'home_pts',
    'fgm': 'home_fgm',
    'fga': 'home_fga',
    'fg_pct': 'home_fg_pct',
    'fg3m': 'home_fg3m',
    'fg3a': 'home_fg3a',
    'fg3_pct': 'home_fg3_pct',
    'ftm': 'home_ftm',
    'fta': 'home_fta',
    'ft_pct': 'home_ft_pct',
    'reb': 'home_reb',
    'ast': 'home_ast',
    'stl': 'home_stl',
    'blk': 'home_blk',
    'tov': 'home_tov',
    'pf': 'home_pf',
    'wl': 'home_wl'
})

# Rename columns for away team
away_games = away_games.rename(columns={
    'team_id': 'away_team_id',
    'pts': 'away_pts',
    'fgm': 'away_fgm',
    'fga': 'away_fga',
    'fg_pct': 'away_fg_pct',
    'fg3m': 'away_fg3m',
    'fg3a': 'away_fg3a',
    'fg3_pct': 'away_fg3_pct',
    'ftm': 'away_ftm',
    'fta': 'away_fta',
    'ft_pct': 'away_ft_pct',
    'reb': 'away_reb',
    'ast': 'away_ast',
    'stl': 'away_stl',
    'blk': 'away_blk',
    'tov': 'away_tov',
    'pf': 'away_pf',
    'wl': 'away_wl'
})

# Merge home and away games
print("\n🔧 Merging home and away data...")

# Select columns to keep from each
home_cols = ['game_id', 'game_date', 'season', 'home_team_id', 'home_pts', 
             'home_fgm', 'home_fga', 'home_fg_pct', 'home_fg3m', 'home_fg3a', 'home_fg3_pct',
             'home_ftm', 'home_fta', 'home_ft_pct', 'home_reb', 'home_ast', 
             'home_stl', 'home_blk', 'home_tov', 'home_pf', 'home_wl',
             'ml_home_avg', 'ml_away_avg', 'spread_avg', 'total_avg']

away_cols = ['game_id', 'away_team_id', 'away_pts',
             'away_fgm', 'away_fga', 'away_fg_pct', 'away_fg3m', 'away_fg3a', 'away_fg3_pct',
             'away_ftm', 'away_fta', 'away_ft_pct', 'away_reb', 'away_ast',
             'away_stl', 'away_blk', 'away_tov', 'away_pf', 'away_wl']

games = home_games[home_cols].merge(away_games[away_cols], on='game_id', how='inner')

print(f"  ✓ Created {len(games):,} complete games")

# Calculate outcomes
print("\n🔧 Calculating game outcomes...")

games['actual_spread'] = games['home_pts'] - games['away_pts']
games['actual_total'] = games['home_pts'] + games['away_pts']
games['home_win'] = (games['actual_spread'] > 0).astype(int)

# Calculate betting outcomes (only for games with odds)
games_with_odds = games[games['spread_avg'].notna()].copy()

games_with_odds['beat_spread'] = (games_with_odds['actual_spread'] > games_with_odds['spread_avg']).astype(int)
games_with_odds['went_over'] = (games_with_odds['actual_total'] > games_with_odds['total_avg']).astype(int)

print(f"  ✓ Games with complete betting data: {len(games_with_odds):,}")

# Calculate statistics
print("\n📊 Data Statistics:")
print(f"   Total games: {len(games):,}")
print(f"   Games with betting odds: {len(games_with_odds):,}")
print(f"   Average home score: {games['home_pts'].mean():.1f}")
print(f"   Average away score: {games['away_pts'].mean():.1f}")
print(f"   Average spread: {games['actual_spread'].mean():.2f} points")
print(f"   Average total: {games['actual_total'].mean():.1f} points")
print(f"   Home win rate: {games['home_win'].mean():.2%}")

if len(games_with_odds) > 0:
    print(f"\n📊 Betting Line Statistics:")
    print(f"   Average betting spread: {games_with_odds['spread_avg'].mean():.2f}")
    print(f"   Average betting total: {games_with_odds['total_avg'].mean():.1f}")
    print(f"   Home team beat spread: {games_with_odds['beat_spread'].mean():.2%}")
    print(f"   Games went over: {games_with_odds['went_over'].mean():.2%}")

# Save restructured data
output_file = Path('../../data/processed/games_restructured.csv')
games.to_csv(output_file, index=False)

print(f"\n✅ Saved restructured data to: {output_file}")
print(f"   Total columns: {len(games.columns)}")

# Also save just the games with betting odds
output_file_odds = Path('../../data/processed/games_with_betting_odds.csv')
games_with_odds.to_csv(output_file_odds, index=False)

print(f"✅ Saved betting games to: {output_file_odds}")

print("\n" + "=" * 80)
print("✅ RESTRUCTURING COMPLETE!")
print("=" * 80)

