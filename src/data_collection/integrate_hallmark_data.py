"""
Integrate NBA Historical Stats and Betting Data (Evan Hallmark)
Process all 7 CSV files and create training data for betting models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("NBA BETTING MODEL - HALLMARK DATASET INTEGRATION")
print("=" * 80)

# Data directory
data_dir = Path('../../data/raw/nba_hallmark/')

print("\n📂 Loading datasets...")

# Load all 7 CSV files
print("  Loading betting data...")
betting_ml = pd.read_csv(data_dir / 'nba_betting_money_line.csv')
betting_spread = pd.read_csv(data_dir / 'nba_betting_spread.csv')
betting_totals = pd.read_csv(data_dir / 'nba_betting_totals.csv')

print(f"    ✓ Moneylines: {len(betting_ml):,} records")
print(f"    ✓ Spreads: {len(betting_spread):,} records")
print(f"    ✓ Totals: {len(betting_totals):,} records")

print("  Loading game data...")
games = pd.read_csv(data_dir / 'nba_games_all.csv')
print(f"    ✓ Games: {len(games):,} records")

print("  Loading player data...")
players = pd.read_csv(data_dir / 'nba_players_all.csv')
print(f"    ✓ Players: {len(players):,} records")

print("  Loading team data...")
teams = pd.read_csv(data_dir / 'nba_teams_all.csv')
print(f"    ✓ Teams: {len(teams):,} records")

# Player game stats is huge, we'll load it selectively
print("  Checking player game stats...")
player_stats_path = data_dir / 'nba_players_game_stats.csv'
print(f"    File size: {player_stats_path.stat().st_size / 1024 / 1024:.1f} MB")

print("\n" + "=" * 80)
print("PROCESSING BETTING DATA")
print("=" * 80)

# Show sample of betting data
print("\n📊 Sample Moneyline Data:")
print(betting_ml.head(3))
print(f"\nColumns: {list(betting_ml.columns)}")

print("\n📊 Sample Spread Data:")
print(betting_spread.head(3))
print(f"\nColumns: {list(betting_spread.columns)}")

print("\n📊 Sample Totals Data:")
print(betting_totals.head(3))
print(f"\nColumns: {list(betting_totals.columns)}")

print("\n" + "=" * 80)
print("AGGREGATING BETTING LINES")
print("=" * 80)

# Group betting data by game_id to get best available lines
print("\n🔧 Finding best available lines from multiple bookmakers...")

# For moneylines - get average odds across bookmakers
ml_avg = betting_ml.groupby('game_id').agg({
    'price1': 'mean',  # Home team price
    'price2': 'mean'   # Away team price
}).reset_index()
ml_avg.columns = ['game_id', 'ml_home_avg', 'ml_away_avg']

# For spreads - get average spread and odds
# Use spread1 (home team spread)
spread_avg = betting_spread.groupby('game_id').agg({
    'spread1': 'mean',
    'price1': 'mean',
    'price2': 'mean'
}).reset_index()
spread_avg.columns = ['game_id', 'spread_avg', 'spread_home_price', 'spread_away_price']

# For totals - get average total
# Use total1 (over/under line)
totals_avg = betting_totals.groupby('game_id').agg({
    'total1': 'mean',
    'price1': 'mean',  # Over price
    'price2': 'mean'   # Under price
}).reset_index()
totals_avg.columns = ['game_id', 'total_avg', 'over_price', 'under_price']

print(f"  ✓ Processed {len(ml_avg):,} games with moneyline data")
print(f"  ✓ Processed {len(spread_avg):,} games with spread data")
print(f"  ✓ Processed {len(totals_avg):,} games with totals data")

print("\n" + "=" * 80)
print("MERGING GAME DATA WITH BETTING LINES")
print("=" * 80)

# Merge all betting data with games
print("\n🔧 Joining datasets...")
games_with_odds = games.copy()

# Merge betting data
games_with_odds = games_with_odds.merge(ml_avg, on='game_id', how='left')
games_with_odds = games_with_odds.merge(spread_avg, on='game_id', how='left')
games_with_odds = games_with_odds.merge(totals_avg, on='game_id', how='left')

print(f"  ✓ Merged data: {len(games_with_odds):,} games")
print(f"  ✓ Games with moneyline odds: {games_with_odds['ml_home_avg'].notna().sum():,}")
print(f"  ✓ Games with spread odds: {games_with_odds['spread_avg'].notna().sum():,}")
print(f"  ✓ Games with totals odds: {games_with_odds['total_avg'].notna().sum():,}")

print("\n📊 Sample merged data:")
print(games_with_odds[['game_id', 'ml_home_avg', 'spread_avg', 'total_avg']].head())

print("\n" + "=" * 80)
print("CALCULATING BETTING FEATURES")
print("=" * 80)

# Calculate actual outcomes
print("\n🔧 Calculating actual game outcomes...")

# Check what columns are available
print(f"\nAvailable columns: {list(games_with_odds.columns)}")

# Find score columns (they might have different names)
score_cols = [col for col in games_with_odds.columns if 'pts' in col.lower() or 'score' in col.lower()]
print(f"Score columns found: {score_cols}")

# Try to identify home/away scores
if len(score_cols) >= 2:
    # Assume first is home, second is away (or try to detect from column names)
    home_score_col = [col for col in score_cols if 'home' in col.lower()]
    away_score_col = [col for col in score_cols if 'away' in col.lower() or 'visitor' in col.lower()]
    
    if home_score_col and away_score_col:
        games_with_odds['home_score'] = games_with_odds[home_score_col[0]]
        games_with_odds['away_score'] = games_with_odds[away_score_col[0]]
    else:
        # Use first two score columns
        games_with_odds['home_score'] = games_with_odds[score_cols[0]]
        games_with_odds['away_score'] = games_with_odds[score_cols[1]]
    
    # Calculate outcomes
    games_with_odds['actual_spread'] = games_with_odds['home_score'] - games_with_odds['away_score']
    games_with_odds['actual_total'] = games_with_odds['home_score'] + games_with_odds['away_score']
    games_with_odds['home_win'] = (games_with_odds['actual_spread'] > 0).astype(int)
    
    # Calculate betting outcomes
    games_with_odds['beat_spread'] = (games_with_odds['actual_spread'] > games_with_odds['spread_avg']).astype(int)
    games_with_odds['went_over'] = (games_with_odds['actual_total'] > games_with_odds['total_avg']).astype(int)
    
    print(f"  ✓ Calculated actual spreads (mean: {games_with_odds['actual_spread'].mean():.2f})")
    print(f"  ✓ Calculated actual totals (mean: {games_with_odds['actual_total'].mean():.2f})")
    print(f"  ✓ Home win rate: {games_with_odds['home_win'].mean():.2%}")
else:
    print("  ⚠️  Could not find score columns automatically")
    print("  Available columns:", list(games_with_odds.columns))

print("\n" + "=" * 80)
print("SAVING PROCESSED DATA")
print("=" * 80)

# Save processed data
output_dir = Path('../../data/processed/')
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / 'hallmark_games_with_odds.csv'
games_with_odds.to_csv(output_file, index=False)

print(f"\n✅ Saved processed data to: {output_file}")
print(f"   Total games: {len(games_with_odds):,}")
print(f"   Total columns: {len(games_with_odds.columns)}")

# Show statistics
print("\n📊 Data Statistics:")
if 'actual_spread' in games_with_odds.columns:
    valid_games = games_with_odds.dropna(subset=['spread_avg', 'total_avg', 'actual_spread'])
    print(f"   Games with complete betting data: {len(valid_games):,}")
    print(f"   Average spread: {valid_games['actual_spread'].mean():.2f} points")
    print(f"   Average total: {valid_games['actual_total'].mean():.2f} points")
    print(f"   Average betting line spread: {valid_games['spread_avg'].mean():.2f}")
    print(f"   Average betting line total: {valid_games['total_avg'].mean():.2f}")

print("\n" + "=" * 80)
print("PROCESSING PLAYER STATS FOR PROPS MODELS")
print("=" * 80)

print("\n🔧 Loading player game stats (this may take a moment)...")
try:
    # Load player stats in chunks to manage memory
    player_stats = pd.read_csv(data_dir / 'nba_players_game_stats.csv', 
                               usecols=['game_id', 'player_id', 'pts', 'reb', 'ast', 'stl', 'blk'],
                               nrows=100000)  # Limit to recent games for now
    
    print(f"  ✓ Loaded {len(player_stats):,} player game records")
    
    # Calculate player averages
    player_avgs = player_stats.groupby('player_id').agg({
        'pts': 'mean',
        'reb': 'mean',
        'ast': 'mean',
        'stl': 'mean',
        'blk': 'mean'
    }).reset_index()
    
    player_avgs.columns = ['player_id', 'avg_pts', 'avg_reb', 'avg_ast', 'avg_stl', 'avg_blk']
    
    # Save player averages
    player_avgs_file = output_dir / 'player_averages.csv'
    player_avgs.to_csv(player_avgs_file, index=False)
    
    print(f"  ✓ Calculated averages for {len(player_avgs):,} players")
    print(f"  ✓ Saved to: {player_avgs_file}")
    
except Exception as e:
    print(f"  ⚠️  Error processing player stats: {e}")
    print("  Continuing without player props data...")

print("\n" + "=" * 80)
print("✅ INTEGRATION COMPLETE!")
print("=" * 80)

print("\nNext steps:")
print("1. Update feature engineering: cd ../features && python build_features.py")
print("2. Retrain models: cd ../models && python spread_model.py")
print("3. Test predictions: python predict.py")
print("4. Push to GitHub: git add . && git commit -m 'Integrated real betting data' && git push")

print("\n" + "=" * 80)

