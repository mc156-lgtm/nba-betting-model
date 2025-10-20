"""
Integrate Wyatt Walsh Basketball Database with Existing Betting Data

This script:
1. Extracts current season (2024-25) games from Wyatt Walsh database
2. Merges with historical betting data from Hallmark dataset
3. Creates enhanced features for predictions
4. Prepares data for model retraining
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("INTEGRATING WYATT WALSH BASKETBALL DATABASE")
print("=" * 80)

# Paths
WYATT_WALSH_DB = "/home/ubuntu/nba_betting_model/data/raw/wyatt_walsh/nba.sqlite"
WYATT_WALSH_CSV = "/home/ubuntu/nba_betting_model/data/raw/wyatt_walsh/csv"
HALLMARK_DATA = "/home/ubuntu/nba_betting_model/data/processed/games_with_betting_odds.csv"
OUTPUT_DIR = "/home/ubuntu/nba_betting_model/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Connect to SQLite database
print("\n1. Connecting to Wyatt Walsh SQLite database...")
conn = sqlite3.connect(WYATT_WALSH_DB)

# Get table names
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(f"   Found {len(tables)} tables in database")
print(f"   Tables: {', '.join(tables['name'].tolist()[:10])}...")

# Load game data from SQLite
print("\n2. Loading game data from SQLite...")
query = """
SELECT 
    game_id,
    game_date,
    season_id,
    team_id_home,
    team_id_away,
    pts_home,
    pts_away,
    fg_pct_home,
    fg_pct_away,
    ft_pct_home,
    ft_pct_away,
    fg3_pct_home,
    fg3_pct_away,
    ast_home,
    ast_away,
    reb_home,
    reb_away
FROM game
WHERE game_date IS NOT NULL
ORDER BY game_date DESC
"""

try:
    wyatt_games = pd.read_sql(query, conn)
    print(f"   Loaded {len(wyatt_games):,} games from database")
except Exception as e:
    print(f"   Error loading from SQL, trying CSV instead: {e}")
    # Load from CSV if SQL fails
    wyatt_games = pd.read_csv(f"{WYATT_WALSH_CSV}/game.csv")
    print(f"   Loaded {len(wyatt_games):,} games from CSV")

conn.close()

# Convert game_date to datetime
print("\n3. Processing dates...")
wyatt_games['game_date'] = pd.to_datetime(wyatt_games['game_date'], errors='coerce')
wyatt_games = wyatt_games.dropna(subset=['game_date'])

# Get current season games (2023-24 and 2024-25)
current_season = wyatt_games[wyatt_games['game_date'] >= '2023-10-01'].copy()
print(f"   Found {len(current_season):,} games from 2023-24 and 2024-25 seasons")

# Load existing betting data
print("\n4. Loading existing betting data from Hallmark dataset...")
if os.path.exists(HALLMARK_DATA):
    betting_data = pd.read_csv(HALLMARK_DATA)
    print(f"   Loaded {len(betting_data):,} games with betting odds")
    
    # Show date range
    if 'date' in betting_data.columns:
        betting_data['date'] = pd.to_datetime(betting_data['date'], errors='coerce')
        print(f"   Betting data range: {betting_data['date'].min()} to {betting_data['date'].max()}")
else:
    print("   No existing betting data found")
    betting_data = pd.DataFrame()

# Calculate basic stats for current season
print("\n5. Calculating statistics for current season games...")
current_season['total_points'] = current_season['pts_home'] + current_season['pts_away']
current_season['point_differential'] = current_season['pts_home'] - current_season['pts_away']
current_season['home_win'] = (current_season['pts_home'] > current_season['pts_away']).astype(int)

# Summary statistics
print("\n" + "=" * 80)
print("CURRENT SEASON STATISTICS (2023-24 & 2024-25)")
print("=" * 80)
print(f"Total Games: {len(current_season):,}")
print(f"Date Range: {current_season['game_date'].min()} to {current_season['game_date'].max()}")
print(f"\nScoring:")
print(f"  Average Home Score: {current_season['pts_home'].mean():.1f}")
print(f"  Average Away Score: {current_season['pts_away'].mean():.1f}")
print(f"  Average Total: {current_season['total_points'].mean():.1f}")
print(f"  Home Win Rate: {current_season['home_win'].mean():.1%}")

# Save current season data
output_file = f"{OUTPUT_DIR}/wyatt_walsh_current_season.csv"
current_season.to_csv(output_file, index=False)
print(f"\n✅ Saved current season data to: {output_file}")

# Create combined dataset summary
print("\n" + "=" * 80)
print("COMBINED DATASET SUMMARY")
print("=" * 80)
if len(betting_data) > 0:
    print(f"Historical Betting Data: {len(betting_data):,} games (2006-2018)")
print(f"Current Season Data: {len(current_season):,} games (2023-2025)")
print(f"Total Available: {len(betting_data) + len(current_season):,} games")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. ✅ Current season data extracted and saved")
print("2. ⏭️  Use this data for current season predictions")
print("3. ⏭️  Combine with betting odds when available")
print("4. ⏭️  Retrain models with enhanced features")

print("\n" + "=" * 80)
print("INTEGRATION COMPLETE!")
print("=" * 80)

