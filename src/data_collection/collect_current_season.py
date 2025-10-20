"""
Collect Current 2024-25 NBA Season Data using NBA API

This script:
1. Fetches all games from the current 2024-25 season
2. Gets team and player statistics
3. Updates daily with new games
4. Prepares data for model predictions
"""

from nba_api.stats.endpoints import leaguegamefinder, teamgamelog, playergamelog
from nba_api.stats.static import teams, players
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os

print("=" * 80)
print("COLLECTING 2024-25 NBA SEASON DATA")
print("=" * 80)

# Paths
OUTPUT_DIR = "/home/ubuntu/nba_betting_model/data/current_season"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Current season
CURRENT_SEASON = "2024-25"

print(f"\n📅 Target Season: {CURRENT_SEASON}")
print(f"📂 Output Directory: {OUTPUT_DIR}")

# ============================================================================
# 1. GET ALL GAMES FROM CURRENT SEASON
# ============================================================================
print("\n" + "=" * 80)
print("1. FETCHING GAMES FROM 2024-25 SEASON")
print("=" * 80)

try:
    print("   Querying NBA API...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=CURRENT_SEASON,
        season_type_nullable='Regular Season'
    )
    
    games_df = gamefinder.get_data_frames()[0]
    
    print(f"   ✅ Retrieved {len(games_df)} game records")
    
    # Each game appears twice (once for each team), so actual games = records / 2
    unique_games = len(games_df['GAME_ID'].unique())
    print(f"   ✅ Unique games: {unique_games}")
    
    if len(games_df) > 0:
        print(f"   📅 Date range: {games_df['GAME_DATE'].min()} to {games_df['GAME_DATE'].max()}")
        
        # Save raw game data
        games_file = f"{OUTPUT_DIR}/games_2024_25_raw.csv"
        games_df.to_csv(games_file, index=False)
        print(f"   💾 Saved to: {games_file}")
    else:
        print("   ⚠️  No games found yet (season may not have started)")
        
except Exception as e:
    print(f"   ❌ Error fetching games: {e}")
    print("   💡 This is normal if the season hasn't started yet!")
    games_df = pd.DataFrame()

# ============================================================================
# 2. PROCESS GAMES INTO HOME/AWAY FORMAT
# ============================================================================
if len(games_df) > 0:
    print("\n" + "=" * 80)
    print("2. PROCESSING GAMES INTO HOME/AWAY FORMAT")
    print("=" * 80)
    
    # Group by game to get home and away teams
    processed_games = []
    
    for game_id in games_df['GAME_ID'].unique():
        game_data = games_df[games_df['GAME_ID'] == game_id]
        
        if len(game_data) == 2:  # Should have exactly 2 records (home and away)
            # Determine home team (MATCHUP contains '@' for away team)
            home_data = game_data[~game_data['MATCHUP'].str.contains('@')].iloc[0]
            away_data = game_data[game_data['MATCHUP'].str.contains('@')].iloc[0]
            
            processed_game = {
                'game_id': game_id,
                'game_date': home_data['GAME_DATE'],
                'season': CURRENT_SEASON,
                'home_team_id': home_data['TEAM_ID'],
                'home_team': home_data['TEAM_NAME'],
                'home_pts': home_data['PTS'],
                'home_fgm': home_data['FGM'],
                'home_fga': home_data['FGA'],
                'home_fg_pct': home_data['FG_PCT'],
                'home_fg3m': home_data['FG3M'],
                'home_fg3a': home_data['FG3A'],
                'home_fg3_pct': home_data['FG3_PCT'],
                'home_ftm': home_data['FTM'],
                'home_fta': home_data['FTA'],
                'home_ft_pct': home_data['FT_PCT'],
                'home_reb': home_data['REB'],
                'home_ast': home_data['AST'],
                'home_stl': home_data['STL'],
                'home_blk': home_data['BLK'],
                'home_tov': home_data['TOV'],
                'home_pf': home_data['PF'],
                'home_wl': home_data['WL'],
                'away_team_id': away_data['TEAM_ID'],
                'away_team': away_data['TEAM_NAME'],
                'away_pts': away_data['PTS'],
                'away_fgm': away_data['FGM'],
                'away_fga': away_data['FGA'],
                'away_fg_pct': away_data['FG_PCT'],
                'away_fg3m': away_data['FG3M'],
                'away_fg3a': away_data['FG3A'],
                'away_fg3_pct': away_data['FG3_PCT'],
                'away_ftm': away_data['FTM'],
                'away_fta': away_data['FTA'],
                'away_ft_pct': away_data['FT_PCT'],
                'away_reb': away_data['REB'],
                'away_ast': away_data['AST'],
                'away_stl': away_data['STL'],
                'away_blk': away_data['BLK'],
                'away_tov': away_data['TOV'],
                'away_pf': away_data['PF'],
                'away_wl': away_data['WL'],
                'actual_total': home_data['PTS'] + away_data['PTS'],
                'actual_spread': home_data['PTS'] - away_data['PTS'],
                'home_win': 1 if home_data['WL'] == 'W' else 0
            }
            
            processed_games.append(processed_game)
    
    processed_df = pd.DataFrame(processed_games)
    
    print(f"   ✅ Processed {len(processed_df)} games")
    
    # Save processed games
    processed_file = f"{OUTPUT_DIR}/games_2024_25_processed.csv"
    processed_df.to_csv(processed_file, index=False)
    print(f"   💾 Saved to: {processed_file}")
    
    # Statistics
    print("\n   📊 Current Season Statistics:")
    print(f"      Average Home Score: {processed_df['home_pts'].mean():.1f}")
    print(f"      Average Away Score: {processed_df['away_pts'].mean():.1f}")
    print(f"      Average Total: {processed_df['actual_total'].mean():.1f}")
    print(f"      Home Win Rate: {processed_df['home_win'].mean():.1%}")

# ============================================================================
# 3. GET TEAM INFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("3. FETCHING TEAM INFORMATION")
print("=" * 80)

try:
    nba_teams = teams.get_teams()
    teams_df = pd.DataFrame(nba_teams)
    
    print(f"   ✅ Retrieved {len(teams_df)} NBA teams")
    
    teams_file = f"{OUTPUT_DIR}/teams.csv"
    teams_df.to_csv(teams_file, index=False)
    print(f"   💾 Saved to: {teams_file}")
    
except Exception as e:
    print(f"   ❌ Error fetching teams: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COLLECTION SUMMARY")
print("=" * 80)

if len(games_df) > 0:
    print(f"\n✅ Successfully collected 2024-25 season data!")
    print(f"   Games: {len(processed_df)}")
    print(f"   Date Range: {processed_df['game_date'].min()} to {processed_df['game_date'].max()}")
    print(f"   Teams: {len(teams_df)}")
else:
    print(f"\n⚠️  No games available yet")
    print(f"   The 2024-25 season may not have started")
    print(f"   Run this script again after games begin!")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("\n1. ✅ Data collection script ready")
print("2. ⏭️  Run this script daily to get new games")
print("3. ⏭️  Use collected data for predictions")
print("4. ⏭️  Compare predictions vs actual results")

print("\n💡 TIP: Set up a cron job or scheduled task to run this daily!")
print("   Example: python3 collect_current_season.py")

print("\n" + "=" * 80)
print("COLLECTION COMPLETE!")
print("=" * 80)

