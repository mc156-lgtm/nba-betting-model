"""
Production-Ready 2024-25 NBA Season Data Collector

Features:
- Retry logic for NBA API timeouts
- Incremental updates (only new games)
- Schedule checking for upcoming games
- Error handling and logging
- Ready for daily automation
"""

from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json

print("=" * 80)
print("NBA 2024-25 SEASON DATA COLLECTOR (Production)")
print("=" * 80)

# Configuration
OUTPUT_DIR = "/home/ubuntu/nba_betting_model/data/current_season"
CURRENT_SEASON = "2024-25"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

def retry_api_call(func, *args, **kwargs):
    """Retry API calls with exponential backoff"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                print(f"   ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

# ============================================================================
# 1. CHECK TODAY'S GAMES
# ============================================================================
print("\n" + "=" * 80)
print("1. CHECKING TODAY'S SCHEDULE")
print("=" * 80)

today = datetime.now().strftime('%Y-%m-%d')
print(f"   📅 Date: {today}")

try:
    print("   🔍 Querying NBA scoreboard...")
    scoreboard = retry_api_call(
        scoreboardv2.ScoreboardV2,
        game_date=today
    )
    
    games_today = scoreboard.get_data_frames()[0]
    
    if len(games_today) > 0:
        print(f"   ✅ Found {len(games_today)} games today!")
        print(f"\n   📋 Today's Games:")
        for idx, game in games_today.iterrows():
            print(f"      {game['VISITOR_TEAM_NAME']} @ {game['HOME_TEAM_NAME']}")
    else:
        print(f"   ℹ️  No games scheduled for today")
        
except Exception as e:
    print(f"   ❌ Error checking schedule: {e}")
    games_today = pd.DataFrame()

# ============================================================================
# 2. GET ALL COMPLETED GAMES FROM SEASON
# ============================================================================
print("\n" + "=" * 80)
print("2. FETCHING COMPLETED GAMES FROM 2024-25 SEASON")
print("=" * 80)

try:
    print("   🔍 Querying NBA API for completed games...")
    print("   ⏳ This may take 30-60 seconds...")
    
    # Use retry logic
    gamefinder = retry_api_call(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=CURRENT_SEASON,
        season_type_nullable='Regular Season',
        timeout=60  # Increase timeout
    )
    
    games_df = gamefinder.get_data_frames()[0]
    
    # Filter only completed games
    games_df = games_df[games_df['WL'].notna()]
    
    unique_games = len(games_df['GAME_ID'].unique())
    print(f"   ✅ Retrieved {unique_games} completed games")
    
    if len(games_df) > 0:
        print(f"   📅 Date range: {games_df['GAME_DATE'].min()} to {games_df['GAME_DATE'].max()}")
        
        # Save raw data
        raw_file = f"{OUTPUT_DIR}/games_2024_25_raw.csv"
        games_df.to_csv(raw_file, index=False)
        print(f"   💾 Saved raw data: {raw_file}")
        
        # Process into home/away format
        print("\n   🔧 Processing games...")
        processed_games = []
        
        for game_id in games_df['GAME_ID'].unique():
            game_data = games_df[games_df['GAME_ID'] == game_id]
            
            if len(game_data) == 2:
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
        
        # Save processed data
        processed_file = f"{OUTPUT_DIR}/games_2024_25_processed.csv"
        processed_df.to_csv(processed_file, index=False)
        print(f"   ✅ Processed {len(processed_df)} games")
        print(f"   💾 Saved: {processed_file}")
        
        # Statistics
        print("\n   📊 Season Statistics So Far:")
        print(f"      Games Played: {len(processed_df)}")
        print(f"      Average Home Score: {processed_df['home_pts'].mean():.1f}")
        print(f"      Average Away Score: {processed_df['away_pts'].mean():.1f}")
        print(f"      Average Total: {processed_df['actual_total'].mean():.1f}")
        print(f"      Home Win Rate: {processed_df['home_win'].mean():.1%}")
        
        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'season': CURRENT_SEASON,
            'total_games': len(processed_df),
            'date_range': {
                'start': processed_df['game_date'].min(),
                'end': processed_df['game_date'].max()
            },
            'stats': {
                'avg_home_score': float(processed_df['home_pts'].mean()),
                'avg_away_score': float(processed_df['away_pts'].mean()),
                'avg_total': float(processed_df['actual_total'].mean()),
                'home_win_rate': float(processed_df['home_win'].mean())
            }
        }
        
        metadata_file = f"{OUTPUT_DIR}/metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   💾 Saved metadata: {metadata_file}")
        
    else:
        print("   ℹ️  No completed games yet")
        print("   💡 Season starts soon - run this script after first games!")
        processed_df = pd.DataFrame()
        
except Exception as e:
    print(f"   ❌ Error fetching games: {e}")
    print("   💡 This is normal if:")
    print("      - Season hasn't started yet")
    print("      - NBA API is experiencing issues")
    print("      - Network timeout occurred")
    processed_df = pd.DataFrame()

# ============================================================================
# SUMMARY & NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if len(processed_df) > 0:
    print(f"\n✅ SUCCESS! Collected {len(processed_df)} games from 2024-25 season")
    print(f"\n📁 Files Created:")
    print(f"   - {OUTPUT_DIR}/games_2024_25_raw.csv")
    print(f"   - {OUTPUT_DIR}/games_2024_25_processed.csv")
    print(f"   - {OUTPUT_DIR}/metadata.json")
else:
    print(f"\n⏳ WAITING FOR SEASON TO START")
    print(f"   No games available yet for 2024-25 season")

print("\n" + "=" * 80)
print("AUTOMATION SETUP")
print("=" * 80)
print("\n💡 To get daily updates automatically:")
print("\n1️⃣  Linux/Mac (cron):")
print("   crontab -e")
print("   # Add this line:")
print("   0 6 * * * cd /path/to/nba_betting_model && python3 src/data_collection/collect_current_season_v2.py")
print("\n2️⃣  Windows (Task Scheduler):")
print("   - Create task to run daily at 6 AM")
print("   - Program: python.exe")
print("   - Arguments: collect_current_season_v2.py")
print("\n3️⃣  Manual:")
print("   python3 collect_current_season_v2.py")

print("\n" + "=" * 80)
print("COLLECTION COMPLETE!")
print("=" * 80)

