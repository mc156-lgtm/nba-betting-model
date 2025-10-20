#!/usr/bin/env python3
"""
Integrate all 4 Kaggle datasets for 2024-25 NBA season
Combines player stats, box scores, player info, and season totals
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 80)
print("INTEGRATING 2024-25 NBA SEASON DATA")
print("=" * 80)

# Paths
raw_dir = '../../data/raw/season_2024_25'
processed_dir = '../../data/processed/season_2024_25'
os.makedirs(processed_dir, exist_ok=True)

# ============================================================================
# DATASET #1: Player Game Stats (Eduardo Palmieri) - PRIMARY DATA
# ============================================================================
print("\n📊 Loading Dataset #1: Player Game Stats (Daily Updates)...")
df1 = pd.read_csv(f'{raw_dir}/dataset1_player_stats/database_24_25.csv')
print(f"   Loaded: {len(df1):,} player game performances")

# Clean and standardize
df1['game_date'] = pd.to_datetime(df1['Data'], errors='coerce')
df1['player_name'] = df1['Player'].str.strip()
df1['team'] = df1['Tm'].str.strip()
df1['opponent'] = df1['Opp'].str.strip()
df1['is_win'] = df1['Res'].str.contains('W', na=False)

# Rename columns to standard format
df1_clean = df1.rename(columns={
    'MP': 'minutes',
    'FG': 'fgm',
    'FGA': 'fga',
    'FG%': 'fg_pct',
    '3P': 'fg3m',
    '3PA': 'fg3a',
    '3P%': 'fg3_pct',
    'FT': 'ftm',
    'FTA': 'fta',
    'FT%': 'ft_pct',
    'ORB': 'oreb',
    'DRB': 'dreb',
    'TRB': 'reb',
    'AST': 'ast',
    'STL': 'stl',
    'BLK': 'blk',
    'TOV': 'tov',
    'PF': 'pf',
    'PTS': 'pts',
    'GmSc': 'game_score'
})

print(f"   Date range: {df1_clean['game_date'].min()} to {df1_clean['game_date'].max()}")
print(f"   Unique players: {df1_clean['player_name'].nunique()}")
print(f"   Unique games: {df1_clean['game_date'].nunique()}")

# ============================================================================
# DATASET #2: Box Scores (Alberto Filosa) - SUPPLEMENTARY
# ============================================================================
print("\n📊 Loading Dataset #2: Box Scores...")
df2 = pd.read_csv(f'{raw_dir}/dataset2_box_scores/NBA-BoxScores-2024-2025.csv')
print(f"   Loaded: {len(df2):,} box score entries")
print(f"   Unique games: {df2['GAME_ID'].nunique()}")
print(f"   Unique players: {df2['PLAYER_NAME'].nunique()}")

# ============================================================================
# DATASET #3: Player Info (Sai Krishnan) - REFERENCE DATA
# ============================================================================
print("\n📊 Loading Dataset #3: Player Info & Salaries...")
df3 = pd.read_csv(f'{raw_dir}/dataset3_player_info/nba_players.csv')
print(f"   Loaded: {len(df3):,} players")
print(f"   Players with salary data: {df3['salary'].notna().sum()}")
print(f"   Average salary: ${df3['salary'].mean():,.0f}")

# Clean player info
df3_clean = df3[['id', 'fullName', 'displayHeight', 'displayWeight', 'age', 'salary']].copy()
df3_clean.columns = ['player_id', 'player_name', 'height', 'weight', 'age', 'salary']
df3_clean['player_name'] = df3_clean['player_name'].str.strip()

# ============================================================================
# DATASET #4: Season Stats (Anu Alli) - SEASON AVERAGES
# ============================================================================
print("\n📊 Loading Dataset #4: Season Totals...")
df4 = pd.read_csv(f'{raw_dir}/dataset4_season_stats/nba_stats_2024.csv')
print(f"   Loaded: {len(df4):,} players")
print(f"   Top scorer: {df4.loc[df4['PTS'].idxmax(), 'Player']} ({df4['PTS'].max():.1f} PPG)")

# Clean season stats
df4_clean = df4.copy()
df4_clean['player_name'] = df4_clean['Player'].str.strip()

# ============================================================================
# AGGREGATE TO TEAM LEVEL FOR GAME PREDICTIONS
# ============================================================================
print("\n🏀 Aggregating player stats to team level...")

# Group by team and game date
team_games = df1_clean.groupby(['team', 'opponent', 'game_date', 'is_win']).agg({
    'pts': 'sum',
    'reb': 'sum',
    'ast': 'sum',
    'stl': 'sum',
    'blk': 'sum',
    'tov': 'sum',
    'fgm': 'sum',
    'fga': 'sum',
    'fg3m': 'sum',
    'fg3a': 'sum',
    'ftm': 'sum',
    'fta': 'sum',
    'minutes': 'sum'
}).reset_index()

# Calculate team shooting percentages
team_games['fg_pct'] = (team_games['fgm'] / team_games['fga'] * 100).round(1)
team_games['fg3_pct'] = (team_games['fg3m'] / team_games['fg3a'] * 100).round(1)
team_games['ft_pct'] = (team_games['ftm'] / team_games['fta'] * 100).round(1)

print(f"   Created {len(team_games):,} team game records")
print(f"   Teams: {team_games['team'].nunique()}")
print(f"   Games per team: {len(team_games) / team_games['team'].nunique():.1f}")

# ============================================================================
# CREATE GAME-LEVEL DATA (HOME vs AWAY)
# ============================================================================
print("\n🏟️  Creating game-level data (home vs away)...")

# Identify home/away (team listed first is typically home, but we'll use both perspectives)
games_list = []

for date in team_games['game_date'].unique():
    date_games = team_games[team_games['game_date'] == date]
    
    # Match teams playing each other
    for _, team_row in date_games.iterrows():
        team = team_row['team']
        opp = team_row['opponent']
        
        # Find opponent's stats for same game
        opp_row = date_games[(date_games['team'] == opp) & (date_games['opponent'] == team)]
        
        if len(opp_row) > 0:
            opp_row = opp_row.iloc[0]
            
            # Create game record (assume first team is home)
            game = {
                'game_date': date,
                'team_home': team,
                'team_away': opp,
                'pts_home': team_row['pts'],
                'pts_away': opp_row['pts'],
                'reb_home': team_row['reb'],
                'reb_away': opp_row['reb'],
                'ast_home': team_row['ast'],
                'ast_away': opp_row['ast'],
                'fg_pct_home': team_row['fg_pct'],
                'fg_pct_away': opp_row['fg_pct'],
                'fg3_pct_home': team_row['fg3_pct'],
                'fg3_pct_away': opp_row['fg3_pct'],
                'total_points': team_row['pts'] + opp_row['pts'],
                'point_diff': team_row['pts'] - opp_row['pts'],
                'home_win': team_row['is_win']
            }
            games_list.append(game)

# Remove duplicates (each game appears twice)
games_df = pd.DataFrame(games_list).drop_duplicates(subset=['game_date', 'team_home', 'team_away'])

print(f"   Created {len(games_df):,} unique games")
print(f"   Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
print(f"   Average total: {games_df['total_points'].mean():.1f} points")
print(f"   Home win rate: {games_df['home_win'].mean():.1%}")

# ============================================================================
# CALCULATE TEAM SEASON AVERAGES
# ============================================================================
print("\n📈 Calculating team season averages...")

team_avg = team_games.groupby('team').agg({
    'pts': 'mean',
    'reb': 'mean',
    'ast': 'mean',
    'stl': 'mean',
    'blk': 'mean',
    'fg_pct': 'mean',
    'fg3_pct': 'mean',
    'ft_pct': 'mean',
    'is_win': 'mean'
}).round(2)

team_avg.columns = ['avg_pts', 'avg_reb', 'avg_ast', 'avg_stl', 'avg_blk', 
                    'avg_fg_pct', 'avg_fg3_pct', 'avg_ft_pct', 'win_pct']
team_avg['games_played'] = team_games.groupby('team').size()

print(f"   Teams: {len(team_avg)}")
print(f"\n   Top 5 Scoring Teams:")
print(team_avg.nlargest(5, 'avg_pts')[['avg_pts', 'games_played', 'win_pct']])

# ============================================================================
# SAVE PROCESSED DATA
# ============================================================================
print("\n💾 Saving processed data...")

# Save player game stats
df1_clean.to_csv(f'{processed_dir}/player_game_stats.csv', index=False)
print(f"   ✅ Saved: player_game_stats.csv ({len(df1_clean):,} rows)")

# Save player info
df3_clean.to_csv(f'{processed_dir}/player_info.csv', index=False)
print(f"   ✅ Saved: player_info.csv ({len(df3_clean):,} rows)")

# Save season averages
df4_clean.to_csv(f'{processed_dir}/season_averages.csv', index=False)
print(f"   ✅ Saved: season_averages.csv ({len(df4_clean):,} rows)")

# Save team game stats
team_games.to_csv(f'{processed_dir}/team_game_stats.csv', index=False)
print(f"   ✅ Saved: team_game_stats.csv ({len(team_games):,} rows)")

# Save game-level data
games_df.to_csv(f'{processed_dir}/games_2024_25.csv', index=False)
print(f"   ✅ Saved: games_2024_25.csv ({len(games_df):,} rows)")

# Save team averages
team_avg.to_csv(f'{processed_dir}/team_averages.csv')
print(f"   ✅ Saved: team_averages.csv ({len(team_avg):,} rows)")

# ============================================================================
# GENERATE STATISTICS REPORT
# ============================================================================
print("\n" + "=" * 80)
print("📊 2024-25 SEASON STATISTICS")
print("=" * 80)

print(f"\n🏀 Games Played: {len(games_df):,}")
print(f"📅 Date Range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
print(f"🏟️  Teams: {team_avg.index.nunique()}")
print(f"👥 Players: {df1_clean['player_name'].nunique()}")

print(f"\n📈 Scoring Statistics:")
print(f"   Average Total: {games_df['total_points'].mean():.1f} points")
print(f"   Average Home Score: {games_df['pts_home'].mean():.1f}")
print(f"   Average Away Score: {games_df['pts_away'].mean():.1f}")
print(f"   Home Win Rate: {games_df['home_win'].mean():.1%}")

print(f"\n⭐ Top 5 Scorers (Season Average):")
top_scorers = df4_clean.nlargest(5, 'PTS')[['Player', 'Team', 'PTS', 'G']]
for idx, row in top_scorers.iterrows():
    print(f"   {row['Player']:25} ({row['Team']:3}): {row['PTS']:.1f} PPG in {int(row['G'])} games")

print(f"\n💰 Highest Paid Players:")
top_salaries = df3_clean.nlargest(5, 'salary')[['player_name', 'salary', 'age']]
for idx, row in top_salaries.iterrows():
    if pd.notna(row['salary']):
        print(f"   {row['player_name']:25}: ${row['salary']:,.0f} (age {int(row['age'])})")

print("\n" + "=" * 80)
print("✅ INTEGRATION COMPLETE!")
print("=" * 80)
print(f"\nProcessed data saved to: {processed_dir}/")
print(f"\nFiles created:")
print(f"  1. player_game_stats.csv - {len(df1_clean):,} player performances")
print(f"  2. player_info.csv - {len(df3_clean):,} player profiles")
print(f"  3. season_averages.csv - {len(df4_clean):,} season stats")
print(f"  4. team_game_stats.csv - {len(team_games):,} team performances")
print(f"  5. games_2024_25.csv - {len(games_df):,} games (ready for predictions!)")
print(f"  6. team_averages.csv - {len(team_avg):,} team season averages")

print(f"\n🎯 Ready for model predictions!")
print("=" * 80)

