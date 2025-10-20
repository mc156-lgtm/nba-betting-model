"""
Simplified NBA Data Collection for Sample Dataset

This script collects a smaller sample of NBA data for model development.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguegamefinder, leaguedashteamstats
from nba_api.stats.static import teams, players
import time
import os


def collect_sample_data():
    """Collect sample NBA data for the current season"""
    
    data_dir = '../../data/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    print("NBA Sample Data Collection")
    print("=" * 60)
    
    # 1. Get teams
    print("\n1. Collecting teams...")
    all_teams = teams.get_teams()
    teams_df = pd.DataFrame(all_teams)
    teams_df.to_csv(f'{data_dir}/teams.csv', index=False)
    print(f"   Saved {len(teams_df)} teams")
    
    # 2. Get players
    print("\n2. Collecting players...")
    all_players = players.get_players()
    players_df = pd.DataFrame(all_players)
    players_df.to_csv(f'{data_dir}/players.csv', index=False)
    print(f"   Saved {len(players_df)} players")
    
    # 3. Get current season games (2023-24 - complete season)
    print("\n3. Collecting 2023-24 season games...")
    try:
        time.sleep(1)
        gamefinder = leaguegamefinder.LeagueGameFinder(
            season_nullable='2023-24',
            season_type_nullable='Regular Season',
            league_id_nullable='00'
        )
        games_df = gamefinder.get_data_frames()[0]
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df = games_df.sort_values('GAME_DATE')
        games_df.to_csv(f'{data_dir}/games_2023-24.csv', index=False)
        print(f"   Saved {len(games_df)} game records")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Get team stats for 2023-24
    print("\n4. Collecting 2023-24 team statistics...")
    try:
        time.sleep(1)
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season='2023-24',
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        )
        team_stats_df = team_stats.get_data_frames()[0]
        team_stats_df.to_csv(f'{data_dir}/team_stats_2023-24.csv', index=False)
        print(f"   Saved stats for {len(team_stats_df)} teams")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. Get a few more recent seasons
    seasons = ['2022-23', '2021-22', '2020-21']
    
    for season in seasons:
        print(f"\n5. Collecting {season} season data...")
        try:
            time.sleep(1)
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00'
            )
            games = gamefinder.get_data_frames()[0]
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games = games.sort_values('GAME_DATE')
            games.to_csv(f'{data_dir}/games_{season}.csv', index=False)
            print(f"   Saved {len(games)} game records for {season}")
            
            time.sleep(1)
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                per_mode_detailed='PerGame'
            )
            team_stats_df = team_stats.get_data_frames()[0]
            team_stats_df.to_csv(f'{data_dir}/team_stats_{season}.csv', index=False)
            print(f"   Saved team stats for {season}")
            
        except Exception as e:
            print(f"   Error for {season}: {e}")
    
    print("\n" + "=" * 60)
    print("Sample data collection complete!")
    print("=" * 60)


if __name__ == "__main__":
    collect_sample_data()

