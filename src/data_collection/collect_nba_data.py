"""
NBA Data Collection Module

This module collects NBA team and player statistics using the nba_api package.
It fetches historical game data, team stats, player stats, and box scores.
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    leaguegamefinder,
    teamgamelog,
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
    boxscoretraditionalv2,
    boxscoreadvancedv2
)
from nba_api.stats.static import teams, players
import time
import json
from datetime import datetime, timedelta
import os


class NBADataCollector:
    """Collects NBA data from various endpoints"""
    
    def __init__(self, data_dir='../../data/raw'):
        self.data_dir = data_dir
        self.delay = 0.6  # API rate limiting delay
        os.makedirs(data_dir, exist_ok=True)
        
    def get_all_teams(self):
        """Get all NBA teams"""
        all_teams = teams.get_teams()
        df = pd.DataFrame(all_teams)
        filepath = os.path.join(self.data_dir, 'teams.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} teams to {filepath}")
        return df
    
    def get_all_players(self):
        """Get all NBA players"""
        all_players = players.get_players()
        df = pd.DataFrame(all_players)
        filepath = os.path.join(self.data_dir, 'players.csv')
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} players to {filepath}")
        return df
    
    def get_season_games(self, season='2024-25', season_type='Regular Season'):
        """
        Get all games for a specific season
        
        Args:
            season: Season in format 'YYYY-YY' (e.g., '2024-25')
            season_type: 'Regular Season', 'Playoffs', or 'All Star'
        """
        print(f"Fetching games for {season} {season_type}...")
        
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable='00'
            )
            time.sleep(self.delay)
            
            games = gamefinder.get_data_frames()[0]
            
            # Each game appears twice (once for each team), so we need to deduplicate
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games = games.sort_values('GAME_DATE')
            
            filepath = os.path.join(self.data_dir, f'games_{season}_{season_type.replace(" ", "_")}.csv')
            games.to_csv(filepath, index=False)
            print(f"Saved {len(games)} game records to {filepath}")
            
            return games
            
        except Exception as e:
            print(f"Error fetching games: {e}")
            return None
    
    def get_team_stats(self, season='2024-25', season_type='Regular Season'):
        """
        Get team statistics for a season
        
        Args:
            season: Season in format 'YYYY-YY'
            season_type: 'Regular Season' or 'Playoffs'
        """
        print(f"Fetching team stats for {season} {season_type}...")
        
        try:
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star=season_type,
                per_mode_detailed='PerGame'
            )
            time.sleep(self.delay)
            
            df = team_stats.get_data_frames()[0]
            
            filepath = os.path.join(self.data_dir, f'team_stats_{season}_{season_type.replace(" ", "_")}.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved team stats for {len(df)} teams to {filepath}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching team stats: {e}")
            return None
    
    def get_player_stats(self, season='2024-25', season_type='Regular Season'):
        """
        Get player statistics for a season
        
        Args:
            season: Season in format 'YYYY-YY'
            season_type: 'Regular Season' or 'Playoffs'
        """
        print(f"Fetching player stats for {season} {season_type}...")
        
        try:
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star=season_type,
                per_mode_detailed='PerGame'
            )
            time.sleep(self.delay)
            
            df = player_stats.get_data_frames()[0]
            
            filepath = os.path.join(self.data_dir, f'player_stats_{season}_{season_type.replace(" ", "_")}.csv')
            df.to_csv(filepath, index=False)
            print(f"Saved player stats for {len(df)} players to {filepath}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return None
    
    def get_historical_seasons(self, start_season='2018-19', end_season='2024-25'):
        """
        Collect data for multiple historical seasons
        
        Args:
            start_season: Starting season (e.g., '2018-19')
            end_season: Ending season (e.g., '2024-25')
        """
        # Parse seasons
        start_year = int(start_season.split('-')[0])
        end_year = int(end_season.split('-')[0])
        
        seasons = []
        for year in range(start_year, end_year + 1):
            next_year = str(year + 1)[-2:]
            season = f"{year}-{next_year}"
            seasons.append(season)
        
        print(f"Collecting data for {len(seasons)} seasons: {seasons}")
        
        all_games = []
        all_team_stats = []
        all_player_stats = []
        
        for season in seasons:
            print(f"\n{'='*60}")
            print(f"Processing season: {season}")
            print(f"{'='*60}")
            
            # Get games
            games = self.get_season_games(season)
            if games is not None:
                games['SEASON'] = season
                all_games.append(games)
            time.sleep(self.delay)
            
            # Get team stats
            team_stats = self.get_team_stats(season)
            if team_stats is not None:
                team_stats['SEASON'] = season
                all_team_stats.append(team_stats)
            time.sleep(self.delay)
            
            # Get player stats
            player_stats = self.get_player_stats(season)
            if player_stats is not None:
                player_stats['SEASON'] = season
                all_player_stats.append(player_stats)
            time.sleep(self.delay)
        
        # Combine all seasons
        if all_games:
            combined_games = pd.concat(all_games, ignore_index=True)
            filepath = os.path.join(self.data_dir, 'all_games_historical.csv')
            combined_games.to_csv(filepath, index=False)
            print(f"\nSaved {len(combined_games)} total game records to {filepath}")
        
        if all_team_stats:
            combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
            filepath = os.path.join(self.data_dir, 'all_team_stats_historical.csv')
            combined_team_stats.to_csv(filepath, index=False)
            print(f"Saved {len(combined_team_stats)} total team stat records to {filepath}")
        
        if all_player_stats:
            combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
            filepath = os.path.join(self.data_dir, 'all_player_stats_historical.csv')
            combined_player_stats.to_csv(filepath, index=False)
            print(f"Saved {len(combined_player_stats)} total player stat records to {filepath}")
        
        return {
            'games': combined_games if all_games else None,
            'team_stats': combined_team_stats if all_team_stats else None,
            'player_stats': combined_player_stats if all_player_stats else None
        }


def main():
    """Main execution function"""
    print("NBA Data Collection Script")
    print("=" * 60)
    
    collector = NBADataCollector()
    
    # Get teams and players
    print("\n1. Collecting teams and players...")
    teams_df = collector.get_all_teams()
    players_df = collector.get_all_players()
    
    # Get current season data
    print("\n2. Collecting current season data (2024-25)...")
    current_games = collector.get_season_games('2024-25')
    current_team_stats = collector.get_team_stats('2024-25')
    current_player_stats = collector.get_player_stats('2024-25')
    
    # Get historical data (last 6 seasons)
    print("\n3. Collecting historical data (2018-19 to 2024-25)...")
    historical_data = collector.get_historical_seasons('2018-19', '2024-25')
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    
    # Summary
    print("\nData Summary:")
    print(f"- Teams: {len(teams_df)}")
    print(f"- Players: {len(players_df)}")
    if historical_data['games'] is not None:
        print(f"- Historical Games: {len(historical_data['games'])}")
    if historical_data['team_stats'] is not None:
        print(f"- Historical Team Stats: {len(historical_data['team_stats'])}")
    if historical_data['player_stats'] is not None:
        print(f"- Historical Player Stats: {len(historical_data['player_stats'])}")


if __name__ == "__main__":
    main()

