"""
Improved NBA API Data Collector

Enhanced version with better error handling, retry logic, and progress tracking.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs, playergamelogs
from nba_api.stats.static import teams, players
import warnings
warnings.filterwarnings('ignore')


class ImprovedNBACollector:
    """Collect NBA data with robust error handling"""
    
    def __init__(self, data_dir='../../data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.request_delay = 3  # seconds between requests
        self.max_retries = 3
    
    def fetch_with_retry(self, fetch_function, description="data"):
        """
        Fetch data with retry logic
        
        Args:
            fetch_function: Function to call for data fetching
            description: Description of what's being fetched
        """
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempting to fetch {description}... (attempt {attempt + 1}/{self.max_retries})")
                result = fetch_function()
                print(f"  ✓ Successfully fetched {description}")
                return result
            except Exception as e:
                error_msg = str(e)
                print(f"  ✗ Error: {error_msg}")
                
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed to fetch {description} after {self.max_retries} attempts")
                    return None
    
    def collect_teams(self):
        """Collect all NBA teams"""
        print("\nCollecting NBA teams...")
        
        def fetch():
            return teams.get_teams()
        
        teams_data = self.fetch_with_retry(fetch, "teams")
        
        if teams_data:
            teams_df = pd.DataFrame(teams_data)
            filepath = os.path.join(self.data_dir, 'nba_teams.csv')
            teams_df.to_csv(filepath, index=False)
            print(f"  Saved {len(teams_df)} teams to {filepath}")
            return teams_df
        return None
    
    def collect_season_games(self, season):
        """
        Collect all games for a season
        
        Args:
            season: Season string (e.g., '2023-24')
        """
        print(f"\nCollecting games for {season} season...")
        
        def fetch():
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable='00',
                timeout=30
            )
            return gamefinder.get_data_frames()[0]
        
        games_df = self.fetch_with_retry(fetch, f"{season} games")
        
        if games_df is not None and not games_df.empty:
            games_df['SEASON'] = season
            filepath = os.path.join(self.data_dir, f'nba_games_{season}.csv')
            games_df.to_csv(filepath, index=False)
            print(f"  Saved {len(games_df)} game records to {filepath}")
            return games_df
        return None
    
    def collect_team_gamelogs(self, season):
        """
        Collect team game logs for a season
        
        Args:
            season: Season string (e.g., '2023-24')
        """
        print(f"\nCollecting team game logs for {season} season...")
        
        def fetch():
            gamelogs = teamgamelogs.TeamGameLogs(
                season_nullable=season,
                timeout=30
            )
            return gamelogs.get_data_frames()[0]
        
        gamelogs_df = self.fetch_with_retry(fetch, f"{season} team game logs")
        
        if gamelogs_df is not None and not gamelogs_df.empty:
            gamelogs_df['SEASON'] = season
            filepath = os.path.join(self.data_dir, f'nba_team_gamelogs_{season}.csv')
            gamelogs_df.to_csv(filepath, index=False)
            print(f"  Saved {len(gamelogs_df)} team game logs to {filepath}")
            return gamelogs_df
        return None
    
    def collect_player_gamelogs(self, season):
        """
        Collect player game logs for a season
        
        Args:
            season: Season string (e.g., '2023-24')
        """
        print(f"\nCollecting player game logs for {season} season...")
        
        def fetch():
            gamelogs = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                timeout=30
            )
            return gamelogs.get_data_frames()[0]
        
        gamelogs_df = self.fetch_with_retry(fetch, f"{season} player game logs")
        
        if gamelogs_df is not None and not gamelogs_df.empty:
            gamelogs_df['SEASON'] = season
            filepath = os.path.join(self.data_dir, f'nba_player_gamelogs_{season}.csv')
            gamelogs_df.to_csv(filepath, index=False)
            print(f"  Saved {len(gamelogs_df)} player game logs to {filepath}")
            return gamelogs_df
        return None
    
    def collect_full_season(self, season):
        """
        Collect all data for a complete season
        
        Args:
            season: Season string (e.g., '2023-24')
        """
        print(f"\n{'='*60}")
        print(f"Collecting NBA Data: {season} Season")
        print(f"{'='*60}")
        
        results = {}
        
        # Collect games
        games = self.collect_season_games(season)
        results['games'] = games
        time.sleep(self.request_delay)
        
        # Collect team game logs
        team_logs = self.collect_team_gamelogs(season)
        results['team_logs'] = team_logs
        time.sleep(self.request_delay)
        
        # Collect player game logs
        player_logs = self.collect_player_gamelogs(season)
        results['player_logs'] = player_logs
        
        print(f"\n{'='*60}")
        print(f"Season {season} collection complete!")
        print(f"{'='*60}")
        
        return results
    
    def collect_multiple_seasons(self, seasons):
        """
        Collect data for multiple seasons
        
        Args:
            seasons: List of season strings (e.g., ['2021-22', '2022-23', '2023-24'])
        """
        print(f"\n{'='*60}")
        print(f"NBA API Data Collector (Improved)")
        print(f"Collecting {len(seasons)} seasons")
        print(f"{'='*60}")
        
        # Collect teams first (only once)
        teams_df = self.collect_teams()
        time.sleep(self.request_delay)
        
        all_results = {}
        
        for season in seasons:
            results = self.collect_full_season(season)
            all_results[season] = results
            
            # Longer pause between seasons
            if season != seasons[-1]:
                print(f"\nPausing before next season...")
                time.sleep(5)
        
        print(f"\n{'='*60}")
        print(f"All seasons collected!")
        print(f"Data saved to: {self.data_dir}")
        print(f"{'='*60}")
        
        return all_results


def main():
    """Main execution function"""
    print("NBA API Data Collector (Improved)")
    print("=" * 60)
    print("\nThis collector includes:")
    print("- Retry logic for failed requests")
    print("- Rate limiting to avoid timeouts")
    print("- Progress tracking")
    print("- Error handling")
    print("\n" + "=" * 60)
    
    collector = ImprovedNBACollector()
    
    # Collect last 3 complete seasons
    seasons = ['2021-22', '2022-23', '2023-24']
    
    print(f"\nCollecting seasons: {', '.join(seasons)}")
    print("This will take approximately 5-10 minutes...")
    print("=" * 60)
    
    results = collector.collect_multiple_seasons(seasons)
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nCollected files:")
    for season in seasons:
        print(f"  - nba_games_{season}.csv")
        print(f"  - nba_team_gamelogs_{season}.csv")
        print(f"  - nba_player_gamelogs_{season}.csv")
    
    print("\nNext steps:")
    print("1. Review collected data in data/raw/")
    print("2. Update feature engineering: cd ../features && python build_features.py")
    print("3. Retrain models: cd ../models && python spread_model.py")


if __name__ == "__main__":
    main()

