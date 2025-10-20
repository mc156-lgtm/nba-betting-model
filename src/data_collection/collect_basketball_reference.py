"""
Basketball Reference Data Collection

This script collects real NBA data from Basketball Reference using web scraping.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from basketball_reference_scraper.seasons import get_schedule, get_standings
from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats
from basketball_reference_scraper.players import get_stats
from basketball_reference_scraper.box_scores import get_box_scores
import warnings
warnings.filterwarnings('ignore')


class BasketballReferenceCollector:
    """Collect NBA data from Basketball Reference"""
    
    def __init__(self, data_dir='../../data/raw'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Team abbreviation mapping
        self.team_abbrevs = [
            'ATL', 'BOS', 'BRK', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
            'OKC', 'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS'
        ]
    
    def collect_season_schedule(self, season):
        """
        Collect full season schedule with game results
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nCollecting {season-1}-{str(season)[-2:]} season schedule...")
        
        try:
            schedule_df = get_schedule(season)
            
            if schedule_df is not None and not schedule_df.empty:
                # Clean and process the schedule
                schedule_df['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                
                # Save to CSV
                filepath = os.path.join(self.data_dir, f'schedule_{season-1}-{str(season)[-2:]}.csv')
                schedule_df.to_csv(filepath, index=False)
                print(f"  Saved {len(schedule_df)} games to {filepath}")
                
                return schedule_df
            else:
                print(f"  No schedule data available for {season}")
                return None
                
        except Exception as e:
            print(f"  Error collecting schedule: {e}")
            return None
    
    def collect_team_stats(self, season):
        """
        Collect team statistics for a season
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nCollecting {season-1}-{str(season)[-2:]} team statistics...")
        
        all_team_stats = []
        
        for team in self.team_abbrevs:
            try:
                print(f"  Collecting {team}...", end=' ')
                
                # Get team stats
                team_stats = get_team_stats(team, season)
                
                if team_stats is not None and not team_stats.empty:
                    team_stats['TEAM'] = team
                    team_stats['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                    all_team_stats.append(team_stats)
                    print("✓")
                else:
                    print("No data")
                
                # Rate limiting
                time.sleep(3)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if all_team_stats:
            combined_df = pd.concat(all_team_stats, ignore_index=True)
            filepath = os.path.join(self.data_dir, f'team_stats_{season-1}-{str(season)[-2:]}.csv')
            combined_df.to_csv(filepath, index=False)
            print(f"\n  Saved stats for {len(all_team_stats)} teams to {filepath}")
            return combined_df
        else:
            print("\n  No team stats collected")
            return None
    
    def collect_player_stats(self, season):
        """
        Collect player statistics for a season
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nCollecting {season-1}-{str(season)[-2:]} player statistics...")
        
        all_player_stats = []
        
        for team in self.team_abbrevs:
            try:
                print(f"  Collecting {team} roster...", end=' ')
                
                # Get roster stats (per game averages)
                roster_stats = get_roster_stats(team, season)
                
                if roster_stats is not None and not roster_stats.empty:
                    roster_stats['TEAM'] = team
                    roster_stats['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                    all_player_stats.append(roster_stats)
                    print(f"✓ ({len(roster_stats)} players)")
                else:
                    print("No data")
                
                # Rate limiting
                time.sleep(3)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        if all_player_stats:
            combined_df = pd.concat(all_player_stats, ignore_index=True)
            filepath = os.path.join(self.data_dir, f'player_stats_{season-1}-{str(season)[-2:]}.csv')
            combined_df.to_csv(filepath, index=False)
            print(f"\n  Saved stats for {len(combined_df)} players to {filepath}")
            return combined_df
        else:
            print("\n  No player stats collected")
            return None
    
    def collect_standings(self, season):
        """
        Collect season standings
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nCollecting {season-1}-{str(season)[-2:]} standings...")
        
        try:
            standings = get_standings(season)
            
            if standings is not None:
                filepath = os.path.join(self.data_dir, f'standings_{season-1}-{str(season)[-2:]}.csv')
                standings.to_csv(filepath, index=False)
                print(f"  Saved standings to {filepath}")
                return standings
            else:
                print("  No standings data available")
                return None
                
        except Exception as e:
            print(f"  Error collecting standings: {e}")
            return None
    
    def collect_full_season(self, season):
        """
        Collect all data for a complete season
        
        Args:
            season: Season year (e.g., 2024 for 2023-24 season)
        """
        print(f"\n{'='*60}")
        print(f"Collecting Full Season Data: {season-1}-{str(season)[-2:]}")
        print(f"{'='*60}")
        
        # Collect all data types
        schedule = self.collect_season_schedule(season)
        standings = self.collect_standings(season)
        team_stats = self.collect_team_stats(season)
        player_stats = self.collect_player_stats(season)
        
        print(f"\n{'='*60}")
        print(f"Season {season-1}-{str(season)[-2:]} collection complete!")
        print(f"{'='*60}")
        
        return {
            'schedule': schedule,
            'standings': standings,
            'team_stats': team_stats,
            'player_stats': player_stats
        }
    
    def collect_multiple_seasons(self, start_year, end_year):
        """
        Collect data for multiple seasons
        
        Args:
            start_year: Starting season year (e.g., 2021 for 2020-21)
            end_year: Ending season year (e.g., 2024 for 2023-24)
        """
        print(f"\n{'='*60}")
        print(f"Basketball Reference Data Collection")
        print(f"Seasons: {start_year-1}-{str(start_year)[-2:]} to {end_year-1}-{str(end_year)[-2:]}")
        print(f"{'='*60}")
        
        all_data = {}
        
        for year in range(start_year, end_year + 1):
            season_data = self.collect_full_season(year)
            all_data[year] = season_data
            
            # Longer pause between seasons
            if year < end_year:
                print("\nPausing before next season...")
                time.sleep(5)
        
        print(f"\n{'='*60}")
        print(f"All seasons collected successfully!")
        print(f"Data saved to: {self.data_dir}")
        print(f"{'='*60}")
        
        return all_data


def main():
    """Main execution function"""
    print("Basketball Reference Data Collector")
    print("=" * 60)
    
    collector = BasketballReferenceCollector()
    
    # Collect last 3 seasons (2021-22, 2022-23, 2023-24)
    # Note: 2024-25 season is currently in progress
    collector.collect_multiple_seasons(start_year=2022, end_year=2024)
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run feature engineering: cd ../features && python build_features.py")
    print("2. Train models: cd ../models && python spread_model.py")


if __name__ == "__main__":
    main()

