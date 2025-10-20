"""
Direct Basketball Reference Web Scraper

This script scrapes NBA data directly from Basketball Reference using BeautifulSoup and pandas.
More reliable than third-party libraries that may break with website updates.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
from datetime import datetime


class BasketballReferenceScraper:
    """Scrape NBA data directly from Basketball Reference"""
    
    def __init__(self, data_dir='../../data/raw'):
        self.data_dir = data_dir
        self.base_url = 'https://www.basketball-reference.com'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        os.makedirs(data_dir, exist_ok=True)
    
    def get_team_stats(self, season):
        """
        Scrape team statistics for a season
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nScraping team stats for {season-1}-{str(season)[-2:]} season...")
        
        url = f'{self.base_url}/leagues/NBA_{season}.html'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse with pandas - it can read HTML tables directly
            tables = pd.read_html(response.content)
            
            if len(tables) >= 2:
                # Team stats are typically in the first few tables
                team_stats = tables[0]  # Per game stats
                team_stats['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                
                # Clean column names
                if isinstance(team_stats.columns, pd.MultiIndex):
                    team_stats.columns = team_stats.columns.droplevel()
                
                # Save to CSV
                filepath = os.path.join(self.data_dir, f'bbref_team_stats_{season-1}-{str(season)[-2:]}.csv')
                team_stats.to_csv(filepath, index=False)
                print(f"  ✓ Saved {len(team_stats)} teams to {filepath}")
                
                return team_stats
            else:
                print(f"  ✗ Could not find team stats tables")
                return None
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def get_team_opponent_stats(self, season):
        """
        Scrape opponent statistics (defensive stats)
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nScraping opponent stats for {season-1}-{str(season)[-2:]} season...")
        
        url = f'{self.base_url}/leagues/NBA_{season}.html'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            tables = pd.read_html(response.content)
            
            # Opponent stats are usually the second table
            if len(tables) >= 4:
                opp_stats = tables[2]  # Opponent per game stats
                opp_stats['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                
                # Clean column names
                if isinstance(opp_stats.columns, pd.MultiIndex):
                    opp_stats.columns = opp_stats.columns.droplevel()
                
                filepath = os.path.join(self.data_dir, f'bbref_opp_stats_{season-1}-{str(season)[-2:]}.csv')
                opp_stats.to_csv(filepath, index=False)
                print(f"  ✓ Saved opponent stats for {len(opp_stats)} teams to {filepath}")
                
                return opp_stats
            else:
                print(f"  ✗ Could not find opponent stats tables")
                return None
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def get_schedule(self, season, month):
        """
        Scrape game schedule for a specific month
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
            month: Month name (e.g., 'october', 'november', etc.)
        """
        print(f"  Scraping {month.capitalize()} {season-1}-{str(season)[-2:]}...", end=' ')
        
        url = f'{self.base_url}/leagues/NBA_{season}_games-{month}.html'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            tables = pd.read_html(response.content)
            
            if tables:
                schedule = tables[0]
                schedule['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                schedule['MONTH'] = month.capitalize()
                print(f"✓ ({len(schedule)} games)")
                return schedule
            else:
                print("✗ No data")
                return None
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def get_full_schedule(self, season):
        """
        Scrape complete season schedule (all months)
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nScraping full schedule for {season-1}-{str(season)[-2:]} season...")
        
        # NBA season months
        months = ['october', 'november', 'december', 'january', 
                  'february', 'march', 'april', 'may', 'june']
        
        all_games = []
        
        for month in months:
            schedule = self.get_schedule(season, month)
            if schedule is not None and not schedule.empty:
                all_games.append(schedule)
            time.sleep(3)  # Rate limiting
        
        if all_games:
            combined_schedule = pd.concat(all_games, ignore_index=True)
            filepath = os.path.join(self.data_dir, f'bbref_schedule_{season-1}-{str(season)[-2:]}.csv')
            combined_schedule.to_csv(filepath, index=False)
            print(f"\n  ✓ Saved {len(combined_schedule)} total games to {filepath}")
            return combined_schedule
        else:
            print("\n  ✗ No schedule data collected")
            return None
    
    def get_standings(self, season):
        """
        Scrape season standings
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
        """
        print(f"\nScraping standings for {season-1}-{str(season)[-2:]} season...")
        
        url = f'{self.base_url}/leagues/NBA_{season}_standings.html'
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            tables = pd.read_html(response.content)
            
            if len(tables) >= 2:
                # Eastern and Western conference standings
                east_standings = tables[0]
                west_standings = tables[1]
                
                east_standings['CONFERENCE'] = 'East'
                west_standings['CONFERENCE'] = 'West'
                
                standings = pd.concat([east_standings, west_standings], ignore_index=True)
                standings['SEASON'] = f'{season-1}-{str(season)[-2:]}'
                
                filepath = os.path.join(self.data_dir, f'bbref_standings_{season-1}-{str(season)[-2:]}.csv')
                standings.to_csv(filepath, index=False)
                print(f"  ✓ Saved standings for {len(standings)} teams to {filepath}")
                
                return standings
            else:
                print(f"  ✗ Could not find standings tables")
                return None
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None
    
    def collect_season(self, season):
        """
        Collect all data for a complete season
        
        Args:
            season: Season ending year (e.g., 2024 for 2023-24 season)
        """
        print(f"\n{'='*60}")
        print(f"Collecting Basketball Reference Data: {season-1}-{str(season)[-2:]}")
        print(f"{'='*60}")
        
        # Collect all data types
        team_stats = self.get_team_stats(season)
        time.sleep(3)
        
        opp_stats = self.get_team_opponent_stats(season)
        time.sleep(3)
        
        standings = self.get_standings(season)
        time.sleep(3)
        
        schedule = self.get_full_schedule(season)
        
        print(f"\n{'='*60}")
        print(f"Season {season-1}-{str(season)[-2:]} collection complete!")
        print(f"{'='*60}")
        
        return {
            'team_stats': team_stats,
            'opp_stats': opp_stats,
            'standings': standings,
            'schedule': schedule
        }
    
    def collect_multiple_seasons(self, start_year, end_year):
        """
        Collect data for multiple seasons
        
        Args:
            start_year: Starting season year (e.g., 2022 for 2021-22)
            end_year: Ending season year (e.g., 2024 for 2023-24)
        """
        print(f"\n{'='*60}")
        print(f"Basketball Reference Web Scraper")
        print(f"Seasons: {start_year-1}-{str(start_year)[-2:]} to {end_year-1}-{str(end_year)[-2:]}")
        print(f"{'='*60}")
        
        all_data = {}
        
        for year in range(start_year, end_year + 1):
            season_data = self.collect_season(year)
            all_data[year] = season_data
            
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
    print("Basketball Reference Web Scraper")
    print("=" * 60)
    print("\nThis will scrape real NBA data from Basketball-Reference.com")
    print("Collecting 3 recent seasons: 2021-22, 2022-23, 2023-24")
    print("\nNote: This will take 5-10 minutes due to rate limiting.")
    print("=" * 60)
    
    scraper = BasketballReferenceScraper()
    
    # Collect last 3 complete seasons
    scraper.collect_multiple_seasons(start_year=2022, end_year=2024)
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review collected data in data/raw/")
    print("2. Update feature engineering to use Basketball Reference data")
    print("3. Retrain models with real data")


if __name__ == "__main__":
    main()

