"""
Feature Engineering for NBA Betting Model

This script processes the raw synthetic NBA data and engineers features for the predictive models.
"""

import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    """Creates features for the NBA betting model"""

    def __init__(self, raw_data_dir='../../data/raw', processed_data_dir='../../data/processed'):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        os.makedirs(processed_data_dir, exist_ok=True)

    def load_data(self):
        """Load all raw data files"""
        print("Loading raw data...")
        
        try:
            games_df = pd.read_csv(os.path.join(self.raw_data_dir, 'all_games_historical.csv'))
            team_stats_df = pd.read_csv(os.path.join(self.raw_data_dir, 'all_team_stats_historical.csv'))
            player_stats_df = pd.read_csv(os.path.join(self.raw_data_dir, 'all_player_stats_historical.csv'))
            teams_df = pd.read_csv(os.path.join(self.raw_data_dir, 'teams.csv'))
            
            print(f"  Loaded {len(games_df)} game records")
            print(f"  Loaded {len(team_stats_df)} team stat records")
            print(f"  Loaded {len(player_stats_df)} player stat records")
            
            return games_df, team_stats_df, player_stats_df, teams_df
        
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return None, None, None, None

    def create_team_features(self, games_df):
        """Create rolling averages and other team-level features"""
        print("\nCreating team features...")
        
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        games_df = games_df.sort_values(by=['TEAM_ID', 'GAME_DATE'])
        
        # Rolling averages for key stats
        rolling_windows = [5, 10, 20]
        stats_to_roll = ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
        
        for window in rolling_windows:
            print(f"  Calculating {window}-game rolling averages...")
            for stat in stats_to_roll:
                games_df[f'{stat}_roll_{window}'] = games_df.groupby('TEAM_ID')[stat].transform(
                    lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                )
        
        # Offensive and Defensive Efficiency (approximations)
        # Possessions = FGA - OREB + TOV + (0.44 * FTA)
        games_df['POSS'] = games_df['FGA'] - games_df['OREB'] + games_df['TOV'] + (0.44 * games_df['FTA'])
        games_df['OFF_EFF'] = 100 * (games_df['PTS'] / games_df['POSS'])
        
        # We need opponent stats to calculate defensive efficiency, which we will do in the matchup creation
        
        print("  Team features created.")
        return games_df

    def create_matchups(self, games_df):
        """Create game matchups by combining home and away team data"""
        print("\nCreating game matchups...")
        
        # Separate home and away games
        games_df['is_home'] = games_df['MATCHUP'].apply(lambda x: 'vs.' in x)
        home_games = games_df[games_df['is_home']].copy()
        away_games = games_df[~games_df['is_home']].copy()
        
        # Rename columns for merging
        home_cols = {col: f'HOME_{col}' for col in home_games.columns}
        away_cols = {col: f'AWAY_{col}' for col in away_games.columns}
        home_games.rename(columns=home_cols, inplace=True)
        away_games.rename(columns=away_cols, inplace=True)
        
        # Merge on game ID and date
        matchups_df = pd.merge(home_games, away_games, left_on=['HOME_GAME_ID', 'HOME_GAME_DATE'], right_on=['AWAY_GAME_ID', 'AWAY_GAME_DATE'], how='inner')
        
        # Calculate defensive efficiency
        matchups_df['HOME_DEF_EFF'] = 100 * (matchups_df['AWAY_PTS'] / matchups_df['HOME_POSS'])
        matchups_df['AWAY_DEF_EFF'] = 100 * (matchups_df['HOME_PTS'] / matchups_df['AWAY_POSS'])
        
        # Create feature differences
        for window in [5, 10, 20]:
            for stat in ['PTS', 'FG_PCT', 'FG3_PCT', 'REB', 'AST']:
                matchups_df[f'DIFF_{stat}_roll_{window}'] = matchups_df[f'HOME_{stat}_roll_{window}'] - matchups_df[f'AWAY_{stat}_roll_{window}']

        # Target variables
        matchups_df['TOTAL_SCORE'] = matchups_df['HOME_PTS'] + matchups_df['AWAY_PTS']
        matchups_df['SPREAD'] = matchups_df['AWAY_PTS'] - matchups_df['HOME_PTS'] # Positive spread means away team wins
        matchups_df['HOME_WIN'] = (matchups_df['HOME_PTS'] > matchups_df['AWAY_PTS']).astype(int)

        print(f"  Created {len(matchups_df)} matchups.")
        return matchups_df

    def run(self):
        """Execute the full feature engineering pipeline"""
        games_df, team_stats_df, player_stats_df, teams_df = self.load_data()
        
        if games_df is None:
            return

        # Team features
        team_featured_df = self.create_team_features(games_df)
        
        # Create matchups
        matchups_df = self.create_matchups(team_featured_df)
        
        # Save processed data
        output_path = os.path.join(self.processed_data_dir, 'nba_matchups_processed.csv')
        matchups_df.to_csv(output_path, index=False)
        print(f"\nProcessed matchup data saved to {output_path}")


def main():
    """Main execution function"""
    print("Feature Engineering Script")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    engineer.run()
    
    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

