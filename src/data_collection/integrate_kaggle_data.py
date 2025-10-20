"""
Integrate Kaggle NBA datasets into the betting model

Supports multiple Kaggle NBA datasets:
1. NBA Database (wyattowalsh/basketball)
2. NBA Games Data (nathanlauga/nba-games)
3. Historical NBA Data (eoinamoore/historical-nba-data-and-player-box-scores)
4. NBA with Betting Data (ehallmar/nba-historical-stats-and-betting-data)
"""

import pandas as pd
import os
from pathlib import Path
import sys

def detect_dataset():
    """Detect which Kaggle dataset is available"""
    data_dir = Path('../../data/raw/')
    
    datasets = {
        'basketball': 'NBA Database (Wyatt Walsh)',
        'nba-games': 'NBA Games Data (nathanlauga)',
        'historical-nba': 'Historical NBA Data (eoinamoore)',
        'nba-betting': 'NBA Historical Stats and Betting Data (ehallmar)'
    }
    
    for folder, name in datasets.items():
        if (data_dir / folder).exists():
            return folder, name
    
    # Check for CSV files directly in raw/
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        return 'raw_csv', f'CSV files in data/raw/ ({len(csv_files)} files)'
    
    return None, None

def load_nba_database():
    """Load NBA Database (Wyatt Walsh)"""
    print("📊 Loading NBA Database...")
    data_dir = Path('../../data/raw/basketball/')
    
    # This dataset uses SQLite, but may also have CSV exports
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("❌ No CSV files found. This dataset may be in SQLite format.")
        print("   Please export tables to CSV or use sqlite3 to query.")
        return None, None, None
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files[:10]:  # Show first 10
        print(f"  - {f.name}")
    
    # Try to find games and stats files
    games = None
    team_stats = None
    player_stats = None
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, nrows=5)
        print(f"\n{csv_file.name} columns: {list(df.columns)[:10]}")
        
        if 'game' in csv_file.name.lower():
            games = pd.read_csv(csv_file)
            print(f"✓ Loaded games: {len(games)} records")
        elif 'team' in csv_file.name.lower():
            team_stats = pd.read_csv(csv_file)
            print(f"✓ Loaded team stats: {len(team_stats)} records")
        elif 'player' in csv_file.name.lower():
            player_stats = pd.read_csv(csv_file)
            print(f"✓ Loaded player stats: {len(player_stats)} records")
    
    return games, team_stats, player_stats

def load_nba_games():
    """Load NBA Games Data (nathanlauga)"""
    print("📊 Loading NBA Games Data...")
    data_dir = Path('../../data/raw/nba-games/')
    
    games = pd.read_csv(data_dir / 'games.csv')
    teams = pd.read_csv(data_dir / 'teams.csv')
    
    print(f"✓ Loaded {len(games)} games (2004-2020)")
    print(f"✓ Loaded {len(teams)} teams")
    print(f"\nGame columns: {list(games.columns)}")
    
    return games, teams

def load_historical_nba():
    """Load Historical NBA Data (eoinamoore)"""
    print("📊 Loading Historical NBA Data...")
    data_dir = Path('../../data/raw/historical-nba/')
    
    csv_files = list(data_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    games = None
    player_boxscores = None
    
    for csv_file in csv_files:
        if 'game' in csv_file.name.lower():
            games = pd.read_csv(csv_file)
            print(f"✓ Loaded games: {len(games)} records")
        elif 'player' in csv_file.name.lower() or 'box' in csv_file.name.lower():
            player_boxscores = pd.read_csv(csv_file)
            print(f"✓ Loaded player boxscores: {len(player_boxscores)} records")
    
    return games, player_boxscores

def load_raw_csv():
    """Load CSV files directly from data/raw/"""
    print("📊 Loading CSV files from data/raw/...")
    data_dir = Path('../../data/raw/')
    
    csv_files = list(data_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files[:5]:
        print(f"\n{csv_file.name}:")
        df = pd.read_csv(csv_file, nrows=3)
        print(f"  Columns: {list(df.columns)[:10]}")
        print(f"  Rows: {len(pd.read_csv(csv_file))}")
    
    # Try to find the most relevant file
    for csv_file in csv_files:
        if 'game' in csv_file.name.lower():
            games = pd.read_csv(csv_file)
            print(f"\n✓ Using {csv_file.name} as games data")
            return games, None
    
    return None, None

def process_for_models(games_df):
    """Process games data for all models"""
    if games_df is None:
        return None
    
    print("\n🔧 Processing data for models...")
    
    # Detect column names (different datasets use different naming)
    col_mapping = {}
    
    for col in games_df.columns:
        col_lower = col.lower()
        if 'home' in col_lower and 'pts' in col_lower or 'home' in col_lower and 'score' in col_lower:
            col_mapping['PTS_home'] = col
        elif 'away' in col_lower and 'pts' in col_lower or 'visitor' in col_lower and 'pts' in col_lower:
            col_mapping['PTS_away'] = col
        elif 'away' in col_lower and 'score' in col_lower or 'visitor' in col_lower and 'score' in col_lower:
            col_mapping['PTS_away'] = col
    
    print(f"Detected columns: {col_mapping}")
    
    # Rename columns to standard format
    if col_mapping:
        games_df = games_df.rename(columns={v: k for k, v in col_mapping.items()})
    
    # Calculate spread
    if 'PTS_home' in games_df.columns and 'PTS_away' in games_df.columns:
        games_df['SPREAD'] = games_df['PTS_home'] - games_df['PTS_away']
        games_df['TOTAL_POINTS'] = games_df['PTS_home'] + games_df['PTS_away']
        games_df['HOME_WIN'] = (games_df['SPREAD'] > 0).astype(int)
        
        print(f"✓ Calculated SPREAD (mean: {games_df['SPREAD'].mean():.2f})")
        print(f"✓ Calculated TOTAL_POINTS (mean: {games_df['TOTAL_POINTS'].mean():.2f})")
        print(f"✓ Calculated HOME_WIN (home win rate: {games_df['HOME_WIN'].mean():.2%})")
    else:
        print("⚠️  Could not find PTS columns. Please check dataset structure.")
        print(f"Available columns: {list(games_df.columns)}")
    
    return games_df

def main():
    """Main integration function"""
    print("=" * 70)
    print("NBA BETTING MODEL - KAGGLE DATA INTEGRATION")
    print("=" * 70)
    
    # Detect dataset
    dataset_type, dataset_name = detect_dataset()
    
    if dataset_type is None:
        print("\n❌ No Kaggle dataset found in data/raw/!")
        print("\nPlease download a dataset first:")
        print("  Option 1: kaggle datasets download -d wyattowalsh/basketball")
        print("  Option 2: kaggle datasets download -d nathanlauga/nba-games")
        print("  Option 3: kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores")
        print("  Option 4: kaggle datasets download -d ehallmar/nba-historical-stats-and-betting-data")
        print("\nThen unzip into data/raw/")
        return
    
    print(f"\n✓ Found dataset: {dataset_name}")
    
    # Load data based on type
    games = None
    
    if dataset_type == 'basketball':
        games, team_stats, player_stats = load_nba_database()
    elif dataset_type == 'nba-games':
        games, teams = load_nba_games()
    elif dataset_type == 'historical-nba':
        games, player_boxscores = load_historical_nba()
    elif dataset_type == 'raw_csv':
        games, _ = load_raw_csv()
    
    if games is None:
        print("\n❌ Could not load games data.")
        print("Please check the dataset structure and file names.")
        return
    
    # Process data
    games = process_for_models(games)
    
    if games is not None:
        # Save processed data
        output_dir = Path('../../data/processed/')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'kaggle_games_processed.csv'
        games.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved processed data to {output_path}")
        print(f"   Total games: {len(games)}")
        
        # Show sample statistics
        if 'SPREAD' in games.columns:
            print(f"\n📊 Data Statistics:")
            print(f"   Average spread: {games['SPREAD'].mean():.2f} points")
            print(f"   Average total: {games['TOTAL_POINTS'].mean():.2f} points")
            print(f"   Home win rate: {games['HOME_WIN'].mean():.2%}")
            print(f"   Date range: {games['GAME_DATE'].min() if 'GAME_DATE' in games.columns else 'N/A'} to {games['GAME_DATE'].max() if 'GAME_DATE' in games.columns else 'N/A'}")
        
        print("\n" + "=" * 70)
        print("✅ DATA INTEGRATION COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Update build_features.py to use: data/processed/kaggle_games_processed.csv")
        print("2. Run: cd ../features && python build_features.py")
        print("3. Retrain models: cd ../models && python spread_model.py")
        print("4. Test predictions: python predict.py")
        print("5. Push to GitHub: git add . && git commit -m 'Updated with real data' && git push")

if __name__ == '__main__':
    main()

