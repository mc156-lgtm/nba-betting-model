# 🏀 Integrating Real NBA Data from Kaggle

## ✅ Best Kaggle NBA Datasets for Your Model

### 1. **NBA Database** (RECOMMENDED - Most Comprehensive)
**URL**: https://www.kaggle.com/datasets/wyattowalsh/basketball
- **Coverage**: 1946-present (updated daily!)
- **Size**: 731 MB
- **Contains**: 
  - 65,000+ games
  - 4,800+ players
  - 30 teams
  - Box scores, stats, betting odds
- **Perfect for**: All predictions (spreads, totals, moneyline, player props)

### 2. **NBA games data** (Good for Game Predictions)
**URL**: https://www.kaggle.com/datasets/nathanlauga/nba-games
- **Coverage**: 2004-2020
- **Contains**:
  - Game results
  - Team stats
  - Rankings
- **Perfect for**: Spread and totals predictions

### 3. **Historical NBA Data and Player Box Scores** (You Found This!)
**URL**: https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores
- **Coverage**: 1947-present
- **Contains**:
  - Player box scores
  - Team statistics
  - Historical data
- **Perfect for**: Player props and game predictions

### 4. **NBA Historical Stats and Betting Data** (BEST FOR BETTING)
**URL**: https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data
- **Coverage**: Historical games with betting odds
- **Contains**:
  - Match stats
  - **Actual betting odds** (spreads, totals, moneylines)
- **Perfect for**: Training models with real betting lines

---

## 🚀 Quick Start: Download and Use Real Data

### Step 1: Install Kaggle CLI

```bash
pip install kaggle
```

### Step 2: Get Kaggle API Token

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Move it to the right location:

**Linux/Mac**:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows**:
```bash
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### Step 3: Download Dataset

Choose one of these datasets:

#### Option A: NBA Database (Most Complete)
```bash
cd nba_betting_model/data/raw/
kaggle datasets download -d wyattowalsh/basketball
unzip basketball.zip
```

#### Option B: NBA Games Data (Simpler)
```bash
cd nba_betting_model/data/raw/
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip
```

#### Option C: Historical NBA + Box Scores (Your Find)
```bash
cd nba_betting_model/data/raw/
kaggle datasets download -d eoinamoore/historical-nba-data-and-player-box-scores
unzip historical-nba-data-and-player-box-scores.zip
```

#### Option D: NBA with Betting Data (Best for Your Use Case)
```bash
cd nba_betting_model/data/raw/
kaggle datasets download -d ehallmar/nba-historical-stats-and-betting-data
unzip nba-historical-stats-and-betting-data.zip
```

---

## 📊 Integration Script

I'll create a script to automatically process the Kaggle data and retrain your models.

### Step 4: Create Data Integration Script

Save this as `src/data_collection/integrate_kaggle_data.py`:

```python
"""
Integrate Kaggle NBA datasets into the betting model
"""

import pandas as pd
import os
from pathlib import Path

def load_kaggle_nba_database():
    """
    Load data from Wyatt Walsh's NBA Database
    Best for comprehensive historical data
    """
    data_dir = Path('data/raw/basketball/')
    
    # Load games
    games = pd.read_csv(data_dir / 'games.csv')
    
    # Load team stats
    team_stats = pd.read_csv(data_dir / 'team_stats.csv')
    
    # Load player stats
    player_stats = pd.read_csv(data_dir / 'player_stats.csv')
    
    print(f"✓ Loaded {len(games)} games")
    print(f"✓ Loaded {len(team_stats)} team stat records")
    print(f"✓ Loaded {len(player_stats)} player stat records")
    
    return games, team_stats, player_stats

def load_kaggle_nba_games():
    """
    Load data from nathanlauga's NBA games dataset
    Good for 2004-2020 game data
    """
    data_dir = Path('data/raw/nba-games/')
    
    games = pd.read_csv(data_dir / 'games.csv')
    teams = pd.read_csv(data_dir / 'teams.csv')
    
    print(f"✓ Loaded {len(games)} games from 2004-2020")
    print(f"✓ Loaded {len(teams)} teams")
    
    return games, teams

def load_kaggle_historical_boxscores():
    """
    Load data from eoinamoore's historical dataset
    Best for player box scores
    """
    data_dir = Path('data/raw/historical-nba/')
    
    # Check what files are available
    files = list(data_dir.glob('*.csv'))
    print(f"Found {len(files)} CSV files:")
    for f in files:
        print(f"  - {f.name}")
    
    # Load the main files
    games = pd.read_csv(data_dir / 'games.csv') if (data_dir / 'games.csv').exists() else None
    player_boxscores = pd.read_csv(data_dir / 'player_boxscores.csv') if (data_dir / 'player_boxscores.csv').exists() else None
    
    return games, player_boxscores

def process_for_spread_model(games_df):
    """
    Process Kaggle data for spread prediction model
    """
    # Ensure required columns exist
    required_cols = ['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'PTS_home', 'PTS_away']
    
    # Calculate point differential (spread)
    games_df['SPREAD'] = games_df['PTS_home'] - games_df['PTS_away']
    
    # Add home win indicator
    games_df['HOME_WIN'] = (games_df['SPREAD'] > 0).astype(int)
    
    return games_df

def process_for_totals_model(games_df):
    """
    Process Kaggle data for totals (over/under) prediction
    """
    # Calculate total points
    games_df['TOTAL_POINTS'] = games_df['PTS_home'] + games_df['PTS_away']
    
    return games_df

def process_for_player_props(player_stats_df):
    """
    Process Kaggle data for player props predictions
    """
    # Ensure we have the key stats
    required_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    
    # Calculate per-game averages if needed
    if 'GP' in player_stats_df.columns:
        for stat in required_stats:
            if stat in player_stats_df.columns:
                player_stats_df[f'{stat}_PER_GAME'] = player_stats_df[stat] / player_stats_df['GP']
    
    return player_stats_df

def main():
    """
    Main integration function
    """
    print("=" * 60)
    print("NBA BETTING MODEL - KAGGLE DATA INTEGRATION")
    print("=" * 60)
    
    # Check which dataset is available
    data_dir = Path('data/raw/')
    
    if (data_dir / 'basketball').exists():
        print("\n📊 Found: NBA Database (Wyatt Walsh)")
        games, team_stats, player_stats = load_kaggle_nba_database()
        
    elif (data_dir / 'nba-games').exists():
        print("\n📊 Found: NBA Games Data (nathanlauga)")
        games, teams = load_kaggle_nba_games()
        
    elif (data_dir / 'historical-nba').exists():
        print("\n📊 Found: Historical NBA Data (eoinamoore)")
        games, player_boxscores = load_kaggle_historical_boxscores()
        
    else:
        print("\n❌ No Kaggle dataset found!")
        print("Please download a dataset first:")
        print("  kaggle datasets download -d wyattowalsh/basketball")
        return
    
    # Process data
    print("\n🔧 Processing data for models...")
    
    if games is not None:
        games = process_for_spread_model(games)
        games = process_for_totals_model(games)
        
        # Save processed data
        output_path = Path('data/processed/kaggle_games_processed.csv')
        games.to_csv(output_path, index=False)
        print(f"✓ Saved processed games to {output_path}")
    
    print("\n✅ Data integration complete!")
    print("\nNext steps:")
    print("1. Update feature engineering: cd src/features && python build_features.py")
    print("2. Retrain models: cd src/models && python spread_model.py")
    print("3. Test predictions: python predict.py")

if __name__ == '__main__':
    main()
```

---

## 🔄 Retrain Models with Real Data

### Step 5: Update Feature Engineering

Edit `src/features/build_features.py` to use Kaggle data:

```python
# At the top of build_features.py, change:
# df = pd.read_csv('data/raw/all_games_historical.csv')

# To:
df = pd.read_csv('data/processed/kaggle_games_processed.csv')
```

### Step 6: Retrain All Models

```bash
# Process Kaggle data
cd src/data_collection
python integrate_kaggle_data.py

# Build features from real data
cd ../features
python build_features.py

# Retrain all models
cd ../models
python spread_model.py
python totals_model.py
python moneyline_model.py
python player_props_model.py

# Test predictions
python predict.py
```

---

## 📈 Expected Improvements

After switching from synthetic to real data:

| Model | Current (Synthetic) | Expected (Real Data) |
|-------|-------------------|---------------------|
| Spread | MAE: 2.37 | MAE: 8-12 points (realistic) |
| Totals | MAE: 12.46 | MAE: 10-15 points |
| Moneyline | Acc: 95% | Acc: 55-65% (realistic) |
| Player Props | MAE: 0.01 | MAE: 2-5 points |

**Note**: Real data will have "worse" metrics because NBA games are inherently unpredictable. The synthetic data was too perfect. Real performance of 55-65% accuracy on moneyline is actually **very good** and profitable!

---

## 🎯 Recommended Dataset

**For your betting model, I recommend:**

### Primary: NBA Historical Stats and Betting Data
- **Why**: Includes actual betting lines (spreads, totals, odds)
- **Benefit**: You can train models to beat the market, not just predict outcomes
- **Download**: `kaggle datasets download -d ehallmar/nba-historical-stats-and-betting-data`

### Secondary: NBA Database (Wyatt Walsh)
- **Why**: Most comprehensive, updated daily
- **Benefit**: Latest data for current season predictions
- **Download**: `kaggle datasets download -d wyattowalsh/basketball`

---

## 🚀 Quick Integration (Copy-Paste)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Download best dataset for betting
cd nba_betting_model/data/raw/
kaggle datasets download -d ehallmar/nba-historical-stats-and-betting-data
unzip nba-historical-stats-and-betting-data.zip

# 3. Integrate and retrain
cd ../../src/data_collection
python integrate_kaggle_data.py

cd ../features
python build_features.py

cd ../models
python spread_model.py
python totals_model.py
python moneyline_model.py
python player_props_model.py

# 4. Push updated models to GitHub
cd ../../
git add models/
git commit -m "Retrained models with real Kaggle NBA data"
git push origin master

# Your Streamlit app will auto-update!
```

---

## 📝 Summary

**You found great datasets!** Here's what to do:

1. ✅ **Download** any of the Kaggle NBA datasets (I recommend the one with betting data)
2. ✅ **Run** the integration script to process the data
3. ✅ **Retrain** your models with real NBA data
4. ✅ **Push** to GitHub - your Streamlit app auto-updates
5. ✅ **Enjoy** realistic predictions based on actual NBA games!

The synthetic data was just a placeholder. Real data will make your model actually useful for betting analysis!

---

**Need help with any step? Let me know which dataset you want to use and I'll create the exact integration code for it!**

