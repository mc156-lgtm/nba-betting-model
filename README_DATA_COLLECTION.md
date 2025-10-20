# Data Collection for NBA Betting Model

## Quick Start: Get Real NBA Data

Basketball Reference blocks automated scraping, but there are better alternatives:

### Option 1: NBA API (Recommended)

Use the improved NBA API collector with retry logic:

```bash
cd src/data_collection
python collect_nba_api_improved.py
```

This will collect:
- All games for 2021-22, 2022-23, 2023-24 seasons
- Team game logs with detailed statistics
- Player game logs for props modeling

**Time**: 5-10 minutes  
**Success Rate**: ~80% (handles timeouts automatically)

### Option 2: Kaggle Datasets (Easiest)

Download pre-compiled NBA datasets:

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API key from kaggle.com/account
mkdir -p ~/.kaggle
# Place your kaggle.json in ~/.kaggle/

# Download NBA games dataset
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip -d data/raw/kaggle/
```

**Time**: 2 minutes  
**Success Rate**: 100%  
**Data**: 2004-2024 seasons

### Option 3: Combine Both (Best Results)

Use Kaggle for historical data + NBA API for current season:

```bash
# Get historical data from Kaggle
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip -d data/raw/kaggle/

# Get current season from NBA API
cd src/data_collection
python collect_nba_api_improved.py
```

## Data Collection Scripts

### 1. `collect_nba_api_improved.py` ✅ WORKS
- **Status**: Working with retry logic
- **Source**: NBA Stats API (stats.nba.com)
- **Data**: Games, team logs, player logs
- **Pros**: Free, official data, comprehensive
- **Cons**: Occasional timeouts (handled automatically)

### 2. `generate_synthetic_data.py` ✅ WORKS
- **Status**: Working (for testing only)
- **Source**: Synthetic/simulated data
- **Data**: Realistic NBA statistics
- **Pros**: Always available, fast
- **Cons**: Not real data, for demo only

### 3. `collect_basketball_reference.py` ❌ BLOCKED
- **Status**: Blocked by Basketball Reference (403 errors)
- **Source**: Basketball-Reference.com
- **Data**: Would have comprehensive stats
- **Issue**: Website blocks automated scraping

### 4. `scrape_basketball_reference.py` ❌ BLOCKED
- **Status**: Also blocked (403 errors)
- **Source**: Direct web scraping
- **Issue**: Same blocking as above

## Recommended Workflow

### For Development/Testing
```bash
# Use synthetic data to test the model pipeline
cd src/data_collection
python generate_synthetic_data.py

cd ../features
python build_features.py

cd ../models
python spread_model.py
python totals_model.py
python moneyline_model.py
```

### For Real Predictions
```bash
# Get real data from Kaggle
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip -d data/raw/kaggle/

# OR use NBA API
cd src/data_collection
python collect_nba_api_improved.py

# Then process and train
cd ../features
python build_features.py

cd ../models
python spread_model.py
python totals_model.py
python moneyline_model.py
```

## Data Format Requirements

Your data needs these columns for the feature engineering pipeline:

### Game Data
- `GAME_ID`: Unique game identifier
- `GAME_DATE`: Date of game
- `TEAM_ID`: Team identifier
- `TEAM_ABBREVIATION`: Team abbrev (e.g., 'LAL', 'GSW')
- `MATCHUP`: Home/away indicator (e.g., 'LAL vs. GSW' or 'LAL @ GSW')
- `WL`: Win/Loss result
- `PTS`: Points scored
- `FG_PCT`, `FG3_PCT`, `FT_PCT`: Shooting percentages
- `REB`, `AST`, `STL`, `BLK`, `TOV`: Box score stats

### Team Stats
- Season averages for all major statistics
- Wins, losses, win percentage
- Offensive and defensive ratings

### Player Stats (for props)
- `PLAYER_ID`, `PLAYER_NAME`
- `GP`: Games played
- `MIN`: Minutes per game
- `PTS`, `REB`, `AST`, `STL`, `BLK`: Per-game averages

## Troubleshooting

### NBA API Timeouts
The improved collector handles this automatically with retries. If it still fails:
- Check your internet connection
- Try again later (NBA API has occasional outages)
- Use Kaggle data as backup

### Kaggle API Not Working
```bash
# Verify kaggle.json is in the right place
ls -la ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json

# Test connection
kaggle datasets list
```

### Basketball Reference 403 Errors
This is expected - the website blocks scrapers. Use NBA API or Kaggle instead.

### Data Format Mismatches
If combining multiple sources:
```python
# Standardize column names
df.rename(columns={'old_name': 'new_name'}, inplace=True)

# Standardize team abbreviations
team_mapping = {'BRK': 'BKN', 'PHX': 'PHO'}  # etc
df['TEAM'] = df['TEAM'].replace(team_mapping)

# Convert dates
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
```

## Data Storage

All collected data goes to `data/raw/`:
```
data/raw/
├── nba_teams.csv                      # Team information
├── nba_games_2023-24.csv              # Game results
├── nba_team_gamelogs_2023-24.csv      # Team game logs
├── nba_player_gamelogs_2023-24.csv    # Player game logs
└── kaggle/                            # Kaggle datasets
    ├── games.csv
    ├── teams.csv
    └── players.csv
```

## Next Steps After Data Collection

1. **Verify Data Quality**
```bash
cd data/raw
wc -l *.csv  # Check row counts
head -5 nba_games_2023-24.csv  # Preview data
```

2. **Run Feature Engineering**
```bash
cd src/features
python build_features.py
```

3. **Train Models**
```bash
cd src/models
python spread_model.py
python totals_model.py
python moneyline_model.py
python player_props_model.py
```

4. **Make Predictions**
```bash
python predict.py
```

## Additional Resources

- **NBA API Documentation**: https://github.com/swar/nba_api
- **Kaggle NBA Datasets**: https://www.kaggle.com/search?q=nba+in%3Adatasets
- **Data Sources Guide**: See `DATA_SOURCES_GUIDE.md` for detailed comparison

## Summary

| Method | Speed | Reliability | Data Quality | Recommendation |
|--------|-------|-------------|--------------|----------------|
| NBA API (Improved) | Medium | Good | Excellent | ⭐ Recommended |
| Kaggle Datasets | Fast | Excellent | Very Good | ⭐ Easiest |
| Synthetic Data | Very Fast | Perfect | N/A (Fake) | Testing Only |
| Basketball Reference | N/A | Blocked | N/A | ❌ Don't Use |

**Best Approach**: Use Kaggle for historical data (2004-2023) + NBA API for current season (2023-24).

