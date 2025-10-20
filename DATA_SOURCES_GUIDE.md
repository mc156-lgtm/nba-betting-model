# NBA Data Sources Guide

Basketball Reference blocks automated scraping (403 Forbidden errors). Here are the best alternatives for getting real NBA data for your betting model.

## Option 1: NBA API (stats.nba.com) - RECOMMENDED

The official NBA Stats API is free and provides comprehensive data.

### Installation
```bash
pip install nba_api
```

### Usage Example
```python
from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs, playergamelogs
from nba_api.stats.static import teams
import pandas as pd
import time

# Get all teams
nba_teams = teams.get_teams()

# Get game logs for a season
gamefinder = leaguegamefinder.LeagueGameFinder(
    season_nullable='2023-24',
    league_id_nullable='00'
)
games = gamefinder.get_data_frames()[0]

# Get team game logs
team_gamelogs = teamgamelogs.TeamGameLogs(
    season_nullable='2023-24'
)
team_games = team_gamelogs.get_data_frames()[0]

# Get player game logs
player_gamelogs = playergamelogs.PlayerGameLogs(
    season_nullable='2023-24'
)
player_games = player_gamelogs.get_data_frames()[0]
```

### Tips for NBA API
- Add delays between requests (3-5 seconds) to avoid rate limiting
- Use try-except blocks for timeout handling
- Cache data locally to minimize API calls
- The API sometimes has connectivity issues - retry failed requests

## Option 2: Kaggle NBA Datasets - EASIEST

Pre-compiled NBA datasets are available on Kaggle with historical data already cleaned.

### Popular Datasets
1. **NBA Games Data (2004-2024)**
   - URL: https://www.kaggle.com/datasets/nathanlauga/nba-games
   - Contains: Game results, team stats, player stats
   - Size: ~50MB
   - Updated: Regularly

2. **NBA Player Stats (1950-2024)**
   - URL: https://www.kaggle.com/datasets/drgilermo/nba-players-stats
   - Contains: Career and season stats for all players
   - Size: ~5MB

3. **NBA Shot Logs**
   - URL: https://www.kaggle.com/datasets/dansbecker/nba-shot-logs
   - Contains: Shot-by-shot data with locations
   - Size: ~300MB

### How to Use Kaggle Data
```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip -d data/raw/
```

## Option 3: Basketball Reference CSV Export

While automated scraping is blocked, you can manually download CSV files.

### Manual Download Steps
1. Go to https://www.basketball-reference.com/leagues/NBA_2024.html
2. Find the table you want (Team Stats, Opponent Stats, etc.)
3. Click "Share & Export" → "Get table as CSV"
4. Copy the CSV data
5. Save to a file in `data/raw/`

### Tables to Download
- **Team Per Game Stats**: Team offensive performance
- **Opponent Per Game Stats**: Team defensive performance  
- **Team Totals**: Season aggregate statistics
- **Advanced Stats**: Efficiency ratings, pace, etc.
- **Standings**: Win-loss records

## Option 4: Sportradar API (Paid)

Official NBA data provider with the most comprehensive and reliable data.

### Features
- Real-time game data
- Play-by-play information
- Injury reports
- Betting odds
- Historical data back to 2010

### Pricing
- Trial: Free with limited requests
- Basic: $50-200/month
- Professional: $500+/month

### Website
https://developer.sportradar.com/

## Option 5: Pre-Downloaded Datasets (GitHub)

Several developers have shared NBA datasets on GitHub.

### Recommended Repositories
1. **NBA Enhanced Box Score**
   - https://github.com/swar/nba_api/tree/master/docs/examples
   - Examples and sample data

2. **NBA Database**
   - https://github.com/bttmly/nba
   - Historical data in JSON format

3. **NBA Stats Tracker**
   - https://github.com/jaebradley/basketball_reference_web_scraper
   - Pre-scraped data files

## Recommended Approach for Your Model

### Step 1: Use Kaggle for Historical Data
Download the Nathan Lauga NBA Games dataset for 2018-2024 seasons. This gives you a solid foundation of historical games for training.

```bash
# Download and extract
kaggle datasets download -d nathanlauga/nba-games
unzip nba-games.zip -d data/raw/kaggle/
```

### Step 2: Use NBA API for Recent/Current Data
Use the NBA API to fetch the current season and recent games that aren't in the Kaggle dataset.

```python
# Your existing collect_nba_data.py script works for this
# Just add better error handling and retries
```

### Step 3: Combine Data Sources
Merge Kaggle historical data with NBA API current data for a complete dataset.

```python
import pandas as pd

# Load Kaggle data
kaggle_games = pd.read_csv('data/raw/kaggle/games.csv')
kaggle_games['SOURCE'] = 'kaggle'

# Load NBA API data  
nba_api_games = pd.read_csv('data/raw/all_games_historical.csv')
nba_api_games['SOURCE'] = 'nba_api'

# Combine and remove duplicates
all_games = pd.concat([kaggle_games, nba_api_games])
all_games = all_games.drop_duplicates(subset=['GAME_ID'], keep='first')

# Save combined dataset
all_games.to_csv('data/raw/combined_games.csv', index=False)
```

## Data You Need for the Betting Model

### Game-Level Data
- Date, home team, away team
- Final scores
- Team statistics (FG%, 3P%, FT%, rebounds, assists, turnovers)
- Pace and possessions
- Home/away indicators

### Team-Level Data
- Season averages for all statistics
- Offensive and defensive ratings
- Win-loss records
- Recent form (last 5, 10, 20 games)

### Player-Level Data (for props)
- Game logs with points, rebounds, assists, etc.
- Minutes played
- Shooting percentages
- Usage rate

### Advanced Metrics (Optional)
- True shooting percentage
- Effective field goal percentage
- Assist-to-turnover ratio
- Plus/minus ratings

## Updated Data Collection Script

I've included an updated `collect_nba_data.py` that:
- Uses NBA API with better error handling
- Implements retry logic
- Adds delays to avoid rate limiting
- Saves data incrementally
- Provides progress updates

## Next Steps

1. **Choose your data source**: I recommend Kaggle + NBA API combination
2. **Download historical data**: Get 3-5 years of games from Kaggle
3. **Fetch recent data**: Use NBA API for current season
4. **Update feature engineering**: Modify `build_features.py` to work with your data format
5. **Retrain models**: Run the model training scripts with real data
6. **Validate performance**: Backtest on held-out season

## Important Notes

### Rate Limiting
All free APIs have rate limits. Always:
- Add delays between requests (3-5 seconds minimum)
- Implement exponential backoff for retries
- Cache data locally
- Don't make unnecessary duplicate requests

### Data Quality
- Check for missing values
- Verify data consistency across sources
- Handle postponed/cancelled games
- Account for trades and roster changes

### Legal Considerations
- Respect terms of service for all data sources
- Don't redistribute copyrighted data
- Use data for personal/educational purposes only
- Commercial use may require licensing

## Troubleshooting

### NBA API Timeouts
```python
import time
from requests.exceptions import Timeout

def fetch_with_retry(fetch_function, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fetch_function()
        except Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Timeout. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### Basketball Reference 403 Errors
- Use Kaggle datasets instead
- Manual CSV export for specific tables
- Consider paid Sportradar API for commercial use

### Data Format Mismatches
- Standardize column names across sources
- Convert dates to consistent format
- Map team abbreviations (BRK vs BKN, etc.)
- Handle missing or null values

## Summary

**Best Option**: Kaggle datasets + NBA API
- **Pros**: Free, comprehensive, regularly updated
- **Cons**: Requires combining multiple sources

**Easiest Option**: Kaggle only
- **Pros**: Pre-cleaned, ready to use, no API calls
- **Cons**: May not have latest games

**Most Reliable**: Sportradar API (paid)
- **Pros**: Official data, real-time, comprehensive
- **Cons**: Expensive for personal use

For your betting model, I recommend starting with Kaggle data to get the model working, then adding NBA API for current season updates.

