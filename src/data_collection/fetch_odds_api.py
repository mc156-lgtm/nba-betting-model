"""
Fetch NBA Odds from The Odds API
Optimized for credit efficiency - FanDuel and DraftKings only
"""

import requests
import json
import os
from datetime import datetime, timedelta
import pandas as pd

# API Configuration
ODDS_API_KEY = "d082b1e452e4604434d17c71edc92255"
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"

# Sport and markets
SPORT = "basketball_nba"
REGIONS = "us"  # US bookmakers only
MARKETS = "h2h,spreads,totals"  # Moneyline, Spreads, Totals
BOOKMAKERS = "fanduel,draftkings,pinnacle"  # FanDuel, DraftKings, and Pinnacle (sharp)

# Cache settings
CACHE_DIR = "data/odds_cache"
CACHE_DURATION_HOURS = 6  # Cache for 6 hours

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_filename():
    """Get cache filename for today's odds"""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CACHE_DIR, f"nba_odds_{today}.json")

def is_cache_valid():
    """Check if cache exists and is still valid"""
    cache_file = get_cache_filename()
    
    if not os.path.exists(cache_file):
        return False
    
    # Check if cache is older than CACHE_DURATION_HOURS
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    age = datetime.now() - cache_time
    
    if age > timedelta(hours=CACHE_DURATION_HOURS):
        return False
    
    return True

def load_from_cache():
    """Load odds from cache"""
    cache_file = get_cache_filename()
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded odds from cache ({cache_file})")
        return data
    except Exception as e:
        print(f"⚠️ Error loading cache: {e}")
        return None

def save_to_cache(data):
    """Save odds to cache"""
    ensure_cache_dir()
    cache_file = get_cache_filename()
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved odds to cache ({cache_file})")
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")

def fetch_nba_odds(use_cache=True):
    """
    Fetch NBA odds from The Odds API
    
    Args:
        use_cache: If True, use cached data if available
    
    Returns:
        dict with odds data
    """
    # Check cache first
    if use_cache and is_cache_valid():
        cached_data = load_from_cache()
        if cached_data:
            return cached_data
    
    # Fetch from API
    print("📡 Fetching NBA odds from The Odds API...")
    print(f"   Bookmakers: {BOOKMAKERS}")
    print(f"   Markets: {MARKETS}")
    
    url = f"{ODDS_API_BASE_URL}/sports/{SPORT}/odds/"
    
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'bookmakers': BOOKMAKERS,
        'oddsFormat': 'american',  # American odds format (-110, +150, etc.)
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check remaining credits
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        print(f"✅ Fetched odds for {len(data)} games")
        print(f"   Credits used: {used}")
        print(f"   Credits remaining: {remaining}")
        
        # Save to cache
        save_to_cache(data)
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching odds: {e}")
        return None

def parse_odds_data(odds_data):
    """
    Parse odds data into a clean DataFrame
    
    Args:
        odds_data: Raw odds data from API
    
    Returns:
        DataFrame with parsed odds
    """
    if not odds_data:
        return pd.DataFrame()
    
    games = []
    
    for game in odds_data:
        game_id = game['id']
        commence_time = game['commence_time']
        home_team = game['home_team']
        away_team = game['away_team']
        
        # Initialize game dict
        game_dict = {
            'game_id': game_id,
            'commence_time': commence_time,
            'home_team': home_team,
            'away_team': away_team
        }
        
        # Parse bookmaker odds
        for bookmaker in game.get('bookmakers', []):
            bookie_name = bookmaker['key']  # 'fanduel' or 'draftkings'
            
            for market in bookmaker.get('markets', []):
                market_key = market['key']  # 'h2h', 'spreads', 'totals'
                
                if market_key == 'h2h':  # Moneyline
                    for outcome in market['outcomes']:
                        team = outcome['name']
                        odds = outcome['price']
                        
                        if team == home_team:
                            game_dict[f'{bookie_name}_ml_home'] = odds
                        else:
                            game_dict[f'{bookie_name}_ml_away'] = odds
                
                elif market_key == 'spreads':  # Spreads
                    for outcome in market['outcomes']:
                        team = outcome['name']
                        point = outcome['point']
                        odds = outcome['price']
                        
                        if team == home_team:
                            game_dict[f'{bookie_name}_spread_home'] = point
                            game_dict[f'{bookie_name}_spread_home_odds'] = odds
                        else:
                            game_dict[f'{bookie_name}_spread_away'] = point
                            game_dict[f'{bookie_name}_spread_away_odds'] = odds
                
                elif market_key == 'totals':  # Totals (Over/Under)
                    for outcome in market['outcomes']:
                        over_under = outcome['name']  # 'Over' or 'Under'
                        point = outcome['point']
                        odds = outcome['price']
                        
                        if over_under == 'Over':
                            game_dict[f'{bookie_name}_total_line'] = point
                            game_dict[f'{bookie_name}_total_over_odds'] = odds
                        else:
                            game_dict[f'{bookie_name}_total_under_odds'] = odds
        
        games.append(game_dict)
    
    df = pd.DataFrame(games)
    
    # Calculate average lines across bookmakers
    if len(df) > 0:
        # Average spread (home team perspective)
        spread_cols = [c for c in df.columns if 'spread_home' in c and 'odds' not in c]
        if spread_cols:
            df['avg_spread'] = df[spread_cols].mean(axis=1)
        
        # Average total
        total_cols = [c for c in df.columns if 'total_line' in c]
        if total_cols:
            df['avg_total'] = df[total_cols].mean(axis=1)
        
        # Average moneyline (convert to implied probability, average, convert back)
        ml_home_cols = [c for c in df.columns if 'ml_home' in c]
        if ml_home_cols:
            # For now, just use first available
            df['avg_ml_home'] = df[ml_home_cols[0]]
        
        ml_away_cols = [c for c in df.columns if 'ml_away' in c]
        if ml_away_cols:
            df['avg_ml_away'] = df[ml_away_cols[0]]
    
    return df

def get_todays_odds(use_cache=True, save_csv=True):
    """
    Get today's NBA odds in a clean format
    
    Args:
        use_cache: Use cached data if available
        save_csv: Save to CSV file
    
    Returns:
        DataFrame with odds
    """
    # Fetch odds
    odds_data = fetch_nba_odds(use_cache=use_cache)
    
    if not odds_data:
        print("⚠️ No odds data available")
        return pd.DataFrame()
    
    # Parse odds
    df = parse_odds_data(odds_data)
    
    # Save to CSV
    if save_csv and len(df) > 0:
        output_file = f"data/processed/nba_odds_{datetime.now().strftime('%Y-%m-%d')}.csv"
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved odds to {output_file}")
    
    return df

def display_odds_summary(df):
    """Display a summary of fetched odds"""
    if len(df) == 0:
        print("No games found")
        return
    
    print(f"\n📊 NBA Odds Summary")
    print(f"   Games: {len(df)}")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("\n")
    
    for idx, row in df.iterrows():
        print(f"🏀 {row['away_team']} @ {row['home_team']}")
        print(f"   Time: {row['commence_time']}")
        
        # Spread
        if 'avg_spread' in row and pd.notna(row['avg_spread']):
            spread = row['avg_spread']
            print(f"   Spread: {row['home_team']} {spread:+.1f}")
        
        # Total
        if 'avg_total' in row and pd.notna(row['avg_total']):
            total = row['avg_total']
            print(f"   Total: {total:.1f}")
        
        # Moneyline
        if 'avg_ml_home' in row and pd.notna(row['avg_ml_home']):
            ml_home = row['avg_ml_home']
            ml_away = row.get('avg_ml_away', 0)
            print(f"   ML: {row['home_team']} {ml_home:+d} | {row['away_team']} {ml_away:+d}")
        
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch NBA odds from The Odds API")
    parser.add_argument('--no-cache', action='store_true', help='Force fetch from API (ignore cache)')
    parser.add_argument('--no-save', action='store_true', help='Do not save to CSV')
    
    args = parser.parse_args()
    
    # Fetch odds
    df = get_todays_odds(
        use_cache=not args.no_cache,
        save_csv=not args.no_save
    )
    
    # Display summary
    display_odds_summary(df)

