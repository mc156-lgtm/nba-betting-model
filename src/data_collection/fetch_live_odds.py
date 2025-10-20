#!/usr/bin/env python3
"""
Hybrid NBA Odds Fetcher
Primary: Free web scraping (OddsShark, ESPN)
Backup: The Odds API (if scraping fails)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import os
import json

# The Odds API configuration (backup)
ODDS_API_KEY = os.getenv('ODDS_API_KEY_NBA', '')  # Set this in environment
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'

# Team abbreviation mapping
TEAM_MAPPING = {
    'Lakers': 'LAL', 'Celtics': 'BOS', 'Warriors': 'GSW', 'Heat': 'MIA',
    'Nets': 'BRK', 'Knicks': 'NYK', 'Bulls': 'CHI', 'Cavaliers': 'CLE',
    'Mavericks': 'DAL', 'Nuggets': 'DEN', 'Pistons': 'DET', 'Rockets': 'HOU',
    'Pacers': 'IND', 'Clippers': 'LAC', 'Grizzlies': 'MEM', 'Bucks': 'MIL',
    'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Thunder': 'OKC', 'Magic': 'ORL',
    '76ers': 'PHI', 'Suns': 'PHO', 'Trail Blazers': 'POR', 'Kings': 'SAC',
    'Spurs': 'SAS', 'Raptors': 'TOR', 'Jazz': 'UTA', 'Wizards': 'WAS',
    'Hawks': 'ATL', 'Hornets': 'CHO'
}

def scrape_espn_odds():
    """
    Scrape NBA odds from ESPN
    Returns: list of dicts with game odds
    """
    print("🕷️  Attempting to scrape ESPN odds...")
    
    try:
        url = "https://www.espn.com/nba/odds"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # ESPN odds structure (this is a simplified example - actual structure may vary)
        games = []
        
        # Find game containers (structure depends on ESPN's current HTML)
        game_rows = soup.find_all('tr', class_='Table__TR')
        
        for row in game_rows[:10]:  # Limit to avoid parsing errors
            try:
                cells = row.find_all('td')
                if len(cells) >= 4:
                    # Extract team names, spread, total
                    # This is simplified - actual parsing depends on ESPN structure
                    game_data = {
                        'source': 'ESPN',
                        'scraped_at': datetime.now().isoformat(),
                        'teams': cells[0].text.strip() if cells else 'Unknown',
                        'spread': cells[1].text.strip() if len(cells) > 1 else None,
                        'total': cells[2].text.strip() if len(cells) > 2 else None,
                        'moneyline': cells[3].text.strip() if len(cells) > 3 else None
                    }
                    games.append(game_data)
            except Exception as e:
                continue
        
        if games:
            print(f"   ✅ Scraped {len(games)} games from ESPN")
            return games
        else:
            print("   ⚠️  No games found on ESPN")
            return []
            
    except Exception as e:
        print(f"   ❌ ESPN scraping failed: {e}")
        return []

def scrape_oddsshark_odds():
    """
    Scrape NBA odds from OddsShark
    Returns: list of dicts with game odds
    """
    print("🕷️  Attempting to scrape OddsShark odds...")
    
    try:
        url = "https://www.oddsshark.com/nba/odds"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        
        # OddsShark structure (simplified - actual structure may vary)
        # This is a placeholder - would need to inspect actual HTML
        game_cards = soup.find_all('div', class_='game')
        
        for card in game_cards[:15]:  # Limit to today's games
            try:
                game_data = {
                    'source': 'OddsShark',
                    'scraped_at': datetime.now().isoformat(),
                    'away_team': None,
                    'home_team': None,
                    'spread': None,
                    'total': None,
                    'moneyline_home': None,
                    'moneyline_away': None
                }
                
                # Extract data (structure depends on actual HTML)
                # This is placeholder logic
                teams = card.find_all('span', class_='team')
                if len(teams) >= 2:
                    game_data['away_team'] = teams[0].text.strip()
                    game_data['home_team'] = teams[1].text.strip()
                
                spread_elem = card.find('span', class_='spread')
                if spread_elem:
                    game_data['spread'] = spread_elem.text.strip()
                
                total_elem = card.find('span', class_='total')
                if total_elem:
                    game_data['total'] = total_elem.text.strip()
                
                if game_data['away_team'] and game_data['home_team']:
                    games.append(game_data)
                    
            except Exception as e:
                continue
        
        if games:
            print(f"   ✅ Scraped {len(games)} games from OddsShark")
            return games
        else:
            print("   ⚠️  No games found on OddsShark")
            return []
            
    except Exception as e:
        print(f"   ❌ OddsShark scraping failed: {e}")
        return []

def fetch_odds_api(api_key=None):
    """
    Fetch NBA odds from The Odds API (backup)
    Returns: list of dicts with game odds
    """
    print("🔑 Attempting to fetch from The Odds API (backup)...")
    
    if not api_key:
        api_key = ODDS_API_KEY
    
    if not api_key:
        print("   ❌ No API key provided")
        return []
    
    try:
        url = f"{ODDS_API_BASE}/sports/basketball_nba/odds/"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',  # Only these 3 markets
            'oddsFormat': 'american'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        games = []
        for game in data:
            game_data = {
                'source': 'The Odds API',
                'game_id': game.get('id'),
                'commence_time': game.get('commence_time'),
                'home_team': game.get('home_team'),
                'away_team': game.get('away_team'),
                'bookmakers': []
            }
            
            # Extract odds from bookmakers
            for bookmaker in game.get('bookmakers', []):
                book_data = {
                    'name': bookmaker.get('title'),
                    'markets': {}
                }
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    outcomes = market.get('outcomes', [])
                    
                    if market_key == 'h2h':  # Moneyline
                        for outcome in outcomes:
                            team = outcome.get('name')
                            price = outcome.get('price')
                            if team == game_data['home_team']:
                                book_data['markets']['moneyline_home'] = price
                            else:
                                book_data['markets']['moneyline_away'] = price
                    
                    elif market_key == 'spreads':
                        for outcome in outcomes:
                            team = outcome.get('name')
                            point = outcome.get('point')
                            if team == game_data['home_team']:
                                book_data['markets']['spread'] = point
                    
                    elif market_key == 'totals':
                        for outcome in outcomes:
                            point = outcome.get('point')
                            book_data['markets']['total'] = point
                            break
                
                game_data['bookmakers'].append(book_data)
            
            games.append(game_data)
        
        # Check remaining credits
        remaining = response.headers.get('x-requests-remaining')
        if remaining:
            print(f"   ✅ Fetched {len(games)} games from API")
            print(f"   📊 Remaining credits: {remaining}")
        
        return games
        
    except Exception as e:
        print(f"   ❌ The Odds API failed: {e}")
        return []

def get_todays_odds(api_key=None, use_api=False):
    """
    Get today's NBA odds using hybrid approach
    
    Args:
        api_key: The Odds API key (optional, for backup)
        use_api: Force use of API instead of scraping
    
    Returns:
        list of dicts with game odds
    """
    print("=" * 80)
    print(f"FETCHING NBA ODDS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    games = []
    
    if use_api:
        # Force API usage
        games = fetch_odds_api(api_key)
    else:
        # Try scraping first (free!)
        print("\n🆓 PRIMARY: Free Web Scraping")
        print("-" * 80)
        
        # Try ESPN first
        games = scrape_espn_odds()
        
        # If ESPN fails, try OddsShark
        if not games:
            games = scrape_oddsshark_odds()
        
        # If both scrapers fail, use API as backup
        if not games:
            print("\n⚠️  All scrapers failed, falling back to API...")
            print("-" * 80)
            games = fetch_odds_api(api_key)
    
    print("\n" + "=" * 80)
    if games:
        print(f"✅ SUCCESS: Retrieved {len(games)} games")
        print(f"   Source: {games[0].get('source', 'Unknown')}")
    else:
        print("❌ FAILED: No odds data retrieved")
    print("=" * 80)
    
    return games

def save_odds_to_file(games, filename='todays_odds.json'):
    """Save odds data to JSON file"""
    filepath = f"../../data/processed/{filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump({
            'fetched_at': datetime.now().isoformat(),
            'game_count': len(games),
            'games': games
        }, f, indent=2)
    
    print(f"\n💾 Saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NBA odds')
    parser.add_argument('--api-key', help='The Odds API key (optional)')
    parser.add_argument('--use-api', action='store_true', help='Force use of API instead of scraping')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    
    args = parser.parse_args()
    
    # Fetch odds
    games = get_todays_odds(api_key=args.api_key, use_api=args.use_api)
    
    # Display results
    if games:
        print(f"\n📊 Retrieved {len(games)} games:")
        for i, game in enumerate(games, 1):
            print(f"\n{i}. {game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}")
            
            if game.get('bookmakers'):
                # API format
                book = game['bookmakers'][0]
                markets = book.get('markets', {})
                print(f"   Spread: {markets.get('spread', 'N/A')}")
                print(f"   Total: {markets.get('total', 'N/A')}")
                print(f"   ML: {markets.get('moneyline_home', 'N/A')} / {markets.get('moneyline_away', 'N/A')}")
            else:
                # Scraper format
                print(f"   Spread: {game.get('spread', 'N/A')}")
                print(f"   Total: {game.get('total', 'N/A')}")
                print(f"   ML: {game.get('moneyline', 'N/A')}")
    
    # Save if requested
    if args.save and games:
        save_odds_to_file(games)

