"""
Fetch NBA Schedule for Any Date
Uses NBA API and The Odds API to get games for today, tomorrow, or custom dates
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
import json

# Try to import NBA API
try:
    from nba_api.live.nba.endpoints import scoreboard
    NBA_API_AVAILABLE = True
except:
    NBA_API_AVAILABLE = False

def get_schedule_for_date(target_date):
    """
    Get NBA schedule for a specific date
    
    Args:
        target_date: datetime object for the target date
    
    Returns:
        list of games with team info
    """
    games = []
    
    # For today, use NBA API if available
    if target_date.date() == datetime.now().date() and NBA_API_AVAILABLE:
        try:
            board = scoreboard.ScoreBoard()
            api_games = board.games.get_dict()
            
            for game in api_games:
                games.append({
                    'game_id': game['gameId'],
                    'game_time': game['gameTimeUTC'],
                    'home_team': game['homeTeam']['teamTricode'],
                    'home_team_name': game['homeTeam']['teamName'],
                    'away_team': game['awayTeam']['teamTricode'],
                    'away_team_name': game['awayTeam']['teamName'],
                    'game_status': game['gameStatusText'],
                    'home_score': game['homeTeam']['score'],
                    'away_score': game['awayTeam']['score'],
                })
            
            return games
        except Exception as e:
            print(f"⚠️ NBA API error: {e}")
    
    # For other dates, try to get from odds data or use typical schedule
    # The Odds API provides upcoming games
    try:
        from fetch_odds_api import fetch_nba_odds, parse_odds_data
        
        odds_data = fetch_nba_odds(use_cache=True)
        if odds_data:
            # Filter games by date (convert UTC to local time for comparison)
            from datetime import timezone
            
            for game in odds_data:
                game_time_utc = datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00'))
                # Convert to local time for date comparison
                game_time_local = game_time_utc.astimezone()
                
                # Check if game is on target date (local time)
                # Allow games from target date or within 12 hours before/after
                time_diff = abs((game_time_local.date() - target_date.date()).days)
                
                if time_diff <= 1:  # Include games within 1 day
                    games.append({
                        'game_id': game['id'],
                        'game_time': game['commence_time'],
                        'home_team': game['home_team'],
                        'home_team_name': game['home_team'],
                        'away_team': game['away_team'],
                        'away_team_name': game['away_team'],
                        'game_status': 'Scheduled',
                        'home_score': 0,
                        'away_score': 0,
                    })
    except Exception as e:
        print(f"⚠️ Odds API error: {e}")
    
    return games

def get_todays_games():
    """Get today's NBA games"""
    return get_schedule_for_date(datetime.now())

def get_tomorrows_games():
    """Get tomorrow's NBA games"""
    tomorrow = datetime.now() + timedelta(days=1)
    return get_schedule_for_date(tomorrow)

def get_games_for_custom_date(date_str):
    """
    Get games for a custom date
    
    Args:
        date_str: Date string in format 'YYYY-MM-DD'
    
    Returns:
        list of games
    """
    target_date = datetime.strptime(date_str, '%Y-%m-%d')
    return get_schedule_for_date(target_date)

if __name__ == "__main__":
    print("🏀 NBA Schedule Fetcher Test\n")
    
    # Test today's games
    print("📅 Today's Games:")
    today_games = get_todays_games()
    if today_games:
        for game in today_games:
            print(f"  {game['away_team']} @ {game['home_team']} - {game['game_status']}")
    else:
        print("  No games today")
    
    print()
    
    # Test tomorrow's games
    print("📅 Tomorrow's Games:")
    tomorrow_games = get_tomorrows_games()
    if tomorrow_games:
        for game in tomorrow_games:
            print(f"  {game['away_team']} @ {game['home_team']} - {game['game_status']}")
    else:
        print("  No games tomorrow (or data not available yet)")

