"""
Generate Synthetic NBA Data for Model Development

This creates realistic synthetic NBA data based on actual NBA statistics distributions.
Replace with real data when API access is available.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)


def generate_teams():
    """Generate NBA teams data"""
    teams = [
        {'id': 1610612737, 'full_name': 'Atlanta Hawks', 'abbreviation': 'ATL', 'city': 'Atlanta', 'state': 'Georgia'},
        {'id': 1610612738, 'full_name': 'Boston Celtics', 'abbreviation': 'BOS', 'city': 'Boston', 'state': 'Massachusetts'},
        {'id': 1610612751, 'full_name': 'Brooklyn Nets', 'abbreviation': 'BKN', 'city': 'Brooklyn', 'state': 'New York'},
        {'id': 1610612766, 'full_name': 'Charlotte Hornets', 'abbreviation': 'CHA', 'city': 'Charlotte', 'state': 'North Carolina'},
        {'id': 1610612741, 'full_name': 'Chicago Bulls', 'abbreviation': 'CHI', 'city': 'Chicago', 'state': 'Illinois'},
        {'id': 1610612739, 'full_name': 'Cleveland Cavaliers', 'abbreviation': 'CLE', 'city': 'Cleveland', 'state': 'Ohio'},
        {'id': 1610612742, 'full_name': 'Dallas Mavericks', 'abbreviation': 'DAL', 'city': 'Dallas', 'state': 'Texas'},
        {'id': 1610612743, 'full_name': 'Denver Nuggets', 'abbreviation': 'DEN', 'city': 'Denver', 'state': 'Colorado'},
        {'id': 1610612765, 'full_name': 'Detroit Pistons', 'abbreviation': 'DET', 'city': 'Detroit', 'state': 'Michigan'},
        {'id': 1610612744, 'full_name': 'Golden State Warriors', 'abbreviation': 'GSW', 'city': 'Golden State', 'state': 'California'},
        {'id': 1610612745, 'full_name': 'Houston Rockets', 'abbreviation': 'HOU', 'city': 'Houston', 'state': 'Texas'},
        {'id': 1610612754, 'full_name': 'Indiana Pacers', 'abbreviation': 'IND', 'city': 'Indiana', 'state': 'Indiana'},
        {'id': 1610612746, 'full_name': 'LA Clippers', 'abbreviation': 'LAC', 'city': 'Los Angeles', 'state': 'California'},
        {'id': 1610612747, 'full_name': 'Los Angeles Lakers', 'abbreviation': 'LAL', 'city': 'Los Angeles', 'state': 'California'},
        {'id': 1610612763, 'full_name': 'Memphis Grizzlies', 'abbreviation': 'MEM', 'city': 'Memphis', 'state': 'Tennessee'},
        {'id': 1610612748, 'full_name': 'Miami Heat', 'abbreviation': 'MIA', 'city': 'Miami', 'state': 'Florida'},
        {'id': 1610612749, 'full_name': 'Milwaukee Bucks', 'abbreviation': 'MIL', 'city': 'Milwaukee', 'state': 'Wisconsin'},
        {'id': 1610612750, 'full_name': 'Minnesota Timberwolves', 'abbreviation': 'MIN', 'city': 'Minneapolis', 'state': 'Minnesota'},
        {'id': 1610612740, 'full_name': 'New Orleans Pelicans', 'abbreviation': 'NOP', 'city': 'New Orleans', 'state': 'Louisiana'},
        {'id': 1610612752, 'full_name': 'New York Knicks', 'abbreviation': 'NYK', 'city': 'New York', 'state': 'New York'},
        {'id': 1610612760, 'full_name': 'Oklahoma City Thunder', 'abbreviation': 'OKC', 'city': 'Oklahoma City', 'state': 'Oklahoma'},
        {'id': 1610612753, 'full_name': 'Orlando Magic', 'abbreviation': 'ORL', 'city': 'Orlando', 'state': 'Florida'},
        {'id': 1610612755, 'full_name': 'Philadelphia 76ers', 'abbreviation': 'PHI', 'city': 'Philadelphia', 'state': 'Pennsylvania'},
        {'id': 1610612756, 'full_name': 'Phoenix Suns', 'abbreviation': 'PHX', 'city': 'Phoenix', 'state': 'Arizona'},
        {'id': 1610612757, 'full_name': 'Portland Trail Blazers', 'abbreviation': 'POR', 'city': 'Portland', 'state': 'Oregon'},
        {'id': 1610612758, 'full_name': 'Sacramento Kings', 'abbreviation': 'SAC', 'city': 'Sacramento', 'state': 'California'},
        {'id': 1610612759, 'full_name': 'San Antonio Spurs', 'abbreviation': 'SAS', 'city': 'San Antonio', 'state': 'Texas'},
        {'id': 1610612761, 'full_name': 'Toronto Raptors', 'abbreviation': 'TOR', 'city': 'Toronto', 'state': 'Ontario'},
        {'id': 1610612762, 'full_name': 'Utah Jazz', 'abbreviation': 'UTA', 'city': 'Utah', 'state': 'Utah'},
        {'id': 1610612764, 'full_name': 'Washington Wizards', 'abbreviation': 'WAS', 'city': 'Washington', 'state': 'District of Columbia'},
    ]
    return pd.DataFrame(teams)


def generate_season_games(season, num_games=1230):
    """Generate synthetic game data for a season (82 games per team = 1230 total games)"""
    
    teams_df = generate_teams()
    team_ids = teams_df['id'].tolist()
    team_abbrevs = teams_df['abbreviation'].tolist()
    
    games = []
    game_id = int(f"002{season.replace('-', '')}00001")
    
    # Generate games from October to April
    start_date = datetime(int(season[:4]), 10, 15)
    
    for i in range(num_games):
        # Random matchup
        home_idx = np.random.randint(0, len(team_ids))
        away_idx = np.random.randint(0, len(team_ids))
        while away_idx == home_idx:
            away_idx = np.random.randint(0, len(team_ids))
        
        home_team_id = team_ids[home_idx]
        away_team_id = team_ids[away_idx]
        home_team_abbrev = team_abbrevs[home_idx]
        away_team_abbrev = team_abbrevs[away_idx]
        
        # Generate realistic scores (NBA average ~110 points)
        home_score = int(np.random.normal(112, 12))
        away_score = int(np.random.normal(108, 12))
        
        # Ensure positive scores
        home_score = max(home_score, 85)
        away_score = max(away_score, 85)
        
        # Game date
        game_date = start_date + timedelta(days=i // 15)
        
        # Home team record
        games.append({
            'SEASON_ID': f'2{season.replace("-", "")}',
            'TEAM_ID': home_team_id,
            'TEAM_ABBREVIATION': home_team_abbrev,
            'TEAM_NAME': teams_df[teams_df['id'] == home_team_id]['full_name'].values[0],
            'GAME_ID': str(game_id),
            'GAME_DATE': game_date.strftime('%Y-%m-%d'),
            'MATCHUP': f'{home_team_abbrev} vs. {away_team_abbrev}',
            'WL': 'W' if home_score > away_score else 'L',
            'PTS': home_score,
            'FGM': int(home_score * 0.38),
            'FGA': int(home_score * 0.85),
            'FG_PCT': round(np.random.uniform(0.42, 0.52), 3),
            'FG3M': int(np.random.uniform(10, 18)),
            'FG3A': int(np.random.uniform(28, 42)),
            'FG3_PCT': round(np.random.uniform(0.32, 0.42), 3),
            'FTM': int(np.random.uniform(12, 22)),
            'FTA': int(np.random.uniform(18, 28)),
            'FT_PCT': round(np.random.uniform(0.72, 0.85), 3),
            'OREB': int(np.random.uniform(8, 14)),
            'DREB': int(np.random.uniform(28, 38)),
            'REB': int(np.random.uniform(38, 50)),
            'AST': int(np.random.uniform(20, 30)),
            'STL': int(np.random.uniform(5, 12)),
            'BLK': int(np.random.uniform(3, 8)),
            'TOV': int(np.random.uniform(10, 18)),
            'PF': int(np.random.uniform(18, 26)),
            'PLUS_MINUS': home_score - away_score,
        })
        
        # Away team record
        games.append({
            'SEASON_ID': f'2{season.replace("-", "")}',
            'TEAM_ID': away_team_id,
            'TEAM_ABBREVIATION': away_team_abbrev,
            'TEAM_NAME': teams_df[teams_df['id'] == away_team_id]['full_name'].values[0],
            'GAME_ID': str(game_id),
            'GAME_DATE': game_date.strftime('%Y-%m-%d'),
            'MATCHUP': f'{away_team_abbrev} @ {home_team_abbrev}',
            'WL': 'W' if away_score > home_score else 'L',
            'PTS': away_score,
            'FGM': int(away_score * 0.38),
            'FGA': int(away_score * 0.85),
            'FG_PCT': round(np.random.uniform(0.42, 0.52), 3),
            'FG3M': int(np.random.uniform(10, 18)),
            'FG3A': int(np.random.uniform(28, 42)),
            'FG3_PCT': round(np.random.uniform(0.32, 0.42), 3),
            'FTM': int(np.random.uniform(12, 22)),
            'FTA': int(np.random.uniform(18, 28)),
            'FT_PCT': round(np.random.uniform(0.72, 0.85), 3),
            'OREB': int(np.random.uniform(8, 14)),
            'DREB': int(np.random.uniform(28, 38)),
            'REB': int(np.random.uniform(38, 50)),
            'AST': int(np.random.uniform(20, 30)),
            'STL': int(np.random.uniform(5, 12)),
            'BLK': int(np.random.uniform(3, 8)),
            'TOV': int(np.random.uniform(10, 18)),
            'PF': int(np.random.uniform(18, 26)),
            'PLUS_MINUS': away_score - home_score,
        })
        
        game_id += 1
    
    return pd.DataFrame(games)


def generate_team_stats(season):
    """Generate season-aggregate team statistics"""
    
    teams_df = generate_teams()
    team_stats = []
    
    for _, team in teams_df.iterrows():
        stats = {
            'TEAM_ID': team['id'],
            'TEAM_NAME': team['full_name'],
            'GP': 82,
            'W': int(np.random.uniform(20, 62)),
            'L': 0,  # Will calculate
            'W_PCT': 0,  # Will calculate
            'MIN': round(np.random.uniform(238, 242), 1),
            'PTS': round(np.random.uniform(105, 120), 1),
            'FGM': round(np.random.uniform(38, 46), 1),
            'FGA': round(np.random.uniform(85, 92), 1),
            'FG_PCT': round(np.random.uniform(0.44, 0.50), 3),
            'FG3M': round(np.random.uniform(11, 16), 1),
            'FG3A': round(np.random.uniform(32, 42), 1),
            'FG3_PCT': round(np.random.uniform(0.34, 0.40), 3),
            'FTM': round(np.random.uniform(15, 22), 1),
            'FTA': round(np.random.uniform(20, 28), 1),
            'FT_PCT': round(np.random.uniform(0.75, 0.82), 3),
            'OREB': round(np.random.uniform(9, 13), 1),
            'DREB': round(np.random.uniform(32, 38), 1),
            'REB': round(np.random.uniform(42, 48), 1),
            'AST': round(np.random.uniform(23, 29), 1),
            'TOV': round(np.random.uniform(12, 16), 1),
            'STL': round(np.random.uniform(7, 10), 1),
            'BLK': round(np.random.uniform(4, 7), 1),
            'PF': round(np.random.uniform(19, 23), 1),
            'PLUS_MINUS': round(np.random.uniform(-8, 8), 1),
        }
        stats['L'] = 82 - stats['W']
        stats['W_PCT'] = round(stats['W'] / 82, 3)
        team_stats.append(stats)
    
    return pd.DataFrame(team_stats)


def generate_player_stats(season, num_players=450):
    """Generate player statistics"""
    
    teams_df = generate_teams()
    team_ids = teams_df['id'].tolist()
    
    players = []
    
    for i in range(num_players):
        team_id = team_ids[i % len(team_ids)]
        
        # Generate player type (star, starter, bench, deep bench)
        player_type = np.random.choice(['star', 'starter', 'bench', 'deep_bench'], 
                                       p=[0.1, 0.3, 0.4, 0.2])
        
        if player_type == 'star':
            pts = round(np.random.uniform(24, 32), 1)
            min_played = round(np.random.uniform(34, 38), 1)
            gp = int(np.random.uniform(65, 82))
        elif player_type == 'starter':
            pts = round(np.random.uniform(12, 20), 1)
            min_played = round(np.random.uniform(28, 34), 1)
            gp = int(np.random.uniform(70, 82))
        elif player_type == 'bench':
            pts = round(np.random.uniform(6, 12), 1)
            min_played = round(np.random.uniform(18, 28), 1)
            gp = int(np.random.uniform(60, 82))
        else:  # deep bench
            pts = round(np.random.uniform(2, 6), 1)
            min_played = round(np.random.uniform(8, 18), 1)
            gp = int(np.random.uniform(40, 70))
        
        player = {
            'PLAYER_ID': 200000 + i,
            'PLAYER_NAME': f'Player_{i}',
            'TEAM_ID': team_id,
            'TEAM_ABBREVIATION': teams_df[teams_df['id'] == team_id]['abbreviation'].values[0],
            'GP': gp,
            'MIN': min_played,
            'PTS': pts,
            'FGM': round(pts * 0.4, 1),
            'FGA': round(pts * 0.9, 1),
            'FG_PCT': round(np.random.uniform(0.42, 0.55), 3),
            'FG3M': round(np.random.uniform(0.5, 3.5), 1),
            'FG3A': round(np.random.uniform(1.5, 9), 1),
            'FG3_PCT': round(np.random.uniform(0.30, 0.42), 3),
            'FTM': round(pts * 0.2, 1),
            'FTA': round(pts * 0.25, 1),
            'FT_PCT': round(np.random.uniform(0.70, 0.90), 3),
            'OREB': round(np.random.uniform(0.3, 2.5), 1),
            'DREB': round(np.random.uniform(2, 8), 1),
            'REB': round(np.random.uniform(2.5, 10), 1),
            'AST': round(np.random.uniform(1, 8), 1),
            'TOV': round(np.random.uniform(0.8, 3.5), 1),
            'STL': round(np.random.uniform(0.4, 2), 1),
            'BLK': round(np.random.uniform(0.2, 2), 1),
            'PF': round(np.random.uniform(1.5, 3.5), 1),
        }
        players.append(player)
    
    return pd.DataFrame(players)


def main():
    """Generate all synthetic data"""
    
    data_dir = '../../data/raw'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Synthetic NBA Data")
    print("=" * 60)
    
    # Generate teams
    print("\n1. Generating teams...")
    teams_df = generate_teams()
    teams_df.to_csv(f'{data_dir}/teams.csv', index=False)
    print(f"   Generated {len(teams_df)} teams")
    
    # Generate data for multiple seasons
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    
    all_games = []
    all_team_stats = []
    all_player_stats = []
    
    for season in seasons:
        print(f"\n2. Generating {season} season data...")
        
        # Games
        games = generate_season_games(season)
        games['SEASON'] = season
        games.to_csv(f'{data_dir}/games_{season}.csv', index=False)
        all_games.append(games)
        print(f"   Generated {len(games)} game records")
        
        # Team stats
        team_stats = generate_team_stats(season)
        team_stats['SEASON'] = season
        team_stats.to_csv(f'{data_dir}/team_stats_{season}.csv', index=False)
        all_team_stats.append(team_stats)
        print(f"   Generated team stats for {len(team_stats)} teams")
        
        # Player stats
        player_stats = generate_player_stats(season)
        player_stats['SEASON'] = season
        player_stats.to_csv(f'{data_dir}/player_stats_{season}.csv', index=False)
        all_player_stats.append(player_stats)
        print(f"   Generated stats for {len(player_stats)} players")
    
    # Combine all seasons
    print("\n3. Combining all seasons...")
    combined_games = pd.concat(all_games, ignore_index=True)
    combined_games.to_csv(f'{data_dir}/all_games_historical.csv', index=False)
    print(f"   Saved {len(combined_games)} total game records")
    
    combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
    combined_team_stats.to_csv(f'{data_dir}/all_team_stats_historical.csv', index=False)
    print(f"   Saved {len(combined_team_stats)} total team stat records")
    
    combined_player_stats = pd.concat(all_player_stats, ignore_index=True)
    combined_player_stats.to_csv(f'{data_dir}/all_player_stats_historical.csv', index=False)
    print(f"   Saved {len(combined_player_stats)} total player stat records")
    
    print("\n" + "=" * 60)
    print("Synthetic data generation complete!")
    print("=" * 60)
    print("\nNote: This is synthetic data for demonstration.")
    print("Replace with real NBA data when API access is available.")


if __name__ == "__main__":
    main()

