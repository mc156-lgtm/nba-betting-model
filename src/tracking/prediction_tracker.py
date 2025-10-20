"""
NBA Prediction Tracker
Records predictions, actual results, and calculates accuracy over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Paths
PREDICTIONS_DB = "data/tracking/predictions.csv"
RESULTS_DB = "data/tracking/results.csv"
PERFORMANCE_DB = "data/tracking/performance.csv"

def ensure_tracking_dirs():
    """Create tracking directories if they don't exist"""
    os.makedirs("data/tracking", exist_ok=True)

def save_prediction(game_date, home_team, away_team, predictions, market_odds=None):
    """
    Save a prediction to the database
    
    Args:
        game_date: Date/time of the game
        home_team: Home team name
        away_team: Away team name
        predictions: Dict with model predictions
            {
                'spread': -6.5,
                'total': 226.3,
                'home_win_prob': 65.0,
                'away_win_prob': 35.0
            }
        market_odds: Dict with market odds (optional)
            {
                'spread': -6.5,
                'total': 224.5,
                'ml_home': -285,
                'ml_away': +230
            }
    """
    ensure_tracking_dirs()
    
    # Create prediction record
    record = {
        'prediction_id': f"{game_date}_{away_team}_{home_team}",
        'timestamp': datetime.now().isoformat(),
        'game_date': game_date,
        'home_team': home_team,
        'away_team': away_team,
        'predicted_spread': predictions.get('spread'),
        'predicted_total': predictions.get('total'),
        'predicted_home_win_prob': predictions.get('home_win_prob'),
        'predicted_away_win_prob': predictions.get('away_win_prob'),
    }
    
    # Add market odds if available
    if market_odds:
        record.update({
            'market_spread': market_odds.get('spread'),
            'market_total': market_odds.get('total'),
            'market_ml_home': market_odds.get('ml_home'),
            'market_ml_away': market_odds.get('ml_away'),
            'spread_edge': abs(predictions.get('spread', 0) - market_odds.get('spread', 0)),
            'total_edge': abs(predictions.get('total', 0) - market_odds.get('total', 0)),
        })
    
    # Load existing predictions or create new DataFrame
    if os.path.exists(PREDICTIONS_DB):
        df = pd.read_csv(PREDICTIONS_DB)
    else:
        df = pd.DataFrame()
    
    # Append new prediction
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    
    # Save
    df.to_csv(PREDICTIONS_DB, index=False)
    print(f"✅ Saved prediction: {away_team} @ {home_team} on {game_date}")
    
    return record['prediction_id']

def save_result(prediction_id, actual_home_score, actual_away_score):
    """
    Save actual game result
    
    Args:
        prediction_id: ID from save_prediction()
        actual_home_score: Home team final score
        actual_away_score: Away team final score
    """
    ensure_tracking_dirs()
    
    # Calculate actual stats
    actual_spread = actual_home_score - actual_away_score
    actual_total = actual_home_score + actual_away_score
    home_won = actual_home_score > actual_away_score
    
    # Create result record
    result = {
        'prediction_id': prediction_id,
        'result_timestamp': datetime.now().isoformat(),
        'actual_home_score': actual_home_score,
        'actual_away_score': actual_away_score,
        'actual_spread': actual_spread,
        'actual_total': actual_total,
        'home_won': home_won
    }
    
    # Load existing results or create new DataFrame
    if os.path.exists(RESULTS_DB):
        df = pd.read_csv(RESULTS_DB)
    else:
        df = pd.DataFrame()
    
    # Append new result
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    # Save
    df.to_csv(RESULTS_DB, index=False)
    print(f"✅ Saved result for prediction {prediction_id}: {actual_away_score}-{actual_home_score}")
    
    return result

def calculate_accuracy(days=None):
    """
    Calculate prediction accuracy
    
    Args:
        days: Number of days to look back (None = all time)
    
    Returns:
        Dict with accuracy metrics
    """
    if not os.path.exists(PREDICTIONS_DB) or not os.path.exists(RESULTS_DB):
        return None
    
    # Load data
    predictions = pd.read_csv(PREDICTIONS_DB)
    results = pd.read_csv(RESULTS_DB)
    
    # Merge predictions with results
    df = predictions.merge(results, on='prediction_id', how='inner')
    
    if len(df) == 0:
        return None
    
    # Filter by date if specified
    if days:
        df['game_date'] = pd.to_datetime(df['game_date'])
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['game_date'] >= cutoff]
    
    if len(df) == 0:
        return None
    
    # Calculate metrics
    metrics = {
        'period': f'Last {days} days' if days else 'All time',
        'total_predictions': len(df),
        'date_range': f"{df['game_date'].min()} to {df['game_date'].max()}"
    }
    
    # Spread accuracy
    df['spread_error'] = abs(df['predicted_spread'] - df['actual_spread'])
    metrics['spread_mae'] = df['spread_error'].mean()
    metrics['spread_rmse'] = np.sqrt((df['spread_error'] ** 2).mean())
    
    # Totals accuracy
    df['total_error'] = abs(df['predicted_total'] - df['actual_total'])
    metrics['total_mae'] = df['total_error'].mean()
    metrics['total_rmse'] = np.sqrt((df['total_error'] ** 2).mean())
    
    # Moneyline accuracy
    df['predicted_home_win'] = df['predicted_home_win_prob'] > 50
    df['correct_ml'] = df['predicted_home_win'] == df['home_won']
    metrics['moneyline_accuracy'] = (df['correct_ml'].sum() / len(df)) * 100
    
    # Spread cover accuracy (did we predict the right side?)
    df['predicted_home_covers'] = df['predicted_spread'] > 0
    df['actual_home_covers'] = df['actual_spread'] > 0
    df['correct_spread_side'] = df['predicted_home_covers'] == df['actual_home_covers']
    metrics['spread_cover_accuracy'] = (df['correct_spread_side'].sum() / len(df)) * 100
    
    # Total over/under accuracy
    if 'market_total' in df.columns:
        df['predicted_over'] = df['predicted_total'] > df['market_total']
        df['actual_over'] = df['actual_total'] > df['market_total']
        df['correct_over_under'] = df['predicted_over'] == df['actual_over']
        metrics['over_under_accuracy'] = (df['correct_over_under'].sum() / len(df[df['market_total'].notna()])) * 100
    
    # Betting edge validation (if market odds available)
    if 'spread_edge' in df.columns:
        metrics['avg_spread_edge'] = df['spread_edge'].mean()
        metrics['avg_total_edge'] = df['total_edge'].mean()
    
    return metrics

def get_performance_summary():
    """
    Get performance summary for multiple time periods
    
    Returns:
        DataFrame with performance across time periods
    """
    periods = [5, 10, 30, 90, None]  # Last 5, 10, 30, 90 days, and all time
    
    summaries = []
    
    for period in periods:
        metrics = calculate_accuracy(days=period)
        if metrics:
            summaries.append(metrics)
    
    if not summaries:
        return pd.DataFrame()
    
    return pd.DataFrame(summaries)

def display_performance(period_days=None):
    """
    Display performance metrics
    
    Args:
        period_days: Number of days to look back (None = all time)
    """
    metrics = calculate_accuracy(days=period_days)
    
    if not metrics:
        print("⚠️ No predictions with results available yet")
        return
    
    print(f"\n📊 Model Performance - {metrics['period']}")
    print(f"   Date Range: {metrics['date_range']}")
    print(f"   Total Predictions: {metrics['total_predictions']}")
    print()
    
    print("🎯 Spread Predictions:")
    print(f"   MAE: {metrics['spread_mae']:.2f} points")
    print(f"   RMSE: {metrics['spread_rmse']:.2f} points")
    print(f"   Cover Accuracy: {metrics['spread_cover_accuracy']:.1f}%")
    print()
    
    print("🎯 Totals Predictions:")
    print(f"   MAE: {metrics['total_mae']:.2f} points")
    print(f"   RMSE: {metrics['total_rmse']:.2f} points")
    if 'over_under_accuracy' in metrics:
        print(f"   Over/Under Accuracy: {metrics['over_under_accuracy']:.1f}%")
    print()
    
    print("🎯 Moneyline Predictions:")
    print(f"   Accuracy: {metrics['moneyline_accuracy']:.1f}%")
    print()
    
    if 'avg_spread_edge' in metrics:
        print("💰 Betting Edge:")
        print(f"   Avg Spread Edge: {metrics['avg_spread_edge']:.2f} points")
        print(f"   Avg Total Edge: {metrics['avg_total_edge']:.2f} points")
        print()

def display_performance_comparison():
    """Display performance comparison across time periods"""
    df = get_performance_summary()
    
    if len(df) == 0:
        print("⚠️ No predictions with results available yet")
        return
    
    print("\n📊 Performance Comparison Across Time Periods\n")
    
    # Format for display
    display_df = df[[
        'period',
        'total_predictions',
        'spread_mae',
        'total_mae',
        'moneyline_accuracy',
        'spread_cover_accuracy'
    ]].copy()
    
    display_df.columns = [
        'Period',
        'Games',
        'Spread MAE',
        'Total MAE',
        'ML Acc %',
        'Spread Acc %'
    ]
    
    print(display_df.to_string(index=False))
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Prediction Tracker")
    parser.add_argument('--save-prediction', action='store_true', help='Save a test prediction')
    parser.add_argument('--save-result', action='store_true', help='Save a test result')
    parser.add_argument('--show-performance', type=int, nargs='?', const=None, help='Show performance (optional: days to look back)')
    parser.add_argument('--compare', action='store_true', help='Compare performance across time periods')
    
    args = parser.parse_args()
    
    if args.save_prediction:
        # Test prediction
        save_prediction(
            game_date="2024-10-23",
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            predictions={
                'spread': -4.9,
                'total': 226.3,
                'home_win_prob': 65.0,
                'away_win_prob': 35.0
            },
            market_odds={
                'spread': -6.5,
                'total': 224.5,
                'ml_home': -285,
                'ml_away': +230
            }
        )
    
    elif args.save_result:
        # Test result
        save_result(
            prediction_id="2024-10-23_Los Angeles Lakers_Boston Celtics",
            actual_home_score=112,
            actual_away_score=108
        )
    
    elif args.compare:
        display_performance_comparison()
    
    elif args.show_performance is not None:
        display_performance(period_days=args.show_performance)
    
    else:
        print("Use --help to see available options")

