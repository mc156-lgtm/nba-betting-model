"""
Unified Prediction Interface for NBA Betting Models

This script provides a simple interface to make predictions using all trained models.
"""

import pandas as pd
import numpy as np
from spread_model import SpreadModel
from totals_model import TotalsModel
from moneyline_model import MoneylineModel
from player_props_model import PlayerPropsModel


class NBABettingPredictor:
    """Unified interface for all NBA betting predictions"""
    
    def __init__(self):
        self.spread_model = SpreadModel()
        self.totals_model = TotalsModel()
        self.moneyline_model = MoneylineModel()
        self.player_props_model = PlayerPropsModel()
        
        # Load trained models
        self.spread_model.load_model()
        self.totals_model.load_model()
        self.moneyline_model.load_model()
        self.player_props_model.load_models()
        
        print("All models loaded successfully!")
    
    def predict_game(self, home_team_features, away_team_features):
        """
        Predict all betting outcomes for a game
        
        Args:
            home_team_features: dict with home team rolling stats
            away_team_features: dict with away team rolling stats
        
        Returns:
            dict with predictions for spread, total, and moneyline
        """
        # Combine features into a single row
        features = {}
        for key, value in home_team_features.items():
            features[f'HOME_{key}'] = value
        for key, value in away_team_features.items():
            features[f'AWAY_{key}'] = value
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        spread_features = df[self.spread_model.feature_columns]
        totals_features = df[self.totals_model.feature_columns]
        moneyline_features = df[self.moneyline_model.feature_columns]
        
        # Make predictions
        spread_pred = self.spread_model.predict(spread_features)[0]
        total_pred = self.totals_model.predict(totals_features)[0]
        win_prob = self.moneyline_model.predict_proba(moneyline_features)[0]
        
        return {
            'spread': {
                'predicted_margin': spread_pred,
                'home_predicted_score': None,  # Would need total to calculate
                'away_predicted_score': None
            },
            'total': {
                'predicted_total': total_pred,
                'over_under_line': None  # User would provide this
            },
            'moneyline': {
                'home_win_probability': win_prob[1],
                'away_win_probability': win_prob[0],
                'predicted_winner': 'Home' if win_prob[1] > 0.5 else 'Away'
            }
        }
    
    def predict_player_prop(self, player_features, stat_type):
        """
        Predict a player prop
        
        Args:
            player_features: dict with player stats
            stat_type: 'PTS', 'REB', 'AST', 'STL', or 'BLK'
        
        Returns:
            float prediction
        """
        df = pd.DataFrame([player_features])
        features = df[self.player_props_model.feature_columns]
        
        prediction = self.player_props_model.predict(features, stat_type)[0]
        
        return {
            'stat_type': stat_type,
            'predicted_value': prediction
        }
    
    def calculate_betting_edge(self, prediction, market_line, bet_type='spread'):
        """
        Calculate the betting edge based on model prediction vs market line
        
        Args:
            prediction: Model prediction
            market_line: Sportsbook line
            bet_type: 'spread', 'total', or 'moneyline'
        
        Returns:
            dict with edge analysis
        """
        if bet_type == 'spread':
            edge = abs(prediction - market_line)
            recommendation = 'Bet Home' if prediction < market_line else 'Bet Away'
        elif bet_type == 'total':
            edge = abs(prediction - market_line)
            recommendation = 'Bet Over' if prediction > market_line else 'Bet Under'
        elif bet_type == 'moneyline':
            # Convert probability to implied odds
            if prediction > 0.5:
                implied_odds = -1 * (prediction / (1 - prediction)) * 100
            else:
                implied_odds = ((1 - prediction) / prediction) * 100
            edge = abs(implied_odds - market_line)
            recommendation = 'Bet Home' if prediction > 0.5 else 'Bet Away'
        
        return {
            'edge': edge,
            'recommendation': recommendation if edge > 2 else 'No bet',
            'confidence': 'High' if edge > 5 else 'Medium' if edge > 2 else 'Low'
        }


def demo_predictions():
    """Demonstrate the prediction system"""
    print("\nNBA Betting Model - Demo Predictions")
    print("=" * 60)
    
    predictor = NBABettingPredictor()
    
    # Example game prediction
    print("\n--- Example Game Prediction ---")
    print("Matchup: Lakers (Home) vs. Warriors (Away)")
    
    # These would come from your feature engineering pipeline
    home_features = {
        'PTS_roll_5': 115.0,
        'PTS_roll_10': 113.5,
        'PTS_roll_20': 112.0,
        'FG_PCT_roll_5': 0.475,
        'FG_PCT_roll_10': 0.470,
        'FG_PCT_roll_20': 0.465,
        'FG3_PCT_roll_5': 0.365,
        'FG3_PCT_roll_10': 0.360,
        'FG3_PCT_roll_20': 0.355,
        'REB_roll_5': 45.0,
        'REB_roll_10': 44.5,
        'REB_roll_20': 44.0,
        'AST_roll_5': 26.0,
        'AST_roll_10': 25.5,
        'AST_roll_20': 25.0,
        'DEF_EFF': 108.5
    }
    
    away_features = {
        'PTS_roll_5': 118.0,
        'PTS_roll_10': 117.0,
        'PTS_roll_20': 116.0,
        'FG_PCT_roll_5': 0.485,
        'FG_PCT_roll_10': 0.480,
        'FG_PCT_roll_20': 0.475,
        'FG3_PCT_roll_5': 0.380,
        'FG3_PCT_roll_10': 0.375,
        'FG3_PCT_roll_20': 0.370,
        'REB_roll_5': 43.0,
        'REB_roll_10': 42.5,
        'REB_roll_20': 42.0,
        'AST_roll_5': 28.0,
        'AST_roll_10': 27.5,
        'AST_roll_20': 27.0,
        'DEF_EFF': 110.0
    }
    
    # Note: This is just a demo - actual prediction would need all required features
    print("\nNote: This demo uses simplified features.")
    print("In production, use the full feature set from the feature engineering pipeline.")
    
    print("\n" + "=" * 60)
    print("Prediction system ready!")
    print("=" * 60)


if __name__ == "__main__":
    demo_predictions()

