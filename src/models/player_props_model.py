"""
NBA Player Props Prediction Model using Ridge Regression

This model predicts individual player statistics (points, rebounds, assists, etc.)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os


class PlayerPropsModel:
    """Ridge Regression model for predicting NBA player props"""
    
    def __init__(self, data_path='../../data/raw/all_player_stats_historical.csv'):
        self.data_path = data_path
        self.models = {}  # One model per stat type
        self.scalers = {}
        self.feature_columns = None
        
    def load_and_prepare_data(self):
        """Load player data and create rolling features"""
        print("Loading player data...")
        df = pd.read_csv(self.data_path)
        
        # Sort by player and season
        df = df.sort_values(by=['PLAYER_ID', 'SEASON'])
        
        # Create rolling averages (simulated since we don't have game-by-game data)
        # In production, you would calculate these from actual game logs
        stats_to_predict = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        
        for stat in stats_to_predict:
            # Season average as feature
            df[f'{stat}_avg'] = df[stat]
            # Add some variance features
            df[f'{stat}_min_played_ratio'] = df['MIN'] / df['MIN'].max()
            df[f'{stat}_gp_ratio'] = df['GP'] / 82.0
        
        # Select features
        feature_cols = ['MIN', 'GP', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
        for stat in stats_to_predict:
            feature_cols.extend([f'{stat}_avg', f'{stat}_min_played_ratio', f'{stat}_gp_ratio'])
        
        # Remove duplicates and keep unique feature columns
        feature_cols = list(set(feature_cols))
        
        # Remove rows with NaN
        df_clean = df[feature_cols + stats_to_predict].dropna()
        
        print(f"Loaded {len(df_clean)} player-season records with {len(feature_cols)} features")
        
        self.feature_columns = feature_cols
        
        return df_clean, stats_to_predict
    
    def train(self, test_size=0.2, random_state=42):
        """Train Ridge Regression models for each stat"""
        print("\nTraining Player Props Prediction Models...")
        print("=" * 60)
        
        df, stats_to_predict = self.load_and_prepare_data()
        
        results = {}
        
        for stat in stats_to_predict:
            print(f"\n--- Training model for {stat} ---")
            
            X = df[self.feature_columns]
            y = df[stat]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Ridge model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"  Training MAE: {train_mae:.2f}")
            print(f"  Test MAE: {test_mae:.2f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test R²: {test_r2:.4f}")
            
            # Store model and scaler
            self.models[stat] = model
            self.scalers[stat] = scaler
            
            results[stat] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary of All Models:")
        print("=" * 60)
        for stat, metrics in results.items():
            print(f"{stat:5s}: Test MAE={metrics['test_mae']:5.2f}, Test R²={metrics['test_r2']:.4f}")
        
        return results
    
    def predict(self, features, stat_type):
        """Make predictions for a specific stat"""
        if stat_type not in self.models:
            raise ValueError(f"No model trained for {stat_type}")
        
        scaler = self.scalers[stat_type]
        model = self.models[stat_type]
        
        features_scaled = scaler.transform(features)
        return model.predict(features_scaled)
    
    def save_models(self, directory='../../models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for stat, model in self.models.items():
            filepath = os.path.join(directory, f'player_props_{stat.lower()}_model.pkl')
            joblib.dump({
                'model': model,
                'scaler': self.scalers[stat],
                'feature_columns': self.feature_columns
            }, filepath)
            print(f"Saved {stat} model to {filepath}")
    
    def load_models(self, directory='../../models'):
        """Load all trained models"""
        stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        
        for stat in stats:
            filepath = os.path.join(directory, f'player_props_{stat.lower()}_model.pkl')
            if os.path.exists(filepath):
                data = joblib.load(filepath)
                self.models[stat] = data['model']
                self.scalers[stat] = data['scaler']
                self.feature_columns = data['feature_columns']
                print(f"Loaded {stat} model from {filepath}")


def main():
    """Main execution function"""
    print("NBA Player Props Prediction Models")
    print("=" * 60)
    
    model = PlayerPropsModel()
    results = model.train()
    model.save_models()
    
    print("\n" + "=" * 60)
    print("Player props models training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

