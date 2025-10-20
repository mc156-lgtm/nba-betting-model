"""
NBA Totals (Over/Under) Prediction Model using XGBoost

This model predicts the total combined score for NBA games.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os


class TotalsModel:
    """XGBoost model for predicting NBA game totals"""
    
    def __init__(self, data_path='../../data/processed/nba_matchups_processed.csv'):
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        self.target_column = 'TOTAL_SCORE'
        
    def load_and_prepare_data(self):
        """Load processed data and prepare features"""
        print("Loading processed data...")
        df = pd.read_csv(self.data_path)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if '_roll_' in col]
        feature_cols += ['HOME_OFF_EFF', 'AWAY_OFF_EFF', 'HOME_DEF_EFF', 'AWAY_DEF_EFF']
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + [self.target_column]].dropna()
        
        print(f"Loaded {len(df_clean)} samples with {len(feature_cols)} features")
        
        self.feature_columns = feature_cols
        
        X = df_clean[feature_cols]
        y = df_clean[self.target_column]
        
        return X, y
    
    def train(self, test_size=0.2, random_state=42):
        """Train the XGBoost model"""
        print("\nTraining Totals Prediction Model...")
        print("=" * 60)
        
        X, y = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective='reg:squarederror'
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\nTraining Performance:")
        print(f"  MAE: {train_mae:.2f} points")
        print(f"  RMSE: {train_rmse:.2f} points")
        print(f"  R²: {train_r2:.4f}")
        
        print(f"\nTest Performance:")
        print(f"  MAE: {test_mae:.2f} points")
        print(f"  RMSE: {test_rmse:.2f} points")
        print(f"  R²: {test_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(features)
    
    def save_model(self, filepath='../../models/totals_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='../../models/totals_model.pkl'):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    print("NBA Totals Prediction Model")
    print("=" * 60)
    
    model = TotalsModel()
    metrics = model.train()
    model.save_model()
    
    print("\n" + "=" * 60)
    print("Totals model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

