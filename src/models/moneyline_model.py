"""
NBA Moneyline (Win Probability) Prediction Model using XGBoost

This model predicts the probability of the home team winning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os


class MoneylineModel:
    """XGBoost model for predicting NBA game winners"""
    
    def __init__(self, data_path='../../data/processed/nba_matchups_processed.csv'):
        self.data_path = data_path
        self.model = None
        self.feature_columns = None
        self.target_column = 'HOME_WIN'
        
    def load_and_prepare_data(self):
        """Load processed data and prepare features"""
        print("Loading processed data...")
        df = pd.read_csv(self.data_path)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if '_roll_' in col or 'DIFF_' in col]
        feature_cols += ['HOME_DEF_EFF', 'AWAY_DEF_EFF']
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + [self.target_column]].dropna()
        
        print(f"Loaded {len(df_clean)} samples with {len(feature_cols)} features")
        print(f"Class distribution: {df_clean[self.target_column].value_counts().to_dict()}")
        
        self.feature_columns = feature_cols
        
        X = df_clean[feature_cols]
        y = df_clean[self.target_column]
        
        return X, y
    
    def train(self, test_size=0.2, random_state=42):
        """Train the XGBoost model"""
        print("\nTraining Moneyline Prediction Model...")
        print("=" * 60)
        
        X, y = self.load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='logloss'
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        test_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        train_auc = roc_auc_score(y_train, train_pred_proba)
        test_auc = roc_auc_score(y_test, test_pred_proba)
        
        print(f"\nTraining Performance:")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  ROC AUC: {train_auc:.4f}")
        
        print(f"\nTest Performance:")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  ROC AUC: {test_auc:.4f}")
        
        print(f"\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['Away Win', 'Home Win']))
        
        print(f"\nTest Set Confusion Matrix:")
        cm = confusion_matrix(y_test, test_pred)
        print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
        print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'feature_importance': feature_importance
        }
    
    def predict(self, features):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(features)
    
    def predict_proba(self, features):
        """Predict win probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict_proba(features)
    
    def save_model(self, filepath='../../models/moneyline_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath='../../models/moneyline_model.pkl'):
        """Load a trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    print("NBA Moneyline Prediction Model")
    print("=" * 60)
    
    model = MoneylineModel()
    metrics = model.train()
    model.save_model()
    
    print("\n" + "=" * 60)
    print("Moneyline model training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

