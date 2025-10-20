"""
Retrain All Models with Combined 2006-2023 Data

Combines:
- Hallmark betting data (2006-2018) with odds
- Wyatt Walsh recent data (2019-2023) without odds

Strategy:
- Use all data for training
- Add era adjustment features
- Test on 2022-23 season
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
from sklearn.linear_model import Ridge
import joblib
import os

print("=" * 80)
print("RETRAINING MODELS WITH COMBINED 2006-2023 DATA")
print("=" * 80)

# Paths
HALLMARK_FILE = "/home/ubuntu/nba_betting_model/data/processed/games_with_betting_odds.csv"
WYATT_FILE = "/home/ubuntu/nba_betting_model/data/processed/recent_games_2019_2023.csv"
MODEL_DIR = "/home/ubuntu/nba_betting_model/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load datasets
print("\n1. Loading datasets...")
hallmark = pd.read_csv(HALLMARK_FILE)
wyatt = pd.read_csv(WYATT_FILE)

print(f"   Hallmark (2006-2018): {len(hallmark):,} games")
print(f"   Wyatt Walsh (2019-2023): {len(wyatt):,} games")

# Prepare Wyatt Walsh data to match Hallmark format
print("\n2. Preparing Wyatt Walsh data...")
wyatt_prepared = pd.DataFrame({
    'home_score': wyatt['pts_home'],
    'away_score': wyatt['pts_away'],
    'total': wyatt['pts_home'] + wyatt['pts_away'],
    'spread': wyatt['pts_home'] - wyatt['pts_away'],
    'home_win': (wyatt['pts_home'] > wyatt['pts_away']).astype(int),
    'date': pd.to_datetime(wyatt['game_date']),
    'is_recent': 1  # Flag for recent era
})

# Prepare Hallmark data
print("\n3. Preparing Hallmark data...")
hallmark_prepared = pd.DataFrame({
    'home_score': hallmark['home_pts'],
    'away_score': hallmark['away_pts'],
    'total': hallmark['actual_total'],
    'spread': hallmark['actual_spread'],
    'home_win': hallmark['home_win'],
    'date': pd.to_datetime(hallmark['game_date']),
    'is_recent': 0  # Flag for older era
})

# Combine datasets
print("\n4. Combining datasets...")
combined = pd.concat([hallmark_prepared, wyatt_prepared], ignore_index=True)
combined = combined.dropna()

print(f"   Combined total: {len(combined):,} games")
print(f"   Date range: {combined['date'].min()} to {combined['date'].max()}")

# Create features
print("\n5. Creating features...")
combined['home_avg_last5'] = combined.groupby(combined.index // 5)['home_score'].transform('mean')
combined['away_avg_last5'] = combined.groupby(combined.index // 5)['away_score'].transform('mean')
combined['total_avg_last5'] = combined.groupby(combined.index // 5)['total'].transform('mean')

# Fill NaN with overall means
combined['home_avg_last5'].fillna(combined['home_score'].mean(), inplace=True)
combined['away_avg_last5'].fillna(combined['away_score'].mean(), inplace=True)
combined['total_avg_last5'].fillna(combined['total'].mean(), inplace=True)

# Features
feature_cols = ['home_avg_last5', 'away_avg_last5', 'total_avg_last5', 'is_recent']
X = combined[feature_cols]

# Split by date for time-series validation
print("\n6. Splitting data (time-series split)...")
# Use 2022-23 season as test set
test_mask = combined['date'] >= '2022-10-01'
train_mask = ~test_mask

X_train = X[train_mask]
X_test = X[test_mask]

print(f"   Training set: {len(X_train):,} games")
print(f"   Test set: {len(X_test):,} games")

# ============================================================================
# SPREAD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SPREAD MODEL")
print("=" * 80)

y_spread = combined['spread']
y_spread_train = y_spread[train_mask]
y_spread_test = y_spread[test_mask]

spread_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
spread_model.fit(X_train, y_spread_train)

spread_pred = spread_model.predict(X_test)
spread_mae = mean_absolute_error(y_spread_test, spread_pred)
spread_rmse = np.sqrt(mean_squared_error(y_spread_test, spread_pred))

print(f"✅ Spread Model Performance:")
print(f"   MAE: {spread_mae:.2f} points")
print(f"   RMSE: {spread_rmse:.2f} points")

joblib.dump(spread_model, f"{MODEL_DIR}/spread_model.pkl")
print(f"   Saved to: {MODEL_DIR}/spread_model.pkl")

# ============================================================================
# TOTALS MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING TOTALS MODEL")
print("=" * 80)

y_total = combined['total']
y_total_train = y_total[train_mask]
y_total_test = y_total[test_mask]

totals_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
totals_model.fit(X_train, y_total_train)

total_pred = totals_model.predict(X_test)
total_mae = mean_absolute_error(y_total_test, total_pred)
total_rmse = np.sqrt(mean_squared_error(y_total_test, total_pred))

print(f"✅ Totals Model Performance:")
print(f"   MAE: {total_mae:.2f} points")
print(f"   RMSE: {total_rmse:.2f} points")
print(f"   Avg actual total: {y_total_test.mean():.1f}")
print(f"   Avg predicted total: {total_pred.mean():.1f}")

joblib.dump(totals_model, f"{MODEL_DIR}/totals_model.pkl")
print(f"   Saved to: {MODEL_DIR}/totals_model.pkl")

# ============================================================================
# MONEYLINE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING MONEYLINE MODEL")
print("=" * 80)

y_win = combined['home_win']
y_win_train = y_win[train_mask]
y_win_test = y_win[test_mask]

moneyline_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
moneyline_model.fit(X_train, y_win_train)

win_pred = moneyline_model.predict(X_test)
win_accuracy = accuracy_score(y_win_test, win_pred)

print(f"✅ Moneyline Model Performance:")
print(f"   Accuracy: {win_accuracy:.2%}")
print(f"   Actual home win rate: {y_win_test.mean():.2%}")

joblib.dump(moneyline_model, f"{MODEL_DIR}/moneyline_model.pkl")
print(f"   Saved to: {MODEL_DIR}/moneyline_model.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE - MODEL COMPARISON")
print("=" * 80)

print("\n📊 OLD MODELS (2006-2018 only):")
print("   Spread MAE: 3.80 points")
print("   Totals MAE: 4.18 points")
print("   Moneyline Accuracy: 91.22%")

print("\n📊 NEW MODELS (2006-2023 combined):")
print(f"   Spread MAE: {spread_mae:.2f} points")
print(f"   Totals MAE: {total_mae:.2f} points")
print(f"   Moneyline Accuracy: {win_accuracy:.2%}")

print("\n🎯 IMPROVEMENTS:")
spread_improvement = ((3.80 - spread_mae) / 3.80) * 100
total_improvement = ((4.18 - total_mae) / 4.18) * 100
win_improvement = ((win_accuracy - 0.9122) / 0.9122) * 100

print(f"   Spread: {spread_improvement:+.1f}%")
print(f"   Totals: {total_improvement:+.1f}%")
print(f"   Moneyline: {win_improvement:+.1f}%")

print("\n" + "=" * 80)
print("ALL MODELS RETRAINED AND SAVED!")
print("=" * 80)
print(f"\n✅ Models saved to: {MODEL_DIR}/")
print("✅ Ready to deploy to GitHub/Streamlit")

