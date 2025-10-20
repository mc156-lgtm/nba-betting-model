# NBA Betting Model: Odds, Spreads, Totals & Player Props

A comprehensive machine learning system for predicting NBA betting outcomes including spreads, totals (over/under), moneylines, and player props.

## Overview

This project implements multiple predictive models using XGBoost and Ridge Regression to forecast NBA game outcomes and player performance. The models are trained on historical NBA data with engineered features including rolling averages, efficiency metrics, and team/player trends.

## Features

### Game Predictions
- **Spread Prediction**: Predicts point spread (margin of victory)
- **Totals Prediction**: Predicts combined score (over/under)
- **Moneyline Prediction**: Predicts win probability for each team

### Player Props
- **Points Prediction**: Individual player scoring forecasts
- **Rebounds Prediction**: Rebounding performance
- **Assists Prediction**: Playmaking metrics
- **Steals & Blocks**: Defensive statistics

## Model Performance

### Spread Model (XGBoost)
- Test MAE: 2.37 points
- Test RMSE: 2.94 points
- Test R²: 0.9687

### Totals Model (XGBoost)
- Test MAE: 12.46 points
- Test RMSE: 15.82 points
- Test R²: 0.1012

### Moneyline Model (XGBoost)
- Test Accuracy: 95.00%
- Test ROC AUC: 0.9920

### Player Props Models (Ridge Regression)
- Points: Test MAE < 0.01, R² = 1.0000
- Rebounds: Test MAE < 0.01, R² = 1.0000
- Assists: Test MAE < 0.01, R² = 1.0000

*Note: These metrics are based on synthetic data. Performance on real NBA data will vary.*

## Project Structure

```
nba_betting_model/
├── data/
│   ├── raw/                    # Raw NBA data
│   └── processed/              # Processed features
├── models/                     # Trained model files (.pkl)
├── notebooks/                  # Jupyter notebooks for analysis
├── src/
│   ├── data_collection/        # Data acquisition scripts
│   │   ├── collect_nba_data.py
│   │   ├── collect_sample_data.py
│   │   └── generate_synthetic_data.py
│   ├── features/               # Feature engineering
│   │   └── build_features.py
│   ├── models/                 # Model training and prediction
│   │   ├── spread_model.py
│   │   ├── totals_model.py
│   │   ├── moneyline_model.py
│   │   ├── player_props_model.py
│   │   └── predict.py
│   └── utils/                  # Utility functions
└── README.md
```

## Installation

### Requirements
- Python 3.11+
- pandas
- numpy
- scikit-learn
- xgboost
- nba_api
- joblib

### Setup

```bash
# Clone or download the project
cd nba_betting_model

# Install dependencies
pip install pandas numpy scikit-learn xgboost nba_api joblib

# Verify installation
python -c "import pandas, numpy, sklearn, xgboost, nba_api; print('All dependencies installed!')"
```

## Usage

### 1. Data Collection

#### Using Real NBA Data (Recommended)
```bash
cd src/data_collection
python collect_nba_data.py
```

This will fetch:
- Team and player information
- Historical game data (2018-19 to current)
- Team statistics by season
- Player statistics by season

#### Using Synthetic Data (For Testing)
```bash
cd src/data_collection
python generate_synthetic_data.py
```

### 2. Feature Engineering

```bash
cd src/features
python build_features.py
```

This creates:
- Rolling averages (5, 10, 20 game windows)
- Offensive and defensive efficiency metrics
- Home/away performance splits
- Feature differences between teams

### 3. Train Models

#### Train All Models
```bash
cd src/models

# Train spread model
python spread_model.py

# Train totals model
python totals_model.py

# Train moneyline model
python moneyline_model.py

# Train player props models
python player_props_model.py
```

### 4. Make Predictions

```python
from src.models.predict import NBABettingPredictor

# Initialize predictor
predictor = NBABettingPredictor()

# Predict game outcome
home_features = {
    'PTS_roll_5': 115.0,
    'FG_PCT_roll_5': 0.475,
    'REB_roll_5': 45.0,
    # ... other features
}

away_features = {
    'PTS_roll_5': 118.0,
    'FG_PCT_roll_5': 0.485,
    'REB_roll_5': 43.0,
    # ... other features
}

predictions = predictor.predict_game(home_features, away_features)
print(predictions)

# Predict player prop
player_features = {
    'MIN': 35.0,
    'GP': 70,
    'FG_PCT': 0.485,
    # ... other features
}

prop_prediction = predictor.predict_player_prop(player_features, 'PTS')
print(prop_prediction)
```

## Feature Engineering Details

### Team-Level Features

**Rolling Averages** (5, 10, 20 game windows):
- Points (PTS)
- Field Goal Percentage (FG_PCT)
- Three-Point Percentage (FG3_PCT)
- Rebounds (REB)
- Assists (AST)
- Steals (STL)
- Blocks (BLK)
- Turnovers (TOV)
- Plus/Minus

**Efficiency Metrics**:
- Offensive Efficiency (points per 100 possessions)
- Defensive Efficiency (points allowed per 100 possessions)
- Possessions per game

**Matchup Features**:
- Feature differences between home and away teams
- Home court advantage indicators

### Player-Level Features

- Minutes per game
- Games played
- Shooting percentages (FG%, 3P%, FT%)
- Usage rate indicators
- Season averages for all major stats

## Model Architecture

### XGBoost Models (Spread, Totals, Moneyline)

**Hyperparameters**:
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

**Key Features** (by importance):
1. Defensive Efficiency (Home & Away)
2. Recent performance (5-game rolling averages)
3. Medium-term trends (10-game rolling averages)
4. Season performance (20-game rolling averages)

### Ridge Regression (Player Props)

**Hyperparameters**:
- alpha: 1.0
- StandardScaler for feature normalization

**Key Features**:
- Minutes played
- Games played
- Shooting efficiency
- Season averages

## Data Sources

### Primary Source: nba_api
Official NBA.com API client providing:
- Real-time game data
- Historical statistics
- Player and team information
- Box scores and play-by-play data

**Installation**:
```bash
pip install nba_api
```

**Documentation**: https://github.com/swar/nba_api

### Alternative Sources
- **Basketball Reference**: Historical data via web scraping
- **Sportradar**: Official NBA data provider (paid)
- **Kaggle NBA Database**: Comprehensive historical dataset

## Betting Integration

### Calculate Betting Edge

```python
# Compare model prediction to sportsbook line
prediction = 5.5  # Model predicts home team wins by 5.5
market_line = 3.0  # Sportsbook has home team -3

edge_analysis = predictor.calculate_betting_edge(
    prediction=prediction,
    market_line=market_line,
    bet_type='spread'
)

print(edge_analysis)
# Output: {'edge': 2.5, 'recommendation': 'Bet Home', 'confidence': 'Medium'}
```

### Kelly Criterion for Bet Sizing

```python
def kelly_criterion(win_probability, odds):
    """
    Calculate optimal bet size using Kelly Criterion
    
    Args:
        win_probability: Model's predicted win probability (0-1)
        odds: Decimal odds (e.g., 2.0 for +100)
    
    Returns:
        Fraction of bankroll to bet
    """
    q = 1 - win_probability
    kelly_fraction = (odds * win_probability - q) / odds
    
    # Use half-Kelly for conservative approach
    return max(0, kelly_fraction * 0.5)

# Example
win_prob = 0.65
decimal_odds = 2.0
bet_size = kelly_criterion(win_prob, decimal_odds)
print(f"Bet {bet_size:.2%} of bankroll")
```

## Limitations & Disclaimers

### Current Limitations
1. **Synthetic Data**: Current models are trained on synthetic data for demonstration
2. **API Timeouts**: NBA API may experience rate limiting or timeouts
3. **Missing Features**: Some advanced metrics (injuries, referee data, betting trends) not included
4. **Market Efficiency**: Sportsbooks have extensive resources and data access
5. **No Guarantee**: Past performance does not guarantee future results

### Important Notes
- **For Educational Purposes**: This model is for learning and research
- **Not Financial Advice**: Do not use for actual betting without extensive validation
- **Responsible Gambling**: Only bet what you can afford to lose
- **Legal Compliance**: Ensure sports betting is legal in your jurisdiction

## Roadmap & Future Enhancements

### Data Improvements
- [ ] Integrate real-time NBA API data
- [ ] Add injury reports and player availability
- [ ] Include referee assignments
- [ ] Incorporate betting market data (line movements, sharp money)
- [ ] Add schedule factors (back-to-back games, travel distance)

### Model Enhancements
- [ ] Deep learning models (LSTM for time series)
- [ ] Ensemble methods combining multiple models
- [ ] Player-specific matchup analysis
- [ ] In-game live betting predictions
- [ ] Confidence intervals for predictions

### Features
- [ ] Web dashboard for predictions
- [ ] Automated daily predictions
- [ ] Backtesting framework
- [ ] Portfolio optimization
- [ ] Alert system for high-edge bets

## Contributing

Contributions are welcome! Areas for improvement:
- Real-time data integration
- Additional feature engineering
- Model optimization
- Visualization and reporting
- Documentation improvements

## License

This project is for educational purposes. Use at your own risk.

## References

### Research & Methodologies
- [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
- [NBA-Betting/NBA_Betting](https://github.com/NBA-Betting/NBA_Betting)
- [chevyphillip/plus-ev-model](https://github.com/chevyphillip/plus-ev-model)

### Data Sources
- [nba_api](https://github.com/swar/nba_api) - Official NBA.com API client
- [Basketball Reference](https://www.basketball-reference.com/) - Historical NBA statistics
- [NBA.com Stats](https://www.nba.com/stats) - Official NBA statistics

### Machine Learning
- XGBoost: Gradient boosting framework
- Scikit-learn: Machine learning library
- Ridge Regression: Linear model with L2 regularization

## Support

For issues, questions, or contributions:
- Review the documentation
- Check existing issues
- Create a new issue with detailed information

## Acknowledgments

- NBA.com for providing comprehensive statistics
- Open source community for data tools and libraries
- Sports analytics researchers for methodological insights

---

**Disclaimer**: This model is for educational and research purposes only. Sports betting involves risk. Always gamble responsibly and within your means. Ensure compliance with local laws and regulations.

