# NBA Betting Model - Quick Start Guide

Get up and running with the NBA betting model in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Terminal/Command line access

## Step 1: Install Dependencies (2 minutes)

```bash
pip install pandas numpy scikit-learn xgboost nba_api joblib
```

## Step 2: Navigate to Project Directory

```bash
cd nba_betting_model
```

## Step 3: Generate Sample Data (1 minute)

Since the NBA API may have timeouts, start with synthetic data:

```bash
cd src/data_collection
python generate_synthetic_data.py
```

**Output**: Creates 4 seasons of synthetic NBA data in `data/raw/`

## Step 4: Build Features (1 minute)

```bash
cd ../features
python build_features.py
```

**Output**: Creates processed matchup data with rolling averages in `data/processed/`

## Step 5: Train Models (2 minutes)

```bash
cd ../models

# Train all models
python spread_model.py
python totals_model.py
python moneyline_model.py
python player_props_model.py
```

**Output**: Trained models saved to `models/` directory

## Step 6: Test Predictions

```bash
python predict.py
```

**Output**: Loads all models and confirms they're ready for predictions

## Making Your First Prediction

### Python Script

Create `test_prediction.py`:

```python
from src.models.spread_model import SpreadModel
from src.models.totals_model import TotalsModel
from src.models.moneyline_model import MoneylineModel

# Load models
spread_model = SpreadModel()
spread_model.load_model()

totals_model = TotalsModel()
totals_model.load_model()

moneyline_model = MoneylineModel()
moneyline_model.load_model()

print("Models loaded successfully!")
print("\nTo make predictions, you need:")
print("1. Home team rolling averages (last 5, 10, 20 games)")
print("2. Away team rolling averages")
print("3. Defensive efficiency for both teams")
print("\nThese features come from the feature engineering pipeline.")
```

Run it:
```bash
python test_prediction.py
```

## Next Steps

### Use Real NBA Data

Replace synthetic data with real NBA statistics:

```bash
cd src/data_collection
python collect_nba_data.py
```

**Note**: This may take 10-15 minutes and requires stable internet connection.

### Customize Models

Edit model hyperparameters in:
- `src/models/spread_model.py`
- `src/models/totals_model.py`
- `src/models/moneyline_model.py`
- `src/models/player_props_model.py`

### Add More Features

Modify `src/features/build_features.py` to add:
- Injury data
- Referee assignments
- Rest days
- Travel distance
- Betting market data

## Common Issues

### Issue: NBA API Timeout
**Solution**: Use synthetic data generator or try again later

### Issue: Missing Dependencies
**Solution**: Run `pip install -r requirements.txt` (if available) or install packages individually

### Issue: Model Not Found
**Solution**: Ensure you've run the training scripts first

### Issue: Feature Mismatch
**Solution**: Retrain models after changing feature engineering

## File Structure Quick Reference

```
nba_betting_model/
├── data/
│   ├── raw/                    # Raw data goes here
│   └── processed/              # Processed features
├── models/                     # Trained models (.pkl files)
├── src/
│   ├── data_collection/        # Data scripts
│   ├── features/               # Feature engineering
│   └── models/                 # Model training & prediction
└── README.md                   # Full documentation
```

## Performance Expectations

### With Synthetic Data
- Spread Model: ~2.4 point MAE
- Totals Model: ~12.5 point MAE
- Moneyline Model: ~95% accuracy

### With Real Data
Performance will vary based on:
- Data quality and completeness
- Feature engineering
- Model hyperparameters
- Market efficiency

## Getting Help

1. Read the full README.md
2. Check model training output for errors
3. Verify data files exist in correct directories
4. Ensure all dependencies are installed

## What's Next?

- **Explore**: Review model performance metrics
- **Experiment**: Try different features and hyperparameters
- **Integrate**: Add real-time data feeds
- **Validate**: Backtest on historical games
- **Deploy**: Create a prediction dashboard

---

**Time to First Prediction**: ~5 minutes  
**Time to Production-Ready Model**: Several hours to days (with real data and validation)

Happy modeling! 🏀📊

