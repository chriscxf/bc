# Understanding NaN in Forecasting Results

## 🎯 Your Question: "Is the model retraining for every time point?"

**Short Answer: NO** - By default, `run_forecasting()` trains the model **ONCE** and predicts all test points at once.

## 📊 Two Forecast Modes

### Mode 1: **STATIC Forecast** (Current Default)
```python
result = run_forecasting(X_full, Y_full, pred_start, target_list, X_to_test)
```

**How it works:**
```
Training data: [0 .................. pred_start]
                                        ↓ Train ONCE
Test data:                             [pred_start .......... end]
                                        ↓ Predict ALL at once
```

- ✅ **Pros**: Very fast (trains once)
- ❌ **Cons**: Model doesn't update with new data
- ⏱️ **Speed**: ~2-5 minutes total

### Mode 2: **ROLLING Forecast** (Realistic)
```python
result = run_rolling_forecasting(
    X_full, Y_full, pred_start, target_list,
    window_size=252,     # Use last 252 days
    retrain_freq=20      # Retrain every 20 steps
)
```

**How it works:**
```
Step 1: Train on [0.........pred_start] → Predict point pred_start
Step 2: Train on [1.........pred_start+1] → Predict point pred_start+1
Step 3: Train on [2.........pred_start+2] → Predict point pred_start+2
...
Step N: Train on [N.........end-1] → Predict point end
```

- ✅ **Pros**: Realistic, adapts to new data
- ❌ **Cons**: VERY slow (retrains many times)
- ⏱️ **Speed**: ~10-30 minutes for 250 test points

## 🔍 Why Are There NaN Values?

NaN values in your predictions come from **NaN in your input features**, NOT from the forecast mode.

### Common Sources of NaN:

1. **Missing feature values**
   ```python
   X_full.isna().sum()  # Check which features have NaN
   ```

2. **Index misalignment**
   ```python
   # If X_to_test has more rows than Y_full after pred_start
   len(X_to_test)  # 300
   len(Y_full) - pred_start  # 280
   # Last 20 predictions will have no actuals (NaN)
   ```

3. **Feature engineering artifacts**
   ```python
   # Example: Rolling window features
   df['ma_20'] = df['price'].rolling(20).mean()  # First 20 rows = NaN
   ```

## 🔧 Diagnosis Steps

### Step 1: Diagnose Your Data
```python
from diagnose_nan import diagnose_data

diagnose_data(X_full, Y_full, pred_start, target_col='pnl')
```

**Output:**
```
DATA DIAGNOSTICS
================
Total samples: 1250
Training samples: 900
Test samples: 350
Features: 250

TRAINING DATA
-------------
⚠️ 15 features have NaN in training data:
   feature_123: 50 NaN
   feature_45: 20 NaN
   ...
   
❌ PROBLEM: Model cannot train with NaN features!
   Solution: Fill NaN with X_full.fillna(method='ffill') or X_full.fillna(0)

TEST DATA
---------
⚠️ 20 features have NaN in test data:
   150 rows (42.9%) have NaN
   These rows will produce NaN predictions!
```

### Step 2: Quick Fix
```python
from diagnose_nan import quick_fix_nan

# Forward fill NaN values
X_clean, Y_clean = quick_fix_nan(X_full, Y_full, method='ffill')

# Now run forecasting
result = run_forecasting(X_clean, Y_clean, pred_start, target_list, X_to_test)
```

### Step 3: Verify No NaN
```python
# Check if NaN is gone
print(f"NaN in predictions: {result['pnl'].isna().sum()}")
print(f"NaN in actuals: {result['actual_pnl'].isna().sum()}")
```

## 📈 Which Mode Should You Use?

### Quick Testing → Use **STATIC**
```python
# Fast, good for initial testing
result = run_forecasting(X_clean, Y_clean, pred_start, target_list, X_to_test)
plot_forecast(result, target_col='pnl', save_path='static_forecast.png')
```

### Realistic Evaluation → Use **ROLLING**
```python
# Slower but realistic (how you'd use in production)
result = run_rolling_forecasting(
    X_clean, Y_clean, pred_start, target_list,
    window_size=252,    # 1 year training window
    retrain_freq=20     # Retrain every 20 days (faster)
)
plot_forecast(result, target_col='pnl', save_path='rolling_forecast.png')
```

### Comparison Table

| Aspect | Static Forecast | Rolling Forecast |
|--------|----------------|------------------|
| **Training** | Once (at pred_start) | Multiple times (rolling window) |
| **Speed** | Fast (~2-5 min) | Slow (~10-30 min) |
| **Realism** | Low (uses old data) | High (adapts to new data) |
| **Use Case** | Quick testing | Production/final eval |
| **Accuracy** | Good short-term | Better long-term |

## 💡 Example Workflow

```python
# 1. Load data
import pandas as pd
from neural_ensemble_forecaster import run_forecasting, run_rolling_forecasting
from diagnose_nan import diagnose_data, quick_fix_nan
from plot_results import plot_forecast

# 2. Diagnose NaN issues
diagnose_data(X_full, Y_full, pred_start=900, target_col='pnl')

# 3. Fix NaN if needed
X_clean, Y_clean = quick_fix_nan(X_full, Y_full, method='ffill')

# 4. Quick test with STATIC forecast
result_static = run_forecasting(
    X_clean, Y_clean, 
    pred_start=900, 
    target_list=['pnl'], 
    X_to_test=X_clean.iloc[900:]
)
plot_forecast(result_static, target_col='pnl', save_path='test_static.png')

# 5. If results look good, run ROLLING forecast for realistic evaluation
result_rolling = run_rolling_forecasting(
    X_clean, Y_clean, 
    pred_start=900, 
    target_list=['pnl'],
    window_size=252,
    retrain_freq=20  # Retrain every 20 days (balance speed vs accuracy)
)
plot_forecast(result_rolling, target_col='pnl', save_path='final_rolling.png')

# 6. Compare
print("\nSTATIC vs ROLLING Comparison:")
print(f"Static RMSE:  {result_static['pnl'].std():.6f}")
print(f"Rolling RMSE: {result_rolling['pnl'].std():.6f}")
```

## 🎯 Summary

**Your NaN issue is likely from:**
1. ✅ NaN in input features (most common)
2. ✅ Index misalignment between X and Y
3. ❌ NOT from the forecast mode

**Solution:**
```python
# Fix NaN first
X_clean, Y_clean = quick_fix_nan(X_full, Y_full, method='ffill')

# Then forecast (choose mode)
result = run_forecasting(X_clean, Y_clean, pred_start, target_list, X_to_test)  # Fast
# OR
result = run_rolling_forecasting(X_clean, Y_clean, pred_start, target_list)  # Realistic
```

**The forecast mode determines:**
- How often model retrains (once vs many times)
- How realistic the predictions are
- How long it takes

**But NaN comes from data quality, not forecast mode!** 🎯
