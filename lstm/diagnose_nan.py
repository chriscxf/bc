"""
Diagnostic script to understand NaN values in forecasting

Usage:
    from diagnose_nan import diagnose_data, compare_forecast_modes
    
    # Check your data
    diagnose_data(X_full, Y_full, pred_start)
    
    # Compare static vs rolling forecast
    compare_forecast_modes(X_full, Y_full, pred_start, target_list)
"""

import numpy as np
import pandas as pd


def diagnose_data(X_full, Y_full, pred_start, target_col=None):
    """
    Diagnose NaN issues in your data
    
    Args:
        X_full: Full feature DataFrame
        Y_full: Full target DataFrame
        pred_start: Prediction start index
        target_col: Target column to check (auto-detect if None)
    """
    print("\n" + "="*80)
    print("DATA DIAGNOSTICS")
    print("="*80)
    
    # Basic info
    print(f"\nTotal samples: {len(X_full)}")
    print(f"Training samples: {pred_start}")
    print(f"Test samples: {len(X_full) - pred_start}")
    print(f"Features: {X_full.shape[1]}")
    
    # Check training data
    X_train = X_full.iloc[:pred_start]
    Y_train = Y_full.iloc[:pred_start]
    
    print("\n" + "-"*80)
    print("TRAINING DATA (Used to fit model)")
    print("-"*80)
    
    # Features
    nan_features_train = X_train.isna().sum()
    nan_features_train = nan_features_train[nan_features_train > 0]
    
    if len(nan_features_train) > 0:
        print(f"⚠️ {len(nan_features_train)} features have NaN in training data:")
        print(nan_features_train.head(10))
        print("\n❌ PROBLEM: Model cannot train with NaN features!")
        print("   Solution: Fill NaN with X_full.fillna(method='ffill') or X_full.fillna(0)")
    else:
        print("✓ No NaN in training features")
    
    # Targets
    if target_col is None:
        target_col = Y_train.columns[0]
    
    nan_target_train = Y_train[target_col].isna().sum()
    if nan_target_train > 0:
        print(f"⚠️ {nan_target_train} NaN values in training target '{target_col}'")
        print("   Solution: Remove rows with NaN targets before training")
    else:
        print(f"✓ No NaN in training target '{target_col}'")
    
    # Check test data
    X_test = X_full.iloc[pred_start:]
    Y_test = Y_full.iloc[pred_start:]
    
    print("\n" + "-"*80)
    print("TEST DATA (What you're predicting)")
    print("-"*80)
    
    # Features
    nan_features_test = X_test.isna().sum()
    nan_features_test = nan_features_test[nan_features_test > 0]
    
    if len(nan_features_test) > 0:
        print(f"⚠️ {len(nan_features_test)} features have NaN in test data:")
        print(nan_features_test.head(10))
        
        # Find rows with NaN
        nan_rows = X_test.isna().any(axis=1)
        print(f"\n   {nan_rows.sum()} rows ({nan_rows.sum()/len(X_test)*100:.1f}%) have NaN")
        print(f"   These rows will produce NaN predictions!")
        print(f"   First NaN row: {X_test.index[nan_rows][0]}")
        
        print("\n❌ PROBLEM: Model gets NaN predictions when features are NaN!")
        print("   Solution: Fill NaN with X_full.fillna(method='ffill') or X_full.fillna(0)")
    else:
        print("✓ No NaN in test features")
    
    # Targets
    nan_target_test = Y_test[target_col].isna().sum()
    if nan_target_test > 0:
        print(f"⚠️ {nan_target_test} NaN values in test target '{target_col}'")
        print("   This is OK if you don't have future values")
        print("   But will affect metric calculation")
    else:
        print(f"✓ No NaN in test target '{target_col}'")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    issues = []
    if len(nan_features_train) > 0:
        issues.append("NaN in training features")
    if nan_target_train > 0:
        issues.append("NaN in training target")
    if len(nan_features_test) > 0:
        issues.append("NaN in test features (causes NaN predictions)")
    
    if issues:
        print("❌ Found issues:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n💡 Quick fix:")
        print("   X_full_clean = X_full.fillna(method='ffill').fillna(0)")
        print("   Y_full_clean = Y_full.fillna(method='ffill')")
    else:
        print("✓ No NaN issues found!")
    
    print("="*80 + "\n")


def compare_forecast_modes(X_full, Y_full, pred_start, target_list):
    """
    Compare static vs rolling forecast
    Shows difference between train-once vs retrain-each-step
    """
    print("\n" + "="*80)
    print("FORECAST MODE COMPARISON")
    print("="*80)
    
    print("\n📊 MODE 1: STATIC FORECAST (Current default)")
    print("-"*80)
    print("How it works:")
    print("  1. Train model ONCE on data up to pred_start")
    print("  2. Predict ALL test points at once")
    print("  3. Fast but uses old training data for later predictions")
    print("\nPros:")
    print("  ✓ Very fast")
    print("  ✓ Consistent model")
    print("  ✓ Good for short-term forecasts")
    print("\nCons:")
    print("  ✗ Doesn't adapt to new data")
    print("  ✗ Performance degrades over time")
    print("  ✗ Not realistic for production")
    print("\nCode:")
    print("  result = run_forecasting(X_full, Y_full, pred_start, target_list, X_test)")
    
    print("\n" + "="*80)
    print("\n📊 MODE 2: ROLLING FORECAST (More realistic)")
    print("-"*80)
    print("How it works:")
    print("  1. For EACH test point:")
    print("     a. Train on last 252 samples (rolling window)")
    print("     b. Predict next point")
    print("     c. Move forward one step")
    print("  2. Can retrain every N steps (e.g., every 20 days)")
    print("\nPros:")
    print("  ✓ Realistic (how you'd use it in production)")
    print("  ✓ Adapts to new data")
    print("  ✓ Better long-term accuracy")
    print("\nCons:")
    print("  ✗ VERY slow (retrains multiple times)")
    print("  ✗ For 250 test points: ~10-30 minutes")
    print("\nCode:")
    print("  result = run_rolling_forecasting(")
    print("      X_full, Y_full, pred_start, target_list,")
    print("      window_size=252,    # 1 year of daily data")
    print("      retrain_freq=20     # Retrain every 20 days")
    print("  )")
    
    print("\n" + "="*80)
    print("\n💡 RECOMMENDATION")
    print("="*80)
    
    n_test = len(Y_full) - pred_start
    
    if n_test < 50:
        print("For your test size (<50 points): Use STATIC forecast")
        print("  Fast and accurate enough for short periods")
    elif n_test < 252:
        print("For your test size (<1 year): Use STATIC or ROLLING")
        print("  STATIC: ~1 minute, good for testing")
        print("  ROLLING: ~5-15 minutes, more realistic")
    else:
        print("For your test size (>1 year): Use ROLLING for realistic results")
        print(f"  Estimated time: ~{n_test * 0.5 / 60:.0f}-{n_test * 2 / 60:.0f} minutes")
        print("  Or use STATIC for quick testing, then ROLLING for final evaluation")
    
    print("\n" + "="*80 + "\n")


def quick_fix_nan(X_full, Y_full, method='ffill'):
    """
    Quick fix for NaN values
    
    Args:
        X_full: Feature DataFrame with NaN
        Y_full: Target DataFrame with NaN
        method: 'ffill' (forward fill) or 'zero' (fill with 0)
    
    Returns:
        X_clean, Y_clean: DataFrames without NaN
    """
    print(f"\n🔧 Fixing NaN values using method='{method}'...")
    
    if method == 'ffill':
        X_clean = X_full.fillna(method='ffill').fillna(0)  # ffill then fill remaining with 0
        Y_clean = Y_full.fillna(method='ffill')
    elif method == 'zero':
        X_clean = X_full.fillna(0)
        Y_clean = Y_full.fillna(0)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ffill' or 'zero'")
    
    # Report
    nan_before_X = X_full.isna().sum().sum()
    nan_after_X = X_clean.isna().sum().sum()
    nan_before_Y = Y_full.isna().sum().sum()
    nan_after_Y = Y_clean.isna().sum().sum()
    
    print(f"✓ Fixed {nan_before_X} NaN values in features")
    print(f"✓ Fixed {nan_before_Y} NaN values in targets")
    print(f"  Remaining NaN: X={nan_after_X}, Y={nan_after_Y}")
    
    return X_clean, Y_clean


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NaN DIAGNOSTIC TOOLS")
    print("="*80)
    
    print("\n1. Diagnose your data:")
    print("""
    from diagnose_nan import diagnose_data
    
    diagnose_data(X_full, Y_full, pred_start, target_col='pnl')
    """)
    
    print("\n2. Compare forecast modes:")
    print("""
    from diagnose_nan import compare_forecast_modes
    
    compare_forecast_modes(X_full, Y_full, pred_start, target_list=['pnl'])
    """)
    
    print("\n3. Quick fix NaN:")
    print("""
    from diagnose_nan import quick_fix_nan
    
    X_clean, Y_clean = quick_fix_nan(X_full, Y_full, method='ffill')
    
    # Then use clean data
    result = run_forecasting(X_clean, Y_clean, pred_start, target_list, X_test)
    """)
    
    print("\n" + "="*80 + "\n")
