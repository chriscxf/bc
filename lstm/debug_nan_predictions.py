"""
Debug script to find why predictions are all NaN
"""

import pandas as pd
import numpy as np

def diagnose_nan_predictions(result_df, X_full=None, Y_full=None):
    """
    Comprehensive diagnosis of NaN predictions
    
    Args:
        result_df: Result DataFrame from run_forecasting
        X_full: Optional - original X data
        Y_full: Optional - original Y data
    """
    
    print("\n" + "="*80)
    print("NaN PREDICTION DIAGNOSIS")
    print("="*80)
    
    # Check 1: Result DataFrame structure
    print("\n1. RESULT DATAFRAME STRUCTURE:")
    print(f"   Shape: {result_df.shape}")
    print(f"   Columns: {result_df.columns.tolist()}")
    print(f"   Index type: {type(result_df.index)}")
    print(f"   Index range: {result_df.index[0]} to {result_df.index[-1]}")
    
    # Check 2: NaN counts in predictions
    print("\n2. NaN COUNTS:")
    for col in result_df.columns:
        nan_count = result_df[col].isna().sum()
        total = len(result_df)
        pct = (nan_count / total) * 100
        print(f"   {col:20s}: {nan_count:4d} / {total} ({pct:5.1f}%)")
    
    # Check 3: Value ranges
    print("\n3. VALUE RANGES:")
    for col in result_df.columns:
        valid_values = result_df[col].dropna()
        if len(valid_values) > 0:
            print(f"   {col:20s}: [{valid_values.min():.2e}, {valid_values.max():.2e}]")
        else:
            print(f"   {col:20s}: ALL NaN")
    
    # Check 4: First few rows
    print("\n4. FIRST 10 ROWS:")
    print(result_df.head(10))
    
    # Check 5: Data types
    print("\n5. DATA TYPES:")
    print(result_df.dtypes)
    
    # Check 6: Original data quality (if provided)
    if X_full is not None and Y_full is not None:
        print("\n6. ORIGINAL DATA QUALITY:")
        print(f"   X_full shape: {X_full.shape}")
        print(f"   Y_full shape: {Y_full.shape}")
        print(f"   X_full NaN: {X_full.isna().sum().sum()} / {X_full.size}")
        print(f"   Y_full NaN: {Y_full.isna().sum().sum()} / {Y_full.size}")
        
        # Check for inf
        if isinstance(X_full, pd.DataFrame):
            inf_count = np.isinf(X_full.values).sum()
            print(f"   X_full inf: {inf_count}")
        if isinstance(Y_full, pd.DataFrame):
            inf_count = np.isinf(Y_full.values).sum()
            print(f"   Y_full inf: {inf_count}")
    
    # Check 7: Identify where NaN starts (if predictions column exists)
    pred_cols = [c for c in result_df.columns if not c.startswith('actual_')]
    if len(pred_cols) > 0:
        pred_col = pred_cols[0]
        print(f"\n7. WHERE DO NaN PREDICTIONS START?")
        
        nan_mask = result_df[pred_col].isna()
        if nan_mask.all():
            print(f"   ALL {len(result_df)} predictions are NaN!")
            print(f"   This means the model failed to produce ANY valid predictions.")
            print(f"\n   Possible causes:")
            print(f"   1. Input data (X_full) has NaN values")
            print(f"   2. Input data has inf values")
            print(f"   3. Model training failed (check console output)")
            print(f"   4. Feature scaling produced NaN")
            print(f"   5. Autoencoder produced NaN")
        elif nan_mask.any():
            first_nan_idx = nan_mask.idxmax()
            print(f"   First NaN at index: {first_nan_idx}")
            print(f"   Total NaN: {nan_mask.sum()} / {len(result_df)}")
        else:
            print(f"   ✓ No NaN predictions!")
    
    print("\n" + "="*80)
    print("RECOMMENDED FIXES:")
    print("="*80)
    
    if X_full is not None:
        x_nan = X_full.isna().sum().sum() if isinstance(X_full, pd.DataFrame) else np.isnan(X_full).sum()
        if x_nan > 0:
            print("1. ❌ X_full has NaN values - FIX THIS FIRST!")
            print("   Solution:")
            print("   X_clean = X_full.fillna(method='ffill').fillna(0)")
    
    if Y_full is not None:
        y_nan = Y_full.isna().sum().sum() if isinstance(Y_full, pd.DataFrame) else np.isnan(Y_full).sum()
        if y_nan > 0:
            print("2. ❌ Y_full has NaN values - FIX THIS FIRST!")
            print("   Solution:")
            print("   Y_clean = Y_full.fillna(method='ffill').fillna(0)")
    
    print("\n3. Check if model training succeeded:")
    print("   Look for 'Training failed' or 'Prediction failed' messages in output")
    
    print("\n4. Try with smaller dataset first:")
    print("   pred_start = len(X_full) - 10  # Just 10 predictions")
    print("   retrain_freq = 1")
    
    print("\n5. Try single model first (faster, easier to debug):")
    print("   use_ensemble = False")
    
    print("="*80 + "\n")
    
    return result_df


# Quick usage function
def quick_check(result_df):
    """Quick check - just show if predictions are valid"""
    pred_cols = [c for c in result_df.columns if not c.startswith('actual_')]
    
    if len(pred_cols) == 0:
        print("❌ No prediction columns found!")
        return False
    
    pred_col = pred_cols[0]
    nan_count = result_df[pred_col].isna().sum()
    total = len(result_df)
    
    if nan_count == total:
        print(f"❌ ALL predictions are NaN ({total}/{total})!")
        print("\nRun diagnose_nan_predictions(result_df, X_full, Y_full) for details")
        return False
    elif nan_count > 0:
        print(f"⚠️  Some predictions are NaN ({nan_count}/{total} = {nan_count/total*100:.1f}%)")
        return False
    else:
        print(f"✓ All predictions are valid ({total}/{total})")
        print(f"  Range: [{result_df[pred_col].min():.2f}, {result_df[pred_col].max():.2f}]")
        return True


if __name__ == "__main__":
    print("Usage:")
    print("""
    from debug_nan_predictions import diagnose_nan_predictions, quick_check
    
    # Quick check
    quick_check(result_df)
    
    # Full diagnosis
    diagnose_nan_predictions(result_df, X_full, Y_full)
    """)
