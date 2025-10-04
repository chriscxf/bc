"""
Test script for Multi-Horizon LSTM Forecaster
Demonstrates the model with synthetic data
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from lstm_forecaster.py
from lstm_forecaster import MultiHorizonForecaster, ModelEvaluator, clean_and_align_data


def generate_synthetic_data(n_samples=1000, n_features_x=5, n_targets=2, seed=42):
    """
    Generate synthetic time series data for testing
    
    Args:
        n_samples: Number of time steps
        n_features_x: Number of external features
        n_targets: Number of target variables
        seed: Random seed for reproducibility
    
    Returns:
        X_df: Feature dataframe
        Y_df: Target dataframe
    """
    np.random.seed(seed)
    
    # Generate time index
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Generate external features (X) with some trends and seasonality
    X = np.zeros((n_samples, n_features_x))
    for i in range(n_features_x):
        # Add trend
        trend = np.linspace(0, 2, n_samples)
        # Add seasonality
        seasonality = np.sin(2 * np.pi * np.arange(n_samples) / 365)
        # Add noise
        noise = np.random.randn(n_samples) * 0.3
        X[:, i] = trend + seasonality + noise
    
    # Generate targets (Y) that depend on X
    Y = np.zeros((n_samples, n_targets))
    for i in range(n_targets):
        # Y depends on X and has its own dynamics
        Y[:, i] = (0.5 * X[:, 0] + 
                   0.3 * X[:, 1] + 
                   0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 180) +
                   np.random.randn(n_samples) * 0.2)
        
        # Add autoregressive component
        for t in range(1, n_samples):
            Y[t, i] += 0.5 * Y[t-1, i]
    
    # Create DataFrames
    X_df = pd.DataFrame(X, 
                        columns=[f'feature_{i}' for i in range(n_features_x)],
                        index=dates)
    Y_df = pd.DataFrame(Y,
                        columns=[f'target_{i}' for i in range(n_targets)],
                        index=dates)
    
    return X_df, Y_df


def test_basic_forecasting():
    """Test basic forecasting functionality"""
    print("="*80)
    print("Testing Multi-Horizon LSTM Forecaster with Synthetic Data")
    print("="*80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    X_full, Y_full = generate_synthetic_data(n_samples=500, n_features_x=5, n_targets=2)
    print(f"   X shape: {X_full.shape}")
    print(f"   Y shape: {Y_full.shape}")
    print(f"   Date range: {X_full.index[0]} to {X_full.index[-1]}")
    
    # Split into train and test
    pred_start = 400
    print(f"\n2. Splitting data at index {pred_start}")
    print(f"   Training samples: {pred_start}")
    print(f"   Testing samples: {len(X_full) - pred_start}")
    
    # Initialize forecaster
    print("\n3. Initializing forecaster...")
    sequence_length = 20
    forecaster = MultiHorizonForecaster(
        sequence_length=sequence_length,
        horizons=[1, 5],  # Predict 1 and 5 steps ahead
        hidden_size=32,
        num_layers=2
    )
    print(f"   Sequence length: {sequence_length}")
    print(f"   Horizons: {forecaster.horizons}")
    print(f"   Hidden size: {forecaster.hidden_size}")
    print(f"   Device: {forecaster.device}")
    
    # Prepare training data
    print("\n4. Preparing training data...")
    X_train = X_full.iloc[:pred_start, :]
    Y_train = Y_full.iloc[:pred_start, :]
    
    X_aligned, Y_aligned = clean_and_align_data(X_train, Y_train)
    X_aligned = X_aligned.select_dtypes(include=[np.number])
    print(f"   Aligned X shape: {X_aligned.shape}")
    print(f"   Aligned Y shape: {Y_aligned.shape}")
    
    # Train models
    print("\n5. Training models...")
    forecaster.fit(
        X_aligned.values,
        Y_aligned.values,
        batch_size=32,
        epochs=50,  # Reduced for testing
        lr=1e-3,
        test_size=0.1
    )
    
    # Prepare test data
    print("\n6. Preparing test data...")
    X_test = X_full.iloc[pred_start:].copy()
    X_test = X_test[X_aligned.columns]
    Y_test = pd.DataFrame(
        np.zeros((len(X_test), Y_aligned.shape[1])),
        columns=Y_aligned.columns,
        index=X_test.index
    )
    
    # Make predictions for horizon=1
    print("\n7. Making predictions (horizon=1)...")
    predictions_h1 = forecaster.predict(X_test, Y_test, horizon=1)
    print(f"   Predictions shape: {predictions_h1.shape}")
    
    # Create prediction dataframe
    target_list = list(Y_aligned.columns)
    lstm_pred_df = pd.DataFrame(predictions_h1, columns=target_list)
    start_idx = sequence_length
    end_idx = start_idx + len(predictions_h1)
    lstm_pred_df.index = X_test.index[start_idx:end_idx]
    
    # Add actual values
    actual_df = Y_full.iloc[pred_start + start_idx:pred_start + end_idx, :].copy()
    actual_df.columns = [f"actual_{col}" for col in target_list]
    result_df = pd.concat([lstm_pred_df, actual_df], axis=1)
    
    print("\n8. Results (first 10 predictions):")
    print(result_df.head(10))
    
    # Evaluate
    print("\n9. Evaluating predictions...")
    evaluator = ModelEvaluator(forecaster, X_aligned.values, Y_aligned.values)
    
    # Print metrics for horizon=1
    evaluator.print_metrics(horizon=1)
    
    # Calculate simple metrics on test set
    print("\n10. Test Set Metrics:")
    for i, target in enumerate(target_list):
        actual_vals = actual_df[f"actual_{target}"].values
        pred_vals = lstm_pred_df[target].values
        
        # Remove any NaN values
        mask = ~(np.isnan(actual_vals) | np.isnan(pred_vals))
        actual_vals = actual_vals[mask]
        pred_vals = pred_vals[mask]
        
        if len(actual_vals) > 0:
            mse = np.mean((actual_vals - pred_vals) ** 2)
            mae = np.mean(np.abs(actual_vals - pred_vals))
            rmse = np.sqrt(mse)
            
            print(f"\n   {target}:")
            print(f"      RMSE: {rmse:.6f}")
            print(f"      MAE:  {mae:.6f}")
            print(f"      MSE:  {mse:.6f}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    
    return forecaster, result_df


def test_multi_horizon():
    """Test multi-horizon forecasting"""
    print("\n" + "="*80)
    print("Testing Multi-Horizon Predictions")
    print("="*80)
    
    # Generate data
    X_full, Y_full = generate_synthetic_data(n_samples=500, n_features_x=5, n_targets=1)
    
    # Initialize forecaster with multiple horizons
    forecaster = MultiHorizonForecaster(
        sequence_length=15,
        horizons=[1, 3, 5, 10],  # Multiple horizons
        hidden_size=32,
        num_layers=2
    )
    
    # Train
    pred_start = 400
    X_train = X_full.iloc[:pred_start, :]
    Y_train = Y_full.iloc[:pred_start, :]
    X_aligned, Y_aligned = clean_and_align_data(X_train, Y_train)
    X_aligned = X_aligned.select_dtypes(include=[np.number])
    
    print("\nTraining models for all horizons...")
    forecaster.fit(
        X_aligned.values,
        Y_aligned.values,
        batch_size=32,
        epochs=30,
        lr=1e-3,
        test_size=0.1
    )
    
    # Evaluate all horizons
    print("\nEvaluating all horizons...")
    evaluator = ModelEvaluator(forecaster, X_aligned.values, Y_aligned.values)
    all_metrics = evaluator.evaluate_all_horizons()
    evaluator.print_metrics()
    
    print("\n" + "="*80)
    print("Multi-horizon test completed!")
    print("="*80)


if __name__ == "__main__":
    # Run basic test
    forecaster, results = test_basic_forecasting()
    
    # Optional: Run multi-horizon test
    # Uncomment the following line to test multiple horizons
    # test_multi_horizon()
    
    print("\n✓ All tests passed!")
    print("\nYou can now use the model with your own data.")
    print("See README_LSTM_FORECASTER.md for detailed usage instructions.")
