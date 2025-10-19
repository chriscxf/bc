"""
Compare LSTM vs Neural Ensemble vs XGBoost

Quick script to compare all three approaches on your data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Import your forecasters
try:
    from neural_ensemble_forecaster import NeuralEnsembleForecaster
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False
    print("⚠️ Neural Ensemble not available")

try:
    from lstm_forecaster import MultiHorizonForecaster
    HAS_LSTM = True
except ImportError:
    HAS_LSTM = False
    print("⚠️ LSTM Forecaster not available")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not available")


def compare_models(X, y, test_size=0.1, random_state=42):
    """
    Compare LSTM, Neural Ensemble, and XGBoost on your data
    
    Args:
        X: Features (DataFrame or array)
        y: Targets (DataFrame or array)
        test_size: Test set proportion
        random_state: Random seed
    
    Returns:
        DataFrame with comparison results
    """
    print("="*80)
    print("MODEL COMPARISON: LSTM vs Neural Ensemble vs XGBoost")
    print("="*80)
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X = X.values
    else:
        feature_names = [f"X{i}" for i in range(X.shape[1])]
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values
    
    if len(y.shape) > 1:
        y = y.ravel()
    
    print(f"\nData: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    # Further split train into train/val for LSTM and ensemble
    split_idx = int(len(X_train) * 0.9)
    X_train_split = X_train[:split_idx]
    y_train_split = y_train[:split_idx]
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    
    print(f"Train: {len(X_train_split)} samples")
    print(f"Val:   {len(X_val)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    results = {}
    
    # ========================================================================
    # 1. XGBoost Baseline
    # ========================================================================
    if HAS_XGBOOST:
        print("\n" + "="*80)
        print("1. TRAINING XGBOOST (Baseline)")
        print("="*80)
        
        start_time = time.time()
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        xgb_pred = xgb_model.predict(X_test)
        xgb_time = time.time() - start_time
        
        results['XGBoost'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'MAE': mean_absolute_error(y_test, xgb_pred),
            'R2': r2_score(y_test, xgb_pred),
            'Time': xgb_time,
            'Predictions': xgb_pred
        }
        
        print(f"✓ XGBoost trained in {xgb_time:.1f}s")
        print(f"  RMSE: {results['XGBoost']['RMSE']:.6f}")
        print(f"  MAE:  {results['XGBoost']['MAE']:.6f}")
        print(f"  R²:   {results['XGBoost']['R2']:.6f}")
    
    # ========================================================================
    # 2. Neural Ensemble
    # ========================================================================
    if HAS_ENSEMBLE:
        print("\n" + "="*80)
        print("2. TRAINING NEURAL ENSEMBLE")
        print("="*80)
        
        start_time = time.time()
        
        ensemble = NeuralEnsembleForecaster(
            n_compressed_features=32,
            n_selected_features=64,
            use_neural_features=True,
            use_stacking=True,
            autoencoder_epochs=50,  # Reduced for faster comparison
            random_state=random_state
        )
        
        ensemble.fit(
            X_train_split, y_train_split,
            X_val=X_val, y_val=y_val,
            feature_names=feature_names
        )
        
        ensemble_pred = ensemble.predict(X_test)
        ensemble_time = time.time() - start_time
        
        results['Neural Ensemble'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'MAE': mean_absolute_error(y_test, ensemble_pred),
            'R2': r2_score(y_test, ensemble_pred),
            'Time': ensemble_time,
            'Predictions': ensemble_pred
        }
        
        print(f"\n✓ Neural Ensemble trained in {ensemble_time:.1f}s")
        print(f"  RMSE: {results['Neural Ensemble']['RMSE']:.6f}")
        print(f"  MAE:  {results['Neural Ensemble']['MAE']:.6f}")
        print(f"  R²:   {results['Neural Ensemble']['R2']:.6f}")
    
    # ========================================================================
    # 3. LSTM
    # ========================================================================
    if HAS_LSTM:
        print("\n" + "="*80)
        print("3. TRAINING LSTM")
        print("="*80)
        
        start_time = time.time()
        
        lstm = MultiHorizonForecaster(
            sequence_length=30,
            horizons=[1],
            hidden_size=96,
            num_layers=2,
            use_attention=True,
            use_tcn=False,
            num_heads=4
        )
        
        lstm.fit(
            X_train, y_train.reshape(-1, 1),
            batch_size=32,
            epochs=200,  # Reduced for faster comparison
            lr=2e-3,
            test_size=0.1
        )
        
        # Create dummy Y for prediction (LSTM interface requires it)
        Y_dummy = np.zeros((len(X_test), 1))
        lstm_pred = lstm.predict(X_test, Y_dummy, horizon=1)
        
        if len(lstm_pred.shape) > 1:
            lstm_pred = lstm_pred.ravel()
        
        # LSTM may return fewer predictions due to sequence length
        min_len = min(len(y_test), len(lstm_pred))
        y_test_trimmed = y_test[:min_len]
        lstm_pred_trimmed = lstm_pred[:min_len]
        
        lstm_time = time.time() - start_time
        
        results['LSTM'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test_trimmed, lstm_pred_trimmed)),
            'MAE': mean_absolute_error(y_test_trimmed, lstm_pred_trimmed),
            'R2': r2_score(y_test_trimmed, lstm_pred_trimmed),
            'Time': lstm_time,
            'Predictions': lstm_pred_trimmed
        }
        
        print(f"\n✓ LSTM trained in {lstm_time:.1f}s")
        print(f"  RMSE: {results['LSTM']['RMSE']:.6f}")
        print(f"  MAE:  {results['LSTM']['MAE']:.6f}")
        print(f"  R²:   {results['LSTM']['R2']:.6f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R²': metrics['R2'],
            'Time (s)': metrics['Time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add relative performance (vs XGBoost)
    if 'XGBoost' in results:
        xgb_rmse = results['XGBoost']['RMSE']
        comparison_df['vs XGBoost'] = comparison_df['RMSE'].apply(
            lambda x: f"{((xgb_rmse - x) / xgb_rmse * 100):+.1f}%"
        )
    
    # Sort by RMSE (best first)
    comparison_df = comparison_df.sort_values('RMSE')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find winner
    best_model = comparison_df.iloc[0]['Model']
    best_rmse = comparison_df.iloc[0]['RMSE']
    
    print("\n" + "="*80)
    print(f"🏆 WINNER: {best_model}")
    print(f"   RMSE: {best_rmse:.6f}")
    
    if best_model != 'XGBoost':
        improvement = (results['XGBoost']['RMSE'] - best_rmse) / results['XGBoost']['RMSE'] * 100
        print(f"   Beats XGBoost by: {improvement:.1f}%")
    
    print("="*80 + "\n")
    
    return comparison_df, results


def plot_predictions(results, y_test, save_path='model_comparison.png'):
    """
    Plot predictions from all models
    
    Args:
        results: Dict from compare_models
        y_test: True test values
        save_path: Where to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping plot")
        return
    
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4*len(results)))
    
    if len(results) == 1:
        axes = [axes]
    
    for ax, (model_name, metrics) in zip(axes, results.items()):
        pred = metrics['Predictions']
        
        # Handle different prediction lengths (LSTM may be shorter)
        min_len = min(len(y_test), len(pred))
        y_plot = y_test[:min_len]
        pred_plot = pred[:min_len]
        
        ax.plot(y_plot, label='Actual', alpha=0.7, linewidth=2)
        ax.plot(pred_plot, label='Predicted', alpha=0.7, linewidth=2)
        ax.set_title(f'{model_name} - RMSE: {metrics["RMSE"]:.6f}, R²: {metrics["R2"]:.4f}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL COMPARISON TOOL")
    print("="*80)
    print("\nThis script compares LSTM, Neural Ensemble, and XGBoost on your data.")
    print("\nUsage:")
    print("  from compare_models import compare_models, plot_predictions")
    print("  comparison_df, results = compare_models(X, y)")
    print("  plot_predictions(results, y_test)")
    print("\n" + "="*80 + "\n")
