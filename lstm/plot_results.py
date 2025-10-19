"""
Simple plotting and evaluation script for forecasting results

Usage:
    from plot_results import plot_forecast
    
    # Option 1: From result DataFrame
    plot_forecast(result_df, target_col='return')
    
    # Option 2: From separate arrays
    plot_forecast(predictions, actuals, dates=None)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def plot_forecast(data, actuals=None, target_col=None, dates=None, save_path=None):
    """
    Plot forecasting results with performance metrics
    
    Args:
        data: Either a result DataFrame or prediction array/Series
        actuals: Actual values (if data is predictions array)
        target_col: Target column name (if data is DataFrame)
        dates: Optional date index for x-axis
        save_path: Path to save plot (e.g., 'forecast.png')
    
    Examples:
        # From DataFrame (result_df from run_forecasting)
        plot_forecast(result_df, target_col='return')
        
        # From arrays
        plot_forecast(predictions, actuals=y_test, dates=test_dates)
    """
    
    # Parse input
    if isinstance(data, pd.DataFrame):
        # DataFrame with predictions and actuals
        if target_col is None:
            target_col = [c for c in data.columns if not c.startswith('actual_')][0]
        
        predictions = data[target_col].values
        actual_col = f'actual_{target_col}' if f'actual_{target_col}' in data.columns else target_col
        actuals = data[actual_col].values if actual_col in data.columns else None
        dates = data.index
        title = f'Forecast vs Actual: {target_col}'
    else:
        # Arrays
        predictions = np.array(data).flatten()
        actuals = np.array(actuals).flatten() if actuals is not None else None
        if dates is None:
            dates = np.arange(len(predictions))
        title = 'Forecast vs Actual'
    
    # Remove NaN values for metrics calculation
    if actuals is not None:
        # Create mask for valid (non-NaN) values
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        n_total = len(predictions)
        n_valid = np.sum(valid_mask)
        n_nan = n_total - n_valid
        
        if n_nan > 0:
            print(f"⚠️ Warning: Found {n_nan} NaN values ({n_nan/n_total*100:.1f}%). Using {n_valid} valid samples for metrics.")
        
        if n_valid > 0:
            pred_valid = predictions[valid_mask]
            actual_valid = actuals[valid_mask]
            
            mse = mean_squared_error(actual_valid, pred_valid)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_valid, pred_valid)
            r2 = r2_score(actual_valid, pred_valid)
        else:
            print("❌ Error: All values are NaN. Cannot calculate metrics.")
            mse = rmse = mae = r2 = None
    else:
        mse = rmse = mae = r2 = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot predictions
    ax.plot(dates, predictions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
    
    # Plot actuals if available
    if actuals is not None:
        ax.plot(dates, actuals, label='Actual', color='#EE5A6F', linewidth=2, alpha=0.8)
        
        # Shade error region
        ax.fill_between(dates, predictions, actuals, alpha=0.2, color='gray', label='Error')
    
    # Add metrics text box
    if rmse is not None:
        # Convert to millions for display
        rmse_millions = rmse / 1e6
        mae_millions = mae / 1e6
        textstr = f'RMSE: {rmse_millions:.4f}M\nMAE:  {mae_millions:.4f}M\nR²:   {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    # Print metrics
    if rmse is not None:
        print("\n" + "="*50)
        print("FORECAST PERFORMANCE")
        print("="*50)
        print(f"RMSE:  {rmse/1e6:.6f} M")
        print(f"MAE:   {mae/1e6:.6f} M")
        print(f"R²:    {r2:.6f}")
        print("="*50 + "\n")
    
    return fig, ax


def plot_multiple_forecasts(result_df, target_cols=None, save_path=None):
    """
    Plot multiple targets in subplots
    
    Args:
        result_df: DataFrame with multiple prediction columns
        target_cols: List of target column names (auto-detect if None)
        save_path: Path to save plot
    """
    # Auto-detect target columns
    if target_cols is None:
        target_cols = [c for c in result_df.columns if not c.startswith('actual_')]
    
    n_targets = len(target_cols)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 4*n_targets))
    
    if n_targets == 1:
        axes = [axes]
    
    for idx, target in enumerate(target_cols):
        ax = axes[idx]
        
        # Get data
        predictions = result_df[target].values
        actual_col = f'actual_{target}'
        actuals = result_df[actual_col].values if actual_col in result_df.columns else None
        dates = result_df.index
        
        # Calculate metrics
        if actuals is not None:
            # Remove NaN values
            valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
            if np.sum(valid_mask) > 0:
                pred_valid = predictions[valid_mask]
                actual_valid = actuals[valid_mask]
                
                rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
                mae = mean_absolute_error(actual_valid, pred_valid)
                r2 = r2_score(actual_valid, pred_valid)
                
                # Plot
                ax.plot(dates, predictions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
                ax.plot(dates, actuals, label='Actual', color='#EE5A6F', linewidth=2, alpha=0.8)
                ax.fill_between(dates, predictions, actuals, alpha=0.2, color='gray')
                
                # Metrics text
                rmse_millions = rmse / 1e6
                mae_millions = mae / 1e6
                textstr = f'RMSE: {rmse_millions:.4f}M  MAE: {mae_millions:.4f}M  R²: {r2:.4f}'
                ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', ha='center', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
            else:
                # Plot without metrics if all NaN
                ax.plot(dates, predictions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
                ax.text(0.5, 0.98, 'All values are NaN', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', ha='center', color='red')
        else:
            ax.plot(dates, predictions, label='Forecast', color='#2E86DE', linewidth=2)
        
        # Formatting
        ax.set_ylabel(target, fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if idx == 0:
            ax.set_title('Multi-Target Forecasts', fontsize=14, fontweight='bold')
        if idx == n_targets - 1:
            ax.set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, axes


def plot_error_distribution(predictions, actuals, save_path=None):
    """
    Plot error distribution histogram
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        save_path: Path to save plot
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    n_nan = np.sum(~valid_mask)
    
    if n_nan > 0:
        print(f"⚠️ Warning: Removed {n_nan} NaN values from error calculation.")
    
    if np.sum(valid_mask) == 0:
        print("❌ Error: All values are NaN. Cannot plot error distribution.")
        return None, None
    
    errors = actuals[valid_mask] - predictions[valid_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(errors, bins=50, color='#54A0FF', alpha=0.7, edgecolor='black')
    
    # Add vertical line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    # Statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    textstr = f'Mean Error: {mean_error:.6f}\nStd Error:  {std_error:.6f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', ha='right', bbox=props, family='monospace')
    
    # Formatting
    ax.set_xlabel('Prediction Error (Actual - Forecast)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


def plot_scatter(predictions, actuals, save_path=None):
    """
    Scatter plot of predictions vs actuals with perfect prediction line
    
    Args:
        predictions: Predicted values
        actuals: Actual values
        save_path: Path to save plot
    """
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    n_nan = np.sum(~valid_mask)
    
    if n_nan > 0:
        print(f"⚠️ Warning: Removed {n_nan} NaN values from scatter plot.")
    
    if np.sum(valid_mask) == 0:
        print("❌ Error: All values are NaN. Cannot create scatter plot.")
        return None, None
    
    predictions = predictions[valid_mask]
    actuals = actuals[valid_mask]
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(actuals, predictions, alpha=0.5, s=30, color='#2E86DE', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Forecast')
    
    # Metrics text
    rmse_millions = rmse / 1e6
    mae_millions = mae / 1e6
    textstr = f'RMSE: {rmse_millions:.6f}M\nMAE:  {mae_millions:.6f}M\nR²:   {r2:.6f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Formatting
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title('Forecast vs Actual Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax


def create_full_report(result_df, target_col=None, save_dir=None):
    """
    Create complete visualization report with all plots
    
    Args:
        result_df: Result DataFrame from forecasting
        target_col: Target column name (auto-detect if None)
        save_dir: Directory to save plots (show if None)
    """
    # Auto-detect target
    if target_col is None:
        target_col = [c for c in result_df.columns if not c.startswith('actual_')][0]
    
    predictions = result_df[target_col].values
    actual_col = f'actual_{target_col}'
    actuals = result_df[actual_col].values if actual_col in result_df.columns else None
    
    if actuals is None:
        print("⚠️ No actual values found. Can only plot forecasts.")
        plot_forecast(result_df, target_col=target_col, save_path=f'{save_dir}/forecast.png' if save_dir else None)
        return
    
    # Check for valid data
    valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
    n_valid = np.sum(valid_mask)
    n_total = len(predictions)
    
    if n_valid == 0:
        print("❌ Error: All values are NaN. Cannot generate report.")
        return
    
    if n_valid < n_total:
        print(f"⚠️ Warning: {n_total - n_valid} NaN values found ({(n_total-n_valid)/n_total*100:.1f}%). Using {n_valid} valid samples.")
    
    print("\n" + "="*70)
    print("GENERATING FORECAST VISUALIZATION REPORT")
    print("="*70)
    
    # 1. Time series plot
    print("\n[1/4] Time series plot...")
    plot_forecast(result_df, target_col=target_col, 
                 save_path=f'{save_dir}/forecast_timeseries.png' if save_dir else None)
    
    # 2. Scatter plot
    print("[2/4] Scatter plot...")
    plot_scatter(predictions, actuals, 
                save_path=f'{save_dir}/forecast_scatter.png' if save_dir else None)
    
    # 3. Error distribution
    print("[3/4] Error distribution...")
    plot_error_distribution(predictions, actuals, 
                           save_path=f'{save_dir}/forecast_errors.png' if save_dir else None)
    
    # 4. Summary metrics
    print("[4/4] Summary metrics...")
    
    # Calculate on valid data only
    pred_valid = predictions[valid_mask]
    actual_valid = actuals[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
    mae = mean_absolute_error(actual_valid, pred_valid)
    r2 = r2_score(actual_valid, pred_valid)
    
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Target:        {target_col}")
    print(f"Total Samples: {n_total}")
    print(f"Valid Samples: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"RMSE:          {rmse/1e6:.6f} M")
    print(f"MAE:           {mae/1e6:.6f} M")
    print(f"R²:            {r2:.6f}")
    print(f"Mean Error:    {np.mean(actual_valid - pred_valid)/1e6:.6f} M")
    print(f"Std Error:     {np.std(actual_valid - pred_valid)/1e6:.6f} M")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FORECAST PLOTTING EXAMPLES")
    print("="*70)
    
    print("\n1. Simple time series plot:")
    print("""
    from plot_results import plot_forecast
    
    # From result DataFrame
    plot_forecast(result_df, target_col='return', save_path='forecast.png')
    
    # From arrays
    plot_forecast(predictions, actuals=y_test, dates=test_dates)
    """)
    
    print("\n2. Multiple targets:")
    print("""
    from plot_results import plot_multiple_forecasts
    
    plot_multiple_forecasts(result_df, save_path='multi_forecast.png')
    """)
    
    print("\n3. Error analysis:")
    print("""
    from plot_results import plot_error_distribution, plot_scatter
    
    plot_error_distribution(predictions, actuals)
    plot_scatter(predictions, actuals)
    """)
    
    print("\n4. Complete report:")
    print("""
    from plot_results import create_full_report
    
    # Creates all plots + metrics summary
    create_full_report(result_df, target_col='return', save_dir='./plots')
    """)
    
    print("\n" + "="*70)
    print("Ready to visualize! 📊")
    print("="*70 + "\n")
