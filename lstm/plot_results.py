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
            
            # Convert to millions for calculation
            pred_valid_millions = pred_valid / 1e6
            actual_valid_millions = actual_valid / 1e6
            
            mse = mean_squared_error(actual_valid_millions, pred_valid_millions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_valid_millions, pred_valid_millions)
            r2 = r2_score(actual_valid_millions, pred_valid_millions)
        else:
            print("❌ Error: All values are NaN. Cannot calculate metrics.")
            mse = rmse = mae = r2 = None
    else:
        mse = rmse = mae = r2 = None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert to millions for plotting
    predictions_millions = predictions / 1e6
    actuals_millions = actuals / 1e6 if actuals is not None else None
    
    # Plot predictions
    ax.plot(dates, predictions_millions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
    
    # Plot actuals if available
    if actuals_millions is not None:
        ax.plot(dates, actuals_millions, label='Actual', color='#EE5A6F', linewidth=2, alpha=0.8)
        
        # Shade error region
        ax.fill_between(dates, predictions_millions, actuals_millions, alpha=0.2, color='gray', label='Error')
    
    # Add metrics text box
    if rmse is not None:
        textstr = f'RMSE: {rmse:.4f}\nMAE:  {mae:.4f}\nR²:   {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value (Millions)', fontsize=12)
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
        print("FORECAST PERFORMANCE (Millions)")
        print("="*50)
        print(f"RMSE:  {rmse:.6f}")
        print(f"MAE:   {mae:.6f}")
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
                
                # Convert to millions for calculation
                pred_valid_millions = pred_valid / 1e6
                actual_valid_millions = actual_valid / 1e6
                
                rmse = np.sqrt(mean_squared_error(actual_valid_millions, pred_valid_millions))
                mae = mean_absolute_error(actual_valid_millions, pred_valid_millions)
                r2 = r2_score(actual_valid_millions, pred_valid_millions)
                
                # Convert to millions for plotting
                predictions_millions = predictions / 1e6
                actuals_millions = actuals / 1e6
                
                # Plot
                ax.plot(dates, predictions_millions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
                ax.plot(dates, actuals_millions, label='Actual', color='#EE5A6F', linewidth=2, alpha=0.8)
                ax.fill_between(dates, predictions_millions, actuals_millions, alpha=0.2, color='gray')
                
                # Metrics text
                textstr = f'RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}'
                ax.text(0.5, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', ha='center', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
            else:
                # Plot without metrics if all NaN
                predictions_millions = predictions / 1e6
                ax.plot(dates, predictions_millions, label='Forecast', color='#2E86DE', linewidth=2, alpha=0.8)
                ax.text(0.5, 0.98, 'All values are NaN', transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', ha='center', color='red')
        else:
            predictions_millions = predictions / 1e6
            ax.plot(dates, predictions_millions, label='Forecast', color='#2E86DE', linewidth=2)
        
        # Formatting
        ax.set_ylabel(f'{target} (Millions)', fontsize=11)
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
    
    # Convert to millions for calculation and plotting
    predictions_millions = predictions / 1e6
    actuals_millions = actuals / 1e6
    
    # Calculate metrics in millions scale
    rmse = np.sqrt(mean_squared_error(actuals_millions, predictions_millions))
    mae = mean_absolute_error(actuals_millions, predictions_millions)
    r2 = r2_score(actuals_millions, predictions_millions)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(actuals_millions, predictions_millions, alpha=0.5, s=30, color='#2E86DE', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actuals_millions.min(), predictions_millions.min())
    max_val = max(actuals_millions.max(), predictions_millions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Forecast')
    
    # Metrics text
    textstr = f'RMSE: {rmse:.6f}\nMAE:  {mae:.6f}\nR²:   {r2:.6f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Formatting
    ax.set_xlabel('Actual Values (Millions)', fontsize=12)
    ax.set_ylabel('Predicted Values (Millions)', fontsize=12)
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
    
    # Convert to millions for calculation
    pred_valid_millions = pred_valid / 1e6
    actual_valid_millions = actual_valid / 1e6
    
    rmse = np.sqrt(mean_squared_error(actual_valid_millions, pred_valid_millions))
    mae = mean_absolute_error(actual_valid_millions, pred_valid_millions)
    r2 = r2_score(actual_valid_millions, pred_valid_millions)
    
    print("\n" + "="*70)
    print("FINAL PERFORMANCE SUMMARY (Millions)")
    print("="*70)
    print(f"Target:        {target_col}")
    print(f"Total Samples: {n_total}")
    print(f"Valid Samples: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"RMSE:          {rmse:.6f}")
    print(f"MAE:           {mae:.6f}")
    print(f"R²:            {r2:.6f}")
    print(f"Mean Error:    {np.mean(actual_valid_millions - pred_valid_millions):.6f}")
    print(f"Std Error:     {np.std(actual_valid_millions - pred_valid_millions):.6f}")
    print("="*70 + "\n")


def save_forecast_results(result_df, target_col=None, save_dir='results', prefix='forecast'):
    """
    Save complete forecasting results to files
    
    Saves:
    1. CSV with predictions, actuals, and dates
    2. CSV with evaluation metrics
    3. Text summary report
    
    Args:
        result_df: DataFrame from run_forecasting with predictions and actuals
        target_col: Target column name (auto-detected if None)
        save_dir: Directory to save results (created if doesn't exist)
        prefix: Prefix for output filenames
    
    Returns:
        dict with paths to saved files
    """
    import os
    from datetime import datetime
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Auto-detect target column
    if target_col is None:
        target_col = [c for c in result_df.columns if not c.startswith('actual_')][0]
    
    actual_col = f'actual_{target_col}'
    
    # Prepare data
    predictions = result_df[target_col].values
    actuals = result_df[actual_col].values if actual_col in result_df.columns else None
    dates = result_df.index
    
    # Remove NaN for metrics
    if actuals is not None:
        valid_mask = ~(np.isnan(predictions) | np.isnan(actuals))
        pred_valid = predictions[valid_mask]
        actual_valid = actuals[valid_mask]
        n_valid = np.sum(valid_mask)
        n_total = len(predictions)
    else:
        n_valid = 0
        n_total = len(predictions)
    
    # Calculate metrics (in millions)
    if n_valid > 0:
        pred_millions = pred_valid / 1e6
        actual_millions = actual_valid / 1e6
        
        mse = mean_squared_error(actual_millions, pred_millions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_millions, pred_millions)
        r2 = r2_score(actual_millions, pred_millions)
        
        # Additional metrics
        mape = np.mean(np.abs((actual_valid - pred_valid) / actual_valid)) * 100
        correlation = np.corrcoef(actual_valid, pred_valid)[0, 1]
    else:
        mse = rmse = mae = r2 = mape = correlation = np.nan
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # ========================================================================
    # 1. SAVE PREDICTIONS CSV
    # ========================================================================
    predictions_file = os.path.join(save_dir, f'{prefix}_predictions_{timestamp}.csv')
    
    # Create comprehensive predictions DataFrame
    save_df = pd.DataFrame({
        'date': dates,
        'predicted': predictions,
        'actual': actuals if actuals is not None else np.nan,
        'error': (predictions - actuals) if actuals is not None else np.nan,
        'abs_error': np.abs(predictions - actuals) if actuals is not None else np.nan,
        'pct_error': ((predictions - actuals) / actuals * 100) if actuals is not None else np.nan
    })
    
    save_df.to_csv(predictions_file, index=False)
    print(f"✓ Predictions saved to: {predictions_file}")
    
    # ========================================================================
    # 2. SAVE METRICS CSV
    # ========================================================================
    metrics_file = os.path.join(save_dir, f'{prefix}_metrics_{timestamp}.csv')
    
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'MAPE', 'Correlation', 
                   'Total Samples', 'Valid Samples', 'NaN Samples'],
        'Value': [mse, rmse, mae, r2, mape, correlation, 
                  n_total, n_valid, n_total - n_valid],
        'Unit': ['M²', 'M', 'M', '', '%', '', '', '', '']
    })
    
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # ========================================================================
    # 3. SAVE TEXT SUMMARY
    # ========================================================================
    summary_file = os.path.join(save_dir, f'{prefix}_summary_{timestamp}.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FORECAST RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Prediction period: {dates[0]} to {dates[-1]}\n")
        f.write(f"Total predictions: {n_total}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("PERFORMANCE METRICS (Millions)\n")
        f.write("-"*80 + "\n\n")
        
        if n_valid > 0:
            f.write(f"Valid samples:   {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)\n")
            f.write(f"NaN samples:     {n_total - n_valid} / {n_total} ({(n_total-n_valid)/n_total*100:.1f}%)\n\n")
            
            f.write(f"MSE:             {mse:.6f} M²\n")
            f.write(f"RMSE:            {rmse:.6f} M\n")
            f.write(f"MAE:             {mae:.6f} M\n")
            f.write(f"R²:              {r2:.6f}\n")
            f.write(f"MAPE:            {mape:.2f}%\n")
            f.write(f"Correlation:     {correlation:.6f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PREDICTION STATISTICS (Original Scale)\n")
            f.write("-"*80 + "\n\n")
            
            f.write(f"Predicted Mean:  {pred_valid.mean():.2f}\n")
            f.write(f"Predicted Std:   {pred_valid.std():.2f}\n")
            f.write(f"Predicted Min:   {pred_valid.min():.2f}\n")
            f.write(f"Predicted Max:   {pred_valid.max():.2f}\n\n")
            
            f.write(f"Actual Mean:     {actual_valid.mean():.2f}\n")
            f.write(f"Actual Std:      {actual_valid.std():.2f}\n")
            f.write(f"Actual Min:      {actual_valid.min():.2f}\n")
            f.write(f"Actual Max:      {actual_valid.max():.2f}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("ERROR STATISTICS\n")
            f.write("-"*80 + "\n\n")
            
            errors = pred_valid - actual_valid
            f.write(f"Error Mean:      {errors.mean():.2f}\n")
            f.write(f"Error Std:       {errors.std():.2f}\n")
            f.write(f"Error Min:       {errors.min():.2f}\n")
            f.write(f"Error Max:       {errors.max():.2f}\n")
            f.write(f"Abs Error Mean:  {np.abs(errors).mean():.2f}\n\n")
            
        else:
            f.write(f"⚠️ WARNING: All {n_total} predictions are NaN!\n")
            f.write(f"Cannot calculate metrics.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("SAMPLE PREDICTIONS (First 10)\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'Date':<20} {'Predicted':>15} {'Actual':>15} {'Error':>15}\n")
        f.write("-"*70 + "\n")
        
        for i in range(min(10, len(save_df))):
            row = save_df.iloc[i]
            f.write(f"{str(row['date']):<20} {row['predicted']:>15.2f} {row['actual']:>15.2f} {row['error']:>15.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    # ========================================================================
    # 4. PRINT SUMMARY TO CONSOLE
    # ========================================================================
    print("\n" + "="*80)
    print("FORECAST RESULTS SUMMARY")
    print("="*80)
    print(f"Target: {target_col}")
    print(f"Period: {dates[0]} to {dates[-1]}")
    print(f"Total predictions: {n_total}")
    
    if n_valid > 0:
        print(f"\nPerformance Metrics (Millions):")
        print(f"  RMSE:        {rmse:.6f}")
        print(f"  MAE:         {mae:.6f}")
        print(f"  R²:          {r2:.6f}")
        print(f"  MAPE:        {mape:.2f}%")
        print(f"  Correlation: {correlation:.6f}")
        print(f"\nValid samples: {n_valid} / {n_total} ({n_valid/n_total*100:.1f}%)")
    else:
        print(f"\n⚠️ WARNING: All predictions are NaN!")
    
    print("="*80 + "\n")
    
    # Return file paths
    return {
        'predictions': predictions_file,
        'metrics': metrics_file,
        'summary': summary_file,
        'directory': save_dir
    }


# Backward compatibility wrapper
def create_full_report(y_true=None, y_pred=None, timestamps=None, target_col='return',
                      result_df=None, save_dir='results', show_plots=False):
    """
    Create comprehensive report with all plots and saved results
    
    Args:
        y_true: Actual values (or use result_df)
        y_pred: Predicted values (or use result_df)
        timestamps: Dates/indices (or use result_df)
        target_col: Target column name
        result_df: DataFrame from run_forecasting (alternative to y_true/y_pred)
        save_dir: Directory to save results
        show_plots: Whether to display plots
    
    Returns:
        dict with paths to all saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle input formats
    if result_df is not None:
        # Use DataFrame
        saved_files = save_forecast_results(result_df, target_col=target_col, 
                                           save_dir=save_dir, prefix='forecast')
        
        # Extract arrays for plotting
        if target_col is None:
            target_col = [c for c in result_df.columns if not c.startswith('actual_')][0]
        
        y_pred = result_df[target_col].values
        actual_col = f'actual_{target_col}'
        y_true = result_df[actual_col].values if actual_col in result_df.columns else None
        timestamps = result_df.index
        
    else:
        # Create DataFrame from arrays
        result_df = pd.DataFrame({
            target_col: y_pred,
            f'actual_{target_col}': y_true,
        }, index=timestamps)
        
        saved_files = save_forecast_results(result_df, target_col=target_col,
                                           save_dir=save_dir, prefix='forecast')
    
    # Create plots
    plot_files = []
    
    # 1. Forecast plot
    fig, ax = plot_forecast(result_df, target_col=target_col, 
                           save_path=os.path.join(save_dir, 'forecast_plot.png'))
    plot_files.append(os.path.join(save_dir, 'forecast_plot.png'))
    if not show_plots:
        plt.close(fig)
    
    # 2. Scatter plot (if valid predictions exist)
    valid_mask = ~(np.isnan(y_pred) | (np.isnan(y_true) if y_true is not None else False))
    if np.sum(valid_mask) > 0 and y_true is not None:
        fig, ax = plot_scatter(y_pred[valid_mask], y_true[valid_mask],
                              save_path=os.path.join(save_dir, 'scatter_plot.png'))
        plot_files.append(os.path.join(save_dir, 'scatter_plot.png'))
        if not show_plots:
            plt.close(fig)
        
        # 3. Error distribution
        fig, ax = plot_error_distribution(y_pred[valid_mask], y_true[valid_mask],
                                          save_path=os.path.join(save_dir, 'error_distribution.png'))
        plot_files.append(os.path.join(save_dir, 'error_distribution.png'))
        if not show_plots:
            plt.close(fig)
    
    saved_files['plots'] = plot_files
    
    print(f"\n✓ Complete report saved to: {save_dir}/")
    print(f"  - Predictions CSV")
    print(f"  - Metrics CSV")
    print(f"  - Text summary")
    print(f"  - {len(plot_files)} plots")
    
    return saved_files


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
