"""
Multi-Horizon LSTM Forecaster
Complete implementation for multivariate time series forecasting
"""

from src.data_loader import DataLoader as src_DataLoader
from src.feature_selection import execute_feature_select
import os
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def _load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.getcwd(), "config_cf.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    return {}


class ModelEvaluator:
    """Class for evaluating and plotting real vs predicted values"""
    def __init__(self, forecaster, X, Y):
        self.forecaster = forecaster
        self.X = X
        self.Y = Y
        self.results = {}
    
    def get_actual_values(self, horizon):
        """Get actual values aligned with predictions"""
        # Calculate the starting index for actual values
        start_idx = self.forecaster.sequence_length + horizon - 1
        end_idx = len(self.Y) - self.forecaster.sequence_length - horizon + 1 + start_idx
        return self.Y[start_idx:end_idx]
    
    def evaluate_horizon(self, horizon, target_col=0):
        """Evaluate predictions for a specific horizon"""
        if horizon not in self.forecaster.models:
            raise ValueError(f"No model trained for horizon {horizon}")

        # Get predictions
        predictions = self.forecaster.predict(self.X, self.Y, horizon)
        # Get actual values
        actual = self.get_actual_values(horizon)
        # Align lengths (predictions might be shorter due to sequence requirements)
        min_len = min(len(predictions), len(actual))
        predictions = predictions[:min_len]
        actual = actual[:min_len]
        
        # Calculate metrics for each target column
        metrics = {}
        for col in range(predictions.shape[1]):
            y_true = actual[:, col]
            y_pred = predictions[:, col]
            metrics[f'target_{col}'] = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
                'correlation': np.corrcoef(y_true, y_pred)[0, 1]
            }
        
        self.results[horizon] = {
            'predictions': predictions,
            'actual': actual,
            'metrics': metrics
        }
        return metrics
    
    def evaluate_all_horizons(self):
        """Evaluate all trained horizons"""
        all_metrics = {}
        for horizon in self.forecaster.horizons:
            all_metrics[horizon] = self.evaluate_horizon(horizon)
        return all_metrics
    
    def plot_predictions(self, horizon, target_col=0, save_path=None):
        """Plot actual vs predicted values for a specific horizon and target"""
        if horizon not in self.results:
            self.evaluate_horizon(horizon)
        
        actual = self.results[horizon]['actual'][:, target_col]
        predictions = self.results[horizon]['predictions'][:, target_col]
        
        plt.figure(figsize=(15, 6))
        plt.plot(actual, label='Actual', linewidth=2, alpha=0.8)
        plt.plot(predictions, label='Predicted', linewidth=2, alpha=0.8)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'Horizon {horizon} - Target {target_col}: Actual vs Predicted', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_all_targets(self, horizon, save_path=None):
        """Plot all target variables for a specific horizon"""
        if horizon not in self.results:
            self.evaluate_horizon(horizon)
        
        actual = self.results[horizon]['actual']
        predictions = self.results[horizon]['predictions']
        n_targets = actual.shape[1]
        
        fig, axes = plt.subplots(n_targets, 1, figsize=(15, 5 * n_targets))
        if n_targets == 1:
            axes = [axes]
        
        for i in range(n_targets):
            axes[i].plot(actual[:, i], label='Actual', linewidth=2, alpha=0.8)
            axes[i].plot(predictions[:, i], label='Predicted', linewidth=2, alpha=0.8)
            axes[i].set_xlabel('Time Steps', fontsize=12)
            axes[i].set_ylabel('Value', fontsize=12)
            axes[i].set_title(f'Target {i} - Horizon {horizon}', fontsize=14)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_metrics(self, horizon=None):
        """Print evaluation metrics in a formatted way"""
        if horizon is not None:
            if horizon not in self.results:
                self.evaluate_horizon(horizon)
            horizons_to_print = [horizon]
        else:
            if not self.results:
                self.evaluate_all_horizons()
            horizons_to_print = sorted(self.results.keys())
        
        for h in horizons_to_print:
            print(f"\n{'='*60}")
            print(f"Metrics for Horizon {h}")
            print(f"{'='*60}")
            metrics = self.results[h]['metrics']
            
            for target, target_metrics in metrics.items():
                print(f"\n{target.upper()}:")
                print(f"  MSE:         {target_metrics['mse']:.6f}")
                print(f"  RMSE:        {target_metrics['rmse']:.6f}")
                print(f"  MAE:         {target_metrics['mae']:.6f}")
                print(f"  R²:          {target_metrics['r2']:.6f}")
                print(f"  MAPE:        {target_metrics['mape']:.2f}%")
                print(f"  Correlation: {target_metrics['correlation']:.6f}")


class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with dilated convolutions for capturing long-range dependencies"""
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Causal padding
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Causal padding
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for capturing important temporal features"""
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # Linear projections in batch from d_model => h x d_k 
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.fc(context)
        output = self.dropout(output)
        
        # Add residual connection and layer norm
        return self.layer_norm(output + residual)


class MultiHorizonLSTM(nn.Module):
    """
    Enhanced LSTM with:
    - Multi-head attention for weak signal detection
    - Temporal Convolutional blocks for long-range dependencies
    - Residual connections throughout
    - Layer normalization for training stability
    
    Based on state-of-the-art architectures for financial time series
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, 
                 use_attention=True, use_tcn=True, num_heads=4):
        super(MultiHorizonLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_tcn = use_tcn
        
        # Input projection to match hidden size
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Temporal Convolutional Network blocks (optional)
        if use_tcn:
            self.tcn_blocks = nn.ModuleList([
                TemporalBlock(hidden_size, hidden_size, kernel_size=3, dilation=2**i, dropout=dropout)
                for i in range(2)  # 2 TCN blocks with increasing dilation
            ])
        
        # Bidirectional LSTM for better context understanding
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size // 2,  # Bidirectional doubles the output size
            num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention layer (optional)
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Enhanced feed-forward network with more capacity
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        # GELU activation (better than ReLU for gradients)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_proj(x)
        x = self.gelu(x)
        
        # TCN blocks (if enabled) - captures long-range dependencies
        if self.use_tcn:
            # TCN expects (batch, channels, seq_len)
            x_tcn = x.transpose(1, 2)
            for tcn_block in self.tcn_blocks:
                x_tcn = tcn_block(x_tcn)
            x = x_tcn.transpose(1, 2) + x  # Residual connection
            x = self.layer_norm1(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        # Multi-head attention (if enabled) - captures weak signals
        if self.use_attention:
            attn_out = self.attention(lstm_out)
            lstm_out = lstm_out + attn_out  # Residual connection
        
        lstm_out = self.layer_norm2(lstm_out)
        
        # Take last output (or use attention pooling)
        if self.use_attention:
            # Attention-weighted pooling over sequence
            attn_weights = torch.softmax(
                torch.bmm(lstm_out, lstm_out[:, -1, :].unsqueeze(2)).squeeze(2),
                dim=1
            )
            last_output = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        else:
            last_output = lstm_out[:, -1, :]
        
        # Enhanced feed-forward network with residuals
        out = self.gelu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.gelu(self.fc2(out))
        out = self.dropout(out)
        out = out + last_output  # Residual connection
        
        out = self.gelu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        
        return out


class MultiHorizonTimeSeriesDataset(Dataset):
    """PyTorch Dataset for multi-horizon time series forecasting"""
    def __init__(self, X, Y, sequence_length, horizon):
        '''
        X: External features (n_samples, n_features_X) - already scaled
        Y: Target time series (n_samples, n_features_Y) - already scaled
        sequence_length: Length of input sequences
        horizon: Prediction horizon (how many steps ahead)
        '''
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.sequence_length = sequence_length
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.sequence_length - self.horizon + 1
    
    def __getitem__(self, idx):
        # Input sequence: X[idx:idx+seq_len] + Y[idx:idx+seq_len-1]
        x_seq = self.X[idx:idx + self.sequence_length]
        y_seq = self.Y[idx:idx + self.sequence_length - 1]  # Y[:-1] for input
        # Combine X and lagged Y as features
        input_seq = torch.cat([x_seq[1:], y_seq], dim=1)  # Align dimensions
        # Target: Y at horizon h
        target = self.Y[idx + self.sequence_length + self.horizon - 1]
        return input_seq, target


class MultiHorizonForecaster:
    """
    Enhanced Multi-Horizon Forecaster with state-of-the-art components:
    - Attention mechanisms for weak signal detection
    - Temporal Convolutional Networks for long-range dependencies
    - Advanced optimization strategies
    """
    def __init__(self, sequence_length=20, horizons=[1, 5, 10], hidden_size=64, num_layers=2,
                 use_attention=True, use_tcn=True, num_heads=4):
        self.sequence_length = sequence_length
        self.horizons = horizons
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_tcn = use_tcn
        self.num_heads = num_heads
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Scalers
        self.scaler_X = RobustScaler(quantile_range=(0, 100))
        self.scaler_Y = RobustScaler(quantile_range=(1.0, 99.0))
        
        # Models for each horizon
        self.models = {}
        
        print(f"Initialized Enhanced Forecaster:")
        print(f"  - Attention: {use_attention} {'(' + str(num_heads) + ' heads)' if use_attention else ''}")
        print(f"  - TCN: {use_tcn}")
        print(f"  - Device: {self.device}")

    def prepare_data(self, X, Y, test_size=0.1):
        """Scale and split data into train and test sets"""
        print("Scaling data with RobustScaler...")
        # Fit and transform the data
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)
        print(f"X scaled range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        print(f"Y scaled range: [{Y_scaled.min():.3f}, {Y_scaled.max():.3f}]")
        
        # Split data
        split_idx = int(len(X) * (1 - test_size))
        print(f"test size = {test_size}, testing range {split_idx}:{len(X_scaled)}")
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        Y_train, Y_test = Y_scaled[:split_idx], Y_scaled[split_idx:]
        return X_train, X_test, Y_train, Y_test

    def create_datasets(self, X_train, X_test, Y_train, Y_test, horizon):
        """Create datasets for specific horizon"""
        # Debug prints
        print(f"Debug - X_test shape: {X_test.shape}")
        print(f"Debug - Y_test shape: {Y_test.shape}")
        print(f"Debug - sequence_length: {self.sequence_length}")
        print(f"Debug - horizon: {horizon}")
        
        # Calculate expected lengths
        train_len = len(Y_train) - self.sequence_length - horizon + 1
        test_len = len(Y_test) - self.sequence_length - horizon + 1
        print(f"Debug - Expected train dataset length: {train_len}")
        print(f"Debug - Expected test dataset length: {test_len}")
        
        # Check if test dataset would be valid
        if test_len <= 0:
            raise ValueError(f"Test dataset too small for the given sequence_length {self.sequence_length} and horizon {horizon}. Please adjust these parameters.")
        
        train_dataset = MultiHorizonTimeSeriesDataset(X_train, Y_train, self.sequence_length, horizon)
        test_dataset = MultiHorizonTimeSeriesDataset(X_test, Y_test, self.sequence_length, horizon)
        return train_dataset, test_dataset

    def train_horizon_model(self, train_loader, val_loader, horizon, epochs, lr):
        """Train enhanced model with advanced optimization for specific horizon"""
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        output_size = sample_batch[1].shape[1]
        print(f"Model architecture: Input={input_size}, Hidden={self.hidden_size}, Output={output_size}")
        
        # Initialize enhanced model with attention and TCN
        model = MultiHorizonLSTM(
            input_size, self.hidden_size, self.num_layers, output_size,
            use_attention=self.use_attention, use_tcn=self.use_tcn, 
            num_heads=self.num_heads
        )
        model.to(self.device)
        
        # Huber loss is more robust to outliers than MSE (better for market data)
        criterion = nn.SmoothL1Loss()  # Also called Huber loss
        
        # AdamW optimizer with better weight decay handling
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Cosine annealing with warm restarts (state-of-the-art for time series)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Initial restart period
            T_mult=2,  # Multiply period after each restart
            eta_min=lr * 0.01  # Minimum learning rate
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # Store in memory instead
        
        print(f"Total epochs {epochs} with Cosine Annealing + Warm Restarts")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step()
            
            # Early stopping - save to memory instead of disk
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()  # Save to memory
            else:
                patience_counter += 1
                if patience_counter >= 40:  # Increased patience for better convergence
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f'Horizon {horizon}, Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Load best model state from memory
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
        
        return model
    
    def fit(self, X, Y, batch_size=32, epochs=100, lr=1e-4, test_size=0.1):
        """Train models for all horizons"""
        print("Preparing data...")
        X_train, X_test, Y_train, Y_test = self.prepare_data(X, Y, test_size=test_size)
        
        for horizon in self.horizons:
            print(f"\nTraining model for horizon {horizon}...")
            train_dataset, test_dataset = self.create_datasets(X_train, X_test, Y_train, Y_test, horizon)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            model = self.train_horizon_model(train_loader, val_loader, horizon, epochs, lr)
            self.models[horizon] = model
        
        print("\nTraining completed for all horizons!")
    
    def predict(self, X, Y, horizon):
        """Make predictions for a specific horizon"""
        if horizon not in self.models:
            raise ValueError(f"No model trained for horizon {horizon}")
        
        model = self.models[horizon]
        model.eval()
        
        # Scale the input data using the fitted scalers
        X_scaled = self.scaler_X.transform(X)
        Y_scaled = self.scaler_Y.transform(Y)
        
        # Create dataset for prediction
        dataset = MultiHorizonTimeSeriesDataset(X_scaled, Y_scaled, self.sequence_length, horizon)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                predictions.append(outputs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Inverse transform predictions to original scale
        predictions = self.scaler_Y.inverse_transform(predictions)
        
        return predictions


def clean_and_align_data(X_df, Y_df):
    """Clean and align X and Y dataframes"""
    # Remove any rows with NaN values
    X_clean = X_df.dropna()
    Y_clean = Y_df.dropna()
    
    # Align indices
    common_index = X_clean.index.intersection(Y_clean.index)
    X_aligned = X_clean.loc[common_index]
    Y_aligned = Y_clean.loc[common_index]
    
    print(f"Cleaned data: {len(common_index)} samples")
    return X_aligned, Y_aligned


def run_forecasting(X_full, Y_full, pred_start, target_list, X_to_test):
    """
    Main function to run the enhanced forecasting pipeline with state-of-the-art architecture
    
    Args:
        X_full: Full feature dataframe
        Y_full: Full target dataframe
        pred_start: Index where testing begins
        target_list: List of target column names
        X_to_test: Test features
    
    Returns:
        DataFrame with predictions and actual values
    """
    # Prepare training data
    X_df = pd.DataFrame(X_full.iloc[:pred_start, :])  # all continuous features
    Y_df = pd.DataFrame(Y_full.iloc[:pred_start, :])
    print(f"Training ends at {Y_df.index[-1]}")
    
    X_aligned, Y_aligned = clean_and_align_data(X_df, Y_df)
    X_aligned = X_aligned.select_dtypes(include=[np.number])
    
    print(f"Data Overview:")
    print(f" X shape: {X_aligned.shape}, Y shape: {Y_aligned.shape}")
    
    # Initialize enhanced forecaster with attention and TCN
    sequence_length = 30
    forecaster = MultiHorizonForecaster(
        sequence_length=sequence_length,
        horizons=[1],
        hidden_size=64,  # Increased for better capacity
        num_layers=2,
        use_attention=True,  # Enable multi-head attention for weak signals
        use_tcn=True,  # Enable temporal convolutional network
        num_heads=4  # 4-head attention
    )
    
    # Train models with enhanced optimization
    test_ratio = 0.05
    print(f"\nTraining enhanced model with attention + TCN, test ratio {test_ratio}")
    forecaster.fit(
        X_aligned.values,
        Y_aligned.values,
        batch_size=64,
        epochs=300,
        lr=2e-3,  # Slightly lower LR for more stable training with attention
        test_size=test_ratio
    )
    
    # Prepare test data
    X_test = X_to_test.copy()[X_aligned.columns]
    Y_test = pd.DataFrame(
        np.zeros((len(X_test), Y_aligned.shape[1])),
        columns=Y_aligned.columns,
        index=X_test.index
    )
    
    # Make predictions
    horizon_1_predictions = forecaster.predict(X_test, Y_test, horizon=1)
    print(f"Predictions shape: {horizon_1_predictions.shape}")
    
    # Create prediction dataframe
    lstm_pred_df = pd.DataFrame(horizon_1_predictions, columns=target_list)
    
    # Fix: Create index that matches the predictions length
    start_idx = sequence_length
    end_idx = start_idx + len(horizon_1_predictions)
    lstm_pred_df.index = X_test.index[start_idx:end_idx]
    
    # Add actual values
    actual_df = Y_full.iloc[start_idx:end_idx, :].copy()
    actual_df.columns = [f"actual_{col}" for col in target_list]
    lstm_pred_df = pd.concat([lstm_pred_df, actual_df], axis=1)
    
    return lstm_pred_df


if __name__ == "__main__":
    # Example usage
    try:
        config = _load_config()
    except:
        config = {}
    
    print("="*80)
    print("ENHANCED LSTM FORECASTER - State-of-the-Art Architecture")
    print("="*80)
    print("\nKey Improvements:")
    print("  ✓ Multi-Head Attention: Captures weak signals in market data")
    print("  ✓ Temporal Convolutional Network: Long-range dependency modeling")
    print("  ✓ Bidirectional LSTM: Better context understanding")
    print("  ✓ Huber Loss: More robust to outliers")
    print("  ✓ AdamW Optimizer: Better weight decay handling")
    print("  ✓ Cosine Annealing: Improved learning rate scheduling")
    print("  ✓ Attention Pooling: Better sequence aggregation")
    print("\nBased on recent research:")
    print("  - Multi-head attention for time series (Vaswani et al.)")
    print("  - Temporal Convolutional Networks (Bai et al.)")
    print("  - Robust loss functions for financial data")
    print("="*80)
    print("\nTo use the model:")
    print("1. Import: from lstm_forecaster import MultiHorizonForecaster, ModelEvaluator, run_forecasting")
    print("2. Prepare your data: X (features) and Y (targets)")
    print("3. Call run_forecasting() or use the forecaster directly")
    print("\nExample usage:")
    print("  forecaster = MultiHorizonForecaster(")
    print("      sequence_length=30,")
    print("      horizons=[1, 5, 10],")
    print("      hidden_size=64,")
    print("      use_attention=True,  # Enable for weak signal detection")
    print("      use_tcn=True,  # Enable for long-range dependencies")
    print("      num_heads=4")
    print("  )")
    print("\nFor baseline comparison, set use_attention=False and use_tcn=False")
    print("\nSee ENHANCED_MODEL_DOCUMENTATION.md for detailed information.")
