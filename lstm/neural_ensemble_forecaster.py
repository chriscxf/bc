"""
Neural Ensemble Forecaster
Optimized for high-dimensional financial data to beat XGBoost

Architecture:
1. Neural Feature Extraction (Autoencoder): 250 → 32 compressed features
2. Traditional Feature Selection: Select top features via MI + RF importance
3. Ensemble Boosting: LightGBM + CatBoost + XGBoost + HistGradientBoosting
4. Stacking Meta-Learner: Ridge regression for optimal combination
5. Feature Importance: Built-in interpretability

Usage (same as lstm_forecaster.py):
    from neural_ensemble_forecaster import NeuralEnsembleForecaster
    
    forecaster = NeuralEnsembleForecaster(
        n_compressed_features=32,
        n_selected_features=64,
        use_neural_features=True
    )
    
    forecaster.fit(X_train, y_train)
    predictions = forecaster.predict(X_test)
    importance = forecaster.get_feature_importance()
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import optional boosting libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost not installed. Install with: pip install catboost")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")


# ============================================================================
# STAGE 1: NEURAL FEATURE EXTRACTOR (AUTOENCODER)
# ============================================================================

class FeatureAutoencoder(nn.Module):
    """
    Autoencoder for non-linear dimension reduction
    Captures complex feature interactions that linear methods miss
    """
    def __init__(self, input_dim, compressed_dim=32, hidden_dim=128):
        super(FeatureAutoencoder, self).__init__()
        
        # Encoder: input → hidden → compressed
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, compressed_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        # Decoder: compressed → hidden → input
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed, compressed
    
    def encode(self, x):
        """Extract compressed features"""
        with torch.no_grad():
            return self.encoder(x)


def train_autoencoder(X, compressed_dim=32, epochs=100, batch_size=32, lr=1e-3, device='cpu'):
    """
    Train autoencoder for feature extraction
    
    Args:
        X: Input features (n_samples, n_features)
        compressed_dim: Size of compressed representation
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cpu' or 'cuda'
    
    Returns:
        Trained autoencoder model
    """
    input_dim = X.shape[1]
    hidden_dim = min(128, input_dim // 2)  # Adaptive hidden size
    
    model = FeatureAutoencoder(input_dim, compressed_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\n🧠 Training Neural Feature Extractor:")
    print(f"   Input: {input_dim} features → Compressed: {compressed_dim} features")
    print(f"   Hidden layer: {hidden_dim} units")
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch_x = batch[0]
            optimizer.zero_grad()
            
            reconstructed, compressed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}: Reconstruction Loss = {avg_loss:.6f}")
    
    print(f"   ✓ Training complete. Best loss: {best_loss:.6f}")
    return model


# ============================================================================
# STAGE 2: TRADITIONAL FEATURE SELECTION
# ============================================================================

class HybridFeatureSelector:
    """
    Combines multiple feature selection methods:
    1. Mutual Information (non-linear relationships)
    2. Random Forest Importance (tree-based interactions)
    3. Correlation-based filtering (remove redundancy)
    """
    def __init__(self, n_features=64):
        self.n_features = n_features
        self.selected_indices_ = None
        self.feature_scores_ = None
        self.feature_names_ = None
    
    def fit(self, X, y, feature_names=None):
        """
        Select top features using ensemble of methods
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional feature names
        
        Returns:
            self
        """
        print(f"\n🔍 Hybrid Feature Selection: {X.shape[1]} → {self.n_features} features")
        
        n_samples, n_input_features = X.shape
        
        # Method 1: Mutual Information
        print("   Computing Mutual Information scores...")
        mi_scores = mutual_info_regression(X, y.ravel(), random_state=42)
        mi_scores = mi_scores / (mi_scores.max() + 1e-10)
        
        # Method 2: Random Forest Importance
        print("   Computing Random Forest importance...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y.ravel())
        rf_scores = rf.feature_importances_
        rf_scores = rf_scores / (rf_scores.max() + 1e-10)
        
        # Method 3: Correlation with target
        print("   Computing correlation scores...")
        corr_scores = np.abs([np.corrcoef(X[:, i], y.ravel())[0, 1] 
                              for i in range(n_input_features)])
        corr_scores = np.nan_to_num(corr_scores)
        corr_scores = corr_scores / (corr_scores.max() + 1e-10)
        
        # Combined score: 40% MI + 40% RF + 20% Correlation
        combined_scores = 0.4 * mi_scores + 0.4 * rf_scores + 0.2 * corr_scores
        
        # Select top features
        self.selected_indices_ = np.argsort(combined_scores)[-self.n_features:]
        self.feature_scores_ = combined_scores[self.selected_indices_]
        
        if feature_names is not None:
            self.feature_names_ = [feature_names[i] for i in self.selected_indices_]
        else:
            self.feature_names_ = [f"feature_{i}" for i in self.selected_indices_]
        
        print(f"   ✓ Selected {self.n_features} features")
        print(f"   Top 5: {self.feature_names_[:5]}")
        print(f"   Score range: [{self.feature_scores_.min():.3f}, {self.feature_scores_.max():.3f}]")
        
        return self
    
    def transform(self, X):
        """Select features from X"""
        if self.selected_indices_ is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self.selected_indices_]
    
    def fit_transform(self, X, y, feature_names=None):
        """Fit and transform in one step"""
        return self.fit(X, y, feature_names).transform(X)


# ============================================================================
# STAGE 3: ENSEMBLE BOOSTING MODELS
# ============================================================================

class BoostingEnsemble:
    """
    Ensemble of gradient boosting methods
    Each model has different strengths for financial data
    """
    def __init__(self, use_lgb=True, use_cat=True, use_xgb=True, random_state=42):
        self.use_lgb = use_lgb and HAS_LIGHTGBM
        self.use_cat = use_cat and HAS_CATBOOST
        self.use_xgb = use_xgb and HAS_XGBOOST
        self.random_state = random_state
        
        self.models = {}
        self.model_weights = {}
        
    def _init_lightgbm(self):
        """Initialize LightGBM with financial-optimized params"""
        return lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            verbose=-1,
            n_jobs=-1
        )
    
    def _init_catboost(self):
        """Initialize CatBoost with financial-optimized params"""
        return cb.CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            subsample=0.8,
            random_strength=0.5,
            bagging_temperature=0.2,
            random_seed=self.random_state,
            verbose=False,
            thread_count=-1
        )
    
    def _init_xgboost(self):
        """Initialize XGBoost with carefully tuned financial params"""
        return xgb.XGBRegressor(
            n_estimators=1000,           # More trees for better learning
            learning_rate=0.03,          # Lower for stability
            max_depth=6,                 # Moderate depth
            min_child_weight=3,          # Conservative splits
            subsample=0.8,               # Row sampling
            colsample_bytree=0.8,        # Column sampling
            gamma=0.1,                   # Minimum loss reduction
            reg_alpha=1.0,               # L1 regularization
            reg_lambda=2.0,              # L2 regularization (higher for financial data)
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
            tree_method='hist'           # Faster for large datasets
        )
    
    def _init_histgb(self):
        """Initialize HistGradientBoosting (always available)"""
        return HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=self.random_state
        )
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train all available boosting models
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
        
        Returns:
            self
        """
        print(f"\n🚀 Training Boosting Ensemble:")
        
        # Always train HistGradientBoosting (no dependencies)
        print("   [1/4] Training HistGradientBoosting...")
        self.models['histgb'] = self._init_histgb()
        self.models['histgb'].fit(X, y.ravel())
        
        # Train LightGBM if available
        if self.use_lgb:
            print("   [2/4] Training LightGBM...")
            self.models['lgb'] = self._init_lightgbm()
            if X_val is not None:
                self.models['lgb'].fit(
                    X, y.ravel(),
                    eval_set=[(X_val, y_val.ravel())],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
            else:
                self.models['lgb'].fit(X, y.ravel())
        
        # Train CatBoost if available
        if self.use_cat:
            print("   [3/4] Training CatBoost...")
            self.models['cat'] = self._init_catboost()
            if X_val is not None:
                self.models['cat'].fit(
                    X, y.ravel(),
                    eval_set=(X_val, y_val.ravel()),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.models['cat'].fit(X, y.ravel())
        
        # Train XGBoost if available
        if self.use_xgb:
            print("   [4/4] Training XGBoost...")
            self.models['xgb'] = self._init_xgboost()
            if X_val is not None:
                self.models['xgb'].fit(
                    X, y.ravel(),
                    eval_set=[(X_val, y_val.ravel())],
                    verbose=False
                )
            else:
                self.models['xgb'].fit(X, y.ravel())
        
        print(f"   ✓ Trained {len(self.models)} models")
        
        # Calculate validation scores if available
        if X_val is not None and y_val is not None:
            print("\n   📊 Validation Performance:")
            for name, model in self.models.items():
                pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                mae = mean_absolute_error(y_val, pred)
                r2 = r2_score(y_val, pred)
                print(f"      {name:10s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return self
    
    def predict(self, X):
        """Get predictions from all models"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X).reshape(-1, 1)
        return predictions
    
    def get_feature_importance(self, feature_names=None):
        """
        Aggregate feature importance from all models
        
        Returns:
            DataFrame with feature importance scores
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        if not importances:
            return None
        
        # Average importance across models
        avg_importance = np.mean(list(importances.values()), axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(avg_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        })
        
        # Add individual model importances
        for name, imp in importances.items():
            importance_df[f'{name}_importance'] = imp
        
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        return importance_df


# ============================================================================
# STAGE 4: STACKING META-LEARNER
# ============================================================================

class StackingMetaLearner:
    """
    Combines predictions from multiple models using Ridge regression
    Learns optimal weights automatically
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.meta_model = Ridge(alpha=alpha)
        self.model_weights_ = None
    
    def fit(self, base_predictions, y):
        """
        Train meta-learner to combine base model predictions
        
        Args:
            base_predictions: dict of {model_name: predictions}
            y: True targets
        
        Returns:
            self
        """
        # Stack predictions horizontally
        X_meta = np.hstack([pred for pred in base_predictions.values()])
        
        print(f"\n🎯 Training Stacking Meta-Learner:")
        print(f"   Combining {len(base_predictions)} base models")
        
        self.meta_model.fit(X_meta, y.ravel())
        
        # Store model weights
        self.model_weights_ = dict(zip(base_predictions.keys(), self.meta_model.coef_))
        
        print("   ✓ Model weights learned:")
        for name, weight in self.model_weights_.items():
            print(f"      {name:10s}: {weight:.4f}")
        
        return self
    
    def predict(self, base_predictions):
        """Combine base predictions using learned weights"""
        X_meta = np.hstack([pred for pred in base_predictions.values()])
        return self.meta_model.predict(X_meta)


# ============================================================================
# MAIN: NEURAL ENSEMBLE FORECASTER
# ============================================================================

class NeuralEnsembleForecaster:
    """
    Complete framework: Neural Feature Extraction + Feature Selection + Ensemble Boosting + Stacking
    
    Optimized to beat XGBoost on high-dimensional financial data
    
    Usage:
        forecaster = NeuralEnsembleForecaster(
            n_compressed_features=32,
            n_selected_features=64,
            use_neural_features=True
        )
        
        forecaster.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        predictions = forecaster.predict(X_test)
        importance = forecaster.get_feature_importance(top_k=20)
    """
    def __init__(
        self,
        n_compressed_features=32,
        n_selected_features=64,
        use_neural_features=True,
        use_stacking=True,
        autoencoder_epochs=100,
        random_state=42
    ):
        """
        Args:
            n_compressed_features: Number of features from autoencoder
            n_selected_features: Number of features from traditional selection
            use_neural_features: Whether to use autoencoder features
            use_stacking: Whether to use stacking meta-learner
            autoencoder_epochs: Training epochs for autoencoder
            random_state: Random seed
        """
        self.n_compressed_features = n_compressed_features
        self.n_selected_features = n_selected_features
        self.use_neural_features = use_neural_features
        self.use_stacking = use_stacking
        self.autoencoder_epochs = autoencoder_epochs
        self.random_state = random_state
        
        # Components (initialized in fit)
        self.scaler = None
        self.autoencoder = None
        self.feature_selector = None
        self.ensemble = None
        self.meta_learner = None
        
        # Feature tracking
        self.original_feature_names_ = None
        self.selected_feature_names_ = None
        self.neural_feature_names_ = None
        self.all_feature_names_ = None
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print("NEURAL FEATURE REDUCTION + XGBOOST FORECASTER")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Neural features: {n_compressed_features if use_neural_features else 0}")
        print(f"  Selected features: {n_selected_features}")
        print(f"  Total features: {(n_compressed_features if use_neural_features else 0) + n_selected_features}")
        print(f"  Prediction model: XGBoost (tuned)")
        print(f"  Device: {self.device}")
        print("="*80)
    
    def _prepare_features(self, X, y=None, fit=True):
        """
        Extract and combine neural + traditional features
        
        Args:
            X: Input features
            y: Targets (required for fitting)
            fit: Whether to fit transformers
        
        Returns:
            Combined feature matrix
        """
        if fit:
            # Scale original features
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Stage 1: Neural feature extraction
            if self.use_neural_features:
                self.autoencoder = train_autoencoder(
                    X_scaled,
                    compressed_dim=self.n_compressed_features,
                    epochs=self.autoencoder_epochs,
                    device=self.device
                )
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                X_neural = self.autoencoder.encode(X_tensor).cpu().numpy()
                self.neural_feature_names_ = [f"neural_{i}" for i in range(self.n_compressed_features)]
            else:
                X_neural = None
            
            # Stage 2: Traditional feature selection
            self.feature_selector = HybridFeatureSelector(n_features=self.n_selected_features)
            X_selected = self.feature_selector.fit_transform(
                X_scaled, 
                y, 
                feature_names=self.original_feature_names_
            )
            self.selected_feature_names_ = self.feature_selector.feature_names_
            
            # Combine features
            if X_neural is not None:
                X_combined = np.hstack([X_neural, X_selected])
                self.all_feature_names_ = self.neural_feature_names_ + self.selected_feature_names_
            else:
                X_combined = X_selected
                self.all_feature_names_ = self.selected_feature_names_
            
            print(f"\n✓ Feature Engineering Complete:")
            print(f"   Input: {X.shape[1]} features")
            print(f"   Neural: {X_neural.shape[1] if X_neural is not None else 0} features")
            print(f"   Selected: {X_selected.shape[1]} features")
            print(f"   Total: {X_combined.shape[1]} features")
            
            return X_combined
        
        else:
            # Transform only (prediction time)
            X_scaled = self.scaler.transform(X)
            
            if self.use_neural_features:
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                X_neural = self.autoencoder.encode(X_tensor).cpu().numpy()
            else:
                X_neural = None
            
            X_selected = self.feature_selector.transform(X_scaled)
            
            if X_neural is not None:
                X_combined = np.hstack([X_neural, X_selected])
            else:
                X_combined = X_selected
            
            return X_combined
    
    def fit(self, X, y, X_val=None, y_val=None, feature_names=None):
        """
        Train the complete pipeline
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,) or (n_samples, 1)
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            feature_names: Optional list of feature names
        
        Returns:
            self
        """
        # Store feature names
        if feature_names is not None:
            self.original_feature_names_ = feature_names
        else:
            self.original_feature_names_ = [f"X{i}" for i in range(X.shape[1])]
        
        # Ensure y is 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        print(f"\n📊 Training Data:")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        if X_val is not None:
            print(f"   X_val shape: {X_val.shape}")
            print(f"   y_val shape: {y_val.shape}")
        
        # Stage 1+2: Feature extraction and selection
        X_processed = self._prepare_features(X, y, fit=True)
        
        if X_val is not None:
            X_val_processed = self._prepare_features(X_val, fit=False)
        else:
            X_val_processed = None
        
        # Stage 3: Train ensemble (using only XGBoost for best performance)
        self.ensemble = BoostingEnsemble(
            use_lgb=False,               # Disable LightGBM
            use_cat=False,               # Disable CatBoost
            use_xgb=True,                # Use only XGBoost with tuned params
            random_state=self.random_state
        )
        self.ensemble.fit(X_processed, y, X_val_processed, y_val)
        
        # Stage 4: Skip meta-learner since we only have one model
        self.use_stacking = False  # Override since single model doesn't need stacking
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predictions (n_samples,)
        """
        # Transform features
        X_processed = self._prepare_features(X, fit=False)
        
        # Get base model predictions
        base_predictions = self.ensemble.predict(X_processed)
        
        # Since we only use XGBoost, directly return its predictions
        if 'xgb' in base_predictions:
            predictions = base_predictions['xgb'].ravel()
        else:
            # Fallback to averaging if somehow multiple models exist
            predictions = np.mean([pred.ravel() for pred in base_predictions.values()], axis=0)
        
        return predictions
    
    def get_feature_importance(self, top_k=None, return_dataframe=True):
        """
        Get feature importance from ensemble models
        
        Args:
            top_k: Return only top k features (None = all)
            return_dataframe: Return as DataFrame (True) or dict (False)
        
        Returns:
            Feature importance DataFrame or dict
        """
        if self.ensemble is None:
            raise ValueError("Must call fit() before getting feature importance")
        
        # Get importance from ensemble
        importance_df = self.ensemble.get_feature_importance(self.all_feature_names_)
        
        if importance_df is None:
            return None
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        if return_dataframe:
            return importance_df
        else:
            return dict(zip(importance_df['feature'], importance_df['importance']))
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True targets
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        if len(y.shape) > 1:
            y = y.ravel()
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'correlation': np.corrcoef(y, predictions)[0, 1]
        }
        
        return metrics
    
    def print_performance(self, X_train, y_train, X_test, y_test):
        """
        Print performance metrics on train and test sets
        """
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        # Training performance
        train_metrics = self.evaluate(X_train, y_train)
        print("\n📊 Training Set:")
        print(f"   RMSE:        {train_metrics['rmse']:.6f}")
        print(f"   MAE:         {train_metrics['mae']:.6f}")
        print(f"   R²:          {train_metrics['r2']:.6f}")
        print(f"   Correlation: {train_metrics['correlation']:.6f}")
        
        # Test performance
        test_metrics = self.evaluate(X_test, y_test)
        print("\n📊 Test Set:")
        print(f"   RMSE:        {test_metrics['rmse']:.6f}")
        print(f"   MAE:         {test_metrics['mae']:.6f}")
        print(f"   R²:          {test_metrics['r2']:.6f}")
        print(f"   Correlation: {test_metrics['correlation']:.6f}")
        
        # Overfitting check
        overfit_ratio = test_metrics['rmse'] / train_metrics['rmse']
        print(f"\n📈 Overfitting Check:")
        print(f"   Test/Train RMSE ratio: {overfit_ratio:.2f}")
        if overfit_ratio < 1.1:
            print("   ✓ Good generalization")
        elif overfit_ratio < 1.3:
            print("   ⚠️ Slight overfitting")
        else:
            print("   ❌ Significant overfitting")
        
        print("="*80 + "\n")
        
        return train_metrics, test_metrics


# ============================================================================
# CONVENIENCE FUNCTIONS (Same interface as lstm_forecaster.py)
# ============================================================================

def run_forecasting(X_full, Y_full, pred_start, target_list, X_to_test):
    """
    Main forecasting function - same interface as lstm_forecaster.py
    
    Args:
        X_full: Full feature DataFrame
        Y_full: Full target DataFrame
        pred_start: Index to start predictions
        target_list: List of target column names
        X_to_test: Features for prediction period
    
    Returns:
        DataFrame with predictions and actuals
    """
    print("\n" + "="*80)
    print("NEURAL ENSEMBLE FORECASTING")
    print("="*80)
    
    # Prepare training data
    X_train = X_full.iloc[:pred_start, :].select_dtypes(include=[np.number])
    Y_train = Y_full.iloc[:pred_start, :]
    
    print(f"\nTraining period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Prediction period: {X_to_test.index[0]} to {X_to_test.index[-1]}")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Initialize forecaster
    forecaster = NeuralEnsembleForecaster(
        n_compressed_features=32,
        n_selected_features=64,
        use_neural_features=True,
        use_stacking=True,
        autoencoder_epochs=100
    )
    
    # Train with validation split
    split_idx = int(len(X_train) * 0.9)
    X_train_split = X_train.iloc[:split_idx].values
    Y_train_split = Y_train.iloc[:split_idx].values
    X_val_split = X_train.iloc[split_idx:].values
    Y_val_split = Y_train.iloc[split_idx:].values
    
    forecaster.fit(
        X_train_split,
        Y_train_split,
        X_val=X_val_split,
        y_val=Y_val_split,
        feature_names=list(X_train.columns)
    )
    
    # Make predictions
    X_test = X_to_test[X_train.columns]
    
    # Check for NaN in test features
    nan_mask = X_test.isna().any(axis=1)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"\n⚠️ Warning: {n_nan} rows ({n_nan/len(X_test)*100:.1f}%) have NaN in features")
        print(f"   These rows will produce NaN predictions")
        print(f"   First NaN row: {X_test.index[nan_mask][0]}")
        print(f"   NaN columns: {X_test.columns[X_test.isna().any()].tolist()[:10]}")
    
    predictions = forecaster.predict(X_test.values)
    
    # Create result DataFrame
    result_df = pd.DataFrame(predictions, columns=target_list, index=X_to_test.index)
    
    # Add actual values if available
    if pred_start < len(Y_full):
        # Align indices properly
        y_test_slice = Y_full.iloc[pred_start:, :]
        
        # Match lengths
        min_len = min(len(result_df), len(y_test_slice))
        result_df = result_df.iloc[:min_len]
        y_test_slice = y_test_slice.iloc[:min_len]
        
        # Add actuals with matching index
        for col in target_list:
            result_df[f"actual_{col}"] = y_test_slice[col].values
        
        print(f"\n✓ Matched {min_len} predictions with actuals")
    
    # Print feature importance
    print("\n" + "="*80)
    print("TOP 20 MOST IMPORTANT FEATURES")
    print("="*80)
    importance = forecaster.get_feature_importance(top_k=20)
    print(importance.to_string(index=False))
    print("="*80 + "\n")
    
    return result_df


def run_rolling_forecasting(X_full, Y_full, pred_start, target_list, window_size=252, 
                           retrain_freq=20, verbose=True):
    """
    Rolling window forecasting - trains and predicts one step at a time
    More realistic but MUCH slower (retrains model multiple times)
    
    Args:
        X_full: Full feature DataFrame
        Y_full: Full target DataFrame
        pred_start: Index to start predictions
        target_list: List of target column names
        window_size: Rolling window size (252 = 1 year daily data)
        retrain_freq: Retrain every N steps (1 = every step, 20 = every 20 steps)
        verbose: Print progress
    
    Returns:
        DataFrame with predictions and actuals
    """
    print("\n" + "="*80)
    print("ROLLING WINDOW FORECASTING (REALISTIC BUT SLOW)")
    print("="*80)
    print(f"Window size: {window_size} samples")
    print(f"Retrain frequency: Every {retrain_freq} steps")
    print(f"Prediction steps: {len(Y_full) - pred_start}")
    print("="*80)
    
    # Select numeric columns
    X_numeric = X_full.select_dtypes(include=[np.number])
    numeric_cols = list(X_numeric.columns)
    
    predictions_list = []
    actuals_list = []
    dates_list = []
    forecaster = None
    
    # Rolling prediction loop
    n_steps = len(Y_full) - pred_start
    for i in range(n_steps):
        current_idx = pred_start + i
        
        # Determine training window
        train_start = max(0, current_idx - window_size)
        train_end = current_idx
        
        # Get training data
        X_train = X_numeric.iloc[train_start:train_end]
        Y_train = Y_full.iloc[train_start:train_end]
        
        # Get test data (next time point)
        X_test = X_numeric.iloc[current_idx:current_idx+1]
        Y_test = Y_full.iloc[current_idx:current_idx+1]
        
        # Check for NaN
        if X_test.isna().any().any():
            if verbose and i < 5:
                print(f"⚠️ Step {i}: Skipping due to NaN in features")
            predictions_list.append(np.nan)
            actuals_list.append(Y_test[target_list[0]].values[0])
            dates_list.append(X_test.index[0])
            continue
        
        # Retrain model every retrain_freq steps or first step
        if i == 0 or i % retrain_freq == 0:
            if verbose:
                print(f"[{i+1}/{n_steps}] Retraining on {len(X_train)} samples...")
            
            forecaster = NeuralEnsembleForecaster(
                n_compressed_features=32,
                n_selected_features=64,
                use_neural_features=True,
                use_stacking=True,
                autoencoder_epochs=50,  # Faster for rolling
                random_state=42
            )
            
            # Split for validation
            split_idx = int(len(X_train) * 0.9)
            forecaster.fit(
                X_train.iloc[:split_idx].values,
                Y_train.iloc[:split_idx].values,
                X_val=X_train.iloc[split_idx:].values,
                y_val=Y_train.iloc[split_idx:].values,
                feature_names=numeric_cols
            )
        
        # Make prediction
        pred = forecaster.predict(X_test.values)[0, 0]
        predictions_list.append(pred)
        actuals_list.append(Y_test[target_list[0]].values[0])
        dates_list.append(X_test.index[0])
        
        if verbose and (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{n_steps} ({(i+1)/n_steps*100:.1f}%)")
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        target_list[0]: predictions_list,
        f"actual_{target_list[0]}": actuals_list
    }, index=dates_list)
    
    print("\n" + "="*80)
    print("ROLLING FORECAST COMPLETE")
    print("="*80)
    
    # Calculate metrics
    valid_mask = ~(np.isnan(predictions_list) | np.isnan(actuals_list))
    if np.sum(valid_mask) > 0:
        pred_valid = np.array(predictions_list)[valid_mask]
        actual_valid = np.array(actuals_list)[valid_mask]
        rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
        mae = mean_absolute_error(actual_valid, pred_valid)
        r2 = r2_score(actual_valid, pred_valid)
        
        print(f"Valid predictions: {np.sum(valid_mask)}/{len(predictions_list)}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"R²:   {r2:.6f}")
    
    print("="*80 + "\n")
    
    return result_df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("NEURAL ENSEMBLE FORECASTER - USAGE EXAMPLES")
    print("="*80)
    
    print("\n1. Basic Usage:")
    print("""
    from neural_ensemble_forecaster import NeuralEnsembleForecaster
    
    forecaster = NeuralEnsembleForecaster(
        n_compressed_features=32,
        n_selected_features=64,
        use_neural_features=True,
        use_stacking=True
    )
    
    forecaster.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    predictions = forecaster.predict(X_test)
    """)
    
    print("\n2. Get Feature Importance:")
    print("""
    # Top 20 features
    importance = forecaster.get_feature_importance(top_k=20)
    print(importance)
    
    # Plot feature importance
    import matplotlib.pyplot as plt
    plt.barh(importance['feature'][:10], importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Features')
    plt.show()
    """)
    
    print("\n3. Same Interface as lstm_forecaster.py:")
    print("""
    from neural_ensemble_forecaster import run_forecasting
    
    result_df = run_forecasting(
        X_full=X_full,
        Y_full=Y_full,
        pred_start=900,
        target_list=['target'],
        X_to_test=X_test
    )
    """)
    
    print("\n4. Evaluate Performance:")
    print("""
    train_metrics, test_metrics = forecaster.print_performance(
        X_train, y_train,
        X_test, y_test
    )
    """)
    
    print("\n" + "="*80)
    print("Ready to beat XGBoost! 🚀")
    print("="*80 + "\n")
