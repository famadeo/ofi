import logging
from dataclasses import dataclass, field
from datetime import time
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Function to perform Linear regression for each symbol
def best_level_regressor(df):
    """
    Perform linear regression on grouped data by symbol and return regression results.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data with columns 'symbol', 'ofi_00', and 'returns_bps'.

    Returns:
    dict: A dictionary where each key is a symbol and the value is another dictionary containing:
        - 'coefficients' (numpy.ndarray): Coefficients of the linear regression model.
        - 'intercept' (float): Intercept of the linear regression model.
        - 'score' (float): R^2 score of the model on the test set.
        - 'y_test' (numpy.ndarray): Actual target values from the test set.
        - 'y_pred' (numpy.ndarray): Predicted target values from the test set.
    """
    results = {}

    # Group by symbol
    grouped = df.groupby("symbol")

    for symbol, group in grouped:
        # Extract features and target
        X = group[["ofi_00"]].values  # Reshape as required
        y = group["returns_bps"].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Fit Linear regression
        linear_reg = LinearRegression()
        linear_reg.fit(X_train, y_train)

        # Predict on test set
        y_pred = linear_reg.predict(X_test)

        # Store results
        results[symbol] = {
            "coefficients": linear_reg.coef_,
            "intercept": linear_reg.intercept_,
            "score": linear_reg.score(X_test, y_test),
            "y_test": y_test,
            "y_pred": y_pred
        }

    return results


# Configure logging to output to screen
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

@dataclass
class ForecastConfig:
    """Configuration for the forecasting model"""
    estimation_window: int = 30  # 30-minute windows as per paper
    lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20, 30])
    prediction_horizon: int = 1  # How many minutes ahead to predict
    lasso_alpha: float = 0.01  # LASSO regularization parameter
    market_start: time = time(10, 0)  # 10:00 AM
    market_end: time = time(15, 30)   # 3:30 PM
    auto_alpha: bool = True  # Whether to automatically determine alpha
    max_iter: int = 10000  # Maximum iterations for LASSO
    tol: float = 1e-4  # Convergence tolerance
    cv_folds: int = 5  # Number of folds for cross-validation
    min_samples: int = 10  # Minimum number of samples required
    skip_sparse_windows: bool = True  # Skip windows with too few samples
    min_nonzero_samples: int = 5  # Minimum number of non-zero samples required

class LassoForecaster:
    """
    Enhanced implementation of the forward-looking cross-impact model (FCII)
    with improved stability and diagnostics
    """
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.lasso = Lasso(
            alpha=config.lasso_alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            fit_intercept=True
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.current_target: str = None  # Track current target symbol
        
    def create_features_for_stock(self, df: pl.DataFrame, target_symbol: str) -> Tuple[np.ndarray, List[str]]:
        """
        Create features for predicting a specific stock following FCII model equation (15)
        with improved feature tracking
        """
        features = []
        feature_names = []
        all_symbols = df['symbol'].unique()
        data_length = len(df.filter(pl.col('symbol') == target_symbol))
        
        # For each lag
        for lag in self.config.lags:
            if lag > 0:
                lag_features = []
                
                # First add target stock's OFI at this lag
                target_ofi = df.filter(pl.col('symbol') == target_symbol)['integrated_ofi'].to_numpy()
                lagged_target = np.roll(target_ofi, lag)
                lagged_target[:lag] = 0
                lag_features.append(lagged_target.reshape(-1, 1))
                feature_names.append(f"{target_symbol}_lag{lag}")
                
                # Then add all other stocks' OFIs at the same lag
                for other_symbol in all_symbols:
                    if other_symbol != target_symbol:
                        other_ofi = df.filter(pl.col('symbol') == other_symbol)['integrated_ofi'].to_numpy()
                        
                        # Ensure same length as target stock data
                        if len(other_ofi) != data_length:
                            if len(other_ofi) > data_length:
                                other_ofi = other_ofi[:data_length]
                            else:
                                other_ofi = np.pad(other_ofi, (0, data_length - len(other_ofi)))
                        
                        lagged_other = np.roll(other_ofi, lag)
                        lagged_other[:lag] = 0
                        lag_features.append(lagged_other.reshape(-1, 1))
                        feature_names.append(f"{other_symbol}_lag{lag}")
                
                combined_features = np.hstack(lag_features)
                features.append(combined_features)
        
        X = np.hstack(features)
        self.feature_names = feature_names
        return X, feature_names
    
    def diagnose_convergence(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Comprehensive diagnostics for LASSO convergence issues
        """
        diagnostics = {}
        
        # Check feature correlations
        corr_matrix = np.corrcoef(X.T)
        high_corr_mask = np.abs(corr_matrix) > 0.95
        np.fill_diagonal(high_corr_mask, False)
        high_corr_pairs = np.where(high_corr_mask)
        
        # Check feature variances
        variances = np.var(X, axis=0)
        low_var_threshold = 1e-6
        low_var_features = np.where(variances < low_var_threshold)[0]
        
        # Condition number
        cond_num = np.linalg.cond(X)
        
        diagnostics['condition_number'] = cond_num
        diagnostics['high_correlation_pairs'] = [
            (feature_names[i], feature_names[j], corr_matrix[i,j])
            for i, j in zip(*high_corr_pairs)
            if i < j  # Avoid duplicates
        ]
        diagnostics['low_variance_features'] = [
            (feature_names[i], variances[i])
            for i in low_var_features
        ]
        
        # Log diagnostics
        if cond_num > 1e5:
            logging.warning(f"High condition number: {cond_num}")
        if len(diagnostics['high_correlation_pairs']) > 0:
            logging.warning(f"Found {len(diagnostics['high_correlation_pairs'])} highly correlated feature pairs")
        if len(diagnostics['low_variance_features']) > 0:
            logging.warning(f"Found {len(diagnostics['low_variance_features'])} low variance features")
            
        return diagnostics
    
    def optimize_alpha(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Find optimal LASSO alpha using time series cross-validation
        """
        if len(y) < self.config.min_samples:
            logging.warning("Insufficient samples for alpha optimization")
            return self.config.lasso_alpha
            
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        # Setup LassoCV with reasonable alpha range
        alphas = np.logspace(-6, 2, 100)
        lasso_cv = LassoCV(
            alphas=alphas,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            cv=tscv
        )
        
        try:
            lasso_cv.fit(X, y)
            logging.info(f"Optimal alpha: {lasso_cv.alpha_}")
            return lasso_cv.alpha_
        except Exception as e:
            logging.error(f"Alpha optimization failed: {str(e)}")
            return self.config.lasso_alpha
    
    def prepare_window_data(self, window_df: pl.DataFrame, target_symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for a given estimation window with enhanced stability
        """
        X, feature_names = self.create_features_for_stock(window_df, target_symbol)
        y = window_df.filter(pl.col('symbol') == target_symbol)['returns_bps'].to_numpy()
        
        # Count non-zero samples
        nonzero_samples = np.sum(np.abs(y) > 1e-10)
        
        if len(y) < self.config.min_samples:
            if self.config.skip_sparse_windows:
                raise ValueError(f"Insufficient samples: {len(y)} < {self.config.min_samples}")
            else:
                logging.warning(f"Low sample count: {len(y)} samples")
        
        if nonzero_samples < self.config.min_nonzero_samples:
            if self.config.skip_sparse_windows:
                raise ValueError(f"Insufficient non-zero samples: {nonzero_samples}")
            else:
                logging.warning(f"Low non-zero sample count: {nonzero_samples} samples")
        
        # Add small noise to avoid perfect collinearity
        epsilon = 1e-10
        X = X + epsilon * np.random.randn(*X.shape)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def fit_window(self, window_df: pl.DataFrame, target_symbol: str):
        """
        Fit model on a single estimation window with comprehensive diagnostics
        """
        self.current_target = target_symbol
        X, y = self.prepare_window_data(window_df, target_symbol)
        
        # Run diagnostics (useful for debugging)
        diagnostics = self.diagnose_convergence(X, self.feature_names)
        
        # Optimize alpha if configured
        if self.config.auto_alpha:
            optimal_alpha = self.optimize_alpha(X, y)
            self.lasso.alpha = optimal_alpha
        
        # Fit model
        try:
            self.lasso.fit(X, y)
            
            # Log model sparsity
            n_nonzero = np.sum(self.lasso.coef_ != 0)
            sparsity = 1 - (n_nonzero / len(self.lasso.coef_))
            logging.info(f"Model sparsity: {sparsity:.2%}")
            
            # Log significant features
            significant_features = [
                (name, coef) 
                for name, coef in zip(self.feature_names, self.lasso.coef_)
                if abs(coef) > 1e-5
            ]
            logging.info(f"Significant features: {len(significant_features)}")
            
            # Analyze cross-asset relationships
            relationships = self.analyze_cross_asset_relationships()
            logging.info("Cross-asset analysis completed")
            
        except Exception as e:
            logging.error(f"Model fitting failed: {str(e)}")
            raise
        
    def predict_window(self, window_df: pl.DataFrame, target_symbol: str) -> np.ndarray:
        """
        Make predictions for a window with error handling
        """
        try:
            X, _ = self.create_features_for_stock(window_df, target_symbol)
            X_scaled = self.scaler.transform(X)
            return self.lasso.predict(X_scaled)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise

    def analyze_cross_asset_relationships(self) -> Dict:
        """
        Analyze the strength of cross-asset relationships based on LASSO coefficients
        """
        relationships = {}
        
        # Get non-zero coefficients and their corresponding feature names
        significant_coefs = [(name, coef) for name, coef in 
                            zip(self.feature_names, self.lasso.coef_) 
                            if abs(coef) > 1e-5]
        
        # Separate own-asset and cross-asset effects
        own_effects = []
        cross_effects = []
        
        for name, coef in significant_coefs:
            if '_lag' in name:
                target, lag = name.split('_lag')
                if target == self.current_target:
                    own_effects.append((int(lag), coef))
                else:
                    cross_effects.append((target, int(lag), coef))
        
        relationships['own_effects'] = sorted(own_effects, key=lambda x: x[0])
        relationships['cross_effects'] = sorted(cross_effects, key=lambda x: abs(x[2]), reverse=True)
        
        # Log findings
        logging.info(f"Own-asset effects: {len(own_effects)} significant lags")
        logging.info(f"Cross-asset effects: {len(cross_effects)} significant relationships")
        
        return relationships

def create_rolling_predictions(df: pl.DataFrame, 
                             model: LassoForecaster,
                             config: ForecastConfig) -> pl.DataFrame:
    """
    Create rolling window predictions with enhanced error handling and logging
    """
    predictions = []
    symbols = df['symbol'].unique()
    
    for symbol in symbols:
        logging.info(f"Processing symbol: {symbol}")
        stock_data = df.filter(pl.col('symbol') == symbol)
        
        valid_windows = 0
        skipped_windows = 0
        
        for window_id in stock_data['window_id'].unique():
            try:
                window_df = df.filter(pl.col('window_id') == window_id)
                
                # Get window size
                window_size = len(window_df.filter(pl.col('symbol') == symbol))
                
                if window_size < config.min_samples:
                    if config.skip_sparse_windows:
                        skipped_windows += 1
                        continue
                    else:
                        logging.warning(f"Small window size for {symbol}, window {window_id}: {window_size} samples")
                
                model.fit_window(window_df, symbol)
                prediction = model.predict_window(window_df, symbol)
                ts_event_start = window_df.filter(pl.col('symbol') == symbol)['ts_event_start'][-1]
                
                predictions.append({
                    'ts_event_start': ts_event_start,
                    'symbol': symbol,
                    'predicted_return': prediction[0],
                    'window_id': window_id
                })
                
                valid_windows += 1
                
            except Exception as e:
                skipped_windows += 1
                if isinstance(e, ValueError) and "Insufficient samples" in str(e):
                    logging.warning(f"Skipped window for {symbol}, window {window_id}: {str(e)}")
                else:
                    logging.error(f"Failed prediction for {symbol}, window {window_id}: {str(e)}")
                continue
        
        logging.info(f"Completed {symbol}: {valid_windows} valid windows, {skipped_windows} skipped")
    
    if not predictions:
        raise ValueError("No valid predictions generated")
        
    pred_df = pl.DataFrame(predictions)
    
    result_df = df.join(pred_df, on=['ts_event_start', 'symbol'], how='left')
    
    # Log summary statistics
    total_windows = len(df['window_id'].unique())
    predicted_windows = len(predictions)
    logging.info(f"Total windows: {total_windows}, Predicted windows: {predicted_windows}")
    logging.info(f"Coverage: {predicted_windows/total_windows:.2%}")
    
    return result_df

def calculate_performance(df: pl.DataFrame, config: ForecastConfig) -> Dict[str, float]:
    """
    Calculate extended performance metrics as described in paper section 4.2
    """
    df = df.drop_nulls('predicted_return')
    metrics = {}
    
    # Per-stock metrics
    for symbol in df['symbol'].unique():
        stock_df = df.filter(pl.col('symbol') == symbol)
        y_true = stock_df['returns_bps'].to_numpy()
        y_pred = stock_df['predicted_return'].to_numpy()
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Additional metrics from paper
        mae = np.mean(np.abs(y_true - y_pred))
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        metrics[f'{symbol}_r2'] = r2
        metrics[f'{symbol}_rmse'] = rmse
        metrics[f'{symbol}_mae'] = mae
        metrics[f'{symbol}_hit_rate'] = hit_rate
    
    # Overall metrics
    y_true_all = df['returns_bps'].to_numpy()
    y_pred_all = df['predicted_return'].to_numpy()
    
    metrics['overall_r2'] = 1 - (np.sum((y_true_all - y_pred_all) ** 2) / 
                                np.sum((y_true_all - np.mean(y_true_all)) ** 2))
    metrics['overall_rmse'] = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
    metrics['overall_mae'] = np.mean(np.abs(y_true_all - y_pred_all))
    metrics['overall_hit_rate'] = np.mean(np.sign(y_true_all) == np.sign(y_pred_all))
    
    return metrics

def run_forecasting(df: pl.DataFrame, config: Optional[ForecastConfig] = None) -> Dict:
    """
    Main function to run the forecasting process with enhanced logging and error handling
    """
    if config is None:
        config = ForecastConfig()
        
    logging.info("Starting forecasting run")
    logging.info(f"Config: {config}")
    
    try:
        model = LassoForecaster(config)
        results_df = create_rolling_predictions(df, model, config)
        metrics = calculate_performance(results_df, config)
        
        logging.info("Forecasting completed successfully")
        logging.info(f"Overall R2: {metrics['overall_r2']:.4f}")
        logging.info(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
        
        return {
            'results': results_df,
            'metrics': metrics,
            'model': model
        }
        
    except Exception as e:
        logging.error(f"Forecasting failed: {str(e)}")
        raise

