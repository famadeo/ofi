import logging
from typing import Dict, Any
import polars as pl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

class ForecastingVisualizer:
    """
    Visualization tools for the cross-impact forecasting results.
    Implementation of visualizations from the paper.
    """
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize with results from run_forecasting.
        
        Parameters:
        - results: Dictionary containing:
            - results: DataFrame with predictions
            - metrics: Dictionary of performance metrics
            - model: Fitted LassoForecaster
        """
        self.results_df = results['results']
        self.metrics = results['metrics']
        self.model = results['model']
        self.symbols = self.results_df['symbol'].unique()

        # Set style for all plots
        plt.style.use('classic')
    
    def plot_coefficient_heatmap(self):
        """
        Create heatmap of cross-impact coefficients with corrected reshaping.
        
        Returns:
        - plt.Figure: The generated heatmap figure.
        """
        plt.figure(figsize=(15, 12))
        
        n_symbols = len(self.symbols)
        n_lags = len(self.model.config.lags)
        total_coefs = len(self.model.lasso.coef_)
        
        logging.info(f"Reshaping coefficient matrix:")
        logging.info(f"Number of symbols: {n_symbols}")
        logging.info(f"Number of lags: {n_lags}")
        logging.info(f"Total coefficients: {total_coefs}")
        
        # For LASSO coefficients with multiple lags, average across lags
        coef_matrix = np.zeros((n_symbols, n_symbols))
        coefs_per_lag = total_coefs // n_lags
        
        # Assuming coefficients are organized as [lag1_sym1, lag1_sym2, ..., lag2_sym1, lag2_sym2, ...]
        for lag in range(n_lags):
            start_idx = lag * coefs_per_lag
            end_idx = start_idx + coefs_per_lag
            lag_coefs = self.model.lasso.coef_[start_idx:end_idx]
            
            # Add contribution from this lag
            coef_matrix += np.reshape(lag_coefs, (n_symbols, 1)) / n_lags
        
        # Create heatmap
        sns.heatmap(coef_matrix, 
                xticklabels=self.symbols,
                yticklabels=self.symbols,
                center=0,
                cmap='RdBu_r')
        
        plt.title('Cross-Impact Coefficient Matrix (Averaged Across Lags)')
        plt.xlabel('Predictor Stock')
        plt.ylabel('Target Stock')
        
        return plt.gcf()

    def plot_coefficient_network(self, threshold_percentile: float = 95):
        """
        Create network visualization with corrected coefficient handling.
        
        Parameters:
        - threshold_percentile: float, the percentile threshold for including edges in the network.
        
        Returns:
        - plt.Figure: The generated network figure.
        """
        plt.figure(figsize=(15, 15))
        
        n_symbols = len(self.symbols)
        n_lags = len(self.model.config.lags)
        total_coefs = len(self.model.lasso.coef_)
        
        # Create the same coefficient matrix as in heatmap
        coef_matrix = np.zeros((n_symbols, n_symbols))
        coefs_per_lag = total_coefs // n_lags
        
        for lag in range(n_lags):
            start_idx = lag * coefs_per_lag
            end_idx = start_idx + coefs_per_lag
            lag_coefs = self.model.lasso.coef_[start_idx:end_idx]
            coef_matrix += np.reshape(lag_coefs, (n_symbols, 1)) / n_lags
        
        # Create network
        G = nx.DiGraph()
        
        # Add nodes
        for symbol in self.symbols:
            G.add_node(symbol)
        
        # Add edges for coefficients above threshold
        threshold = np.percentile(np.abs(coef_matrix), threshold_percentile)
        logging.info(f"Network threshold: {threshold}")
        
        edges_added = 0
        for i, from_symbol in enumerate(self.symbols):
            for j, to_symbol in enumerate(self.symbols):
                if abs(coef_matrix[i, j]) > threshold:
                    G.add_edge(from_symbol, to_symbol, 
                            weight=coef_matrix[i, j],
                            color='g' if coef_matrix[i, j] > 0 else 'r')
                    edges_added += 1
        
        logging.info(f"Added {edges_added} edges to network")
        
        # Draw network
        pos = nx.spring_layout(G)
        edges = G.edges()
        
        if edges:
            colors = [G[u][v]['color'] for u, v in edges]
            weights = [abs(G[u][v]['weight']) * 3 for u, v in edges]
            
            nx.draw(G, pos, node_color='lightblue', 
                    node_size=500, arrowsize=20,
                    edge_color=colors, width=weights,
                    with_labels=True)
        else:
            nx.draw(G, pos, node_color='lightblue', 
                    node_size=500, with_labels=True)
            plt.text(0.5, 0.5, "No significant connections", 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.title(f'Cross-Impact Network (threshold: {threshold_percentile}th percentile)')
        
        return plt.gcf()

    def inspect_coefficient_structure(self):
        """
        Enhanced debug helper to understand coefficient structure.
        """
        n_symbols = len(self.symbols)
        n_lags = len(self.model.config.lags)
        total_coefs = len(self.model.lasso.coef_)
        
        logging.info("=== Coefficient Structure Analysis ===")
        logging.info(f"Number of symbols: {n_symbols}")
        logging.info(f"Number of lags: {n_lags}")
        logging.info(f"Coefficient shape: {self.model.lasso.coef_.shape}")
        logging.info(f"Total coefficients: {total_coefs}")
        logging.info(f"Coefficients per lag: {total_coefs // n_lags}")
        
        # Print coefficients by lag
        for lag in range(n_lags):
            start_idx = lag * (total_coefs // n_lags)
            end_idx = start_idx + (total_coefs // n_lags)
            logging.info(f"\nLag {lag + 1} coefficients:")
            logging.info(f"Indices {start_idx}:{end_idx}")
            logging.info(f"Values: {self.model.lasso.coef_[start_idx:end_idx]}")
        
        # Print non-zero coefficients
        non_zero_mask = self.model.lasso.coef_ != 0
        non_zero_coefs = self.model.lasso.coef_[non_zero_mask]
        non_zero_indices = np.where(non_zero_mask)[0]
        
        logging.info(f"\nNumber of non-zero coefficients: {len(non_zero_coefs)}")
        if len(non_zero_coefs) > 0:
            logging.info("Non-zero coefficients:")
            for idx, coef in zip(non_zero_indices, non_zero_coefs):
                lag_num = idx // (total_coefs // n_lags) + 1
                sym_num = idx % (total_coefs // n_lags)
                logging.info(f"Lag {lag_num}, Symbol {sym_num}: {coef}")

    def plot_performance_metrics(self):
        """
        Plot various performance metrics based on performance analysis in paper Section 4.2.
        
        Returns:
        - plt.Figure: The generated performance metrics figure.
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Create a 2x2 grid
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # R-squared by stock
        r2_metrics = {k: v for k, v in self.metrics.items() if '_r2' in k and 'overall' not in k}
        if r2_metrics:
            ax1 = fig.add_subplot(gs[0, 0])
            r2_values = list(r2_metrics.values())
            r2_symbols = [k.split('_')[0] for k in r2_metrics.keys()]
            ax1.bar(range(len(r2_values)), r2_values)
            ax1.set_xticks(range(len(r2_values)))
            ax1.set_xticklabels(r2_symbols, rotation=45)
            ax1.set_title('R-squared by Stock')
        
        # RMSE by stock
        rmse_metrics = {k: v for k, v in self.metrics.items() if '_rmse' in k and 'overall' not in k}
        if rmse_metrics:
            ax2 = fig.add_subplot(gs[0, 1])
            rmse_values = list(rmse_metrics.values())
            rmse_symbols = [k.split('_')[0] for k in rmse_metrics.keys()]
            ax2.bar(range(len(rmse_values)), rmse_values)
            ax2.set_xticks(range(len(rmse_values)))
            ax2.set_xticklabels(rmse_symbols, rotation=45)
            ax2.set_title('RMSE by Stock')
        
        # Predicted vs Actual Returns
        ax3 = fig.add_subplot(gs[1, 0])
        actual_returns = self.results_df['returns_bps'].to_numpy()
        predicted_returns = self.results_df['predicted_return'].to_numpy()
        
        # Remove NaN values
        mask = ~(np.isnan(actual_returns) | np.isnan(predicted_returns))
        actual_returns = actual_returns[mask]
        predicted_returns = predicted_returns[mask]
        
        if len(actual_returns) > 0:
            ax3.scatter(actual_returns, predicted_returns, alpha=0.5)
            ax3.plot([min(actual_returns), max(actual_returns)], 
                    [min(actual_returns), max(actual_returns)], 'r--')
            ax3.set_title('Predicted vs Actual Returns')
            ax3.set_xlabel('Actual Returns (bps)')
            ax3.set_ylabel('Predicted Returns (bps)')
        
        # Hit Rate by stock
        hit_metrics = {k: v for k, v in self.metrics.items() if '_hit_rate' in k and 'overall' not in k}
        if hit_metrics:
            ax4 = fig.add_subplot(gs[1, 1])
            hit_values = list(hit_metrics.values())
            hit_symbols = [k.split('_')[0] for k in hit_metrics.keys()]
            ax4.bar(range(len(hit_values)), hit_values)
            ax4.set_xticks(range(len(hit_values)))
            ax4.set_xticklabels(hit_symbols, rotation=45)
            ax4.set_title('Hit Rate by Stock')
        
        plt.tight_layout()
        return fig  
    
    def plot_temporal_prediction_analysis(self):
        """
        Analyze prediction performance over time using Polars DataFrame.
        
        Returns:
        - plt.Figure: The generated temporal prediction analysis figure.
        """
        plt.figure(figsize=(15, 10))
        
        # Use with_columns for Polars DataFrame
        plot_df = (self.results_df
                .with_columns(pl.col('ts_event_start')
                            .cast(pl.Datetime).alias('datetime')))
        
        # Convert to pandas for rolling calculation
        # First select only needed columns
        plot_data = (plot_df
                    .select(['datetime', 'returns_bps', 'predicted_return'])
                    .to_pandas())
        
        # Calculate rolling accuracy
        window_size = 100
        rolling_rmse = np.sqrt(
            (plot_data['returns_bps'] - plot_data['predicted_return'])**2
        ).rolling(window_size).mean()
        
        plt.plot(plot_data['datetime'], rolling_rmse)
        plt.title(f'Rolling RMSE (window size: {window_size})')
        plt.xlabel('Time')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        return plt.gcf()
    
    def plot_feature_importance(self):
        """
        Visualize feature importance based on LASSO coefficients.
        
        Returns:
        - plt.Figure: The generated feature importance figure.
        """
        plt.figure(figsize=(15, 8))
        
        # Get non-zero coefficients and their feature names
        coef_mask = self.model.lasso.coef_ != 0
        significant_coefs = self.model.lasso.coef_[coef_mask]
        significant_features = np.array(self.model.feature_names)[coef_mask]
        
        # Sort by absolute coefficient value
        sort_idx = np.argsort(np.abs(significant_coefs))
        significant_coefs = significant_coefs[sort_idx]
        significant_features = significant_features[sort_idx]
        
        plt.barh(range(len(significant_coefs)), significant_coefs)
        plt.yticks(range(len(significant_coefs)), significant_features)
        plt.title('Significant Feature Coefficients')
        plt.xlabel('Coefficient Value')
        
        return plt.gcf()
            
    def plot_all(self, save_path: str = None):
        """
        Generate all visualizations and optionally save to file.
        
        Parameters:
        - save_path: str, optional path to save the generated figures.
        
        Returns:
        - Dict[str, plt.Figure]: Dictionary of generated figures.
        """
        figs = {}
        
        try:
            #figs['network'] = self.plot_coefficient_network()
            figs['heatmap'] = self.plot_coefficient_heatmap()
            figs['metrics'] = self.plot_performance_metrics()
            #figs['temporal'] = self.plot_temporal_prediction_analysis()
            figs['importance'] = self.plot_feature_importance()
            
            if save_path:
                for name, fig in figs.items():
                    fig.savefig(f"{save_path}/{name}.png")
                    plt.close(fig)
                    
            logging.info("Successfully generated all visualizations")
            return figs
            
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            raise
            
    def generate_summary_report(self) -> str:
        """
        Generate text summary of key findings.
        
        Returns:
        - str: The generated summary report.
        """
        summary = []
        
        # Overall performance
        summary.append("=== Overall Performance ===")
        summary.append(f"R-squared: {self.metrics['overall_r2']:.4f}")
        summary.append(f"RMSE: {self.metrics['overall_rmse']:.4f}")
        summary.append(f"Hit Rate: {self.metrics['overall_hit_rate']:.4f}")
        
        # Model sparsity
        n_features = len(self.model.lasso.coef_)
        n_nonzero = np.sum(self.model.lasso.coef_ != 0)
        sparsity = 1 - (n_nonzero / n_features)
        summary.append(f"\nModel uses {n_nonzero} of {n_features} features ({sparsity:.1%} sparsity)")
        
        # Top positive and negative impacts
        coef_df = pd.DataFrame({
            'feature': self.model.feature_names,
            'coefficient': self.model.lasso.coef_
        })
        
        top_pos = coef_df.nlargest(5, 'coefficient')
        top_neg = coef_df.nsmallest(5, 'coefficient')
        
        summary.append("\n=== Top Positive Impacts ===")
        for _, row in top_pos.iterrows():
            summary.append(f"{row['feature']}: {row['coefficient']:.4f}")
            
        summary.append("\n=== Top Negative Impacts ===")
        for _, row in top_neg.iterrows():
            summary.append(f"{row['feature']}: {row['coefficient']:.4f}")
        
        return "\n".join(summary)