from sys import displayhook
from typing import List, Tuple, Dict, Any
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

class OFIConfig:
    def __init__(self, num_levels: int, chunk_seconds: int):
        """
        Configuration class for Order Flow Imbalance (OFI) calculations.

        Parameters:
        -----------
        num_levels : int
            Number of price levels to consider.
        chunk_seconds : int
            Duration of each time chunk in seconds.
        """
        self.num_levels = num_levels
        self.chunk_seconds = chunk_seconds


def process_response(df: pl.DataFrame, config: OFIConfig) -> pl.DataFrame:
    """
    Process the response DataFrame by selecting, renaming, and adjusting columns.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing the aggregated order flow data.
    config : OFIConfig
        Configuration object containing the number of levels and chunk duration.

    Returns:
    --------
    pl.DataFrame
        Processed DataFrame with adjusted timestamps and renamed columns.
    """
    # Select relevant columns
    selected_df = df.select([
        'symbol', 'ts_event_start', 'ts_event_end', 'log_returns', 'returns_bps', 'window_id'
    ] + [f'normalized_ofi_{i:02d}' for i in range(config.num_levels)])

    # Rename columns for clarity
    renamed_df = selected_df.rename({
        f'normalized_ofi_{i:02d}': f'ofi_{i:02d}' for i in range(config.num_levels)
    })

    # Add row count as an index
    renamed_df = renamed_df.with_row_count("index")

    # Adjust timestamps to match actual minute starts and ends
    renamed_df = renamed_df.with_columns([
        (pl.col("ts_event_start").dt.truncate("1m")).alias("ts_event_start"),
        (pl.col("ts_event_end").dt.truncate("1m") + pl.duration(minutes=1) - pl.duration(milliseconds=1)).alias("ts_event_end")
    ])
    
    return renamed_df


def create_order_flow_expressions(level: int) -> List[pl.Expr]:
    """
    Generate expressions for calculating order flow imbalance (OFI) at a specific level.

    Parameters:
    -----------
    level : int
        The price level for which to generate the expressions.

    Returns:
    --------
    List[pl.Expr]
        List of expressions for bid, ask, and depth calculations.
    """
    b_px_col = f"bid_px_{level:02d}"
    b_sz_col = f"bid_sz_{level:02d}"
    a_px_col = f"ask_px_{level:02d}"
    a_sz_col = f"ask_sz_{level:02d}"

    # Expression for bid order flow imbalance
    b_expr = (
        pl.when(pl.col(b_px_col).shift(1).is_null())
        .then(0)
        .when(pl.col(b_px_col) > pl.col(b_px_col).shift(1))
        .then(pl.col(b_sz_col))
        .when(pl.col(b_px_col) == pl.col(b_px_col).shift(1))
        .then(pl.col(b_sz_col) - pl.col(b_sz_col).shift(1))
        .otherwise(-pl.col(b_sz_col))
        .alias(f"ofb_n_{level:02d}")
    )

    # Expression for ask order flow imbalance
    a_expr = (
        pl.when(pl.col(a_px_col).shift(1).is_null())
        .then(0)
        .when(pl.col(a_px_col) > pl.col(a_px_col).shift(1))
        .then(pl.col(a_sz_col))
        .when(pl.col(a_px_col) == pl.col(a_px_col).shift(1))
        .then(pl.col(a_sz_col) - pl.col(a_sz_col).shift(1))
        .otherwise(-pl.col(a_sz_col))
        .alias(f"ofa_n_{level:02d}")
    )

    # Expression for average depth
    depth_expr = ((pl.col(b_sz_col) + pl.col(a_sz_col)) * 0.5).alias(f"depth_{level:02d}")

    return [b_expr, a_expr, depth_expr]


def create_aggregation_expressions(config: OFIConfig) -> List[pl.Expr]:
    """
    Generate expressions for aggregating order flow data.

    Parameters:
    -----------
    config : OFIConfig
        Configuration object containing the number of levels and chunk duration.

    Returns:
    --------
    List[pl.Expr]
        List of expressions for aggregating time, price, OFI, and depth data.
    """
    # Expressions for time aggregation
    time_exprs = [
        pl.col("ts_event").min().alias("ts_event_start"),
        pl.col("ts_event").max().alias("ts_event_end")
    ]

    # Expressions for price aggregation
    price_exprs = [
        pl.col("mid_price").first().alias("mid_price_start"),
        pl.col("mid_price").last().alias("mid_price_end")
    ]

    # Expressions for summing OFI
    sum_exprs = [
        (pl.col(f"ofb_n_{i:02d}") - pl.col(f"ofa_n_{i:02d}"))
        .sum().alias(f"ofi_{i:02d}")
        for i in range(config.num_levels)
    ]

    # Expressions for averaging depth
    avg_depth_exprs = [
        (pl.col(f"depth_{i:02d}").mean()).alias(f"avg_depth_{i:02d}")
        for i in range(config.num_levels)
    ]

    return time_exprs + price_exprs + sum_exprs + avg_depth_exprs


def process_time_chunks(df: pl.DataFrame, config: OFIConfig) -> pl.DataFrame:
    """
    Process order book data in time chunks following the paper's methodology:
    - 1-minute intervals for OFI calculation
    - Trading hours: 10:00 AM - 3:30 PM
    - 30-minute estimation windows
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing raw order book data.
    config : OFIConfig
        Configuration object containing the number of levels.
        
    Returns:
    --------
    pl.DataFrame
        DataFrame with aggregated and processed order flow data.
    """
    # Ensure 1-minute chunks as per paper
    CHUNK_SECONDS = config.chunk_seconds
    
    # Convert timestamps and create chunk IDs
    df = df.with_columns([
        pl.col('symbol'),
        pl.col("ts_event").str.strptime(pl.Datetime),
        (pl.col("ts_event").dt.timestamp("ms") // (CHUNK_SECONDS * 1000)).cast(pl.Int64).alias("chunk_id")
    ])
    
    # Calculate order flows for each level
    expressions = []
    for i in range(config.num_levels):
        expressions.extend(create_order_flow_expressions(i))

    # Add mid price calculation
    expressions.append(
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) * 0.5).alias("mid_price")
    )

    df = df.with_columns(expressions)

    # Aggregate data by 1-minute chunks
    result = (df.group_by("chunk_id", "symbol")
              .agg(create_aggregation_expressions(config))
              .sort("ts_event_start"))

    # Calculate average depth (Q_m)
    result = result.with_columns(
        pl.concat_list([pl.col(f"avg_depth_{i:02d}") for i in range(config.num_levels)])
        .list.mean()
        .alias(f"Q_{config.num_levels:02d}")
    )

    # Calculate normalized OFIs and log returns
    result = result.with_columns([
        *[(pl.col(f"ofi_{i:02d}") / pl.col(f"Q_{config.num_levels:02d}"))
          .alias(f"normalized_ofi_{i:02d}")
          for i in range(config.num_levels)],
        (pl.col("mid_price_end") / pl.col("mid_price_start"))
        .log().alias("log_returns")
    ])

    result = result.with_columns((10000 * (np.exp(pl.col("log_returns")) - 1)).alias("returns_bps"))

    # Create 30-minute rolling windows
    # Each window contains 30 one-minute observations
    result = result.with_columns(
        (pl.col("chunk_id") // 30).alias("window_id")
    )

    # Process the response DataFrame
    response = process_response(result, config)

    return response


def calculate_integrated_ofi(
    df: pl.DataFrame,
    num_levels: int = 10,
    window_size: int = 100
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Calculate Integrated OFI using dynamic PCA estimation on a rolling window.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing normalized OFI columns.
    num_levels : int
        Number of OFI levels to consider.
    window_size : int
        Number of rows in the rolling window for dynamic PCA.

    Returns:
    --------
    Tuple[pl.DataFrame, Dict[str, Any]]
        Input DataFrame with an additional 'dynamic_integrated_ofi' column and a dictionary with PCA results.
    """
    # Get normalized OFI columns
    ofi_cols = [f'ofi_{i:02d}' for i in range(num_levels)]
    X = df.select(ofi_cols).to_numpy()

    # Initialize array to store dynamic integrated OFI
    dynamic_integrated_ofis = np.empty(len(X))
    dynamic_integrated_ofis[:] = np.nan  # Fill with NaN for rows where PCA isn't computed

    # Loop through rolling windows
    for i in range(window_size, len(X)):
        # Get the rolling window data
        window_data = X[i - window_size:i]

        # Standardize the data within the window
        scaler = RobustScaler()
        window_data_scaled = scaler.fit_transform(window_data)

        # Perform PCA
        pca = PCA()
        pca.fit(window_data_scaled)

        # Extract PCA results
        explained_variance_ratios = pca.explained_variance_ratio_
        cumulative_variance_ratios = np.cumsum(explained_variance_ratios)
        first_component_weights = pca.components_[0]

        result_dict = {
            'explained_variance_ratios': explained_variance_ratios,
            'cumulative_variance_ratios': cumulative_variance_ratios,
            'first_component_weights': first_component_weights
        }

        # Get normalized weights for the first principal component
        w1 = pca.components_[0]
        w1_normalized = w1 / np.abs(w1).sum()

        # Calculate integrated OFI for the current observation
        dynamic_integrated_ofis[i] = np.dot(X[i], w1_normalized)

    # Add dynamic integrated OFI to the DataFrame
    result = df.with_columns(
        pl.Series("integrated_ofi", dynamic_integrated_ofis)
    )

    return result, result_dict