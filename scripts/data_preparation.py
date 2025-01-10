import os
from datetime import time
from dataclasses import dataclass
from functools import reduce
from typing import List
from urllib.parse import urljoin

import polars as pl
import pytz
import requests
import yaml
from bs4 import BeautifulSoup

def download_parquet_files(base_url, download_dir):
    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Get the list of files from the base URL
    response = requests.get(base_url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links ending with .parquet
    parquet_links = [urljoin(base_url, link.get('href')) for link in soup.find_all('a') if link.get('href').endswith('.parquet')]

    # Download each parquet file
    for link in parquet_links:
        file_name = os.path.join(download_dir, os.path.basename(link))
        print(f"Downloading {link} to {file_name}")
        with requests.get(link, stream=True) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")


@dataclass
class OFIConfig:
    num_levels: int = 10
    chunk_seconds: int = 60

def load_symbol_data(current_dir: str, symbol_name: str) -> pl.DataFrame:
    """
    Load symbol data from a parquet file specified in the config.yaml file.

    Parameters:
        symbol_name (str): The name of the symbol to load data for.

    Returns:
        pl.DataFrame: The loaded data as a Polars DataFrame.

    Raises:
        ValueError: If the symbol name is not found in the config or if the file path is invalid.
        IOError: If there is an error loading the parquet file.
    """
    # Construct the path to the config file
    config_path = os.path.join(current_dir, '..', 'notebooks', 'config.yaml')
    
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Check if the symbol name exists in the config
    if symbol_name not in config:
        raise ValueError(f"Symbol '{symbol_name}' not found in config.yaml")
    
    # Get the file path for the symbol
    filename = config[symbol_name]
    
    # Check if the file path is valid
    if not filename or not isinstance(filename, str):
        raise ValueError(f"Invalid file path for symbol '{symbol_name}' in config.yaml")
    
    # Load the parquet file into a Polars DataFrame
    try:
        # Construct the relative path to the file
        data_path = os.path.join(current_dir, '..', 'data', filename)
        
        # Print the constructed file path for debugging
        print(f"Loading file from path: {data_path}")

        df = pl.read_parquet(data_path)
        df = df.with_columns(pl.lit(symbol_name).alias("symbol"))
        
    except Exception as e:
        raise IOError(f"Error loading file '{data_path}': {e}")
    
    return df

def filter_by_time(data: pl.DataFrame, start_time: time, end_time: time) -> pl.DataFrame:
    """
    Filters a Polars DataFrame based on the time of day in the Eastern Time zone.

    Parameters:
        data (pl.DataFrame): The Polars DataFrame with a ts_event_start column.
        start_time (time): The start time for filtering (Eastern Time).
        end_time (time): The end time for filtering (Eastern Time).

    Returns:
        pl.DataFrame: Filtered DataFrame with rows within the specified time range.

    Raises:
        ValueError: If the DataFrame does not have a 'ts_event_start' column.
    """
    if 'ts_event_start' not in data.columns:
        raise ValueError("DataFrame must have a 'ts_event_start' column.")

    # Ensure the 'ts_event_start' column is parsed as ts_event_start if not already
    if data['ts_event_start'].dtype != pl.ts_event_start:
        data = data.with_columns(pl.col('ts_event_start').cast(pl.ts_event_start))

    # Convert ts_event_start column to Eastern Time zone
    eastern = pytz.timezone("US/Eastern")
    data = data.with_columns(
        pl.col('ts_event_start').apply(lambda dt: dt.astimezone(eastern)).alias('ts_event_start_eastern')
    )

    # Add a time column derived from the ts_event_start column in Eastern Time
    data = data.with_columns(data['ts_event_start_eastern'].dt.time().alias('time'))

    # Filter the DataFrame based on the time range
    filtered_data = data.filter((pl.col('time') >= start_time) & (pl.col('time') <= end_time))

    # Drop the temporary 'time' column
    filtered_data = filtered_data.drop(['time', 'ts_event_start_eastern'])

    return filtered_data


def remove_outliers(df: pl.DataFrame, columns: list, z_thresh: float = 3.0) -> pl.DataFrame:
    """
    Remove outliers from the specified columns in the DataFrame using the Z-score method.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame from which to remove outliers.
    columns : list
        List of column names to check for outliers.
    z_thresh : float
        Z-score threshold to identify outliers.
    
    Returns:
    --------
    pl.DataFrame
        DataFrame with outliers removed.
    """
    for column in columns:
        mean = df.select(column).mean()
        std = df.select(column).std()
        
        df = df.with_columns(((pl.col(column) - mean) / std).alias(f"{column}_z_score"))
        
        df = df.filter(pl.col(f"{column}_z_score").abs() <= z_thresh)
                
        df = df.drop(f"{column}_z_score")
    
    return df

def filter_and_stack_common_ids(datasets: dict, symbols: list) -> dict:
    """
    Filter and stack dataframes in the datasets dictionary based on common IDs across specified symbols.

    Parameters:
        datasets (dict): Dictionary of datasets where the key is a string and the value is a Polars DataFrame.
        symbols (list): List of symbols to filter and stack.

    Returns:
        dict: Updated dictionary with filtered and stacked dataframes.
    """

    for key in datasets:
        data = datasets[key]
        dfs = []

        for symbol in symbols:
            symbol_df = data.filter(pl.col('symbol') == symbol)
            dfs.append(symbol_df)

        # Find common IDs across all dataframes
        common_ids = reduce(lambda x, y: x.intersection(y), 
                            [set(df['ts_event_start']) for df in dfs])

        # Filter and stack
        result = pl.concat([
            df.filter(pl.col('ts_event_start').is_in(common_ids))
            for df in dfs
        ], how='vertical')

        datasets[key] = result.sort('ts_event_start')
    
    return datasets