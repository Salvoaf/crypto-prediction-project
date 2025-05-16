"""
Data loading and preprocessing module for cryptocurrency trading.

This module provides classes and functions for loading, cleaning, and preprocessing
cryptocurrency market data. It handles data loading from various sources, feature
engineering, and data normalization.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime
from src.features.feature_engineering import FeatureEngineer

class CryptoDataLoader:
    """Handles loading and preprocessing of cryptocurrency market data."""
    
    def __init__(self, config: Union[str, dict]):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Path to the YAML configuration file or configuration dictionary
        """
        if isinstance(config, str):
            self.config = self._load_config(config)
        else:
            self.config = config
            
        self.data_dir = Path(self.config['data']['data_dir'])
        self.sequence_length = self.config['features']['sequence_length']
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(self.config)
        
        # Set price index (Close price is typically used for trading)
        self.price_idx = 3  # Index of 'Close' in the feature columns
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, mode: str = 'train') -> pd.DataFrame:
        """
        Load BTC data for training (until 2020) or testing (2020 onwards).
        
        Args:
            mode: 'train' or 'test' to specify which period to load
            
        Returns:
            DataFrame containing the loaded data
        """
        try:
            file_path = self.data_dir / "BTCUSDT.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) == 0:
                raise ValueError(f"File {file_path} is empty or has no columns")
            
            # Debug: stampa i nomi delle colonne
            logging.info(f"Original columns: {df.columns.tolist()}")
            
            # Convert timestamp to datetime
            df['Open time'] = pd.to_datetime(df['Open time'])
            
            # Filter by period
            if mode == 'train':
                df = df[df['Open time'] < self.config['evaluation']['test_start_date']]
            else:  # test
                df = df[
                    (df['Open time'] >= self.config['evaluation']['test_start_date']) & 
                    (df['Open time'] <= self.config['evaluation']['test_end_date'])
                ]
            
            # Sort by date
            df = df.sort_values('Open time')
            
            # Debug: stampa i nomi delle colonne prima del feature engineering
            logging.info(f"Columns before feature engineering: {df.columns.tolist()}")
            
            # Build features
            df = self.feature_engineer.build_features(df)
            
            # Debug: stampa i nomi delle colonne dopo il feature engineering
            logging.info(f"Columns after feature engineering: {df.columns.tolist()}")
            
            # Update price_idx based on the actual column order
            feature_cols = [col for col in df.columns if col not in ['Open time']]
            
            # Try different possible column names for the close price
            close_price_names = ['Close', 'close', 'CLOSE', 'close_price', 'Close_price']
            for name in close_price_names:
                if name in feature_cols:
                    self.price_idx = feature_cols.index(name)
                    logging.info(f"Found close price column: {name} at index {self.price_idx}")
                    break
            else:
                raise ValueError(f"Could not find close price column. Available columns: {feature_cols}")
            
            logging.info(f"Loaded {len(df)} rows for {mode} period")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for training.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (sequences, timestamps)
        """
        try:
            # Get feature columns (exclude timestamp and target)
            feature_cols = [col for col in df.columns if col not in ['open_time']]
            
            # Convert features to numpy array and handle NaN values
            features_array = df[feature_cols].ffill().bfill().values.astype(np.float32)
            timestamps_array = df['open_time'].values
            
            # Normalize features
            mean = np.nanmean(features_array, axis=0)
            std = np.nanstd(features_array, axis=0)
            std[std < 1e-8] = 1.0  # Avoid division by zero
            features_array = (features_array - mean) / std
            
            # Create sequences
            sequences = []
            sequence_timestamps = []
            
            for i in range(len(features_array) - self.config['features']['sequence_length'] + 1):
                sequence = features_array[i:i + self.config['features']['sequence_length']]
                sequences.append(sequence)
                sequence_timestamps.append(timestamps_array[i + self.config['features']['sequence_length'] - 1])
            
            sequences = np.array(sequences, dtype=np.float32)
            sequence_timestamps = np.array(sequence_timestamps)
            
            logging.info(f"Generated sequences shape: {sequences.shape}")
            logging.info(f"Generated timestamps shape: {sequence_timestamps.shape}")
            
            return sequences, sequence_timestamps
            
        except Exception as e:
            logging.error(f"Error preparing sequences: {e}")
            raise
    
    def split_data(self, X: np.ndarray, timestamps: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train and validation sets based on dates.
        
        Args:
            X: Feature sequences
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary containing train and validation splits
        """
        validation_months = self.config['data']['validation_months']
        validation_mask = np.array([
            any((pd.Timestamp(ts).month == month and pd.Timestamp(ts).year == year)
                for month, year in validation_months)
            for ts in timestamps
        ])
        
        return {
            'train': (X[~validation_mask], timestamps[~validation_mask]),
            'validation': (X[validation_mask], timestamps[validation_mask])
        }
    
    def aggregate_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggrega i dati orari a giornalieri usando la libreria ta.
        Usa Open dell'ora 0, Close dell'ora 23, High e Low come massimi e minimi delle 24 ore.
        """
        # Assicurati che 'Open time' sia datetime
        df['Open time'] = pd.to_datetime(df['Open time'])
        # Raggruppa per data (ignorando l'ora)
        df['date'] = df['Open time'].dt.date
        # Aggrega usando Open dell'ora 0, Close dell'ora 23, e max/min per High/Low
        daily = df.groupby('date').agg({
            'Open': 'first',  # Open dell'ora 0
            'High': 'max',    # Massimo delle 24 ore
            'Low': 'min',     # Minimo delle 24 ore
            'Close': 'last',  # Close dell'ora 23
            'Volume': 'sum',  # Volume totale giornaliero
            'symbol': 'first'  # Mantieni il simbolo
        }).reset_index()
        # Rinomina 'date' in 'Open time' per compatibilità
        daily.rename(columns={'date': 'Open time'}, inplace=True)
        return daily 