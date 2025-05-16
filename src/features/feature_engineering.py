"""
Feature engineering module for cryptocurrency trading.

This module implements comprehensive feature engineering for cryptocurrency data,
including multi-timeframe analysis, technical indicators, and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import logging
from datetime import datetime, timedelta

class FeatureEngineer:
    """Handles feature engineering for cryptocurrency data."""
    
    def __init__(self, config: dict):
        """
        Initialize the feature engineer.
        
        Args:
            config: Dictionary containing feature engineering configuration
        """
        self.config = config
        self.timeframes = ['1h', '2h', '4h', '12h', '24h', '7d']
        self.sma_periods = [5, 10, 20, 50, 100, 200]
        self.ema_periods = [5, 10, 20, 50, 100, 200]
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added features
        """
        try:
            # Ensure datetime index
            if 'Open time' not in df.columns:
                raise ValueError("DataFrame must have 'Open time' column")
            
            # Convert column names to lowercase for consistency
            df.columns = df.columns.str.lower()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open time': 'open_time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Convert Open time to datetime and set as index
            df['open_time'] = pd.to_datetime(df['open_time'])
            df.set_index('open_time', inplace=True)
            
            # Store original price columns
            df_original = df.copy()
            
            # Build features for each timeframe
            for tf in self.timeframes:
                df = self._build_timeframe_features(df_original, df, tf)
            
            # Add candlestick patterns using original timeframe data
            df = self._add_candlestick_patterns(df)
            
            # Add risk metrics using original timeframe data
            df = self._add_risk_metrics(df)
            
            # Add cyclical features
            df = self._add_cyclical_features(df)
            
            # Handle NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            # Reset index
            df.reset_index(inplace=True)
            
            logging.info(f"Generated {len(df.columns)} features")
            return df
            
        except Exception as e:
            logging.error(f"Error in build_features: {e}")
            raise
    
    def _build_timeframe_features(self, df_original: pd.DataFrame, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Build features for a specific timeframe.
        
        Args:
            df_original: Original DataFrame with OHLCV data
            df: DataFrame with features
            timeframe: Timeframe to build features for
            
        Returns:
            DataFrame with added features
        """
        try:
            # Assicura che i nomi delle colonne siano in minuscolo
            df_original = df_original.copy()
            df_original.columns = df_original.columns.str.lower()
            # Resample data to target timeframe
            resampled = df_original.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Calculate features
            resampled = self._add_trend_features(resampled, timeframe)
            resampled = self._add_momentum_features(resampled, timeframe)
            resampled = self._add_volatility_features(resampled, timeframe)
            
            # Add timeframe suffix to price columns
            rename_dict = {
                'open': f'open_{timeframe}',
                'high': f'high_{timeframe}',
                'low': f'low_{timeframe}',
                'close': f'close_{timeframe}',
                'volume': f'volume_{timeframe}'
            }
            resampled = resampled.rename(columns=rename_dict)
            
            # Merge back to original timeframe
            df = pd.merge_asof(
                df.reset_index(),
                resampled.reset_index(),
                on='open_time',
                direction='backward'
            ).set_index('open_time')
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _build_timeframe_features for {timeframe}: {e}")
            raise
    
    def _add_trend_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add trend-based features."""
        try:
            # SMA
            for period in self.sma_periods:
                sma = SMAIndicator(close=df['close'], window=period)
                df[f'sma_{period}_{timeframe}'] = sma.sma_indicator()
            
            # EMA
            for period in self.ema_periods:
                ema = EMAIndicator(close=df['close'], window=period)
                df[f'ema_{period}_{timeframe}'] = ema.ema_indicator()
            
            # MACD (only for 1h and 4h)
            if timeframe in ['1h', '4h']:
                macd = MACD(close=df['close'])
                df[f'macd_{timeframe}'] = macd.macd()
                df[f'macd_signal_{timeframe}'] = macd.macd_signal()
                df[f'macd_hist_{timeframe}'] = macd.macd_diff()
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_trend_features: {e}")
            raise
    
    def _add_momentum_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add momentum-based features."""
        try:
            # RSI
            if timeframe in ['1h', '6h']:
                rsi = RSIIndicator(close=df['close'])
                df[f'rsi_{timeframe}'] = rsi.rsi()
                
                # Stochastic RSI
                stoch_rsi = StochasticOscillator(
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )
                df[f'stoch_k_{timeframe}'] = stoch_rsi.stoch()
                df[f'stoch_d_{timeframe}'] = stoch_rsi.stoch_signal()
            
            # ROC and Momentum
            if timeframe in ['1h', '3h', '12h']:
                df[f'roc_{timeframe}'] = df['close'].pct_change(periods=1, fill_method=None)
                df[f'momentum_{timeframe}'] = df['close'].diff(periods=1)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_momentum_features: {e}")
            raise
    
    def _add_volatility_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add volatility-based features."""
        try:
            if len(df) == 0:
                logging.warning(f"Empty DataFrame for timeframe {timeframe}")
                return df
                
            # ATR
            if timeframe in ['1h', '4h']:
                try:
                    atr = AverageTrueRange(
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        window=14,
                        fillna=True
                    )
                    df[f'atr_{timeframe}'] = atr.average_true_range()
                except Exception as e:
                    logging.error(f"Error calculating ATR for {timeframe}: {e}")
                    df[f'atr_{timeframe}'] = np.nan
            
            # Bollinger Bands
            if timeframe in ['1h', '24h']:
                try:
                    bb = BollingerBands(close=df['close'])
                    df[f'bb_upper_{timeframe}'] = bb.bollinger_hband()
                    df[f'bb_lower_{timeframe}'] = bb.bollinger_lband()
                    df[f'bb_width_{timeframe}'] = bb.bollinger_wband()
                    df[f'bb_pct_b_{timeframe}'] = bb.bollinger_pband()
                except Exception as e:
                    logging.error(f"Error calculating Bollinger Bands for {timeframe}: {e}")
                    for col in ['upper', 'lower', 'width', 'pct_b']:
                        df[f'bb_{col}_{timeframe}'] = np.nan
            
            # Rolling standard deviation
            for period in [20, 50, 100]:
                try:
                    df[f'returns_std_{period}_{timeframe}'] = (
                        df['close'].pct_change(fill_method=None)
                        .rolling(period, min_periods=1)
                        .std()
                        .fillna(0)
                    )
                except Exception as e:
                    logging.error(f"Error calculating rolling std for period {period} and timeframe {timeframe}: {e}")
                    df[f'returns_std_{period}_{timeframe}'] = np.nan
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_volatility_features: {e}")
            raise
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        try:
            # Get the base columns (without timeframe suffix)
            base_cols = {
                'open': next(col for col in df.columns if col.startswith('open_')),
                'high': next(col for col in df.columns if col.startswith('high_')),
                'low': next(col for col in df.columns if col.startswith('low_')),
                'close': next(col for col in df.columns if col.startswith('close_')),
                'volume': next(col for col in df.columns if col.startswith('volume_'))
            }
            
            # Basic candlestick properties
            df['body_size'] = abs(df[base_cols['close']] - df[base_cols['open']])
            df['upper_wick'] = df[base_cols['high']] - df[[base_cols['open'], base_cols['close']]].max(axis=1)
            df['lower_wick'] = df[[base_cols['open'], base_cols['close']]].min(axis=1) - df[base_cols['low']]
            df['range'] = df[base_cols['high']] - df[base_cols['low']]
            
            # Pattern detection
            df['is_doji'] = (df['body_size'] <= 0.1 * df['range']).astype(int)
            df['is_hammer'] = (
                (df['lower_wick'] > 2 * df['body_size']) &
                (df['upper_wick'] < df['body_size'])
            ).astype(int)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_candlestick_patterns: {e}")
            raise
    
    def _add_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add risk-based features."""
        try:
            # Get the base close column (without timeframe suffix)
            close_col = next(col for col in df.columns if col.startswith('close_'))
            
            # Rolling max drawdown
            for window in [24, 168]:  # 24h e 7d
                returns = df[close_col].pct_change(fill_method=None)
                rolling_max = returns.rolling(window=window).max()
                drawdown = (returns - rolling_max) / (rolling_max + 1e-10)  # Evita divisione per zero
                df[f'max_drawdown_{window}h'] = drawdown.rolling(window=window).min()
            
            # Sharpe and Sortino ratios
            for window in [24, 168]:  # 24h e 7d
                returns = df[close_col].pct_change(fill_method=None)
                excess_returns = returns - 0.02/365  # 2% risk-free rate annualizzato
                
                # Sharpe ratio
                df[f'sharpe_{window}h'] = (
                    excess_returns.rolling(window=window).mean() /
                    (returns.rolling(window=window).std() + 1e-10)  # Evita divisione per zero
                )
                
                # Sortino ratio (solo rendimenti negativi)
                negative_returns = returns[returns < 0]
                df[f'sortino_{window}h'] = (
                    excess_returns.rolling(window=window).mean() /
                    (negative_returns.rolling(window=window).std() + 1e-10)  # Evita divisione per zero
                )
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_risk_metrics: {e}")
            raise
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical features."""
        try:
            # Hour of day (0-23)
            df['hour'] = df.index.hour
            hour_dummies = pd.get_dummies(df['hour'], prefix='hour')
            df = pd.concat([df, hour_dummies], axis=1)
            
            # Day of week (0-6)
            df['day'] = df.index.dayofweek
            day_dummies = pd.get_dummies(df['day'], prefix='day')
            df = pd.concat([df, day_dummies], axis=1)
            
            # Get the base close column (without timeframe suffix)
            close_col = next(col for col in df.columns if col.startswith('close_'))
            
            # FFT for periodicity detection
            for window in [24, 168]:  # 24h e 7d
                returns = df[close_col].pct_change(fill_method=None)
                # Calculate FFT for each window
                fft = np.fft.fft(returns.fillna(0))
                # Take absolute value and normalize
                fft_abs = np.abs(fft)[:len(df)]
                df[f'fft_amp_{window}h'] = fft_abs / np.max(fft_abs)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in _add_cyclical_features: {e}")
            raise 