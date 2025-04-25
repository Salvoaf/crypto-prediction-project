# scripts/feature_engineering.py

import os
import pandas as pd
from typing import List, Optional
from scripts.preprocessing import load_hourly_csv, normalize_timestamp
from scripts.technical_indicators import TechnicalIndicators

def build_features(
    df: pd.DataFrame,
    add_basic: bool = True,
    advanced_indicators: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    df: DataFrame normalizzato e indicizzato
    add_basic: se True aggiunge SMA, EMA, RSI, MACD, BB, ATR, lag
    advanced_indicators: lista di indicatori da TechnicalIndicators.calculate_indicators
                         (es. ['bollinger_bands','ema','rsi','stoch_rsi','wave_trend','pps_killer','macd'])
    """
    df_feat = df.copy()

    if add_basic:
        # --- BASIC FEATURES ---
        # SMA ed EMA
        for w in [7, 14, 30]:
            df_feat[f'SMA_{w}'] = df_feat['Close'].rolling(w).mean()
            df_feat[f'EMA_{w}'] = df_feat['Close'].ewm(span=w, adjust=False).mean()

        # RSI
        delta = df_feat['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        df_feat['RSI_14'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))

        # MACD
        ema_fast = df_feat['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = df_feat['Close'].ewm(span=26, adjust=False).mean()
        df_feat['MACD_line']   = ema_fast - ema_slow
        df_feat['MACD_signal'] = df_feat['MACD_line'].ewm(span=9, adjust=False).mean()
        df_feat['MACD_hist']   = df_feat['MACD_line'] - df_feat['MACD_signal']

        # Bollinger Bands
        sma20 = df_feat['Close'].rolling(20).mean()
        std20 = df_feat['Close'].rolling(20).std()
        df_feat['BB_upper'] = sma20 + 2*std20
        df_feat['BB_lower'] = sma20 - 2*std20

        # ATR
        hl = df_feat['High'] - df_feat['Low']
        hc = (df_feat['High'] - df_feat['Close'].shift()).abs()
        lc = (df_feat['Low']  - df_feat['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df_feat['ATR_14'] = tr.rolling(14).mean()

        # Lag features
        for lag in [1,2,3,6,12,24]:
            df_feat[f'Close_lag_{lag}'] = df_feat['Close'].shift(lag)

    # --- ADVANCED INDICATORS via TechnicalIndicators ---
    if advanced_indicators is not None:
        df_feat = TechnicalIndicators.calculate_indicators(df_feat, advanced_indicators)

    # Rimuovi NaN dovuti a rolling/shift
    df_feat = df_feat.dropna()

    return df_feat

from scripts.preprocessing import load_hourly_csv, normalize_timestamp
from scripts.feature_engineering import build_features

# Carica e normalizza
df = load_hourly_csv('BTC')
df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')

# Richiama build_features senza salvataggi interni
features = build_features(
    df,
    add_basic=True,
    advanced_indicators=[
        'bollinger_bands',
        'ema',
        'rsi',
        'stoch_rsi',
        'wave_trend',
        'pps_killer',
        'macd'
    ]
)

# A questo punto 'features' è il tuo DataFrame completo, in memoria,
# e puoi salvarlo o usarlo come preferisci nel flusso di lavoro.

