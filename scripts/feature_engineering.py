# scripts/feature_engineering.py

import os
import pandas as pd
from typing import List, Optional
from scripts.preprocessing import load_hourly_csv, normalize_timestamp
from scripts.technical_indicators import TechnicalIndicators
from indicators.frb import compute_frb
from indicators.porco_zia import compute_zia
from tqdm import tqdm

def build_features(
    df: pd.DataFrame,
    add_basic: bool = True,
    advanced_indicators: Optional[List[str]] = None,
    df_weekly: Optional[pd.DataFrame] = None,
    use_ema: bool = True
) -> pd.DataFrame:
    """
    df: DataFrame normalizzato e indicizzato
    add_basic: se True aggiunge SMA, EMA, RSI, MACD, BB, ATR, lag
    advanced_indicators: lista di indicatori da TechnicalIndicators.calculate_indicators
                         (es. ['bollinger_bands','ema','rsi','stoch_rsi','wave_trend','pps_killer','macd'])
    """
    df_feat = df.copy()

    # --- FEATURE TEMPORALI ---
    if 'Open time' in df_feat.columns:
        df_feat['hour'] = pd.to_datetime(df_feat['Open time']).dt.hour
        df_feat['day_of_week'] = pd.to_datetime(df_feat['Open time']).dt.dayofweek
        # Sessione: 0=Asia, 1=Europa, 2=USA (approssimato)
        df_feat['session'] = df_feat['hour'].apply(
            lambda h: 0 if 0 <= h < 8 else (1 if 8 <= h < 16 else 2)
        )
        # Giorno/Notte (UTC): 8-20 giorno, altrimenti notte
        df_feat['is_day'] = df_feat['hour'].between(8, 20).astype(int)

    if add_basic:
        # --- BASIC FEATURES ---
        # SMA
        for w in [7, 14, 30]:
            df_feat[f'SMA_{w}'] = df_feat['Close'].rolling(w).mean()
        # EMA solo se richiesto
        if use_ema:
            for w in [7, 14, 30]:
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

        # --- FEATURE AVANZATE ROLLING ---
        # Volatilità rolling (std su 5, 10, 20 periodi)
        for w in [5, 10, 20]:
            df_feat[f'volatility_{w}'] = df_feat['Close'].rolling(w).std()
        # Momentum (Close - Close_n_periods_fa)
        for w in [3, 7, 14]:
            df_feat[f'momentum_{w}'] = df_feat['Close'] - df_feat['Close'].shift(w)
        # Z-score (normalizzazione rolling)
        for w in [7, 14, 30]:
            mean = df_feat['Close'].rolling(w).mean()
            std = df_feat['Close'].rolling(w).std()
            df_feat[f'zscore_{w}'] = (df_feat['Close'] - mean) / std

        # Drawdown massimo su 14 periodi
        df_feat['max_drawdown_14'] = (
            df_feat['Close'].rolling(14).apply(lambda x: (x - x.max()).min())
        )

        # Return cumulativo su 3, 7, 14 periodi
        for w in [3, 7, 14]:
            df_feat[f'return_{w}'] = df_feat['Close'].pct_change(w)

        # Pattern di breakout: rottura massimi/minimi ultimi 7 periodi
        df_feat['breakout_high_7'] = (df_feat['Close'] > df_feat['High'].rolling(7).max().shift(1)).astype(int)
        df_feat['breakout_low_7'] = (df_feat['Close'] < df_feat['Low'].rolling(7).min().shift(1)).astype(int)

        # Media, max, min, varianza rolling su Volume (5, 14 periodi)
        for w in [5, 14]:
            df_feat[f'vol_mean_{w}'] = df_feat['Volume'].rolling(w).mean()
            df_feat[f'vol_max_{w}'] = df_feat['Volume'].rolling(w).max()
            df_feat[f'vol_min_{w}'] = df_feat['Volume'].rolling(w).min()
            df_feat[f'vol_var_{w}'] = df_feat['Volume'].rolling(w).var()

    # --- ADVANCED INDICATORS via TechnicalIndicators ---
    if advanced_indicators is not None:
        df_feat = TechnicalIndicators.calculate_indicators(df_feat, advanced_indicators)

    # --- FRB ---
    try:
        frb_df = compute_frb(df_feat)
        for col in frb_df.columns:
            df_feat[col] = frb_df[col]
        for col in frb_df.columns:
            df_feat[f'dist_{col}'] = df_feat['Close'] - df_feat[col]
    except Exception as e:
        print(f"Errore nel calcolo FRB: {e}")

    # --- Zia Levels ---
    # if df_weekly is not None:
    #     try:
    #         zia_df = compute_zia(df_feat, df_weekly)
    #         for col in zia_df.columns:
    #             df_feat[col] = zia_df[col]
    #         for col in zia_df.columns:
    #             df_feat[f'dist_{col}'] = df_feat['Close'] - df_feat[col]
    #     except Exception as e:
    #         print(f"Errore nel calcolo Zia: {e}")

    # Rimuovi NaN dovuti a rolling/shift
    df_feat = df_feat.dropna()

    print(f"[DEBUG] Dopo basic: {df_feat.shape}")
    print(f"[DEBUG] Dopo advanced: {df_feat.shape}")
    print(f"[DEBUG] Dopo FRB: {df_feat.shape}")
    print(f"[DEBUG] Dopo Zia: {df_feat.shape}")

    return df_feat

def make_weekly_df(df):
    """
    Aggrega un DataFrame orario in dati settimanali (OHLCV).
    """
    df = df.copy()
    df = df.set_index('Open time')
    df_weekly = df.resample('1W', label='right', closed='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna().reset_index()
    return df_weekly

#from scripts.preprocessing import load_hourly_csv, normalize_timestamp
#from scripts.feature_engineering import build_features

# Carica e normalizza
#df = load_hourly_csv('BTC')
#df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')

# Richiama build_features senza salvataggi interni
#features = build_features(
#    df,
#    add_basic=True,
#    advanced_indicators=[
#        'bollinger_bands',
#        'ema',
#        'rsi',
#        'stoch_rsi',
#        'wave_trend',
#        'pps_killer',
#        'macd'
#    ]
#)

# A questo punto 'features' è il tuo DataFrame completo, in memoria,
# e puoi salvarlo o usarlo come preferisci nel flusso di lavoro.

