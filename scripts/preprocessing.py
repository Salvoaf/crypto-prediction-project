# scripts/preprocessing.py

import os
from dotenv import load_dotenv
import pandas as pd

# carica le variabili da config/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))
CACHE_DIR = os.environ['CACHE_DIR']


def load_hourly_csv(symbol: str) -> pd.DataFrame:
    """
    Legge il file CSV orario per la symbol specificata, e restituisce un DataFrame.
    """
    path = os.path.join(CACHE_DIR, f"{symbol}USDT.csv")
    df = pd.read_csv(path)
    return df


def normalize_timestamp(
        df: pd.DataFrame,
        ts_col: str = 'timestamp',
        tz_from: str = None,
        tz_to: str = 'UTC'
) -> pd.DataFrame:
    """
    Converte la colonna timestamp (stringa o numerica) in datetime,
    imposta timezone e restituisce un nuovo DataFrame con index datetime normalizzato.
    - tz_from: timezone originale (es. 'Europe/Rome'); se None assume UTC.
    - tz_to: timezone di destinazione (default 'UTC').
    """
    # 1) parse: pandas riconosce automaticamente il formato
    df[ts_col] = pd.to_datetime(df[ts_col], errors='raise')

    # 2) localizza (se non ha già tz) e converte
    if df[ts_col].dt.tz is None:
        if tz_from:
            df[ts_col] = df[ts_col].dt.tz_localize(tz_from)
        else:
            df[ts_col] = df[ts_col].dt.tz_localize('UTC')
    df[ts_col] = df[ts_col].dt.tz_convert(tz_to)

    # 3) imposta come index e ordina
    df = df.set_index(ts_col).sort_index()
    return df


#from scripts.preprocessing import load_hourly_csv, normalize_timestamp

# carica e normalizza
#df = load_hourly_csv('BTC')
#df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')

#print(df)
