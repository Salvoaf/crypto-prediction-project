# scripts/label_generation.py

import pandas as pd

def create_labels(df: pd.DataFrame, target_col: str = 'Close', horizon: int = 24, threshold: float = 0.04) -> pd.DataFrame:
    """
    Aggiunge al DataFrame la colonna 'label' definita come:
      2  se (Close_{t+horizon} - Close_t) / Close_t >= threshold   (UP)
      0  se (Close_{t+horizon} - Close_t) / Close_t <= -threshold  (DOWN)
      1  altrimenti (HOLD)

    df: DataFrame con indice temporale e colonna target_col
    target_col: nome della colonna prezzo da usare
    horizon: orizzonte in ore/giorni
    threshold: variazione percentuale per classificare up/down
    """
    df = df.copy()
    # shift: per ogni riga, prendi il prezzo a t + horizon
    df[f'{target_col}_future'] = df[target_col].shift(-horizon)
    # calcola la variazione percentuale
    df['pct_change'] = (df[f'{target_col}_future'] - df[target_col]) / df[target_col]
    # crea la label multiclass: 2=UP, 0=DOWN, 1=HOLD
    df['label'] = 1  # default hold
    df.loc[df['pct_change'] >= threshold, 'label'] = 2
    df.loc[df['pct_change'] <= -threshold, 'label'] = 0
    # opzionale: togli le ultime `horizon` righe che non hanno label
    df = df.iloc[:-horizon]
    # rimuovi le colonne temporanee se non servono più
    df.drop(columns=[f'{target_col}_future', 'pct_change'], inplace=True)
    # Propaga la colonna temporale se esiste
    if 'Open time' in df.columns:
        df['Open time'] = df['Open time']
    return df


