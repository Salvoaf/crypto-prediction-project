# scripts/label_generation.py

import pandas as pd

def create_labels(df: pd.DataFrame, target_col: str = 'Close', horizon: int = 24) -> pd.DataFrame:
    """
    Aggiunge al DataFrame la colonna 'label' definita come:
      1 se Close_{t+horizon} > Close_t
      0 altrimenti

    df: DataFrame con indice temporale e colonna target_col
    target_col: nome della colonna prezzo da usare
    horizon: orizzonte in ore
    """
    df = df.copy()
    # shift: per ogni riga, prendi il prezzo a t + horizon
    df[f'{target_col}_future'] = df[target_col].shift(-horizon)
    # crea la label
    df['label'] = (df[f'{target_col}_future'] > df[target_col]).astype(int)
    # opzionale: togli le ultime `horizon` righe che non hanno label
    df = df.iloc[:-horizon]
    # rimuovi la colonna futura se non serve più
    df.drop(columns=[f'{target_col}_future'], inplace=True)
    return df
