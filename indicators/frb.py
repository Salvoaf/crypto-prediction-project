import numpy as np
import pandas as pd

# -------------------------------
# Funzione per calcolare FRB
# -------------------------------
def compute_frb(df, period=144):
    """
    Calcola i Fractal Regression Bands (FRB) su un DataFrame.
    Si assume che df abbia le colonne: High, Low, Close.
    """
    # Condizioni per individuare i fractal (usando shift per simulare [4], [3], [2], [1], [0])
    ufract = (df['High'].shift(4) < df['High'].shift(2)) & \
             (df['High'].shift(3) < df['High'].shift(2)) & \
             (df['High'].shift(1) < df['High'].shift(2)) & \
             (df['High'] < df['High'].shift(2))
    dfract = (df['Low'].shift(4) > df['Low'].shift(2)) & \
             (df['Low'].shift(3) > df['Low'].shift(2)) & \
             (df['Low'].shift(1) > df['Low'].shift(2)) & \
             (df['Low'] > df['Low'].shift(2))

    # Ottieni l'ultimo valore "High" quando ufract è True (valuewhen)
    hval = df['High'].where(ufract).ffill()
    # Ottieni l'ultimo valore "Low" quando dfract è True
    lval = df['Low'].where(dfract).ffill()

    # Calcola il valore centrale
    mval_frb = (hval + lval) / 2.0
    # Calcola l'EMA del valore centrale con il periodo specificato
    bl_frb = mval_frb.ewm(span=period, adjust=False).mean()

    phi = (1 + np.sqrt(5)) / 2.0

    # Calcola l'ATR (True Range) su 'period' barre
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    atr_frb = tr.rolling(window=period, min_periods=1).mean()

    # Calcola le bande
    hbhalf = bl_frb + atr_frb * phi / 2.0
    hb1 = bl_frb + atr_frb * phi
    hb2 = bl_frb + atr_frb * phi * 2.0
    hb3 = bl_frb + atr_frb * phi * 3.0

    lbhalf = bl_frb - atr_frb * phi / 2.0
    lb1 = bl_frb - atr_frb * phi
    lb2 = bl_frb - atr_frb * phi * 2.0
    lb3 = bl_frb - atr_frb * phi * 3.0

    result = pd.DataFrame({
        'bl_frb': bl_frb,
        'hbhalf': hbhalf,
        'hb1': hb1,
        'hb2': hb2,
        'hb3': hb3,
        'lbhalf': lbhalf,
        'lb1': lb1,
        'lb2': lb2,
        'lb3': lb3
    }, index=df.index)

    return result

