import numpy as np
import pandas as pd


# -------------------------------
# Funzione per calcolare Zia
# -------------------------------
def compute_zia(df_current, df_weekly, window=265, high_slope=0.0008, low_slope=0.00107):
    """
    Calcola gli indicatori Zia (livelli Fibonacci e deviazioni) su df_current (es. timeframe 2H)
    utilizzando anche dati settimanali (df_weekly) per i calcoli storici.
    """
    df = df_current.copy()
    # Assicurati che l'indice sia datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Open time' in df.columns:
            df['Open time'] = pd.to_datetime(df['Open time'])
            df = df.set_index('Open time')
        else:
            raise ValueError("Manca la colonna 'Open time' per settare l'indice temporale.")

    df_week = df_weekly.copy()
    if not isinstance(df_week.index, pd.DatetimeIndex):
        if 'Open time' in df_week.columns:
            df_week['Open time'] = pd.to_datetime(df_week['Open time'])
            df_week = df_week.set_index('Open time')
        else:
            raise ValueError("Manca la colonna 'Open time' per settare l'indice temporale.")

    # Calcola highest e lowest rolling su 'window' barre
    df['highest_close'] = df['Close'].rolling(window=window, min_periods=1).max()
    df['lowest_close'] = df['Close'].rolling(window=window, min_periods=1).min()
    df['HighIntercept'] = np.log(df['highest_close'])
    df['LowIntercept'] = np.log(df['lowest_close'])
    df['LogRange'] = np.log(df['highest_close']) - np.log(df['lowest_close'])

    # Definisci il timestamp di riferimento (1279670400000 ms -> 2010-07-21)
    threshold = pd.Timestamp(1279670400000, unit='ms').tz_localize(None)
    df.index = df.index.tz_localize(None)

    df['TimeIndex'] = df.index.to_series().apply(
        lambda t: 3.0 if t < threshold else (t - threshold).total_seconds() / 86400)

    df['Weight'] = (np.log10(df['TimeIndex'] + 10) * (df['TimeIndex'] ** 2) - df['TimeIndex']) / 30000
    df['HighLogDev'] = np.where(df['TimeIndex'] > 2,
                                np.log(df['Weight']) + df['HighIntercept'] + high_slope * df['TimeIndex'], np.nan)
    df['LowLogDev'] = np.where(df['TimeIndex'] > 2,
                               np.log(df['Weight']) + df['LowIntercept'] + low_slope * df['TimeIndex'], np.nan)

    # Su df_weekly (dati settimanali) calcola highest/lowest rolling su 'window' barre
    df_week['highest_close'] = df_week['Close'].rolling(window=window, min_periods=1).max()
    df_week['lowest_close'] = df_week['Close'].rolling(window=window, min_periods=1).min()
    df_week['LogHighWeekly'] = np.log(df_week['highest_close'])
    df_week['LogLowWeekly'] = np.log(df_week['lowest_close'])
    df_week['LogRangeWeekly'] = df_week['LogHighWeekly'] - df_week['LogLowWeekly']

    # Prima del merge
    df = df.reset_index()
    df['Open time'] = pd.to_datetime(df['Open time']).dt.tz_localize(None)
    assert df['Open time'].is_unique, "Dati orari con duplicati su 'Open time'!"

    df_week = df_week.reset_index()
    df_week['Open time'] = pd.to_datetime(df_week['Open time']).dt.tz_localize(None)

    df = pd.merge_asof(
        df.sort_values('Open time'),
        df_week[['Open time', 'LogRangeWeekly', 'LogLowWeekly']].sort_values('Open time'),
        left_on='Open time', right_on='Open time', direction='backward'
    )

    assert df['Open time'].is_unique, "Dopo il merge ci sono duplicati su 'Open time'!"
    df.set_index('Open time', inplace=True)

    # Calcola i livelli Fibonacci (in scala logaritmica)
    df['Fib10000Calc'] = df['LogRange'] * 1.0 + df['LowIntercept']
    df['Fib9236Calc'] = df['LogRangeWeekly'] * 0.9236 + df['LogLowWeekly']
    df['Fib9000Calc'] = df['LogRange'] * 0.90 + df['LowIntercept']
    df['Fib8786Calc'] = df['LogRange'] * 0.8786 + df['LowIntercept']
    df['Fib8618Calc'] = df['LogRange'] * 0.8618 + df['LowIntercept']
    df['Fib8541Calc'] = df['LogRange'] * 0.8541 + df['LowIntercept']
    df['Fib7639Calc'] = df['LogRange'] * 0.7639 + df['LowIntercept']
    df['Fib618Calc'] = df['LogRange'] * 0.618 + df['LowIntercept']
    df['MidCalc'] = df['LogRange'] * 0.5 + df['LowIntercept']
    df['Fib382Calc'] = df['LogRange'] * 0.382 + df['LowIntercept']
    df['Fib2361Calc'] = df['LogRange'] * 0.2361 + df['LowIntercept']
    df['Fib1459Calc'] = df['LogRange'] * 0.1459 + df['LowIntercept']
    df['Fib0902Calc'] = df['LogRange'] * 0.0902 + df['LowIntercept']
    df['Fib0000Calc'] = df['LogRange'] * 0.0 + df['LowIntercept']
    df['Fibneg1459Calc'] = df['LogRange'] * -0.1459 + df['LowIntercept']
    df['Fibneg0902Calc'] = df['LogRange'] * -0.0902 + df['LowIntercept']
    df['Fibneg0236Calc'] = df['LogRange'] * -0.236 + df['LowIntercept']
    df['Fibneg0382Calc'] = df['LogRange'] * -0.382 + df['LowIntercept']

    # Esponenzia per ottenere i livelli finali
    df['HighDev'] = np.exp(df['HighLogDev'])
    df['Fib10000Dev'] = np.exp(df['Fib10000Calc'])
    df['Fib9236Dev'] = np.exp(df['Fib9236Calc'])
    df['Fib9000Dev'] = np.exp(df['Fib9000Calc'])
    df['Fib8786Dev'] = np.exp(df['Fib8786Calc'])
    df['Fib8618Dev'] = np.exp(df['Fib8618Calc'])
    df['Fib8541Dev'] = np.exp(df['Fib8541Calc'])
    df['Fib7639Dev'] = np.exp(df['Fib7639Calc'])
    df['Fib618Dev'] = np.exp(df['Fib618Calc'])
    df['MidDev'] = np.exp(df['MidCalc'])
    df['Fib382Dev'] = np.exp(df['Fib382Calc'])
    df['Fib2361Dev'] = np.exp(df['Fib2361Calc'])
    df['Fib1459Dev'] = np.exp(df['Fib1459Calc'])
    df['Fib0902Dev'] = np.exp(df['Fib0902Calc'])
    df['Fib0000Dev'] = np.exp(df['Fib0000Calc'])
    df['Fibneg1459Dev'] = np.exp(df['Fibneg1459Calc'])
    df['Fibneg0902Dev'] = np.exp(df['Fibneg0902Calc'])
    df['Fibneg0236Dev'] = np.exp(df['Fibneg0236Calc'])
    df['Fibneg0382Dev'] = np.exp(df['Fibneg0382Calc'])
    df['LowDev'] = np.exp(df['LowLogDev'])

    # Seleziona le colonne di interesse
    cols = ['HighDev', 'Fib10000Dev', 'Fib9236Dev', 'Fib9000Dev', 'Fib8786Dev',
            'Fib8618Dev', 'Fib8541Dev', 'Fib7639Dev', 'Fib618Dev', 'MidDev',
            'Fib382Dev', 'Fib2361Dev', 'Fib1459Dev', 'Fib0902Dev', 'Fib0000Dev',
            'Fibneg1459Dev', 'Fibneg0902Dev', 'Fibneg0236Dev', 'Fibneg0382Dev', 'LowDev']

    return df[cols]
