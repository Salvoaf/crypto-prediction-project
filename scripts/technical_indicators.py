# technical_indicators.py
import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    """
    Class for calculating technical indicators.
    """

    # Thresholds as class attributes
    obLevel1 = 60  # Overbought Level 1
    obLevel2 = 53  # Overbought Level 2
    osLevel1 = -60  # Oversold Level 1
    osLevel2 = -53  # Oversold Level 2

    @staticmethod
    def laguerre_smoothing(series, gamma):
        """
        Applies Laguerre filtering to the provided series.
        """
        L0 = L1 = L2 = L3 = 0
        output = []
        for p in series:
            L0_new = (1 - gamma) * p + gamma * L0
            L1_new = -gamma * L0_new + L0 + gamma * L1
            L2_new = -gamma * L1_new + L1 + gamma * L2
            L3_new = -gamma * L2_new + L2 + gamma * L3
            smoothed = (L0_new + 2 * L1_new + 2 * L2_new + L3_new) / 6
            output.append(smoothed)
            L0, L1, L2, L3 = L0_new, L1_new, L2_new, L3_new
        return pd.Series(output, index=series.index)

    @staticmethod
    def calculate_wave_trend(data):
        """
        Calculates the Wave Trend indicator based on ChuckBanger's Pine Script.
        """
        # Parameters as in the Pine Script
        n1 = 10  # Channel Length
        n2 = 21  # Average Length
        useLag = True  # Apply Laguerre Smoothing
        gamma = 0.02  # Laguerre Gamma

        # Calculate hlc3
        data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3

        # Calculate ESA
        data['esa'] = data['hlc3'].ewm(span=n1, adjust=False).mean()

        # Calculate absolute deviation
        data['apesa'] = (data['hlc3'] - data['esa']).abs()

        # Calculate d
        data['d'] = data['apesa'].ewm(span=n1, adjust=False).mean()

        # Handle division by zero by adding a small constant
        data['d'] = data['d'].replace(0, 1e-10)

        # Calculate CI
        data['ci'] = (data['hlc3'] - data['esa']) / (0.015 * data['d'])

        # Calculate TCI
        data['tci'] = data['ci'].ewm(span=n2, adjust=False).mean()

        # Apply Laguerre filtering
        if useLag:
            data['wt1'] = TechnicalIndicators.laguerre_smoothing(data['tci'], gamma)
        else:
            data['wt1'] = data['tci']

        # Calculate wt2 as SMA of wt1 with window=4
        data['wt2'] = data['wt1'].rolling(window=4).mean()

        # Calculate wt3 as difference between wt1 and wt2
        data['wt3'] = data['wt1'] - data['wt2']

        # Generate buy and sell signals
        data['wt_buy_signal'] = (
            (data['wt1'] > data['wt2']) &
            (data['wt1'].shift(1) <= data['wt2'].shift(1)) &
            (data['wt1'] < TechnicalIndicators.osLevel2)
        ).astype(int)

        data['wt_sell_signal'] = (
            (data['wt1'] < data['wt2']) &
            (data['wt1'].shift(1) >= data['wt2'].shift(1)) &
            (data['wt1'] > TechnicalIndicators.obLevel2)
        ).astype(int)

        return data

    @staticmethod
    def calculate_pps_killer(data):
        """
        Simplified implementation of John Ehlers' PPS Killer.
        """
        # Calculate PPS Killer (this is a simplification)
        data['PPS'] = (data['Close'] - data['Close'].shift(3)) / data['Close'].shift(3) * 100
        return data

    @staticmethod
    def calculate_indicators(data, indicators_to_calculate):
        """
        Calculates the specified technical indicators and adds them to the DataFrame.
        """
        # Ensure 'Close' is a Series
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        # Verify that critical columns do not contain NaN
        critical_columns = ['Open', 'High', 'Low', 'Close']
        data = data.dropna(subset=critical_columns)

        if data.empty:
            raise ValueError("Dati insufficienti dopo aver rimosso i NaN dalle colonne critiche.")

        if 'bollinger_bands' in indicators_to_calculate:
            # Bollinger Bands with standard deviation 2 and 3
            if len(data) >= 20:
                for dev in [2, 3]:
                    indicator_bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=dev)
                    data[f'bb{dev}_high'] = indicator_bb.bollinger_hband()
                    data[f'bb{dev}_low'] = indicator_bb.bollinger_lband()
                    data[f'bb{dev}_mid'] = indicator_bb.bollinger_mavg()
            else:
                print("Warning: Dati insufficienti per calcolare le Bollinger Bands.")

        if 'ema' in indicators_to_calculate:
            # EMA for periods 21, 50, 100, 200, and 500
            for window in [21, 50, 100, 200, 500]:
                if window <= len(data):
                    indicator_ema = ta.trend.EMAIndicator(close=close, window=window)
                    data[f'EMA_{window}'] = indicator_ema.ema_indicator()
                else:
                    print(f"Warning: Dati insufficienti per calcolare l'EMA a {window} periodi.")

        if 'rsi' in indicators_to_calculate:
            # RSI with a 14-period window
            indicator_rsi = ta.momentum.RSIIndicator(close=close, window=14)
            data['RSI'] = indicator_rsi.rsi()
            # Moving averages of RSI
            data['RSI_MA14'] = data['RSI'].rolling(window=14).mean()
            data['RSI_MA21'] = data['RSI'].rolling(window=21).mean()

        if 'stoch_rsi' in indicators_to_calculate:
            # Stochastic RSI
            stoch_rsi_indicator = ta.momentum.StochRSIIndicator(close=close, window=14)
            data['Stoch_RSI'] = stoch_rsi_indicator.stochrsi() * 100
            data['Stoch_RSI_K'] = stoch_rsi_indicator.stochrsi_k() * 100
            data['Stoch_RSI_D'] = stoch_rsi_indicator.stochrsi_d() * 100
            data['Stoch_RSI_MA21'] = data['Stoch_RSI'].rolling(window=21).mean()

        if 'wave_trend' in indicators_to_calculate:
            data = TechnicalIndicators.calculate_wave_trend(data)

        if 'pps_killer' in indicators_to_calculate:
            data = TechnicalIndicators.calculate_pps_killer(data)

        if 'macd' in indicators_to_calculate:
            macd = ta.trend.MACD(close=close)
            # Controlla se i valori non sono None
            if macd.macd_signal() is not None:
                data['MACD'] = macd.macd()
                data['MACD_signal'] = macd.macd_signal()
                data['MACD_diff'] = macd.macd_diff()
            else:
                print("Errore: MACD_signal non è stato calcolato correttamente.")
        # Do not remove rows with NaN in indicators
        # Keep rows intact for the candlestick chart
        return data
