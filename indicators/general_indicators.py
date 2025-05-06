import pandas as pd
import numpy as np
import ta

# ===============================
# CLASSE PER INDICATORI TECNICI
# ===============================
# -------------------------------
# Classe per il calcolo degli indicatori tecnici
# -------------------------------
class TechnicalIndicators:
    # Soglie per Wave Trend
    obLevel1 = 60
    obLevel2 = 53
    osLevel1 = -60
    osLevel2 = -53

    @staticmethod
    def laguerre_smoothing(series, gamma):
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
        n1 = 10
        n2 = 21
        useLag = True
        gamma = 0.02

        data['hlc3'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['esa'] = data['hlc3'].ewm(span=n1, adjust=False).mean()
        data['apesa'] = (data['hlc3'] - data['esa']).abs()
        data['d'] = data['apesa'].ewm(span=n1, adjust=False).mean().replace(0, 1e-10)
        data['ci'] = (data['hlc3'] - data['esa']) / (0.015 * data['d'])
        data['tci'] = data['ci'].ewm(span=n2, adjust=False).mean()
        if useLag:
            data['wt1'] = TechnicalIndicators.laguerre_smoothing(data['tci'], gamma)
        else:
            data['wt1'] = data['tci']
        data['wt2'] = data['wt1'].rolling(window=4).mean()
        data['wt3'] = data['wt1'] - data['wt2']
        data['wt_buy_signal'] = ((data['wt1'] > data['wt2']) &
                                 (data['wt1'].shift(1) <= data['wt2'].shift(1)) &
                                 (data['wt1'] < TechnicalIndicators.osLevel2)).astype(int)
        data['wt_sell_signal'] = ((data['wt1'] < data['wt2']) &
                                  (data['wt1'].shift(1) >= data['wt2'].shift(1)) &
                                  (data['wt1'] > TechnicalIndicators.obLevel2)).astype(int)
        return data

    @staticmethod
    def calculate_pps_killer(data):
        data['PPS'] = (data['Close'] - data['Close'].shift(3)) / data['Close'].shift(3) * 100
        return data

    @staticmethod
    def calculate_indicators(data, indicators_to_calculate):
        close = data['Close']
        # Rimuove eventuali NaN nelle colonne critiche
        critical_columns = ['Open', 'High', 'Low', 'Close']
        data.dropna(subset=critical_columns, inplace=True)
        if data.empty:
            print("Dati insufficienti dopo aver rimosso i NaN nelle colonne critiche.")
            return data
        if 'bollinger_bands' in indicators_to_calculate:
            if len(data) >= 20:
                for dev in [2, 3]:
                    indicator_bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=dev)
                    data[f'bb{dev}_high'] = indicator_bb.bollinger_hband()
                    data[f'bb{dev}_low'] = indicator_bb.bollinger_lband()
                    data[f'bb{dev}_mid'] = indicator_bb.bollinger_mavg()
            else:
                print("Dati insufficienti per calcolare le Bollinger Bands.")
        if 'ema' in indicators_to_calculate:
            for window in [21, 50, 100, 200, 500]:
                if window <= len(data):
                    data[f'EMA_{window}'] = ta.trend.EMAIndicator(close=close, window=window).ema_indicator()
                else:
                    print(f"Dati insufficienti per calcolare l'EMA a {window} periodi.")
        if 'rsi' in indicators_to_calculate:
            data['RSI'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
            data['RSI_MA14'] = data['RSI'].rolling(window=14).mean()
            data['RSI_MA21'] = data['RSI'].rolling(window=21).mean()
        if 'stoch_rsi' in indicators_to_calculate:
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
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_diff'] = macd.macd_diff()
        return data