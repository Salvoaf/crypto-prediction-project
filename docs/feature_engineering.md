# Feature Engineering for Cryptocurrency Trading

## Introduction

This document describes the feature engineering pipeline for cryptocurrency trading data. The dataset consists of hourly OHLCV (Open, High, Low, Close, Volume) data for various cryptocurrency pairs. The goal is to create a comprehensive set of technical indicators and features that can help predict future price movements.

## Multi-timeframe Methodology

The feature engineering process handles multiple timeframes (1h, 2h, 4h, 12h, 24h, 7d) by:

1. Resampling the data to each timeframe
2. Calculating features at that timeframe
3. Merging back to the original hourly data using `merge_asof`

Example code:
```python
# Resample to target timeframe
resampled = df.resample('4h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Calculate features
resampled = add_features(resampled)

# Merge back to hourly data
df = pd.merge_asof(
    df.reset_index(),
    resampled.reset_index(),
    on='Open time',
    direction='backward'
)
```

## Feature Categories

### 1. Trend & Momentum

#### Moving Averages
- **SMA (Simple Moving Average)**
  - Formula: SMA = (P1 + P2 + ... + Pn) / n
  - Periods: [5,10,20,50,100,200]
  - Timeframes: 1h, 2h, 4h, 12h, 24h
  - Rationale: Identifies trend direction and potential support/resistance levels
  - Code: `ta.trend.SMAIndicator(close=df['Close'], window=period)`

- **EMA (Exponential Moving Average)**
  - Formula: EMA = (P * k) + (EMA_prev * (1 - k)) where k = 2/(n+1)
  - Periods: [5,10,20,50,100,200]
  - Timeframes: 1h, 2h, 4h, 12h, 24h
  - Rationale: More responsive to recent price changes than SMA
  - Code: `ta.trend.EMAIndicator(close=df['Close'], window=period)`

#### MACD (Moving Average Convergence Divergence)
- **Components**: MACD Line, Signal Line, Histogram
- **Timeframes**: 1h, 4h
- **Rationale**: Identifies trend changes and momentum
- **Code**: 
```python
macd = MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_hist'] = macd.macd_diff()
```

#### RSI (Relative Strength Index)
- **Period**: 14
- **Timeframes**: 1h, 6h
- **Formula**: RSI = 100 - (100 / (1 + RS)) where RS = Avg Gain / Avg Loss
- **Rationale**: Identifies overbought/oversold conditions
- **Code**: `ta.momentum.RSIIndicator(close=df['Close'])`

#### Stochastic RSI
- **Components**: %K, %D
- **Timeframes**: 1h, 6h
- **Rationale**: Combines RSI and Stochastic oscillator for better signal
- **Code**: `ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])`

### 2. Volatility

#### ATR (Average True Range)
- **Period**: 14
- **Timeframes**: 1h, 4h
- **Formula**: ATR = max(high - low, |high - prev_close|, |low - prev_close|)
- **Rationale**: Measures market volatility
- **Code**: `ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'])`

#### Bollinger Bands
- **Parameters**: 20 periods, 2 standard deviations
- **Timeframes**: 1h, 24h
- **Components**: Upper Band, Lower Band, Width, %B
- **Rationale**: Identifies volatility and potential reversals
- **Code**: `ta.volatility.BollingerBands(close=df['Close'])`

#### Rolling Standard Deviation
- **Periods**: 20, 50, 100
- **Formula**: σ = sqrt(Σ(x - μ)² / n)
- **Rationale**: Measures price volatility
- **Code**: `df['Close'].pct_change().rolling(period).std()`

### 3. Candlestick Patterns

#### Basic Properties
- **Body Size**: |Close - Open|
- **Upper Wick**: High - max(Open, Close)
- **Lower Wick**: min(Open, Close) - Low
- **Range**: High - Low

#### Pattern Detection
- **Doji**: Body size ≤ 10% of range
- **Hammer**: Long lower wick (>2x body) and short upper wick
- **Code**:
```python
df['Is_Doji'] = (df['Body_size'] <= 0.1 * df['Range']).astype(int)
df['Is_Hammer'] = (
    (df['Lower_wick'] > 2 * df['Body_size']) &
    (df['Upper_wick'] < df['Body_size'])
).astype(int)
```

### 4. Risk Metrics

#### Max Drawdown
- **Windows**: 24h, 7d
- **Formula**: (Price - Rolling Max) / Rolling Max
- **Rationale**: Measures maximum loss from peak
- **Code**:
```python
returns = df['Close'].pct_change()
rolling_max = returns.rolling(window=window).max()
drawdown = (returns - rolling_max) / rolling_max
```

#### Sharpe Ratio
- **Windows**: 24h, 7d
- **Formula**: (Returns - Risk-free Rate) / Std Dev
- **Rationale**: Risk-adjusted return measure
- **Code**:
```python
excess_returns = returns - 0.02/365  # 2% risk-free rate
sharpe = excess_returns.rolling(window).mean() / returns.rolling(window).std()
```

#### Sortino Ratio
- **Windows**: 24h, 7d
- **Formula**: (Returns - Risk-free Rate) / Downside Std Dev
- **Rationale**: Focuses on downside risk
- **Code**:
```python
negative_returns = returns[returns < 0]
sortino = excess_returns.rolling(window).mean() / negative_returns.rolling(window).std()
```

### 5. Cyclical Features

#### Time-based Features
- **Hour of Day**: One-hot encoded (0-23)
- **Day of Week**: One-hot encoded (0-6)
- **Rationale**: Captures intraday and weekly patterns
- **Code**:
```python
df['Hour'] = df.index.hour
hour_dummies = pd.get_dummies(df['Hour'], prefix='Hour')
```

#### FFT Analysis
- **Windows**: 24h, 168h (7d)
- **Rationale**: Identifies periodic patterns
- **Code**:
```python
fft = np.fft.fft(returns.rolling(window).mean().fillna(0))
df['FFT_amp'] = np.abs(fft)[:window//2]
```

## Normalization and Scaling

### Z-score Normalization
- Applied to all numerical features
- Prevents data leakage by calculating statistics on training data only
- Code:
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Min-Max Scaling
- Applied to bounded features (e.g., RSI, %B)
- Scales to [0,1] range
- Code:
```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## Best Practices

1. **Data Leakage Prevention**
   - Calculate rolling statistics using only past data
   - Use `shift()` for lagged features
   - Split data before scaling

2. **Feature Selection**
   - Remove highly correlated features
   - Use feature importance from models
   - Consider computational cost

3. **Maintenance**
   - Keep feature calculation modular
   - Document all feature formulas
   - Version control feature sets

4. **Performance**
   - Use vectorized operations
   - Cache intermediate calculations
   - Profile memory usage

## Conclusion

This feature engineering pipeline provides a comprehensive set of technical indicators and features for cryptocurrency trading. The multi-timeframe approach allows capturing patterns at different scales, while the various feature categories help identify different aspects of market behavior.

The modular design makes it easy to add new features or modify existing ones. Regular monitoring and updating of the feature set is recommended to maintain its effectiveness. 