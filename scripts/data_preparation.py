import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(df: pd.DataFrame, feature_cols: list, scaler_type: str = 'standard') -> (pd.DataFrame, object):
    """
    Scales the specified feature columns of df using StandardScaler or MinMaxScaler.

    Args:
        df: DataFrame containing the features and label.
        feature_cols: List of column names to scale.
        scaler_type: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.

    Returns:
        A tuple (scaled_df, scaler) where scaled_df is a DataFrame with the same index and columns,
        and scaler is the fitted scaler object for later inference.
    """
    X = df[feature_cols].copy()
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Use 'standard' or 'minmax'.")

    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=feature_cols)
    return X_scaled_df, scaler


def train_test_split_time(df: pd.DataFrame, date_col: str, split_date: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a time-indexed DataFrame into train and test sets based on a split timestamp.

    Args:
        df: DataFrame containing features and 'label'.
        date_col: Name of the datetime column to use as index (if not already indexed).
        split_date: Cutoff date (inclusive) in 'YYYY-MM-DD' format; data before this goes to train,
                    data from this date onward goes to test.

    Returns:
        A tuple (train_df, test_df)
    """
    df_copy = df.copy()
    # Ensure datetime index
    if date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.set_index(date_col)
    elif not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame must have a DatetimeIndex or column '{date_col}' must exist.")

    # Parse cutoff and align timezone with index
    cutoff = pd.to_datetime(split_date)
    idx_tz = df_copy.index.tz
    if idx_tz is not None:
        # If cutoff is naive, localize; else convert
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(idx_tz)
        else:
            cutoff = cutoff.tz_convert(idx_tz)

    train_df = df_copy[df_copy.index < cutoff]
    test_df = df_copy[df_copy.index >= cutoff]
    return train_df, test_df


def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Prepares and returns split features and labels without saving to disk.

    Args:
        train_df: DataFrame with features + 'label' for training.
        test_df: DataFrame with features + 'label' for testing.

    Returns:
        Dictionary containing:
          - 'train_features': DataFrame
          - 'train_labels': Series
          - 'test_features': DataFrame
          - 'test_labels': Series
    """
    train_features = train_df.drop(columns=['label'])
    train_labels = train_df['label']
    test_features = test_df.drop(columns=['label'])
    test_labels = test_df['label']

    return {
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels,
    }
