from scripts.preprocessing import load_hourly_csv, normalize_timestamp
from scripts.feature_engineering import build_features, make_weekly_df
from scripts.label_generation import create_labels
from scripts.data_preparation import scale_features, train_test_split_time, prepare_data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import os
from datetime import datetime
import glob
from tqdm import tqdm
from typing import List, Optional

# Constants
HORIZON = 1
USE_EMA = True  # <--- Variabile booleana per EMA
AGGREGATE_TO_DAILY = True  # <--- Per attivare l'aggregazione a daily
SPLIT_DATE = '2023-01-01'
SCALER_TYPE = 'standard'
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs/graphs'
REPORTS_DIR = 'reports'
LOGS_DIR = 'logs'
N_COINS = 30
N_SELECTED_FEATURES = None

# Create necessary directories
for dir_path in [MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'giorno_7_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

SELECTED_FEATURES_PATH = os.path.join(MODELS_DIR, 'selected_features.txt')

def select_features_xgb(xgb_model, feature_names, max_features=None):
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    if max_features is None:
        selected = feature_names[sorted_idx]
    else:
        selected = feature_names[sorted_idx][:max_features]
    return list(selected)

def save_selected_features(feature_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for feat in feature_list:
            f.write(f"{feat}\n")

def load_selected_features(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        features = [line.strip() for line in f if line.strip()]
    return features if features else None

def load_all_cryptos(data_dir, pattern="*USDT.csv", n_coins=None, test_symbol="BTCUSDT"):
    """Load and prepare crypto data"""
    all_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    test_file = None
    for file in all_files:
        if os.path.basename(file).startswith(test_symbol):
            test_file = file
            break
    
    selected_files = all_files[:n_coins] if n_coins is not None else all_files
    if test_file and test_file not in selected_files:
        selected_files = selected_files[:-1] + [test_file]
    
    if not test_file:
        logging.warning(f"File for {test_symbol} not found!")
    
    logging.info(f"Loading files: {[os.path.basename(f) for f in selected_files]}")
    
    dfs = []
    for file in tqdm(selected_files, desc="Loading files", unit="file"):
        symbol = os.path.basename(file).replace('.csv', '')
        try:
            df = pd.read_csv(file)
            if df.empty or df.columns.size == 0:
                tqdm.write(f"[WARNING] Empty file or no columns: {file} - skipping.")
                continue
            df['symbol'] = symbol
            df['Open time'] = pd.to_datetime(df['Open time'])
            dfs.append(df)
        except pd.errors.EmptyDataError:
            tqdm.write(f"[WARNING] Empty file (EmptyDataError): {file} - skipping.")
            continue
        except Exception as e:
            tqdm.write(f"[ERROR] Error loading {file}: {e} - skipping.")
            continue
    
    if not dfs:
        raise ValueError("No valid files found! Check the directory and pattern.")
    
    return pd.concat(dfs, ignore_index=True)

def aggregate_to_daily(df):
    # Se 'Open time' non è tra le colonne ma è l'indice, resetta l'indice
    if 'Open time' not in df.columns and df.index.name == 'Open time':
        df = df.reset_index()
    # Ora puoi lavorare come prima
    df['Open time'] = pd.to_datetime(df['Open time'])
    df = df.set_index('Open time')
    daily_list = []
    for symbol, group in df.groupby('symbol'):
        daily = group.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        daily['symbol'] = symbol
        daily_list.append(daily)
    daily_df = pd.concat(daily_list)
    daily_df = daily_df.reset_index()
    return daily_df

def prepare_data_for_training():
    """Prepare data for model training"""
    logging.info("[STEP 1] Loading multi-crypto data")
    data_dir = 'E:/data_cache'  # <-- Update this path!
    df = load_all_cryptos(data_dir, n_coins=N_COINS, test_symbol="BTCUSDT")
    
    if df.empty:
        raise ValueError("Empty DataFrame after loading!")
    logging.info(f"[INFO] Loaded data shape: {df.shape}")
    
    logging.info("[STEP 2] Normalizing timestamps")
    df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')
    
    if AGGREGATE_TO_DAILY:
        logging.info("Aggrego i dati a daily...")
        df = aggregate_to_daily(df)
    
    logging.info("[STEP 3] Feature engineering")
    advanced_indicators = [
        'bollinger_bands',
        'ema',
        'rsi',
        'stoch_rsi',
        'wave_trend',
        'pps_killer',
        'macd'
    ]
    
    symbols = df['symbol'].unique()
    features_list = []
    for symbol in tqdm(symbols, desc="Feature engineering", unit="coin"):
        df_symbol = df[df['symbol'] == symbol].copy()
        df_symbol = df_symbol.sort_values('Open time')
        if df_symbol.index.name == 'Open time':
            df_symbol = df_symbol.reset_index()
        df_weekly = make_weekly_df(df_symbol)
        features_symbol = build_features(
            df_symbol,
            add_basic=True,
            advanced_indicators=advanced_indicators,
            df_weekly=df_weekly,
            use_ema=USE_EMA
        )
        if features_symbol.empty:
            print(f"[WARNING] features_symbol vuoto per {symbol}, salto.")
            continue
        features_symbol['symbol'] = symbol
        features_list.append(features_symbol)
    features = pd.concat(features_list, ignore_index=True)
    print(f"[DEBUG] Shape features dopo concat: {features.shape}")
    
    logging.info("[STEP 4] Label generation")
    labeled = create_labels(features, target_col='Close', horizon=HORIZON, threshold=0.02)
    if 'Open time' in features.columns:
        labeled['Open time'] = features['Open time'].iloc[:-HORIZON].values
    logging.info(f"Distribuzione classi:\n{labeled['label'].value_counts(normalize=True)}")
    print(f"[DEBUG] Shape labeled dopo label generation: {labeled.shape}")
    
    logging.info("[STEP 5] Feature scaling")
    feature_cols = [c for c in labeled.columns if c not in ['label', 'symbol', 'Open time']]
    X_scaled, scaler = scale_features(labeled, feature_cols, scaler_type=SCALER_TYPE)
    print(f"[DEBUG] Shape X_scaled dopo scaling: {X_scaled.shape}")
    
    logging.info("[STEP 6] Reconstructing scaled DataFrame")
    df_scaled = pd.concat([X_scaled, labeled[['label', 'symbol', 'Open time']].reset_index(drop=True)], axis=1)
    
    logging.info("[STEP 7] Train/test split")
    btc_mask = df_scaled['symbol'] == 'BTCUSDT'
    train_df = df_scaled[~btc_mask]
    test_df = df_scaled[btc_mask]
    
    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    exclude_cols = ['label', 'symbol', 'Open time']
    feature_cols = [c for c in df_scaled.columns if c not in exclude_cols]
    
    train_X = train_df[feature_cols]
    test_X = test_df[feature_cols]
    train_y = train_df['label']
    test_y = test_df['label']
    
    # Handle NaN values
    train_X = train_X.dropna()
    train_y = train_y.loc[train_X.index]
    test_X = test_X.dropna()
    test_y = test_y.loc[test_X.index]
    
    # Dopo aver creato train_X, train_y
    selected_features = select_top_features_rf(train_X, train_y, n_features=N_SELECTED_FEATURES, plot=True)
    train_X_sel = train_X[selected_features]
    test_X_sel = test_X[selected_features]
    
    return train_X_sel, test_X_sel, train_y, test_y

def train_random_forest(X_train, y_train):
    """Train Random Forest with GridSearchCV"""
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10, None],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
    logging.info("Starting Random Forest GridSearchCV...")
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_xgboost(X_train, y_train):
    """Train XGBoost directly with best parameters (no GridSearch)"""
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.2,
        'max_depth': 5,
        'n_estimators': 200,
        'subsample': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    xgb = XGBClassifier(**best_params)
    logging.info("Training XGBoost with best parameters (no GridSearch)...")
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, digits=3)
    }
    logging.info(f"\n{model_name} Metrics:")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
    logging.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    logging.info(f"\nClassification Report:\n{metrics['classification_report']}")
    return metrics

def plot_roc_curves(models_metrics, y_test, output_path):
    pass  # <-- placeholder

def plot_confusion_matrices(models_metrics, output_dir):
    """Plot confusion matrices for all models"""
    for model_name, metrics in models_metrics.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'))
        plt.close()

def generate_report(models_metrics, output_path):
    """Generate markdown report with results"""
    report = f"""# Day 7 Report: Advanced Models Evaluation

## Model Comparison

| Model | Accuracy | F1 Score |
|-------|----------|----------|
"""
    
    for model_name, metrics in models_metrics.items():
        report += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} |\n"
    
    report += """
## Improvements over Baseline

| Model | Δ Accuracy |
|-------|------------|
"""
    
    baseline_metrics = models_metrics['XGBoost']
    for model_name, metrics in models_metrics.items():
        if model_name != 'XGBoost':
            delta_acc = metrics['accuracy'] - baseline_metrics['accuracy']
            report += f"| {model_name} | {delta_acc:+.4f} |\n"
    
    report += """
## Quality Gates

- Accuracy improvement threshold: 1%

## Next Steps

1. Feature importance analysis
2. Error analysis
3. Hyperparameter tuning if needed
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

def select_top_features_rf(X, y, n_features=6, plot=True):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    if n_features is None:
        top_features = X.columns[indices]
        n_plot = len(top_features)
    else:
        top_features = X.columns[indices][:n_features]
        n_plot = n_features
    if plot:
        plt.figure(figsize=(max(8, n_plot // 2), 4))
        plt.title("Feature importances (Random Forest)")
        plt.bar(range(n_plot), importances[indices][:n_plot], align="center")
        plt.xticks(range(n_plot), X.columns[indices][:n_plot], rotation=45)
        plt.tight_layout()
        plt.show()
    return list(top_features)

def main():
    try:
        # Prepare data
        train_X, test_X, train_y, test_y = prepare_data_for_training()

        # 1. Allena XGBoost su tutte le feature per la feature selection
        xgb_model_full = train_xgboost(train_X, train_y)

        # 2. Seleziona le feature migliori
        feature_names = np.array(train_X.columns)
        selected_features = load_selected_features(SELECTED_FEATURES_PATH)
        if selected_features is None:
            if N_SELECTED_FEATURES is None:
                logging.info("Feature selection: uso tutte le feature (nessun filtro).")
            else:
                logging.info(f"Feature selection: calcolo le top {N_SELECTED_FEATURES} feature con XGBoost...")
            selected_features = select_features_xgb(xgb_model_full, feature_names, max_features=N_SELECTED_FEATURES)
        else:
            logging.info(f"Feature selection: uso feature già salvate ({len(selected_features)} feature)")

        # 3. Filtra i dataset
        train_X_sel = train_X[selected_features]
        test_X_sel = test_X[selected_features]

        # --- COMMENTA QUESTE DUE RIGHE ---
        # logging.info("Retrain Logistic Regression sulle feature selezionate...")
        # model_lr = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        # model_lr.fit(train_X_sel, train_y)
        # metrics_lr = evaluate_model(model_lr, test_X_sel, test_y, 'Logistic Regression')

        # logging.info("Retrain Random Forest sulle feature selezionate...")
        # model_rf = train_random_forest(train_X_sel, train_y)
        # metrics_rf = evaluate_model(model_rf, test_X_sel, test_y, 'Random Forest')

        # --- SOLO XGBOOST ---
        logging.info("Retrain XGBoost sulle feature selezionate...")
        model_xgb = train_xgboost(train_X_sel, train_y)
        metrics_xgb = evaluate_model(model_xgb, test_X_sel, test_y, 'XGBoost')

        # Salva il modello
        joblib.dump(model_xgb, os.path.join(MODELS_DIR, 'xgboost.pkl'))

        # Prepara dizionario metriche
        models_metrics = {
            'XGBoost': metrics_xgb
        }

        # Plots e report
        plot_confusion_matrices(models_metrics, OUTPUTS_DIR)
        generate_report(models_metrics, os.path.join(REPORTS_DIR, 'giorno_7.md'))

        logging.info("Day 7 completed successfully!")

    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 