from scripts.preprocessing import load_hourly_csv, normalize_timestamp
from scripts.feature_engineering import build_features
from scripts.label_generation import create_labels
from scripts.data_preparation import scale_features, train_test_split_time, prepare_data
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import glob
from tqdm import tqdm
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parametri
HORIZON = 24
SPLIT_DATE = '2023-01-01'
SCALER_TYPE = 'standard'
MODEL_PATH = 'models/logistic_regression.pkl'
SCALER_PATH = 'models/scaler.pkl'
PLOT_DIR = 'outputs/graphs/'

os.makedirs(PLOT_DIR, exist_ok=True)

def load_all_cryptos(data_dir, pattern="*USDT.csv", n_coins=None, test_symbol="BTCUSDT"):
    all_files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    # Trova il file del test_symbol
    test_file = None
    for file in all_files:
        if os.path.basename(file).startswith(test_symbol):
            test_file = file
            break
    # Prendi i primi n_coins file (escludendo il test_file se già incluso)
    selected_files = all_files[:n_coins] if n_coins is not None else all_files
    if test_file and test_file not in selected_files:
        selected_files = selected_files[:-1] + [test_file]  # Sostituisci l'ultimo con BTCUSDT
    # Se test_file non c'è, warning
    if not test_file:
        print(f"[WARNING] Il file per {test_symbol} non è stato trovato!")
    print(f"[INFO] File caricati: {[os.path.basename(f) for f in selected_files]}")
    # Carica i file come prima
    dfs = []
    for file in tqdm(selected_files, desc="Caricamento file", unit="file"):
        symbol = os.path.basename(file).replace('.csv', '')
        try:
            df = pd.read_csv(file)
            if df.empty or df.columns.size == 0:
                tqdm.write(f"[WARNING] File vuoto o senza colonne: {file} - salto.")
                continue
            df['symbol'] = symbol
            df['Open time'] = pd.to_datetime(df['Open time'])
            dfs.append(df)
        except pd.errors.EmptyDataError:
            tqdm.write(f"[WARNING] File vuoto (EmptyDataError): {file} - salto.")
            continue
        except Exception as e:
            tqdm.write(f"[ERROR] Errore nel caricamento di {file}: {e} - salto.")
            continue
    if not dfs:
        print("[ERRORE] Nessun file valido trovato! Controlla la cartella e il pattern.")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# 1. Carica e normalizza
print("[STEP 1] Caricamento dati multi-crypto")
data_dir = 'E:/data_cache'  # <-- Modifica qui il path!
N_COINS = 20
df = load_all_cryptos(data_dir, n_coins=N_COINS, test_symbol="BTCUSDT")
if df.empty:
    print("[ERRORE] DataFrame vuoto dopo il caricamento! Esco.")
    exit(1)
print(f"[INFO] Dati caricati: {df.shape}")

print("[STEP 2] Normalizzazione timestamp")
df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')

print("[STEP 3] Feature engineering")
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
    features_symbol = build_features(df_symbol, add_basic=True, advanced_indicators=advanced_indicators)
    # Se 'Open time' è indice, riportalo a colonna
    if features_symbol.index.name == 'Open time':
        features_symbol = features_symbol.reset_index()
    features_symbol['symbol'] = symbol
    features_list.append(features_symbol)
features = pd.concat(features_list, ignore_index=True)

print("[STEP 4] Generazione label")
labeled = create_labels(features, target_col='Close', horizon=HORIZON)
# Propaga 'Open time' tagliando le ultime righe come fa create_labels
if 'Open time' in features.columns:
    labeled['Open time'] = features['Open time'].iloc[:-HORIZON].values


print("[INFO] Ultime righe con label:")
print(labeled[['Close', 'label']].tail())
print('Distribuzione label:')
print(labeled['label'].value_counts(normalize=True))



print("[STEP 5] Scaling delle feature")
# Seleziono solo le colonne numeriche da scalare
feature_cols = [c for c in labeled.columns if c not in ['label', 'symbol', 'Open time']]
X_scaled, scaler = scale_features(labeled, feature_cols, scaler_type=SCALER_TYPE)

print("[STEP 6] Ricomposizione DataFrame scalato")
# Ricompongo il DataFrame con label, symbol e Open time
df_scaled = pd.concat([X_scaled, labeled[['label', 'symbol', 'Open time']].reset_index(drop=True)], axis=1)
print(f"[INFO] Dati scalati: {df_scaled.shape}")

print("[STEP 7] Split train/test: allena su tutte le crypto tranne BTCUSDT, testa su BTCUSDT")
btc_mask = df_scaled['symbol'] == 'BTCUSDT'
train_df = df_scaled[~btc_mask]
test_df = df_scaled[btc_mask]

print(f"[INFO] Shape train_df: {train_df.shape}")
print(f"[INFO] Shape test_df: {test_df.shape}")

print("[STEP 8] Preparo dizionari finali di features e labels")
data_dict = prepare_data(train_df, test_df)
exclude_cols = ['label', 'symbol', 'Open time']
feature_cols = [c for c in df_scaled.columns if c not in exclude_cols]
train_X = train_df[feature_cols]
test_X = test_df[feature_cols]
train_y = train_df['label']
test_y = test_df['label']

print("Shape train_X:", train_X.shape)
print("Shape test_X:", test_X.shape)
print("\nDistribuzione label in train:")
print(train_X.value_counts(normalize=True))
print("\nDistribuzione label in test:")
print(test_X.value_counts(normalize=True))

print("Colonne in train_X:", train_X.columns)

print("[DEBUG] Numero di NaN in train_X:", train_X.isna().sum().sum())
print("[DEBUG] Numero di NaN in test_X:", test_X.isna().sum().sum())

train_X = train_X.dropna()
train_y = train_y.loc[train_X.index]
test_X = test_X.dropna()
test_y = test_y.loc[test_X.index]

print("[STEP 9] Logistic Regression")
model = LogisticRegression(max_iter=1000)
model.fit(train_X, train_y)

y_pred = model.predict(test_X)
y_proba = model.predict_proba(test_X)[:,1]

acc = accuracy_score(test_y, y_pred)
f1  = f1_score(test_y, y_pred)
roc_auc = roc_auc_score(test_y, y_proba)
cm = confusion_matrix(test_y, y_pred)
print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
print("Confusion Matrix:\n", cm)

# ROC
fpr, tpr, _ = roc_curve(test_y, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))
plt.close()

# Confusion matrix a colori
plt.figure()
plt.matshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.close()

# Salva modello e scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("[STEP 10] Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train_X, train_y)
rf_pred = rf.predict(test_X)
rf_proba = rf.predict_proba(test_X)[:,1]

print("[STEP 11] XGBoost")
xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb.fit(train_X, train_y)
xgb_pred = xgb.predict(test_X)
xgb_proba = xgb.predict_proba(test_X)[:,1]

results = {
    'Random Forest': {
        'y_pred': rf_pred,
        'y_proba': rf_proba,
        'acc': accuracy_score(test_y, rf_pred),
        'f1': f1_score(test_y, rf_pred),
        'roc_auc': roc_auc_score(test_y, rf_proba),
        'cm': confusion_matrix(test_y, rf_pred)
    },
    'XGBoost': {
        'y_pred': xgb_pred,
        'y_proba': xgb_proba,
        'acc': accuracy_score(test_y, xgb_pred),
        'f1': f1_score(test_y, xgb_pred),
        'roc_auc': roc_auc_score(test_y, xgb_proba),
        'cm': confusion_matrix(test_y, xgb_pred)
    }
}

for model_name, res in results.items():
    print(f"\n--- {model_name} ---")
    print(f"Accuracy: {res['acc']:.3f}, F1: {res['f1']:.3f}, ROC-AUC: {res['roc_auc']:.3f}")
    print("Confusion Matrix:\n", res['cm'])

    # ROC Curve
    fpr, tpr, _ = roc_curve(test_y, res['y_proba'])
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(PLOT_DIR, f"roc_curve_{model_name.replace(' ', '_').lower()}.png"))
    plt.close()

    # Confusion Matrix
    plt.figure()
    plt.matshow(res['cm'], cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOT_DIR, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"))
    plt.close()

# --- FEATURE IMPORTANCE (XGBoost) ---
importances = xgb.feature_importances_
feature_names = np.array(train_X.columns)
top_n = 20  # puoi cambiare questo numero

sorted_idx = np.argsort(importances)[::-1][:top_n]
selected_features = feature_names[sorted_idx]

print("\nTop 20 feature importances (XGBoost):")
for i in sorted_idx:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# --- Riallena i modelli solo con le feature selezionate ---
train_X_sel = train_X[selected_features]
test_X_sel = test_X[selected_features]

print("\n[Feature Selection] Riallena Logistic Regression solo con le top feature...")
model_sel = LogisticRegression(max_iter=1000)
model_sel.fit(train_X_sel, train_y)
y_pred_sel = model_sel.predict(test_X_sel)
y_proba_sel = model_sel.predict_proba(test_X_sel)[:,1]
print("Accuracy (selected):", accuracy_score(test_y, y_pred_sel))
print("F1 (selected):", f1_score(test_y, y_pred_sel))
print("ROC-AUC (selected):", roc_auc_score(test_y, y_proba_sel))

