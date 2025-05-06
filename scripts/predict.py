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

# 1. Carica e normalizza
df = load_hourly_csv('BTC')
df = normalize_timestamp(df, ts_col='Open time', tz_from='UTC', tz_to='UTC')

# 2. Costruisci le feature
advanced_indicators = [
    'bollinger_bands',
    'ema',
    'rsi',
    'stoch_rsi',
    'wave_trend',
    'pps_killer',
    'macd'
]
features = build_features(df, add_basic=True, advanced_indicators=advanced_indicators)

# 3. Crea le label per un orizzonte di 24 ore
labeled = create_labels(features, target_col='Close', horizon=24)

print(labeled[['Close', 'label']].tail())
print('Distribuzione label:')
print(labeled['label'].value_counts(normalize=True))

# 4. Scaling delle feature
# Seleziono tutte le colonne tranne 'label'
feature_cols = [c for c in labeled.columns if c != 'label']
X_scaled, scaler = scale_features(labeled, feature_cols, scaler_type=SCALER_TYPE)

# Ricompongo il DataFrame con la colonna 'label'
df_scaled = pd.concat([X_scaled, labeled['label']], axis=1)

# 5. Split train/test basato sulla data
# Supponiamo di usare come cutoff il 1 gennaio 2023
train_df, test_df = train_test_split_time(df_scaled, date_col='timestamp', split_date=SPLIT_DATE)

# 6. Preparo i dizionari finali di features e labels
data_dict = prepare_data(train_df, test_df)
train_X = data_dict['train_features']
train_y = data_dict['train_labels']
test_X  = data_dict['test_features']
test_y  = data_dict['test_labels']

# 7. Verifiche rapide
print("Shape train_X:", train_X.shape)
print("Shape test_X:", test_X.shape)
print("\nDistribuzione label in train:")
print(train_y.value_counts(normalize=True))
print("\nDistribuzione label in test:")
print(test_y.value_counts(normalize=True))

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

# Salva i plot
plt.savefig(os.path.join(PLOT_DIR, "roc_curve.png"))
plt.close()

# Confusion matrix a colori
plt.figure()
plt.matshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Salva i plot
plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
plt.close()

# Salva modello e scaler
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

results = {}

# --- RANDOM FOREST ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train_X, train_y)
rf_pred = rf.predict(test_X)
rf_proba = rf.predict_proba(test_X)[:,1]

results['Random Forest'] = {
    'y_pred': rf_pred,
    'y_proba': rf_proba,
    'acc': accuracy_score(test_y, rf_pred),
    'f1': f1_score(test_y, rf_pred),
    'roc_auc': roc_auc_score(test_y, rf_proba),
    'cm': confusion_matrix(test_y, rf_pred)
}

# --- XGBOOST ---
xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
xgb.fit(train_X, train_y)
xgb_pred = xgb.predict(test_X)
xgb_proba = xgb.predict_proba(test_X)[:,1]

results['XGBoost'] = {
    'y_pred': xgb_pred,
    'y_proba': xgb_proba,
    'acc': accuracy_score(test_y, xgb_pred),
    'f1': f1_score(test_y, xgb_pred),
    'roc_auc': roc_auc_score(test_y, xgb_proba),
    'cm': confusion_matrix(test_y, xgb_pred)
}

# --- RISULTATI ---
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
feature_names = train_X.columns if hasattr(train_X, 'columns') else [f'feat_{i}' for i in range(train_X.shape[1])]
sorted_idx = importances.argsort()[::-1][:20]  # Top 20

plt.figure(figsize=(10,6))
plt.barh([feature_names[i] for i in sorted_idx][::-1], importances[sorted_idx][::-1])
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "xgboost_feature_importance.png"))
plt.close()

