

# 📈 Crypto Price Movement Prediction

Benvenuto nel progetto **Crypto Price Movement Prediction**!  
In questo progetto applichiamo tecniche di **machine learning** e **deep learning** per prevedere il **movimento di prezzo** (salita/discesa) di criptovalute, basandoci su dati storici di mercato e indicatori tecnici.

---

## 📚 Descrizione del Progetto

L'obiettivo di questo progetto è sviluppare un sistema che predica se il prezzo di una criptovaluta salirà o scenderà nelle prossime 24 ore.  
Utilizziamo dati OHLCV (Open, High, Low, Close, Volume) e vari indicatori tecnici come feature per allenare modelli di classificazione.

Il progetto è strutturato per essere facilmente estendibile a:
- Altri timeframe di previsione (6h, 12h, 1 settimana)
- Altre criptovalute
- Modelli più complessi (es: LSTM, reinforcement learning)

---

## 🛠️ Tecnologie utilizzate

- Python 3.x
- Pandas, Numpy
- Scikit-learn
- XGBoost
- Tensorflow / Keras
- Matplotlib, Seaborn
- TA-Lib (per il calcolo degli indicatori tecnici)

---

## 📂 Struttura del progetto

```
/crypto-prediction-project
│
├── data/
│   ├── raw/          # Dati grezzi
│   ├── processed/    # Dati preprocessati
│   └── indicators/   # Dati con feature tecniche
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_training_baseline.ipynb
│   ├── 03_training_xgboost.ipynb
│   ├── 04_training_mlp.ipynb
│   └── 05_backtesting.ipynb
│
├── models/
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── mlp_model.h5
│
├── scripts/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── predict.py
│   └── backtest.py
│
├── outputs/
│   ├── graphs/
│   └── reports/
│
├── requirements.txt
└── README.md
```

---

## 📈 Dataset

- Fonte dati: [Binance API]
- Periodo: [Lo storico offerto da Binance]
- Frequenza: [oraria]

---

## ⚙️ Come eseguire il progetto

1. Clonare il repository:
   ```bash
   git clone https://github.com/tuo_username/crypto-prediction-project.git
   cd crypto-prediction-project
   ```

2. Installare i requisiti:
   ```bash
   pip install -r requirements.txt
   ```

3. Eseguire i notebook in ordine:
   - `01_preprocessing.ipynb` ➔ prepara i dati
   - `02_training_baseline.ipynb` ➔ primo modello
   - `03_training_xgboost.ipynb` ➔ modello avanzato
   - `04_training_mlp.ipynb` ➔ deep learning
   - `05_backtesting.ipynb` ➔ simulazione trading

---

## 🔥 Risultati principali

- Accuracy [XX%] sul test set.
- F1-score [YY%].
- Ritorno cumulativo [ZZ%] con la strategia basata sul modello vs Buy&Hold.

*(Aggiorna questi dati man mano che hai i risultati!)*

---

## 🚀 Prossimi sviluppi

- Provare modelli LSTM e GRU per gestione sequenziale dei dati.
- Integrare dati esterni (sentiment analysis da social media).
- Applicare reinforcement learning per l'ottimizzazione del portafoglio.

---



---

# 

---

🎯 **Consiglio pratico**: appena finisci ogni task (tipo allenare un modello, fare un grafico), **torna nel README** e aggiorna subito i risultati o aggiungi note, così ti trovi tutto pronto alla fine!

---
