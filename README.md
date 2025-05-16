# 🤖 Crypto Trading with Genetic Algorithm Optimization

## 📚 Descrizione del Progetto

Questo progetto implementa un sistema di trading automatico per criptovalute utilizzando Reinforcement Learning (RL) ottimizzato tramite algoritmi genetici. Il sistema combina:

- **PPO (Proximal Policy Optimization)** come algoritmo di RL
- **Algoritmo Genetico** per l'ottimizzazione degli iperparametri
- **Feature Engineering** avanzato per l'analisi tecnica
- **Backtesting** completo per la validazione delle strategie

## 🎯 Obiettivo

L'obiettivo è sviluppare un sistema di trading che:
1. Impara autonomamente strategie di trading ottimali
2. Ottimizza automaticamente i suoi iperparametri
3. Gestisce il rischio in modo efficiente
4. Si adatta a diverse condizioni di mercato

## 🛠️ Tecnologie Utilizzate

- Python 3.x
- Stable-Baselines3 (PPO implementation)
- PyTorch
- Pandas, NumPy
- TA-Lib (Technical Analysis)
- Matplotlib, Seaborn
- Gymnasium (Trading Environment)

## 📂 Struttura del Progetto

```
/crypto-prediction-project
│
├── src/
│   ├── optimization/          # Algoritmo genetico e ottimizzazione
│   │   ├── genetic_optimizer.py
│   │   ├── evaluator.py
│   │   └── run_optimization.py
│   │
│   ├── models/               # Implementazioni dei modelli RL
│   │   └── rl_agents.py
│   │
│   ├── env/                 # Ambiente di trading
│   │   └── trading_env.py
│   │
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py
│   │
│   └── utils/             # Utility functions
│       └── logging.py
│
├── config/                # File di configurazione
│   └── config.yaml
│
├── docs/                # Documentazione
│   └── feature_engineering.md
│
├── run_training.py      # Script per il training del modello
├── run_backtest.py      # Script per il backtesting
├── check_gpu.py         # Script per la verifica della GPU
└── requirements.txt
```

## ⚙️ Iperparametri Ottimizzati

### 1. Parametri del Modello (PPO)
```python
'model': {
    'learning_rate': (1e-5, 1e-3),      # Tasso di apprendimento
    'n_steps': (512, 4096),             # Step per update
    'batch_size': (32, 512),            # Dimensione batch
    'n_epochs': (3, 20),                # Epoche per update
    'gamma': (0.9, 0.999),              # Fattore di sconto
    'gae_lambda': (0.9, 0.999),         # Parametro GAE
    'clip_ratio': (0.1, 0.3),           # Clipping PPO
    'entropy_coef': (0.0, 0.1),         # Coefficiente entropia
    'value_coef': (0.5, 1.0),           # Coefficiente valore
    'max_grad_norm': (0.3, 1.0)         # Norma gradiente
}
```

### 2. Parametri dell'Ambiente
```python
'env': {
    'initial_balance': (1000, 10000),    # Saldo iniziale
    'commission_rate': (0.0001, 0.001),  # Commissioni
    'slippage': (0.0001, 0.001),        # Slippage
    'max_position': (0.1, 1.0),         # Posizione max
    'reward_scaling': (0.001, 0.1),     # Scaling reward
    'min_trade_threshold': (0.01, 0.1), # Soglia min trade
    'stop_loss': (0.01, 0.05),          # Stop loss
    'take_profit': (0.02, 0.1),         # Take profit
    'running_mean_window': (10, 50),    # Finestra media
    'running_std_window': (10, 50)      # Finestra std
}
```

## 🔄 Algoritmo Genetico

L'ottimizzazione genetica funziona attraverso:

1. **Inizializzazione**
   - Crea una popolazione iniziale di configurazioni
   - Ogni configurazione è un set di iperparametri

2. **Valutazione**
   - Addestra il modello PPO con ogni configurazione
   - Valuta le performance su un periodo di test
   - Calcola una fitness score

3. **Evoluzione**
   - **Selezione**: Mantiene le migliori configurazioni (elite)
   - **Crossover**: Combina parametri tra configurazioni
   - **Mutazione**: Modifica casualmente alcuni parametri

4. **Terminazione**
   - Dopo un numero prefissato di generazioni
   - Quando si raggiunge una fitness target
   - Quando non ci sono miglioramenti

## 🚀 Come Eseguire il Progetto

### 1. Setup Iniziale
```bash
# Clona il repository
git clone https://github.com/tuo_username/crypto-prediction-project.git
cd crypto-prediction-project

# Installa le dipendenze
pip install -r requirements.txt
```

### 2. Verifica GPU
```bash
python check_gpu.py
```

### 3. Ottimizzazione Genetica
```bash
python src/optimization/run_optimization.py \
    --config config/config.yaml \
    --population-size 10 \
    --generations 20 \
    --mutation-rate 0.1 \
    --elite-size 2 \
    --train-timesteps 100000
```

### 4. Training del Modello
```bash
python run_training.py --config config/config.yaml
```

### 5. Backtesting
```bash
python run_backtest.py
```

## Feature Engineering

The trading environment implements a sophisticated reward system that guides the agent's learning process through various incentives and penalties:

### Core Components

1. **Time-based Penalties**
   - `delta_time_penalty`: Penalizes the agent for extended periods of inactivity
   - `end_episode_penalty`: Applied if the agent makes fewer trades than the minimum threshold
   - Purpose: Encourages active trading while maintaining reasonable frequency

2. **Trade Execution Rewards**
   - `trade_execution_bonus`: Rewards the agent for executing trades
   - `min_trade_threshold`: Defines the minimum action magnitude to be considered a trade
   - Purpose: Balances the time penalties and encourages decisive actions

3. **Position Management**
   - `holding_cost_alpha`: Penalizes maintaining open positions
   - `unrealized_pnl`: Tracks unrealized profit/loss for open positions
   - Purpose: Encourages timely position closure and efficient capital usage

4. **Risk Management**
   - `max_drawdown_threshold`: Maximum allowed drawdown (10%)
   - `drawdown_penalty`: Applied when drawdown exceeds threshold
   - `stop_loss` and `take_profit`: Automatic position closure levels
   - Purpose: Enforces risk management and prevents excessive losses

5. **Exploration Incentives**
   - `curiosity_coef`: Rewards the agent for exploring new states
   - Based on state visit frequency
   - Purpose: Promotes strategy diversity and prevents local optima

### Reward Calculation

The total reward is a weighted sum of:
```python
total_reward = (
    pnl_reward * reward_scaling +
    trade_execution_bonus * (if trade executed) -
    delta_time_penalty * (if no trade) -
    holding_cost_alpha * abs(position) -
    drawdown_penalty * (if drawdown > threshold) +
    curiosity_bonus
)
```

### Key Parameters

```yaml
env:
  reward_scaling: 0.01        # Overall reward scaling factor
  delta_time_penalty: 0.001   # Penalty for inactivity
  trade_execution_bonus: 0.01 # Reward for executing trades
  holding_cost_alpha: 0.0001  # Cost of maintaining positions
  max_drawdown_threshold: 0.1 # Maximum allowed drawdown (10%)
  drawdown_penalty: 0.1       # Penalty for exceeding drawdown
  min_trades_per_episode: 5   # Minimum required trades
  curiosity_coef: 0.01        # Exploration bonus coefficient
```

This reward system is designed to:
- Balance trading activity (not too frequent, not too rare)
- Manage risk effectively (prevent excessive losses)
- Promote exploration (discover new strategies)
- Simulate real trading costs (commissions, slippage, holding costs)

## Genetic Algorithm Reward Optimization

The genetic algorithm optimizes the reward system parameters to find the optimal balance between different trading behaviors. Here's how it works:

### Reward Parameters Optimization

The genetic algorithm searches for optimal values of these reward-related parameters:

```yaml
'env': {
    'reward_scaling': (0.001, 0.1),     # Range: 0.001 to 0.1
    'delta_time_penalty': (0.0001, 0.01), # Range: 0.0001 to 0.01
    'trade_execution_bonus': (0.001, 0.1), # Range: 0.001 to 0.1
    'holding_cost_alpha': (0.00001, 0.001), # Range: 0.00001 to 0.001
    'max_drawdown_threshold': (0.05, 0.2),  # Range: 5% to 20%
    'drawdown_penalty': (0.05, 0.2),    # Range: 0.05 to 0.2
    'min_trades_per_episode': (3, 10),  # Range: 3 to 10 trades
    'curiosity_coef': (0.001, 0.1)      # Range: 0.001 to 0.1
}
```

### Optimization Process

1. **Initialization**
   - Creates a population of reward configurations
   - Each configuration has different values for reward parameters
   - Values are randomly selected within the specified ranges

2. **Evaluation**
   - Trains the PPO agent with each reward configuration
   - Evaluates performance using multiple metrics:
     - Total return
     - Sharpe ratio
     - Maximum drawdown
     - Win rate
     - Profit factor
     - Average trade duration

3. **Fitness Calculation**
   ```python
   fitness_score = (
       total_return * 0.3 +
       sharpe_ratio * 0.2 +
       (1 - max_drawdown) * 0.2 +
       win_rate * 0.15 +
       profit_factor * 0.1 +
       (1 / avg_trade_duration) * 0.05
   )
   ```

4. **Evolution**
   - **Selection**: Keeps the best reward configurations
   - **Crossover**: Combines parameters from different configurations
   - **Mutation**: Randomly adjusts reward parameters
   - **Elitism**: Preserves the best configurations

5. **Convergence**
   - Stops when:
     - Maximum generations reached
     - Fitness score plateaus
     - Target performance achieved

### Benefits of Reward Optimization

1. **Adaptive Behavior**
   - System learns optimal balance between:
     - Trading frequency
     - Risk management
     - Position holding time
     - Exploration vs exploitation

2. **Market Adaptation**
   - Different reward configurations for:
     - Different market conditions
     - Different trading pairs
     - Different timeframes

3. **Risk Management**
   - Optimizes risk-reward ratios
   - Balances profit targets with drawdown limits
   - Adjusts position sizing parameters

4. **Performance Metrics**
   - Tracks optimization progress
   - Monitors parameter convergence
   - Evaluates strategy robustness