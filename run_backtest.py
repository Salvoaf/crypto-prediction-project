"""
Script to run backtest on the trained cryptocurrency trading model.
"""

import os
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.data.data_loader import CryptoDataLoader
from src.env.trading_env import CryptoTradingEnv
from src.models.rl_agents import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv

def calculate_metrics(equity_curve: pd.Series, trades: list) -> dict:
    """Calculate trading performance metrics."""
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
    max_drawdown = (equity_curve / equity_curve.cummax() - 1).min() * 100
    
    # Trade statistics
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
    profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
    
    # Calculate additional metrics
    avg_trade_duration = np.mean([t.exit_time - t.entry_time for t in trades]) if trades else 0
    avg_profit_per_trade = np.mean([t.pnl for t in trades]) if trades else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade_duration': avg_trade_duration,
        'avg_profit_per_trade': avg_profit_per_trade
    }

def plot_results(equity_curve: pd.Series, trades: list, save_path: str = None):
    """Plot backtest results."""
    plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.subplot(2, 1, 1)
    plt.plot(equity_curve.index, equity_curve.values, label='Equity')
    plt.title('Equity Curve')
    plt.grid(True)
    plt.legend()
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
    plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3, label='Drawdown')
    plt.title('Drawdown (%)')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load configuration
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize data loader
        data_loader = CryptoDataLoader(config)
        
        # Load test data
        logging.info("Loading test data...")
        test_df = data_loader.load_data(mode='test')
        
        # Prepare sequences
        logging.info("Preparing sequences...")
        sequences, timestamps = data_loader.prepare_sequences(test_df)
        
        # Create test environment
        base_env = CryptoTradingEnv(config)
        base_env.set_data(sequences, timestamps, price_idx=3)  # Close price index
        
        # Wrap environment
        test_env = DummyVecEnv([lambda: base_env])
        
        # Load trained model
        model_path = Path("models/ppo_crypto_trading.zip")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        agent = PPOAgent(config)
        agent.load(str(model_path))
        
        # Run backtest
        logging.info("Running backtest...")
        obs = test_env.reset()  # This now returns just the observation
        done = False
        equity_curve = []
        current_equity = config['env']['initial_balance']
        
        while not done:
            try:
                action, _ = agent.predict(obs)
                obs, rewards, dones, infos = test_env.step(action)
                done = dones[0]  # Get the first done flag since we're using VecEnv
                current_equity = infos[0]['balance'] + infos[0]['unrealized_pnl']
                equity_curve.append(current_equity)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                raise
        
        # Convert equity curve to pandas Series
        equity_curve = pd.Series(equity_curve, index=timestamps[:len(equity_curve)])
        
        # Get the underlying environment
        env_trades = test_env.envs[0].trades
        
        # Calculate metrics using the base environment's trades
        metrics = calculate_metrics(equity_curve, env_trades)
        
        # Print results
        logging.info("\nBacktest Results:")
        logging.info(f"Total Return: {metrics['total_return']:.2f}%")
        logging.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logging.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        logging.info(f"Win Rate: {metrics['win_rate']:.2f}%")
        logging.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logging.info(f"Total Trades: {metrics['total_trades']}")
        logging.info(f"Winning Trades: {metrics['winning_trades']}")
        logging.info(f"Losing Trades: {metrics['losing_trades']}")
        logging.info(f"Average Win: ${metrics['avg_win']:.2f}")
        logging.info(f"Average Loss: ${metrics['avg_loss']:.2f}")
        logging.info(f"Average Trade Duration: {metrics['avg_trade_duration']:.1f} periods")
        logging.info(f"Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}")
        
        # Plot results
        plot_results(equity_curve, env_trades, "results/backtest_results.png")
        logging.info("Results plot saved to results/backtest_results.png")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 