"""
Model evaluator for genetic optimization.

This module implements the evaluation of model configurations
by training and testing them on the trading environment.
"""

import numpy as np
from typing import Dict, Tuple
import logging
from pathlib import Path
import yaml
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.trading_env import CryptoTradingEnv
from src.models.rl_agents import PPOAgent
from src.data.data_loader import CryptoDataLoader

class ModelEvaluator:
    """Evaluates model configurations by training and testing them."""
    
    def __init__(self, 
                 data_loader: CryptoDataLoader,
                 train_ratio: float = 0.8,
                 eval_episodes: int = 5,
                 train_timesteps: int = 100000):
        """
        Initialize the evaluator.
        
        Args:
            data_loader: DataLoader instance for loading market data
            train_ratio: Ratio of data to use for training
            eval_episodes: Number of episodes to evaluate on
            train_timesteps: Number of timesteps to train for
        """
        self.data_loader = data_loader
        self.train_ratio = train_ratio
        self.eval_episodes = eval_episodes
        self.train_timesteps = train_timesteps
        
        # Load and prepare data
        self.train_data = None
        self.test_data = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Load and prepare training and testing data."""
        # Load training data
        train_df = self.data_loader.load_data(mode='train')
        train_sequences, train_timestamps = self.data_loader.prepare_sequences(train_df)
        
        # Load test data
        test_df = self.data_loader.load_data(mode='test')
        test_sequences, test_timestamps = self.data_loader.prepare_sequences(test_df)
        
        self.train_data = {
            'sequences': train_sequences,
            'timestamps': train_timestamps
        }
        
        self.test_data = {
            'sequences': test_sequences,
            'timestamps': test_timestamps
        }
        
        logging.info(f"Prepared {len(train_sequences)} training sequences and {len(test_sequences)} test sequences")
    
    def _create_env(self, data: Dict, config: Dict) -> CryptoTradingEnv:
        """Create a trading environment with the given data and configuration."""
        env = CryptoTradingEnv(config)
        env.set_data(
            sequences=data['sequences'],
            timestamps=data['timestamps'],
            price_idx=self.data_loader.price_idx
        )
        return env
    
    def evaluate_config(self, config: Dict) -> float:
        """
        Evaluate a configuration by training and testing the model.
        
        Args:
            config: Model configuration to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Create training environment
            train_env = self._create_env(self.train_data, config)
            train_env = DummyVecEnv([lambda: train_env])
            
            # Create test environment
            test_env = self._create_env(self.test_data, config)
            test_env = DummyVecEnv([lambda: test_env])
            
            # Create and train model
            model = PPOAgent(config)
            model.train(
                env=train_env,
                total_timesteps=self.train_timesteps,
                eval_env=test_env,
                eval_freq=10000,
                n_eval_episodes=self.eval_episodes
            )
            
            # Evaluate on test environment
            mean_reward, std_reward = model.evaluate(test_env, n_eval_episodes=self.eval_episodes)
            
            # Get performance metrics from the environment
            total_return = test_env.envs[0].balance / config['env']['initial_balance'] - 1
            sharpe_ratio = mean_reward / (std_reward + 1e-6)  # Avoid division by zero
            max_drawdown = test_env.envs[0].max_drawdown
            win_rate = test_env.envs[0].win_rate
            profit_factor = test_env.envs[0].profit_factor
            avg_trade_duration = test_env.envs[0].avg_trade_duration
            
            # Calculate fitness score using all metrics
            fitness = self.calculate_fitness(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration
            )
            
            logging.info(f"Configuration evaluated - Fitness: {fitness:.2f}")
            logging.info(f"Metrics - Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Drawdown: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}")
            
            return fitness
            
        except Exception as e:
            logging.error(f"Error evaluating configuration: {e}")
            return float('-inf')  # Return worst possible score on error 

    def calculate_fitness(self, total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, avg_trade_duration):
        fitness_score = (
            total_return * 0.3 +          # 30% peso
            sharpe_ratio * 0.2 +          # 20% peso
            (1 - max_drawdown) * 0.2 +    # 20% peso
            win_rate * 0.15 +             # 15% peso
            profit_factor * 0.1 +         # 10% peso
            (1 / avg_trade_duration) * 0.05  # 5% peso
        )
        return fitness_score 