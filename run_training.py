"""
Script to run the cryptocurrency trading model training.
"""

import os
import logging
from pathlib import Path
import yaml
import numpy as np
from src.data.data_loader import CryptoDataLoader
from src.features.feature_engineering import FeatureEngineer
from src.env.trading_env import CryptoTradingEnv
from src.models.rl_agents import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Load configuration
        config_path = Path("src/config/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"CONFIG DEBUG: {config}")
        
        # Initialize data loader
        data_loader = CryptoDataLoader(config)
        
        # Load and prepare data
        logging.info("Loading training data...")
        train_df = data_loader.load_data(mode='train')
        
        # Prepare sequences
        logging.info("Preparing sequences...")
        sequences, timestamps = data_loader.prepare_sequences(train_df)
        
        # Split data
        data_splits = data_loader.split_data(sequences, timestamps)
        train_sequences, train_timestamps = data_splits['train']
        val_sequences, val_timestamps = data_splits['validation']
        
        # Create training environment
        train_env = CryptoTradingEnv(config)
        train_env.set_data(train_sequences, train_timestamps, price_idx=3)  # Close price index
        
        # Create validation environment
        val_env = CryptoTradingEnv(config)
        val_env.set_data(val_sequences, val_timestamps, price_idx=3)
        
        # Wrap environments
        train_env = DummyVecEnv([lambda: train_env])
        val_env = DummyVecEnv([lambda: val_env])
        
        # Initialize and train agent
        agent = PPOAgent(config)
        logging.info("Starting training...")
        
        # Train with validation
        total_timesteps = config['training']['total_timesteps']
        eval_freq = config['training'].get('eval_freq', 10000)
        n_eval_episodes = config['training'].get('n_eval_episodes', 5)
        
        try:
            # Reset environments before training
            train_env.reset()
            val_env.reset()
            
            agent.train(
                env=train_env,
                total_timesteps=total_timesteps,
                eval_env=val_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes
            )
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
        
        # Save model
        model_path = Path("models/ppo_crypto_trading")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(model_path))
        logging.info(f"Model saved to {model_path}")
        
        # Evaluate final performance
        logging.info("Evaluating final performance...")
        mean_reward, std_reward = agent.evaluate(val_env, n_eval_episodes=10)
        logging.info(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 