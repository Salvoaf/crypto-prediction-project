"""
Reinforcement Learning agents for cryptocurrency trading.

This module implements various RL agents (PPO, A2C, DQN) that can be used
for trading cryptocurrencies. Each agent is implemented as a separate class
with a common interface.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.a2c import MlpPolicy as A2CMlpPolicy
from stable_baselines3.dqn import MlpPolicy as DQNMlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import logging
import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import get_linear_fn
from tqdm import tqdm
import torch

def check_gpu_availability():
    """Check GPU availability and print detailed diagnostics."""
    logging.info("Checking GPU availability...")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        logging.info(f"CUDA version: {cuda_version}")
        
        # Get number of GPUs
        n_gpus = torch.cuda.device_count()
        logging.info(f"Number of GPUs: {n_gpus}")
        
        # Get GPU information
        for i in range(n_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            logging.info(f"GPU {i}: {gpu_name}")
            logging.info(f"GPU {i} Memory: {gpu_memory:.2f} GB")
        
        # Check if PyTorch was built with CUDA
        logging.info(f"PyTorch built with CUDA: {torch.backends.cudnn.enabled}")
        
        # Check cuDNN version
        if torch.backends.cudnn.is_available():
            logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        else:
            logging.warning("cuDNN is not available")
    else:
        logging.warning("CUDA is not available. Possible reasons:")
        logging.warning("1. No NVIDIA GPU found")
        logging.warning("2. NVIDIA drivers not installed")
        logging.warning("3. CUDA toolkit not installed")
        logging.warning("4. PyTorch not built with CUDA support")
        
        # Check PyTorch installation
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"PyTorch CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")
    
    return cuda_available

class ProgressBarCallback(BaseCallback):
    """Callback to display a progress bar during training."""
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
    
    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

class BaseRLAgent:
    """Base class for all RL agents."""
    
    def __init__(self, config: dict):
        """
        Initialize the base agent.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        self.env = None
    
    def train(self, env: gym.Env, total_timesteps: int) -> None:
        """
        Train the agent.
        
        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train for
        """
        raise NotImplementedError
    
    def predict(self, observation) -> np.ndarray:
        """
        Make a prediction for the given observation.
        
        Args:
            observation: Current environment observation (np.ndarray or dict)
            
        Returns:
            Action to take
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError

class PPOAgent(BaseRLAgent):
    """Proximal Policy Optimization agent."""
    
    def __init__(self, config: dict):
        """
        Initialize the PPO agent.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__(config)
        
        # PPO specific parameters
        self.learning_rate = config['model']['learning_rate']
        self.n_steps = config['model']['n_steps']
        self.batch_size = config['model']['batch_size']
        self.n_epochs = config['model']['n_epochs']
        self.gamma = config['model']['gamma']
        self.gae_lambda = config['model']['gae_lambda']
        self.clip_ratio = config['model']['clip_ratio']
        self.value_coef = config['model']['value_coef']
        self.entropy_coef = config['model']['entropy_coef']
        self.max_grad_norm = config['model']['max_grad_norm']
        
        # Initialize model as None
        self.model = None
    
    def train(self, env: gym.Env, total_timesteps: int, eval_env: Optional[gym.Env] = None,
              eval_freq: int = 10000, n_eval_episodes: int = 5):
        """Train the PPO agent."""
        # Create model with custom optimizer
        learning_rate = float(self.config['model']['learning_rate'])
        
        # Check GPU availability with detailed diagnostics
        device = "cuda" if check_gpu_availability() else "cpu"
        logging.info(f"Using device: {device}")
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=get_linear_fn(
                start=learning_rate,
                end=learning_rate * 0.1,
                end_fraction=0.8
            ),
            n_steps=self.config['model']['n_steps'],
            batch_size=self.config['model']['batch_size'],
            n_epochs=self.config['model']['n_epochs'],
            gamma=self.config['model']['gamma'],
            gae_lambda=self.config['model']['gae_lambda'],
            clip_range=self.config['model']['clip_ratio'],
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=self.config['model']['entropy_coef'],
            vf_coef=self.config['model']['value_coef'],
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=0.015,
            tensorboard_log=self.config['training']['tensorboard_log'],
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                activation_fn=torch.nn.ReLU,
                log_std_init=0.0
            ),
            device=device,
            verbose=1
        )
        
        # Create callbacks
        callbacks = [ProgressBarCallback(total_timesteps)]
        
        # Add evaluation callback if eval_env is provided
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path="models/",
                log_path="logs/",
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1
        )
    
    def evaluate(self, env: VecEnv, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the agent.
        
        Args:
            env: Evaluation environment
            n_eval_episodes: Number of episodes for evaluation
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        try:
            episode_rewards = []
            for _ in range(n_eval_episodes):
                obs = env.reset()[0]  # Get only the observation from reset
                done = False
                episode_reward = 0.0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward[0]  # Take first reward since we're using VecEnv
                    done = done or truncated
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            return mean_reward, std_reward
            
        except Exception as e:
            logging.error(f"Error during evaluation: {e}")
            raise
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Make a prediction for the given observation.
        
        Args:
            observation: Current observation from the environment
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, info)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        try:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            return action, {}
            
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        
        try:
            self.model.save(path)
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        try:
            self.model = PPO.load(path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

class A2CAgent(BaseRLAgent):
    """Advantage Actor-Critic agent."""
    
    def __init__(self, config: dict):
        """
        Initialize the A2C agent.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__(config)
        
        # A2C specific parameters
        self.learning_rate = config['model']['learning_rate']
        self.n_steps = config['model']['n_steps']
        self.gamma = config['model']['gamma']
        self.value_coef = config['model']['value_coef']
        self.entropy_coef = config['model']['entropy_coef']
        self.max_grad_norm = config['model']['max_grad_norm']
    
    def train(self, env: gym.Env, total_timesteps: int) -> None:
        """
        Train the A2C agent.
        
        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train for
        """
        # Create vectorized environment
        self.env = DummyVecEnv([lambda: env])
        
        # Initialize A2C model
        self.model = A2C(
            policy=A2CMlpPolicy,
            env=self.env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            gamma=self.gamma,
            vf_coef=self.value_coef,
            ent_coef=self.entropy_coef,
            max_grad_norm=self.max_grad_norm,
            tensorboard_log=self.config['training']['tensorboard_log'],
            verbose=1
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, observation) -> np.ndarray:
        """
        Make a prediction using the A2C model.
        Args:
            observation: Current environment observation (np.ndarray or dict)
        Returns:
            Action to take
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        if isinstance(observation, dict):
            obs = self._process_observation(observation)
        else:
            obs = observation
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def load(self, path: str) -> None:
        """
        Load the A2C model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = A2C.load(path)
    
    def _process_observation(self, observation: Dict) -> np.ndarray:
        """
        Process the observation into the format expected by the model.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation
        """
        # Concatenate market data and account state
        market_data = observation['market_data'].flatten()
        account_state = observation['account_state']
        
        return np.concatenate([market_data, account_state])

class DQNAgent(BaseRLAgent):
    """Deep Q-Network agent."""
    
    def __init__(self, config: dict):
        """
        Initialize the DQN agent.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__(config)
        
        # DQN specific parameters
        self.learning_rate = config['model']['learning_rate']
        self.gamma = config['model']['gamma']
        self.buffer_size = 100000
        self.learning_starts = 1000
        self.target_update_interval = 1000
        self.train_freq = 4
        self.gradient_steps = 1
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
    
    def train(self, env: gym.Env, total_timesteps: int) -> None:
        """
        Train the DQN agent.
        
        Args:
            env: Training environment
            total_timesteps: Total number of timesteps to train for
        """
        # Create vectorized environment
        self.env = DummyVecEnv([lambda: env])
        
        # Initialize DQN model
        self.model = DQN(
            policy=DQNMlpPolicy,
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            target_update_interval=self.target_update_interval,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            tensorboard_log=self.config['training']['tensorboard_log'],
            verbose=1
        )
        
        # Train the model
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, observation) -> np.ndarray:
        """
        Make a prediction using the DQN model.
        Args:
            observation: Current environment observation (np.ndarray or dict)
        Returns:
            Action to take
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        if isinstance(observation, dict):
            obs = self._process_observation(observation)
        else:
            obs = observation
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def load(self, path: str) -> None:
        """
        Load the DQN model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = DQN.load(path)
    
    def _process_observation(self, observation: Dict) -> np.ndarray:
        """
        Process the observation into the format expected by the model.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation
        """
        # Concatenate market data and account state
        market_data = observation['market_data'].flatten()
        account_state = observation['account_state']
        
        return np.concatenate([market_data, account_state])

class CryptoTradingAgent:
    """Trading agent using PPO algorithm."""
    
    def __init__(self, env: VecEnv, config: dict):
        """
        Initialize the trading agent.
        
        Args:
            env: Training environment
            config: Configuration dictionary
        """
        self.env = env
        self.config = config
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config['model']['learning_rate'],
            n_steps=2048,  # Numero di step per ogni update
            batch_size=config['model']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            gae_lambda=config['model']['gae_lambda'],
            clip_range=config['model']['clip_range'],
            ent_coef=config['model']['entropy_coef'],
            verbose=1
        )
    
    def train(self) -> None:
        """Train the agent."""
        try:
            total_timesteps = self.config['training']['total_timesteps']
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True
            )
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make a prediction for the given observation.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Tuple of (action, info)
        """
        return self.model.predict(observation, deterministic=True)
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = PPO.load(path, env=self.env) 