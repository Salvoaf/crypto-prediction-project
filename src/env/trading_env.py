"""
Trading environment for cryptocurrency trading using Gymnasium interface.

This module implements a trading environment that simulates cryptocurrency trading
with realistic constraints like commission fees, slippage, and position limits.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
import pandas as pd
from dataclasses import dataclass
import logging

@dataclass
class Trade:
    """Represents a single trade with its details."""
    entry_price: float
    entry_time: int
    position_size: float
    direction: int  # 1 for long, -1 for short
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: int = 0
    is_closed: bool = False

class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment following OpenAI Gym interface.
    
    The environment simulates trading with:
    - Realistic commission fees
    - Slippage modeling
    - Position size limits
    - Continuous action space for position sizing
    """
    
    def __init__(self, config: dict):
        """
        Initialize the trading environment.
        
        Args:
            config: Dictionary containing environment configuration
        """
        super().__init__()
        self.config = config
        self.initial_balance = config['env']['initial_balance']
        self.commission_rate = config['env']['commission_rate']
        self.slippage = config['env']['slippage']
        self.max_position = config['env']['max_position']
        self.reward_scaling = config['env']['reward_scaling']
        self.min_trade_threshold = config['env']['min_trade_threshold']
        self.stop_loss = config['env']['stop_loss']
        self.take_profit = config['env']['take_profit']
        self.sequence_length = config['features']['sequence_length']
        
        # Nuove costanti per il sistema di reward
        self.delta_time_penalty = config['env']['delta_time_penalty']
        self.trade_execution_bonus = config['env']['trade_execution_bonus']
        self.holding_cost_alpha = config['env']['holding_cost_alpha']
        self.max_drawdown_threshold = config['env']['max_drawdown_threshold']
        self.drawdown_penalty = config['env']['drawdown_penalty']
        self.min_trades_per_episode = config['env']['min_trades_per_episode']
        self.end_episode_penalty = config['env']['end_episode_penalty']
        self.curiosity_coef = config['env']['curiosity_coef']
        
        # Inizializzazione reward normalization
        self.running_mean_window = config['env']['running_mean_window']
        self.running_std_window = config['env']['running_std_window']
        self.reward_buffer = []
        self.running_mean = 0.0
        self.running_std = 1.0
        
        # Nuove variabili per il tracking
        self.peak_balance = self.initial_balance
        self.total_trades = 0
        self.state_visit_count = {}
        self.current_state = None
        
        # Initialize state
        self.sequences = None
        self.timestamps = None
        self.price_idx = None
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        self.trades = []
        self.unrealized_pnl = 0.0
        self.last_price = 0.0
        
        # Define action space
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Observation space will be properly initialized in set_data()
        self.observation_space = None
    
    def set_data(self, sequences: np.ndarray, timestamps: np.ndarray, price_idx: int):
        """Set the market data for the environment."""
        self.sequences = sequences
        self.timestamps = timestamps
        self.price_idx = price_idx
        
        # Initialize observation space
        n_features = sequences.shape[2]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, n_features + 3),
            dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        if self.current_step >= len(self.sequences):
            return None
            
        # Get market data
        market_data = self.sequences[self.current_step]
        
        # Get current price (from the last timestep)
        current_price = float(market_data[-1, self.price_idx])
        self.last_price = current_price
        
        # Create account state with safe normalization
        balance_ratio = np.clip(self.balance / self.initial_balance, -1e6, 1e6)
        pnl_ratio = np.clip(self.unrealized_pnl / self.initial_balance, -1e6, 1e6)
        
        account_state = np.array([
            balance_ratio,  # Normalized balance
            self.position,  # Current position
            pnl_ratio      # Normalized unrealized PnL
        ])
        
        # Combine market data and account state
        obs = np.column_stack([market_data, np.tile(account_state, (len(market_data), 1))])
        
        # Safe conversion to float32
        obs = np.clip(obs, -1e6, 1e6)
        return obs.astype(np.float32)
    
    def _calculate_commission(self, amount: float) -> float:
        """Calculate commission for a trade."""
        return amount * self.commission_rate
    
    def _calculate_slippage(self, price: float, amount: float) -> float:
        """Calculate slippage for a trade."""
        return price * self.slippage * np.sign(amount)
    
    def _check_stop_loss_take_profit(self, current_price: float) -> bool:
        """Check if any open positions hit stop loss or take profit."""
        for trade in self.trades:
            if not trade.is_closed:
                if trade.direction == 1:  # Long position
                    if current_price <= trade.stop_loss:
                        self._close_trade(trade, current_price, "stop_loss")
                    elif current_price >= trade.take_profit:
                        self._close_trade(trade, current_price, "take_profit")
                else:  # Short position
                    if current_price >= trade.stop_loss:
                        self._close_trade(trade, current_price, "stop_loss")
                    elif current_price <= trade.take_profit:
                        self._close_trade(trade, current_price, "take_profit")
        return True
    
    def _close_trade(self, trade: Trade, current_price: float, reason: str):
        """Close a trade and update account state."""
        trade.exit_price = current_price
        trade.exit_time = self.current_step
        trade.is_closed = True
        
        # Calculate PnL
        price_diff = (trade.exit_price - trade.entry_price) * trade.direction
        trade.pnl = trade.position_size * price_diff
        
        # Update balance and position
        self.balance += trade.pnl
        self.position -= trade.position_size * trade.direction
        
        # Apply commission
        commission = self._calculate_commission(abs(trade.position_size))
        self.balance -= commission
        
        logging.debug(f"Closed trade: {reason}, PnL: {trade.pnl:.2f}, Commission: {commission:.2f}")
    
    def _update_state_visit_count(self, state):
        """Aggiorna il contatore di visite per lo stato corrente."""
        state_key = str(state.tobytes())
        self.state_visit_count[state_key] = self.state_visit_count.get(state_key, 0) + 1
        self.current_state = state_key

    def _calculate_curiosity_bonus(self):
        """Calcola il bonus di curiosità basato sul numero di visite dello stato."""
        if self.current_state is None:
            return 0.0
        visit_count = self.state_visit_count.get(self.current_state, 1)
        return self.curiosity_coef / np.sqrt(visit_count)

    def _calculate_drawdown_penalty(self):
        """Calcola la penalità per drawdown eccessivo."""
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if current_drawdown > self.max_drawdown_threshold:
            return self.drawdown_penalty * current_drawdown
        return 0.0

    def _calculate_holding_cost(self):
        """Calcola il costo di mantenimento della posizione."""
        return self.holding_cost_alpha * abs(self.position)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_step = 0
        self.trades = []
        self.unrealized_pnl = 0.0
        self.peak_balance = self.initial_balance
        self.total_trades = 0
        self.current_state = None
        
        # Get initial observation
        obs = self._get_observation()
        if obs is not None:
            self._update_state_visit_count(obs)
        
        # Get current price
        current_price = float(self.sequences[self.current_step][-1, self.price_idx])
        self.last_price = current_price
        
        info = {
            'balance': float(self.balance),
            'position': float(self.position),
            'unrealized_pnl': float(self.unrealized_pnl),
            'current_price': float(current_price)
        }
        
        return obs, info
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        # Update buffer
        self.reward_buffer.append(reward)
        if len(self.reward_buffer) > self.running_mean_window:
            self.reward_buffer.pop(0)
        
        # Calculate running statistics
        if len(self.reward_buffer) > 1:
            self.running_mean = np.mean(self.reward_buffer)
            self.running_std = np.std(self.reward_buffer) + 1e-5
        
        # Normalize reward
        normalized_reward = (reward - self.running_mean) / self.running_std
        
        # Scale reward using the new formula
        reward_scaled = normalized_reward / (1 + abs(normalized_reward))
        
        return reward_scaled
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        if self.current_step >= len(self.sequences):
            return self._get_observation(), 0, True, False, {}
        
        # Get current price (from the last timestep)
        current_price = float(self.sequences[self.current_step][-1, self.price_idx])
        
        # Check stop loss and take profit
        self._check_stop_loss_take_profit(current_price)
        
        # Handle both scalar and array actions
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Initialize reward components
        reward_components = {
            'pnl': 0.0,
            'time_penalty': 0.0,
            'trade_bonus': 0.0,
            'holding_cost': 0.0,
            'drawdown_penalty': 0.0,
            'curiosity_bonus': 0.0
        }
        
        # Apply time penalty for HOLD
        if abs(action_value) <= self.min_trade_threshold:
            reward_components['time_penalty'] = -self.delta_time_penalty
        
        # Execute trade if action magnitude exceeds threshold
        trade_executed = False
        if abs(action_value) > self.min_trade_threshold:
            # Calculate position size
            position_size = abs(action_value) * self.max_position * self.balance / current_price
            
            # Calculate entry price with slippage
            entry_price = current_price + self._calculate_slippage(current_price, position_size)
            
            # Create new trade
            trade = Trade(
                entry_price=entry_price,
                entry_time=self.current_step,
                position_size=position_size,
                direction=int(np.sign(action_value)),
                stop_loss=entry_price * (1 - self.stop_loss) if action_value > 0 else entry_price * (1 + self.stop_loss),
                take_profit=entry_price * (1 + self.take_profit) if action_value > 0 else entry_price * (1 - self.take_profit)
            )
            
            # Update position and balance
            self.position += position_size * trade.direction
            commission = self._calculate_commission(position_size)
            self.balance -= commission
            
            self.trades.append(trade)
            self.total_trades += 1
            trade_executed = True
            reward_components['trade_bonus'] = self.trade_execution_bonus
            
            logging.debug(f"Opened trade: Size={position_size:.4f}, Price={entry_price:.2f}")
        
        # Calculate unrealized PnL
        self.unrealized_pnl = self.position * (current_price - self.last_price)
        
        # Update peak balance
        self.peak_balance = max(self.peak_balance, self.balance + self.unrealized_pnl)
        
        # Calculate reward components
        reward_components['pnl'] = float((self.unrealized_pnl + sum(t.pnl for t in self.trades)) * 0.01)
        reward_components['holding_cost'] = -self._calculate_holding_cost()
        reward_components['drawdown_penalty'] = -self._calculate_drawdown_penalty()
        reward_components['curiosity_bonus'] = self._calculate_curiosity_bonus()
        
        # Calculate total reward
        total_reward = sum(reward_components.values())
        
        # Normalize reward
        reward = self._normalize_reward(total_reward)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.sequences)
        truncated = False
        
        # Apply end-of-episode penalty if needed
        if done and self.total_trades < self.min_trades_per_episode:
            reward -= self.end_episode_penalty
        
        # Get next observation
        obs = self._get_observation()
        if obs is not None:
            self._update_state_visit_count(obs)
        
        info = {
            'balance': float(self.balance),
            'position': float(self.position),
            'unrealized_pnl': float(self.unrealized_pnl),
            'current_price': float(current_price),
            'raw_reward': total_reward,
            'normalized_reward': reward,
            'reward_components': reward_components,
            'total_trades': self.total_trades,
            'drawdown': (self.peak_balance - (self.balance + self.unrealized_pnl)) / self.peak_balance
        }
        
        return obs, reward, done, truncated, info
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Unrealized P&L: ${self.unrealized_pnl:.2f}")
            print(f"Current Price: ${self._get_observation()[self.price_idx]:.2f}")
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented") 