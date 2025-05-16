"""
Evaluation module for cryptocurrency trading strategies.

This module provides functions to evaluate trading strategies using various
metrics like Sharpe ratio, Sortino ratio, maximum drawdown, win rate, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class TradeMetrics:
    """Container for trade-level metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    largest_win: float
    largest_loss: float

@dataclass
class PerformanceMetrics:
    """Container for overall performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    trade_metrics: TradeMetrics

def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate returns from an equity curve.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Series of returns
    """
    return equity_curve.pct_change().dropna()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    if len(excess_returns) < 2:
        return 0.0
    
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) < 2:
        return 0.0
    
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate the maximum drawdown.
    
    Args:
        equity_curve: Series of equity values
        
    Returns:
        Maximum drawdown as a percentage
    """
    rolling_max = equity_curve.expanding().max()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return abs(drawdowns.min())

def calculate_trade_metrics(trades: List[Dict]) -> TradeMetrics:
    """
    Calculate metrics for individual trades.
    
    Args:
        trades: List of trade dictionaries containing P&L information
        
    Returns:
        TradeMetrics object
    """
    if not trades:
        return TradeMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    pnls = [trade['pnl'] for trade in trades]
    winning_trades = [p for p in pnls if p > 0]
    losing_trades = [p for p in pnls if p < 0]
    
    total_trades = len(trades)
    winning_trades_count = len(winning_trades)
    losing_trades_count = len(losing_trades)
    
    win_rate = winning_trades_count / total_trades if total_trades > 0 else 0.0
    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = np.mean(losing_trades) if losing_trades else 0.0
    
    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    largest_win = max(winning_trades) if winning_trades else 0.0
    largest_loss = min(losing_trades) if losing_trades else 0.0
    
    return TradeMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades_count,
        losing_trades=losing_trades_count,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        largest_win=largest_win,
        largest_loss=largest_loss
    )

def evaluate_strategy(
    equity_curve: pd.Series,
    trades: List[Dict],
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Evaluate a trading strategy using various metrics.
    
    Args:
        equity_curve: Series of equity values
        trades: List of trade dictionaries
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        PerformanceMetrics object
    """
    # Calculate returns
    returns = calculate_returns(equity_curve)
    
    # Calculate performance metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(equity_curve)
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else float('inf')
    
    # Calculate trade metrics
    trade_metrics = calculate_trade_metrics(trades)
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        trade_metrics=trade_metrics
    )

def plot_equity_curve(equity_curve: pd.Series, trades: List[Dict], save_path: str = None) -> None:
    """
    Plot the equity curve with trade markers.
    
    Args:
        equity_curve: Series of equity values
        trades: List of trade dictionaries
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.plot(equity_curve.index, equity_curve.values, label='Equity', color='blue')
    
    # Plot trades
    for trade in trades:
        if trade['pnl'] > 0:
            plt.scatter(trade['timestamp'], equity_curve[trade['timestamp']],
                       color='green', marker='^', alpha=0.6)
        else:
            plt.scatter(trade['timestamp'], equity_curve[trade['timestamp']],
                       color='red', marker='v', alpha=0.6)
    
    plt.title('Equity Curve with Trades')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_performance_report(
    equity_curve: pd.Series,
    trades: List[Dict],
    risk_free_rate: float = 0.0,
    save_path: str = None
) -> None:
    """
    Generate a comprehensive performance report.
    
    Args:
        equity_curve: Series of equity values
        trades: List of trade dictionaries
        risk_free_rate: Risk-free rate (annualized)
        save_path: Optional path to save the report
    """
    # Calculate metrics
    metrics = evaluate_strategy(equity_curve, trades, risk_free_rate)
    
    # Create report
    report = f"""
    Performance Report
    =================
    
    Overall Performance:
    -------------------
    Total Return: {metrics.total_return:.2%}
    Annualized Return: {metrics.annualized_return:.2%}
    Sharpe Ratio: {metrics.sharpe_ratio:.2f}
    Sortino Ratio: {metrics.sortino_ratio:.2f}
    Maximum Drawdown: {metrics.max_drawdown:.2%}
    Calmar Ratio: {metrics.calmar_ratio:.2f}
    
    Trade Statistics:
    ----------------
    Total Trades: {metrics.trade_metrics.total_trades}
    Winning Trades: {metrics.trade_metrics.winning_trades}
    Losing Trades: {metrics.trade_metrics.losing_trades}
    Win Rate: {metrics.trade_metrics.win_rate:.2%}
    Average Win: ${metrics.trade_metrics.avg_win:.2f}
    Average Loss: ${metrics.trade_metrics.avg_loss:.2f}
    Profit Factor: {metrics.trade_metrics.profit_factor:.2f}
    Largest Win: ${metrics.trade_metrics.largest_win:.2f}
    Largest Loss: ${metrics.trade_metrics.largest_loss:.2f}
    """
    
    # Print report
    print(report)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    # Plot equity curve
    plot_equity_curve(equity_curve, trades, save_path=save_path.replace('.txt', '.png') if save_path else None) 