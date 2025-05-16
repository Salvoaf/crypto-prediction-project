"""
Script to run genetic optimization of model hyperparameters.
"""

import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent
# Get the project root (two levels up from script_dir)
project_root = script_dir.parent.parent

# Add project root to Python path
sys.path.append(str(project_root))

import logging
import yaml
import argparse

from src.optimization.genetic_optimizer import GeneticOptimizer
from src.optimization.evaluator import ModelEvaluator
from src.data.data_loader import CryptoDataLoader

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('genetic_optimization.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Run the genetic optimization process."""
    parser = argparse.ArgumentParser(description='Run genetic optimization of model hyperparameters')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to base configuration file')
    parser.add_argument('--population-size', type=int, default=10, help='Size of population')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--elite-size', type=int, default=2, help='Number of elite configurations to preserve')
    parser.add_argument('--train-timesteps', type=int, default=100000, help='Number of timesteps to train each model')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get absolute path to config file
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logging.info(f"Using config file: {config_path}")
    
    # Create results directory
    results_dir = Path("results/genetic")
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created results directory: {results_dir}")
    
    # Load base configuration
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    logging.info(f"Loaded base configuration")
    
    # Initialize data loader
    data_loader = CryptoDataLoader(base_config)
    data_loader.load_data()
    logging.info("Data loaded successfully")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_loader=data_loader,
        train_timesteps=args.train_timesteps
    )
    logging.info("Initialized model evaluator")
    
    # Initialize genetic optimizer
    optimizer = GeneticOptimizer(
        config_path=config_path,
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        elite_size=args.elite_size
    )
    logging.info("Initialized genetic optimizer")
    
    # Override the evaluate_config method with our evaluator
    optimizer._evaluate_config = evaluator.evaluate_config
    
    # Run optimization
    logging.info("Starting genetic optimization...")
    logging.info(f"Population size: {args.population_size}")
    logging.info(f"Number of generations: {args.generations}")
    logging.info(f"Mutation rate: {args.mutation_rate}")
    logging.info(f"Elite size: {args.elite_size}")
    logging.info(f"Training timesteps per model: {args.train_timesteps}")
    
    best_config, best_fitness = optimizer.optimize()
    
    # Save results
    logging.info("Saving final results...")
    
    # Save best configuration
    best_config_path = results_dir / "best_config.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f)
    logging.info(f"Saved best configuration to {best_config_path}")
    
    # Save optimization history
    history_path = results_dir / "optimization_history.json"
    optimizer.save_history(history_path)
    logging.info(f"Saved optimization history to {history_path}")
    
    # Plot and save optimization history
    plot_path = results_dir / "optimization_history.png"
    optimizer.plot_history()
    logging.info(f"Saved optimization plot to {plot_path}")
    
    logging.info(f"Optimization completed. Best fitness: {best_fitness:.4f}")
    logging.info(f"Best configuration saved to {best_config_path}")

if __name__ == "__main__":
    main() 