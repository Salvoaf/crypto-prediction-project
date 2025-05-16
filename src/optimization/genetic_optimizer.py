"""
Genetic Algorithm for optimizing trading model hyperparameters.

This module implements a genetic algorithm to find optimal hyperparameters
for the trading model by evolving a population of configurations.
"""

import numpy as np
import yaml
import random
from typing import List, Dict, Tuple, Any
import logging
from pathlib import Path
import copy
from concurrent.futures import ProcessPoolExecutor
import json
from datetime import datetime

class GeneticOptimizer:
    """Genetic Algorithm for hyperparameter optimization."""
    
    def __init__(self, 
                 config_path: str,
                 population_size: int = 10,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2):
        """
        Initialize the genetic optimizer.
        
        Args:
            config_path: Path to the base configuration file
            population_size: Number of configurations in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutating each parameter
            elite_size: Number of best configurations to preserve
        """
        self.config_path = config_path
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        # Define parameter ranges
        self.param_ranges = {
            'model': {
                'learning_rate': (1e-5, 1e-3),
                'n_steps': (512, 4096),
                'batch_size': (32, 512),
                'n_epochs': (3, 20),
                'gamma': (0.9, 0.999),
                'gae_lambda': (0.9, 0.999),
                'clip_ratio': (0.1, 0.3),
                'entropy_coef': (0.0, 0.1),
                'value_coef': (0.5, 1.0),
                'max_grad_norm': (0.3, 1.0)
            },
            'env': {
                'initial_balance': (1000, 10000),
                'commission_rate': (0.0001, 0.001),
                'slippage': (0.0001, 0.001),
                'max_position': (0.1, 1.0),
                'reward_scaling': (0.001, 0.1),
                'min_trade_threshold': (0.01, 0.1),
                'stop_loss': (0.01, 0.05),
                'take_profit': (0.02, 0.1),
                'running_mean_window': (10, 50),
                'running_std_window': (10, 50)
            }
        }
        
        # Initialize population
        self.population = []
        self.best_config = None
        self.best_fitness = float('-inf')
        self.history = []
    
    def _create_random_config(self) -> Dict:
        """Create a random configuration within parameter ranges."""
        config = copy.deepcopy(self.base_config)
        
        for section, params in self.param_ranges.items():
            for param, (min_val, max_val) in params.items():
                if isinstance(min_val, int):
                    config[section][param] = random.randint(min_val, max_val)
                else:
                    config[section][param] = random.uniform(min_val, max_val)
        
        return config
    
    def _mutate_config(self, config: Dict) -> Dict:
        """Apply random mutations to a configuration."""
        mutated = copy.deepcopy(config)
        
        for section, params in self.param_ranges.items():
            for param, (min_val, max_val) in params.items():
                if random.random() < self.mutation_rate:
                    if isinstance(min_val, int):
                        mutated[section][param] = random.randint(min_val, max_val)
                    else:
                        mutated[section][param] = random.uniform(min_val, max_val)
        
        return mutated
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Create a child configuration by combining two parents."""
        child = copy.deepcopy(parent1)
        
        for section in self.param_ranges.keys():
            for param in self.param_ranges[section].keys():
                if random.random() < 0.5:
                    child[section][param] = parent2[section][param]
        
        return child
    
    def _evaluate_config(self, config: Dict) -> float:
        """
        Evaluate a configuration by training the model and returning the fitness score.
        This should be implemented based on your specific needs.
        """
        # TODO: Implement actual model training and evaluation
        # For now, return a random score for demonstration
        return random.random()
    
    def _save_config(self, config: Dict, fitness: float, generation: int) -> None:
        """Save a configuration and its fitness score."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_dir = Path("configs/genetic")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            'config': config,
            'fitness': fitness,
            'generation': generation,
            'timestamp': timestamp
        }
        
        config_path = config_dir / f"config_gen{generation}_fit{fitness:.4f}_{timestamp}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
    
    def optimize(self) -> Tuple[Dict, float]:
        """
        Run the genetic optimization process.
        
        Returns:
            Tuple of (best_config, best_fitness)
        """
        # Initialize population
        self.population = [self._create_random_config() for _ in range(self.population_size)]
        logging.info(f"Initialized population of size {self.population_size}")
        
        for generation in range(self.generations):
            logging.info(f"\nGeneration {generation + 1}/{self.generations}")
            
            # Evaluate all configurations
            fitness_scores = []
            for i, config in enumerate(self.population):
                logging.info(f"Evaluating configuration {i+1}/{len(self.population)} in generation {generation + 1}")
                fitness = self._evaluate_config(config)
                fitness_scores.append(fitness)
                self._save_config(config, fitness, generation)
                logging.info(f"Configuration {i+1} fitness: {fitness:.4f}")
            
            # Update best configuration
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_config = copy.deepcopy(self.population[best_idx])
                logging.info(f"New best fitness found: {self.best_fitness:.4f}")
            
            # Create next generation
            next_generation = []
            
            # Keep elite configurations
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                next_generation.append(copy.deepcopy(self.population[idx]))
            logging.info(f"Preserved {self.elite_size} elite configurations")
            
            # Create rest of population through crossover and mutation
            while len(next_generation) < self.population_size:
                # Select parents using tournament selection
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                
                # Create child through crossover
                child = self._crossover(parent1, parent2)
                
                # Apply mutation
                child = self._mutate_config(child)
                
                next_generation.append(child)
            
            self.population = next_generation
            
            # Log progress
            logging.info(f"Generation {generation + 1} completed")
            logging.info(f"Best fitness: {self.best_fitness:.4f}")
            logging.info(f"Mean fitness: {np.mean(fitness_scores):.4f}")
            logging.info(f"Std fitness: {np.std(fitness_scores):.4f}")
            
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            })
        
        return self.best_config, self.best_fitness
    
    def save_history(self, path: str = "genetic_optimization_history.json") -> None:
        """Save optimization history to a file."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self) -> None:
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            generations = [h['generation'] for h in self.history]
            best_fitness = [h['best_fitness'] for h in self.history]
            mean_fitness = [h['mean_fitness'] for h in self.history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(generations, best_fitness, label='Best Fitness')
            plt.plot(generations, mean_fitness, label='Mean Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Genetic Optimization Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig('genetic_optimization_history.png')
            plt.close()
            
        except ImportError:
            logging.warning("Matplotlib not installed. Skipping plot generation.") 