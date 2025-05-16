"""
Logging utilities for the project.
"""

import os
import logging
from pathlib import Path

def setup_logging() -> None:
    """
    Set up logging configuration.
    
    Creates a logs directory and configures logging to both file and console.
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler()
        ]
    ) 