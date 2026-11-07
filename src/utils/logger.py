"""
src/utils/logger.py

Logging configuration and seed management for reproducible research experiments.
"""

import logging
import random
import sys
from pathlib import Path
import numpy as np

from src.config import LOGS_DIR, RANDOM_STATE


def setup_logger(name: str = "load_aware_sage", log_file: str = "experiment.log") -> logging.Logger:
    """
    Configure a structured logger with both console and file handlers.
    
    Args:
        name: Name of the logger module.
        log_file: Name of the output log file inside LOGS_DIR.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    # Formatter: timestamp | level | module | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Enforce deterministic random seeds across Python, NumPy, and PyTorch.
    
    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

