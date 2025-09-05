#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger utility for the Report Generator

This module provides logging setup for the report generator.
"""

import logging
import sys
from pathlib import Path


def setup_logger(level=logging.INFO, log_file=None):
    """Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (default: None, logs to console only)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger