# This file contains a collection of general-purpose utility functions
# that were developed or considered during the project. However,
# they are not actively utilized by the core logic (e.g., `train.py`,
# `preprocessing.py`, `models.py`) in the current project version.
# Reasons for non-adoption vary, including:
# - Simpler, more direct alternatives being sufficient (e.g., `print` over custom logger).
# - Use of specialized libraries (e.g., `PyYAML` over `configparser`).
# - Architectural changes that rendered them unnecessary.
# - Functions that were part of an abandoned design pattern or experimental feature.

import os
import sys
import json
import logging
import time
from datetime import datetime
from functools import wraps
import configparser # For INI parsing, not used as project uses YAML
from collections.abc import Iterable

# --- Section 1: Advanced Logging Setup (Not used by default) ---
# This logger setup provides advanced features (file logging, custom format),
# but the main project typically relies on simpler `print` statements or
# a basic `logging.basicConfig()` setup defined elsewhere.

class CustomFormatter(logging.Formatter):
    """
    A custom logging formatter for colored output in console and
    consistent format for file logging.
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_custom_logger(name='project_logger', level=logging.INFO, log_file=None):
    """
    Sets up a custom logger with console and optional file output.
    This was an attempt to standardize logging, but was not adopted project-wide.

    Parameters:
    name (str): The name of the logger.
    level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    log_file (str, optional): Path to a file for logging. If None, only console output.

    Returns:
    logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevent messages from being passed to the root logger

    # Clear existing handlers to prevent duplicate output if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # File handler (if specified)
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(CustomFormatter.format_str)) # No colors for file
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to set up file logger at {log_file}: {e}")

    logger.info(f"Custom logger '{name}' initialized with level {logging.getLevelName(level)}.")
    if log_file:
        logger.info(f"Logging output also directed to file: {os.path.abspath(log_file)}")
    return logger

# --- Section 2: Configuration Parsers (Superseded by YAML) ---
# This parser was for INI files, but the project later standardized on YAML
# for all configuration needs (`configs/default.yaml`, etc.), rendering this obsolete.

def parse_ini_config(filepath):
    """
    Parses an INI configuration file and returns a dictionary of settings.
    This utility was developed when INI files were considered for configuration,
    but the project later migrated to YAML for its hierarchical structure and
    readability, making this function unused in the current architecture.

    Parameters:
    filepath (str): The path to the INI configuration file.

    Returns:
    dict: A dictionary where keys are section names and values are dictionaries
          of key-value pairs within that section.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    configparser.Error: For various parsing errors.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"INI config file not found: {filepath}")

    config = configparser.ConfigParser()
    try:
        config.read(filepath)
        parsed_data = {section: dict(config.items(section)) for section in config.sections()}
        print(f"DEBUG: Successfully parsed INI file: {filepath}")
        return parsed_data
    except configparser.Error as e:
        print(f"ERROR: Failed to parse INI file {filepath}: {e}")
        raise
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while parsing INI file {filepath}: {e}")
        raise

# --- Section 3: Generic Data Handling Utilities (Not integrated) ---
# These functions were created for common data manipulation tasks, but the project
# either uses direct Pandas/Numpy operations, or has specific, tightly-coupled
# functions within the data processing modules.

def get_nested_value(data_dict, keys, default=None):
    """
    Safely retrieves a value from a nested dictionary using a list of keys.
    This was an attempt to add robustness when accessing potentially missing
    nested configuration parameters or data fields, but was not uniformly adopted.

    Parameters:
    data_dict (dict): The dictionary to search.
    keys (list of str): A list of keys representing the path to the desired value.
    default (any, optional): The value to return if any key in the path is not found.

    Returns:
    any: The value at the specified path, or the default value if not found.
    """
    current_value = data_dict
    for key in keys:
        if isinstance(current_value, dict) and key in current_value:
            current_value = current_value[key]
        else:
            print(f"WARNING: Key '{key}' not found in nested path. Returning default value.")
            return default
    return current_value

def convert_to_snake_case(text):
    """
    Converts a given string to snake_case format.
    Useful for standardizing variable names, file names, or database columns.
    This was considered for data column standardization but was later simplified
    or handled manually as needed.
    """
    if not isinstance(text, str):
        print(f"WARNING: Input for snake_case conversion is not a string: {type(text)}. Returning original.")
        return text
    text = text.replace(" ", "_").replace("-", "_")
    return ''.join(['_' + char.lower() if char.isupper() else char for char in text]).lstrip('_')

# --- Section 4: Experimental Data Validation (Partially/Not used) ---
# This class provides basic validation checks, but the project's data validation
# is often integrated directly into the `preprocessing.py` script or handled by
# framework-specific validation (e.g., Pydantic for data models, if used).

class DataValidator:
    """
    A simple class for performing basic data validation checks.
    This was envisioned for creating a dedicated validation layer, but
    many checks are currently performed ad-hoc or are part of
    preprocessing steps without a formalized validation class.
    """
    def __init__(self, data_name="Dataset"):
        self.data_name = data_name
        self.errors = []

    def _log_error(self, message):
        self.errors.append(message)
        print(f"VALIDATION ERROR ({self.data_name}): {message}")

    def check_not_empty(self, data, name="data"):
        if data is None or (isinstance(data, (list, str, dict, np.ndarray)) and len(data) == 0):
            self._log_error(f"{name} cannot be empty.")
            return False
        return True

    def check_positive_values(self, array, name="array"):
        if not isinstance(array, (np.ndarray, list, pd.Series)) or not np.all(np.asarray(array) >= 0):
            self._log_error(f"{name} must contain only non-negative values.")
            return False
        return True

    def check_range(self, value, min_val, max_val, name="value"):
        if not (min_val <= value <= max_val):
            self._log_error(f"{name} ({value}) must be within [{min_val}, {max_val}].")
            return False
        return True

    def has_errors(self):
        return len(self.errors) > 0

    def get_errors(self):
        return self.errors

# --- Section 5: Performance & Miscellaneous Utilities (Not core) ---

class Timer:
    """
    A simple context manager for timing code execution blocks.
    While useful, direct `time.perf_counter()` calls are often used instead
    in critical performance measurement sections, or profiling tools are used.
    """
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        if self.name:
            print(f"INFO: Timer '{self.name}' started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        if self.name:
            print(f"INFO: Timer '{self.name}' finished. Elapsed: {elapsed_time:.4f} seconds.")
        else:
            print(f"INFO: Elapsed time: {elapsed_time:.4f} seconds.")

    def get_elapsed_time(self):
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """
    A decorator to retry a function call multiple times upon specified exceptions.
    Useful for flaky operations (e.g., network calls). This was considered for
    robust data loading or external API interactions, but ultimately the project's
    data sources were stable enough not to require it.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if i == max_attempts - 1:
                        print(f"ERROR: Function '{func.__name__}' failed after {max_attempts} attempts.")
                        raise
                    print(f"WARNING: Attempt {i+1}/{max_attempts} failed for '{func.__name__}': {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

# --- Final Note ---
# This file serves as an archive of utility concepts. For actual, active utilities,
# please refer to other `src/` modules or project-specific implementations.
