"""
Miscellaneous utility functions for GFlowNet training and evaluation.

This module provides various helper functions and classes used throughout the GFlowNet
codebase, including logging utilities, random number generation for multiprocessing,
device management, and strict dataclass implementation.
"""

# Standard library imports for logging and system utilities
import logging
import sys

# Third-party imports for numerical computing and PyTorch
import numpy as np
import torch


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    """
    Create a configured logger instance with optional file and console output.
    
    This function sets up a logger with customizable output destinations and formatting.
    It removes any existing handlers to avoid duplicate logging and ensures consistent
    formatting across all log messages.
    
    Parameters
    ----------
    name : str, default="logger"
        The name identifier for this logger instance
    loglevel : int, default=logging.INFO  
        The minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logfile : str or None, default=None
        Path to log file for persistent logging; if None, no file logging
    streamHandle : bool, default=True
        Whether to enable console output to stdout
        
    Returns
    -------
    logging.Logger
        Configured logger instance ready for use
    """
    # Get or create a logger instance with the specified name
    logger = logging.getLogger(name)
    # Set the minimum logging level for this logger
    logger.setLevel(loglevel)
    
    # Remove all existing handlers to avoid duplication during debugging/reloading
    while len([logger.removeHandler(i) for i in logger.handlers]):
        pass  # Remove all handlers (only useful when debugging)
    
    # Create a consistent formatter for all log messages
    # Format: timestamp - level - logger_name - message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",  # European date format with time
    )

    # Initialize empty list to hold handler objects
    handlers = []
    
    # Add file handler if log file path is specified
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))  # Append mode
    
    # Add console handler if stream handling is enabled
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    # Apply formatter and add each handler to the logger
    for handler in handlers:
        handler.setFormatter(formatter)  # Apply consistent formatting
        logger.addHandler(handler)       # Register handler with logger

    return logger



# Global variables for managing random number generators across multiple workers
# These are used to ensure reproducible results in multiprocessing environments
_worker_rngs = {}  # Dictionary mapping worker IDs to their RandomState instances
_worker_rng_seed = [142857]  # Base seed stored in list for mutability across functions
_main_process_device = [torch.device("cpu")]  # Default device stored in list for mutability


def get_worker_rng():
    """
    Get a numpy RandomState instance specific to the current worker process.
    
    This function ensures that each worker process in PyTorch's DataLoader has its own
    independent random number generator, preventing race conditions and ensuring
    reproducible results across different numbers of workers.
    
    Returns
    -------
    np.random.RandomState
        Random number generator instance specific to current worker
        Each worker gets a unique seed based on: base_seed + worker_id
    """
    # Get information about the current worker process
    worker_info = torch.utils.data.get_worker_info()
    # Use worker ID if in multiprocessing context, otherwise use 0 for main process
    wid = worker_info.id if worker_info is not None else 0
    
    # Create new RandomState for this worker if it doesn't exist
    if wid not in _worker_rngs:
        # Each worker gets a unique seed by adding worker ID to base seed
        _worker_rngs[wid] = np.random.RandomState(_worker_rng_seed[0] + wid)
    
    return _worker_rngs[wid]


def set_worker_rng_seed(seed):
    """
    Set the base random seed for all current and future worker processes.
    
    This function updates the global seed and re-seeds all existing worker
    random number generators to ensure consistent randomization across runs.
    
    Parameters
    ----------
    seed : int
        Base seed value; each worker will use seed + worker_id as their actual seed
    """
    # Update the global base seed
    _worker_rng_seed[0] = seed
    
    # Re-seed all existing worker random number generators
    for wid in _worker_rngs:
        _worker_rngs[wid].seed(seed + wid)  # Unique seed per worker


def set_main_process_device(device):
    """
    Set the default device for the main process and workers.
    
    This function stores the device to be used by the main process. Worker processes
    typically use CPU regardless of this setting for memory efficiency.
    
    Parameters
    ----------
    device : torch.device
        The PyTorch device (CPU or GPU) to use for the main process
    """
    _main_process_device[0] = device


def get_worker_device():
    """
    Get the appropriate PyTorch device for the current process context.
    
    This function returns the main process device if running in the main process,
    or CPU device if running in a worker process (for memory efficiency and
    to avoid CUDA context issues in multiprocessing).
    
    Returns
    -------
    torch.device
        Device for current process: main_device for main process, CPU for workers
    """
    # Check if running in a worker process
    worker_info = torch.utils.data.get_worker_info()
    # Return main device if in main process, CPU if in worker process
    return _main_process_device[0] if worker_info is None else torch.device("cpu")


class StrictDataClass:
    """
    A base class that enforces strict attribute control for dataclass-like objects.
    
    This class prevents dynamic attribute creation outside of the class definition,
    helping to catch typos and maintain code quality. It's particularly useful for
    configuration objects where accidental attribute creation can lead to silent bugs.
    
    The class allows setting attributes that are either:
    1. Already defined as instance attributes (from __init__)
    2. Declared in type annotations (__annotations__)
    
    Any attempt to set an undeclared attribute raises an AttributeError with
    a descriptive message explaining the restriction.
    """

    def __setattr__(self, name, value):
        """
        Override attribute setting to enforce strict attribute control.
        
        This method is called whenever an attribute is set on the instance.
        It checks if the attribute is allowed before setting it.
        
        Parameters
        ----------
        name : str
            The name of the attribute being set
        value : Any
            The value to assign to the attribute
            
        Raises
        ------
        AttributeError
            If attempting to set an attribute that doesn't exist in the class
            definition or hasn't been initialized in __init__
        """
        # Check if attribute already exists or is declared in type annotations
        if hasattr(self, name) or name in self.__annotations__:
            # Allow setting the attribute using parent class behavior
            super().__setattr__(name, value)
        else:
            # Raise descriptive error for undeclared attributes
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'."
                f" '{type(self).__name__}' is a StrictDataClass object."
                f" Attributes can only be defined in the class definition."
            )
