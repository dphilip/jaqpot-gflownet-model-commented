"""
Configuration management for GFlowNet training and evaluation.

This module defines the hierarchical configuration structure used throughout the GFlowNet
framework. It uses dataclasses with OmegaConf integration to provide type-safe,
structured configuration management with support for YAML files and command-line overrides.

The configuration is organized into several main sections:
- Training parameters (steps, validation, checkpointing)
- Hardware and system settings (device, workers, random seeds)
- Algorithm-specific parameters (from algo.config)
- Model architecture parameters (from models.config) 
- Task and environment settings (from tasks.config)
- Data handling and replay buffer settings (from data.config)
- Conditioning and multi-objective settings (from utils.config)

Key features:
- Hierarchical structure with sensible defaults
- Integration with OmegaConf for YAML support and CLI overrides
- Type safety through dataclass annotations
- Support for missing values that can be filled at runtime
- Utility functions for dynamic configuration initialization
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

from omegaconf import MISSING

from gflownet.algo.config import AlgoConfig
from gflownet.data.config import ReplayConfig
from gflownet.models.config import ModelConfig
from gflownet.tasks.config import TasksConfig
from gflownet.utils.config import ConditionalsConfig
from gflownet.utils.misc import StrictDataClass


@dataclass
class OptimizerConfig(StrictDataClass):
    """
    Configuration for optimization algorithms and training dynamics.
    
    This class contains all parameters related to the optimization process,
    including the choice of optimizer, learning rate scheduling, regularization,
    and gradient clipping. These settings directly affect training stability
    and convergence behavior.

    Attributes
    ----------
    opt : str, default="adam"
        The optimizer algorithm to use. Supported values:
        - "adam": Adam optimizer with adaptive learning rates
        - "sgd": Stochastic gradient descent with momentum
    learning_rate : float, default=1e-4
        Initial learning rate for the optimizer
        This is the step size used for parameter updates
    lr_decay : float, default=20_000
        Learning rate decay schedule parameter (in training steps)
        Learning rate decays as: lr = lr_initial * 2^(-steps / lr_decay)
    weight_decay : float, default=1e-8
        L2 regularization strength to prevent overfitting
        Higher values add more penalty on large weights
    momentum : float, default=0.9
        Momentum parameter for SGD optimizer (ignored for Adam)
        Helps accelerate convergence in relevant directions
    clip_grad_type : str, default="norm"
        Type of gradient clipping to prevent exploding gradients:
        - "norm": Clip gradients by their L2 norm
        - "value": Clip gradients by their absolute values
    clip_grad_param : float, default=10.0
        Parameter for gradient clipping:
        - For "norm": maximum allowed gradient norm
        - For "value": maximum allowed absolute gradient value
    adam_eps : float, default=1e-8
        Epsilon parameter for Adam optimizer numerical stability
        Prevents division by zero in adaptive learning rate computation
    """

    opt: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 20_000
    weight_decay: float = 1e-8
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8


@dataclass
class Config(StrictDataClass):
    """
    Main configuration class for GFlowNet training and evaluation.
    
    This is the root configuration object that contains all parameters needed
    for running GFlowNet experiments. It combines training logistics, system
    settings, and algorithm-specific configurations into a single structured
    object that can be serialized to/from YAML files.
    
    The configuration is hierarchical, with specialized config objects for
    different components (algorithms, models, tasks, etc.) contained as fields.

    Attributes
    ----------
    desc : str, default="noDesc"
        Human-readable description of the experiment for logging and identification
    log_dir : str, required (MISSING)
        Directory path where logs, checkpoints, and generated samples are stored
        Must be specified when creating a config instance
    device : str, default="cuda"
        PyTorch device string for computation:
        - "cuda": Use default GPU
        - "cuda:0", "cuda:1", etc.: Use specific GPU
        - "cpu": Use CPU only
    seed : int, default=0
        Random seed for reproducibility across numpy, torch, and python random
    validate_every : int, default=1000
        Number of training steps between validation runs
        Validation assesses model performance on held-out data
    checkpoint_every : Optional[int], default=None
        Number of training steps between model checkpoints
        If None, checkpointing is disabled
    store_all_checkpoints : bool, default=False
        Whether to keep all checkpoints (True) or only the latest (False)
        Affects disk usage for long training runs
    print_every : int, default=100
        Number of training steps between progress printouts to console/logs
    start_at_step : int, default=0
        Training step to start at, useful for resuming interrupted training
    num_final_gen_steps : Optional[int], default=None
        Number of generation steps to run after training completes
        Used for final evaluation and sample collection
    num_validation_gen_steps : Optional[int], default=None
        Number of generation steps during each validation run
        Controls the size of validation sample sets
    num_training_steps : int, default=10_000
        Total number of training steps to perform
        Each step processes one batch of trajectories
    num_workers : int, default=0
        Number of parallel worker processes for data generation
        0 means no multiprocessing, >0 enables parallel trajectory sampling
    hostname : Optional[str], default=None
        Machine hostname for tracking distributed experiments
        Automatically populated at runtime if not specified
    pickle_mp_messages : bool, default=False
        Whether to pickle inter-process messages (relevant only if num_workers > 0)
        May be needed for complex data structures in multiprocessing
    git_hash : Optional[str], default=None
        Git commit hash for experiment reproducibility tracking
        Automatically populated at runtime if in a git repository
    overwrite_existing_exp : bool, default=False
        Whether to overwrite existing files in log_dir if it already exists
        Safety feature to prevent accidental overwriting of results
    
    # Nested configuration objects for different components:
    algo : AlgoConfig
        Algorithm-specific parameters (trajectory balance, flow matching, etc.)
    model : ModelConfig
        Neural network architecture parameters (layers, dimensions, etc.)
    opt : OptimizerConfig
        Optimization parameters (learning rate, decay, clipping, etc.)
    replay : ReplayConfig
        Replay buffer parameters for off-policy training
    task : TasksConfig
        Task and environment parameters (rewards, objectives, etc.)
    cond : ConditionalsConfig
        Conditional generation parameters (temperature, preferences, etc.)
    """

    desc: str = "noDesc"
    log_dir: str = MISSING
    device: str = "cuda"
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: Optional[int] = None
    store_all_checkpoints: bool = False
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: Optional[int] = None
    num_validation_gen_steps: Optional[int] = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: Optional[str] = None
    pickle_mp_messages: bool = False
    git_hash: Optional[str] = None
    overwrite_existing_exp: bool = False
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    task: TasksConfig = field(default_factory=TasksConfig)
    cond: ConditionalsConfig = field(default_factory=ConditionalsConfig)


def init_empty(cfg):
    """
    Initialize a dataclass configuration with all fields set to MISSING.
    
    This utility function recursively traverses a configuration dataclass and
    sets all fields to OmegaConf's MISSING sentinel value. This is particularly
    useful for creating configuration templates where users only need to specify
    the parameters they want to override, leaving others to be filled with
    defaults later.
    
    The function handles nested dataclass configurations by recursively calling
    itself on any field that is itself a dataclass. This ensures that the entire
    configuration hierarchy is properly initialized with MISSING values.
    
    Typical usage pattern:
    ```python
    config = init_empty(Config())
    config.log_dir = "./my_experiment"
    config.num_training_steps = 50_000
    # All other fields remain MISSING and will use defaults
    ```
    
    Parameters
    ----------
    cfg : dataclass instance
        A dataclass configuration object to initialize with MISSING values
        Must be an instance of a dataclass, not the class itself
        
    Returns
    -------
    cfg : dataclass instance
        The same configuration object with all fields set to MISSING
        Nested dataclass fields are also recursively initialized
        
    Notes
    -----
    This function modifies the input configuration object in-place and also
    returns it for convenience. The MISSING values will later be replaced
    with actual defaults by OmegaConf during configuration merging.
    """
    # Iterate through all fields defined in the dataclass
    for f in fields(cfg):
        if is_dataclass(f.type):
            # Recursively initialize nested dataclass configurations
            setattr(cfg, f.name, init_empty(f.type()))
        else:
            # Set primitive fields to MISSING sentinel value
            setattr(cfg, f.name, MISSING)

    return cfg
