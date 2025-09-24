"""
Configuration classes for conditional generation in GFlowNets.

This module defines the configuration structure for various conditional generation
mechanisms used in GFlowNet training. These configurations control how the model
generates diverse outputs by conditioning on different types of information.

The conditional system enables:
- Temperature-controlled exploration (hot/cold sampling)
- Multi-objective optimization with weighted preferences  
- Focus region sampling for targeted generation
- Advanced preference learning and adaptation

Each conditional type has its own configuration class with specific parameters
for controlling the sampling behavior and encoding methods.
"""

# Standard library imports for dataclass functionality
from dataclasses import dataclass, field
from typing import Any, List, Optional

# GFlowNet utility imports
from gflownet.utils.misc import StrictDataClass


@dataclass
class TempCondConfig(StrictDataClass):
    """
    Configuration for temperature-based conditional generation.
    
    Temperature conditioning controls the exploration-exploitation trade-off in
    GFlowNet sampling by modulating the sampling temperature. Different temperature
    values lead to different sampling behaviors:
    - High temperature (β → 0): More exploratory, diverse sampling
    - Low temperature (β → ∞): More exploitative, greedy sampling
    
    The temperature can be sampled from various distributions to provide
    curriculum learning or adaptive exploration during training.
    
    Attributes
    ----------
    sample_dist : str, default="uniform"  
        Distribution for sampling inverse temperature β values. Options:
        - "uniform": Uniform distribution over specified range
        - "loguniform": Log-uniform distribution (good for wide ranges)
        - "gamma": Gamma distribution (flexible shape control)
        - "constant": Fixed temperature (no variability)
        - "beta": Beta distribution (bounded support)
        
    dist_params : List[Any], default=[0.5, 32]
        Parameters for the temperature distribution
        Format depends on sample_dist:
        - uniform/loguniform: [min_value, max_value]
        - gamma: [shape, scale] or [shape, scale, offset]
        - constant: [temperature_value]
        - beta: [alpha, beta, min, max]
        
    num_thermometer_dim : int, default=32
        Dimensionality of thermometer encoding for temperature values
        Higher dimensions provide finer resolution but increase model complexity
        Typical values: 16-64 depending on precision requirements
    """

    sample_dist: str = "uniform"                              # Temperature sampling distribution
    dist_params: List[Any] = field(default_factory=lambda: [0.5, 32])  # Distribution parameters
    num_thermometer_dim: int = 32                             # Thermometer encoding dimension


@dataclass
class MultiObjectiveConfig(StrictDataClass):
    """
    Configuration for multi-objective optimization conditioning.
    
    This configuration controls the basic multi-objective setup where multiple
    objectives need to be balanced. It provides the foundational parameters
    for encoding multi-objective information in the neural network.
    
    Attributes
    ----------
    num_objectives : int, default=2
        Total number of objectives to optimize simultaneously
        Common values: 2-5 objectives (higher dimensions become challenging)
        TODO: This may conflict with task-specific num_objectives settings
        
    num_thermometer_dim : int, default=16
        Dimensionality of thermometer encoding for objective values
        Lower than temperature encoding since objectives are typically bounded
        Typical values: 8-32 depending on objective precision needs
    """
    
    num_objectives: int = 2     # Number of simultaneous objectives
    num_thermometer_dim: int = 16  # Encoding dimension for objectives


@dataclass
class WeightedPreferencesConfig(StrictDataClass):
    """
    Configuration for weighted preference-based conditional generation.
    
    This system allows for flexible multi-objective optimization by sampling
    different preference weights that specify the relative importance of
    each objective. This enables a single model to generate solutions
    across the entire Pareto front by varying preferences at inference time.
    
    Attributes
    ----------
    preference_type : Optional[str], default="dirichlet"
        Type of preference weight sampling distribution. Options:
        - "dirichlet": Dirichlet distribution (ensures weights sum to 1)
        - "dirichlet_exponential": Dirichlet with exponential temperature scaling
        - "seeded": Predefined enumerated preference vectors
        - None: Equal weights for all objectives (no preference)
        
    preference_param : Optional[float], default=1.5
        Parameter controlling the preference distribution shape
        For Dirichlet: concentration parameter (α)
        - α < 1: Sparse preferences (favors corners of simplex)
        - α = 1: Uniform over simplex
        - α > 1: Dense preferences (favors center of simplex)
        Typical values: 0.1-5.0 depending on desired preference diversity
    """

    preference_type: Optional[str] = "dirichlet"     # Preference sampling method
    preference_param: Optional[float] = 1.5          # Distribution shape parameter


@dataclass
class FocusRegionConfig(StrictDataClass):
    """
    Configuration for focus region-based conditional generation.
    
    Focus regions allow the model to concentrate generation efforts on specific
    areas of the objective space. This is useful for targeted exploration or
    when interested in particular trade-off regions between objectives.
    
    The focus mechanism can use various strategies from simple geometric
    regions to learned models that adapt based on training data.
    
    Attributes
    ----------
    focus_type : Optional[str], default="centered"
        Type of focus region definition. Options:
        - None: No focus regions (uniform sampling)
        - "centered": Focus on center of objective space
        - "partitioned": Divide space into discrete regions
        - "dirichlet": Dirichlet-distributed focus directions
        - "hyperspherical": Focus directions on hypersphere
        - "learned-gfn": GFlowNet-learned focus regions
        - "learned-tabular": Tabular model for focus learning
        
    use_steer_thermometer : bool, default=False
        Whether to use thermometer encoding for steering/focus strength
        Provides continuous control over focus intensity
        
    focus_cosim : float, default=0.98
        Cosine similarity threshold for focus region boundaries
        Range: [0, 1] where higher values create tighter focus regions
        Typical values: 0.9-0.99 for meaningful focusing
        
    focus_limit_coef : float, default=0.1
        Coefficient for focus boundary strength
        Range: (0, 1] where lower values create sharper boundaries
        Controls the decay rate outside focus regions
        
    focus_model_training_limits : tuple[float, float], default=(0.25, 0.75)
        Training limits for learned focus models
        Defines the region of objective space used for model training
        Format: (lower_percentile, upper_percentile)
        
    focus_model_state_space_res : int, default=30
        Resolution of state space discretization for tabular focus models
        Higher values provide finer control but increase memory usage
        Typical values: 20-50 depending on objective dimensionality
        
    max_train_it : int, default=20_000
        Maximum training iterations for learned focus models
        Controls the training budget for adaptive focus learning
        Should be balanced with main GFlowNet training duration
    """

    focus_type: Optional[str] = "centered"                    # Focus region strategy
    use_steer_thermomether: bool = False                      # Thermometer steering encoding
    focus_cosim: float = 0.98                                 # Focus boundary similarity threshold
    focus_limit_coef: float = 0.1                             # Focus boundary strength coefficient
    focus_model_training_limits: tuple[float, float] = (0.25, 0.75)  # Training region limits
    focus_model_state_space_res: int = 30                     # State space resolution
    max_train_it: int = 20_000                                # Maximum training iterations


@dataclass
class ConditionalsConfig(StrictDataClass):
    """
    Master configuration for all conditional generation mechanisms.
    
    This class aggregates all conditional generation configurations into a
    single structure for easy management and passing to the training system.
    It provides a unified interface for configuring complex conditional
    generation behaviors.
    
    Attributes
    ----------
    valid_sample_cond_info : bool, default=True
        Whether to sample fresh conditioning information during validation
        If False, uses fixed conditioning for consistent validation metrics
        If True, samples new conditioning to test generalization
        
    temperature : TempCondConfig
        Temperature-based conditioning configuration
        Factory method ensures each instance gets independent config
        
    moo : MultiObjectiveConfig
        Multi-objective optimization configuration
        Factory method ensures each instance gets independent config
        
    weighted_prefs : WeightedPreferencesConfig
        Weighted preferences conditioning configuration
        Factory method ensures each instance gets independent config
        
    focus_region : FocusRegionConfig
        Focus region conditioning configuration
        Factory method ensures each instance gets independent config
    """
    
    valid_sample_cond_info: bool = True                       # Sample conditioning during validation
    
    # Nested conditional configurations (use factories for independence)
    temperature: TempCondConfig = field(default_factory=TempCondConfig)
    moo: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    weighted_prefs: WeightedPreferencesConfig = field(default_factory=WeightedPreferencesConfig)
    focus_region: FocusRegionConfig = field(default_factory=FocusRegionConfig)
