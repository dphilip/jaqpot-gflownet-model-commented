"""
Configuration classes for data handling and replay buffer management in GFlowNet training.

This module defines the configuration structure for data-related components of the
GFlowNet training pipeline, with a focus on replay buffer settings for off-policy
learning. The replay buffer is a key component for improving sample efficiency by
reusing previously generated trajectories during training.

The configuration allows fine-grained control over:
- Replay buffer capacity and warmup behavior
- On-policy vs off-policy sampling ratios
- Hindsight experience replay for exploration
- Batch composition and sampling strategies
"""

# Standard library imports for dataclass definition
from dataclasses import dataclass
from typing import Optional

# GFlowNet utility imports
from gflownet.utils.misc import StrictDataClass


@dataclass
class ReplayConfig(StrictDataClass):
    """
    Configuration for replay buffer in off-policy GFlowNet training.
    
    The replay buffer stores previously generated trajectories and samples from them
    during training to improve sample efficiency and learning stability. This configuration
    class provides detailed control over replay buffer behavior, sampling strategies,
    and integration with on-policy learning.
    
    The replay buffer supports several advanced features:
    - Mixed on-policy/off-policy training batches
    - Hindsight experience replay for better exploration
    - Flexible capacity management and warmup periods
    - Configurable sampling ratios and batch composition
    
    Attributes
    ----------
    use : bool, default=False
        Whether to enable the replay buffer for training
        When False, training is purely on-policy using only current samples
        When True, training mixes on-policy and off-policy samples
        
    capacity : Optional[int], default=None
        Maximum number of trajectories the replay buffer can store
        When None, buffer size is unlimited (limited only by memory)
        Larger capacity allows more diverse experience but uses more memory
        
    warmup : Optional[int], default=None
        Number of trajectories to collect before starting replay buffer sampling
        During warmup, training is purely on-policy to build initial buffer
        Should be large enough to provide diverse experiences for stable training
        Typical values: 1000-10000 depending on trajectory complexity
        
    hindsight_ratio : float, default=0
        Fraction of replay samples to use hindsight experience replay
        Range: [0.0, 1.0] where 0 means no hindsight, 1 means all hindsight
        Hindsight replay modifies past trajectories with current knowledge
        Useful for exploration in sparse reward environments
        
    num_from_replay : Optional[int], default=None
        Number of trajectories sampled from replay buffer per training batch
        When None, defaults to cfg.algo.num_from_policy (50/50 on/off-policy split)
        Lower values favor on-policy learning, higher values favor off-policy
        Must be ≤ total batch size
        
    num_new_samples : Optional[int], default=None
        Number of new on-policy trajectories generated and added to buffer per step
        When None, defaults to cfg.algo.num_from_policy
        Can be different from num_from_policy to control buffer growth rate
        If > num_from_policy: buffer grows faster than training batch consumes
        If < num_from_policy: not all on-policy samples are stored in buffer
        
    Examples
    --------
    # Basic replay buffer setup
    replay_config = ReplayConfig(
        use=True,
        capacity=50000,
        warmup=1000,
        hindsight_ratio=0.1
    )
    
    # Advanced setup with asymmetric sampling
    replay_config = ReplayConfig(
        use=True,
        capacity=100000,
        warmup=5000,
        num_from_replay=16,      # 16 off-policy samples
        num_new_samples=32,      # 32 new samples per step
        hindsight_ratio=0.2      # 20% hindsight experience
    )
    """

    use: bool = False                        # Enable/disable replay buffer
    capacity: Optional[int] = None           # Maximum buffer size (None = unlimited)
    warmup: Optional[int] = None             # Warmup samples before replay starts  
    hindsight_ratio: float = 0               # Fraction of hindsight experience samples
    num_from_replay: Optional[int] = None    # Off-policy samples per batch
    num_new_samples: Optional[int] = None    # New samples added to buffer per step
