"""
Configuration classes for GFlowNet training algorithms.

This module defines the configuration structure for various GFlowNet training algorithms
including Trajectory Balance (TB), Flow Matching (FM), Multi-Objective Q-Learning (MOQL),
Advantage Actor-Critic (A2C), and SQL-based methods. Each algorithm has its own specific
hyperparameters while sharing common base parameters.

The configuration system uses enums for categorical choices and dataclasses for
structured parameter organization, ensuring type safety and clear documentation
of all available options.
"""

# Standard library imports for dataclass and enum functionality
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

# GFlowNet utility imports
from gflownet.utils.misc import StrictDataClass


class Backward(IntEnum):
    """
    Enumeration of backward policy types for Trajectory Balance algorithms.
    
    The backward policy P_B(s|s') defines the probability of transitioning from
    state s' back to state s. Different backward policies have different properties
    and computational requirements:
    
    - Uniform: Simple uniform distribution (baseline approach)
    - Free: Fully parameterized backward policy (most flexible)
    - Maxent: Maximum entropy formulation for exploration
    - MaxentA/GSQLA: Advanced variants requiring environment-provided n values
    
    See TrajectoryBalance class documentation for detailed mathematical formulations.
    Note: The 'A' variants (MaxentA, GSQLA) require the environment to provide
    the number of valid actions n, which is available for sEH but not QM9 tasks.
    """
    
    Uniform = 1   # Uniform backward policy (simple baseline)
    Free = 2      # Fully parameterized backward policy  
    Maxent = 3    # Maximum entropy backward policy
    MaxentA = 4   # Maxent variant requiring environment n values
    GSQL = 5      # Graph SQL backward policy
    GSQLA = 6     # Graph SQL variant requiring environment n values


class NLoss(IntEnum):
    """
    Enumeration of loss functions for learning the number of paths n.
    
    These loss functions are used when learning to predict the number of valid
    action sequences from each state, which is important for flow conservation
    in GFlowNet algorithms. Different formulations provide different trade-offs
    between computational efficiency and learning accuracy.
    
    See TrajectoryBalance class for mathematical details of each loss type.
    """
    
    none = 0        # No n-loss (don't learn path counts)
    Transition = 1  # Transition-based n-loss  
    SubTB1 = 2      # Sub-trajectory balance variant 1
    TermTB1 = 3     # Terminal trajectory balance variant 1
    StartTB1 = 4    # Starting trajectory balance variant 1
    TB = 5          # Full trajectory balance n-loss


class TBVariant(IntEnum):
    """
    Enumeration of Trajectory Balance algorithm variants.
    
    These variants represent different formulations of the trajectory balance
    objective, each with different properties for learning efficiency and
    computational requirements:
    
    - TB: Standard trajectory balance (full trajectories)
    - SubTB1: Sub-trajectory balance (partial trajectories) 
    - DB: Detailed balance (single transitions)
    
    See TrajectoryBalance class documentation for detailed mathematical formulations.
    """
    
    TB = 0      # Standard Trajectory Balance (full trajectories)
    SubTB1 = 1  # Sub-Trajectory Balance variant 1 (partial trajectories)
    DB = 2      # Detailed Balance (single transitions)


class LossFN(IntEnum):
    """
    Enumeration of loss functions for training GFlowNet models.
    
    Different loss functions provide different properties for optimization:
    - MSE: Standard mean squared error (smooth, differentiable)
    - MAE: Mean absolute error (robust to outliers)
    - HUB: Huber loss (smooth near zero, linear for large errors)
    - GHL: Generalized Huber Loss (differentiable variant of Huber)
    
    References:
    GHL formulation from Kaan Gokcesu, Hakan Gokcesu
    https://arxiv.org/pdf/2108.12627.pdf
    Note: GHL can be used as a differentiable version of Huber loss.
    """
    
    MSE = 0  # Mean Squared Error (L2 loss)
    MAE = 1  # Mean Absolute Error (L1 loss)  
    HUB = 2  # Huber Loss (smooth L1)
    GHL = 3  # Generalized Huber Loss (differentiable Huber variant)


@dataclass
class TBConfig(StrictDataClass):
    """Trajectory Balance config.

    Attributes
    ----------
    bootstrap_own_reward : bool
        Whether to bootstrap the reward with the own reward. (deprecated)
    epsilon : Optional[float]
        The epsilon parameter in log-flow smoothing (see paper)
    reward_loss_multiplier : float
        The multiplier for the reward loss when bootstrapping the reward. (deprecated)
    variant : TBVariant
        The loss variant. See algo.trajectory_balance.TrajectoryBalance for details.
    do_correct_idempotent : bool
        Whether to correct for idempotent actions
    do_parameterize_p_b : bool
        Whether to parameterize the P_B distribution (otherwise it is uniform)
    do_predict_n : bool
        Whether to predict the number of paths in the graph
    do_length_normalize : bool
        Whether to normalize the loss by the length of the trajectory
    subtb_max_len : int
        The maximum length trajectories, used to cache subTB computation indices
    Z_learning_rate : float
        The learning rate for the logZ parameter (only relevant when do_subtb is False)
    Z_lr_decay : float
        The learning rate decay for the logZ parameter (only relevant when do_subtb is False)
    loss_fn: LossFN
        The loss function to use
    loss_fn_par: float
        The loss function parameter in case of Huber loss, it is the delta
    n_loss: NLoss
        The $n$ loss to use (defaults to NLoss.none i.e., do not learn $n$)
    n_loss_multiplier: float
        The multiplier for the $n$ loss
    backward_policy: Backward
        The backward policy to use
    """

    bootstrap_own_reward: bool = False
    epsilon: Optional[float] = None
    reward_loss_multiplier: float = 1.0
    variant: TBVariant = TBVariant.TB
    do_correct_idempotent: bool = False
    do_parameterize_p_b: bool = False
    do_predict_n: bool = True
    do_sample_p_b: bool = False
    do_length_normalize: bool = False
    subtb_max_len: int = 128
    Z_learning_rate: float = 1e-4
    Z_lr_decay: float = 50_000
    cum_subtb: bool = True
    loss_fn: LossFN = LossFN.MSE
    loss_fn_par: float = 1.0
    n_loss: NLoss = NLoss.none
    n_loss_multiplier: float = 1.0
    backward_policy: Backward = Backward.Uniform


@dataclass
class MOQLConfig(StrictDataClass):
    gamma: float = 1
    num_omega_samples: int = 32
    num_objectives: int = 2
    lambda_decay: int = 10_000
    penalty: float = -10


@dataclass
class A2CConfig(StrictDataClass):
    entropy: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class FMConfig(StrictDataClass):
    epsilon: float = 1e-38
    balanced_loss: bool = False
    leaf_coef: float = 10
    correct_idempotent: bool = False


@dataclass
class SQLConfig(StrictDataClass):
    alpha: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class AlgoConfig(StrictDataClass):
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    num_from_policy : int
        The number of on-policy samples for a training batch.
        If using a replay buffer, see `replay.num_from_replay` for the number of samples from the replay buffer, and
        `replay.num_new_samples` for the number of new samples to add to the replay buffer (e.g. `num_from_policy=0`,
        and `num_new_samples=N` inserts `N` new samples in the replay buffer at each step, but does not make that data
        part of the training batch).
    num_from_dataset : int
        The number of samples from the dataset for a training batch
    valid_num_from_policy : int
        The number of on-policy samples for a validation batch
    valid_num_from_dataset : int
        The number of samples from the dataset for a validation batch
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    train_random_action_prob : float
        The probability of taking a random action during training
    train_det_after: Optional[int]
        Do not take random actions after this number of steps
    valid_random_action_prob : float
        The probability of taking a random action during validation
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    num_from_policy: int = 64
    num_from_dataset: int = 0
    valid_num_from_policy: int = 64
    valid_num_from_dataset: int = 0
    max_len: int = 128
    max_nodes: int = 128
    max_edges: int = 128
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.0
    train_det_after: Optional[int] = None
    valid_random_action_prob: float = 0.0
    sampling_tau: float = 0.0
    tb: TBConfig = field(default_factory=TBConfig)
    moql: MOQLConfig = field(default_factory=MOQLConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
