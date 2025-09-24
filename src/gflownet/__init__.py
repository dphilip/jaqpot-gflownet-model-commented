"""
Core GFlowNet package initialization and base classes.

This module defines the fundamental types and abstract base classes used throughout
the GFlowNet framework. It establishes the core interfaces for algorithms and tasks
that all implementations must follow.

Key components:
- Type definitions for scalar properties and rewards
- GFNAlgorithm: Base class for all GFlowNet training algorithms
- GFNTask: Base class for all GFlowNet tasks/environments

The module uses PyTorch and PyTorch Geometric as the underlying tensor and graph
frameworks, providing typed interfaces for better code documentation and IDE support.
"""

from typing import Any, Dict, List, NewType, Optional, Tuple

import torch_geometric.data as gd
from torch import Tensor, nn

from .config import Config

# This type represents a set of scalar properties attached to each object in a batch.
ObjectProperties = NewType("ObjectProperties", Tensor)  # type: ignore

# This type represents log-scalars, in particular log-rewards at the scale we operate with with GFlowNets
# for example, converting a reward ObjectProperties to a log-scalar with log [(sum R_i omega_i) ** beta]
LogScalar = NewType("LogScalar", Tensor)  # type: ignore
# This type represents linear-scalars
LinScalar = NewType("LinScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    """
    Abstract base class for all GFlowNet training algorithms.
    
    This class defines the interface that all GFlowNet algorithms must implement,
    including trajectory balance, flow matching, and other variants. It handles
    the core training loop operations like loss computation and batch construction.
    
    Attributes:
        updates (int): Counter for the number of training updates performed
        global_cfg (Config): Global configuration object containing all hyperparameters
        is_eval (bool): Flag indicating whether the algorithm is in evaluation mode
    """
    updates: int = 0
    global_cfg: Config
    is_eval: bool = False

    def step(self):
        """Increment the update counter. Currently not used elsewhere in the codebase."""
        self.updates += 1  # This isn't used anywhere?

    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute the loss for a batch of data and provide logging information.
        
        This is the core method that each algorithm must implement to define how
        losses are computed from model predictions on trajectory data.

        Parameters
        ----------
        model : nn.Module
            The GFlowNet model being trained or evaluated
        batch : gd.Batch
            A batch of graph trajectories with associated conditioning and reward info
        num_bootstrap : Optional[int], default=0
            The number of trajectories with reward targets in the batch (if applicable)
            Used for algorithms that mix on-policy and off-policy data

        Returns
        -------
        loss : Tensor
            The scalar loss for this batch to be used for backpropagation
        info : Dict[str, Tensor]
            Dictionary of logged information about model predictions for monitoring
            training progress (e.g., predicted rewards, action probabilities, etc.)
        """
        raise NotImplementedError()

    def construct_batch(self, trajs, cond_info, log_rewards):
        """
        Construct a training batch from trajectories and their associated information.

        This method converts a list of trajectory data into a PyTorch Geometric batch
        that can be processed by the model. It typically calls the environment context's
        graph_to_Data and collate methods to handle the conversion.

        Parameters
        ----------
        trajs : List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories, where each trajectory is a sequence of
            (graph_state, action) pairs representing the construction process
        cond_info : Tensor
            Conditional information tensor for each trajectory. Shape (N, n_info)
            Contains conditioning like temperature, preferences, or other task-specific info
        log_rewards : Tensor  
            The log-transformed rewards for each trajectory. Shape (N,)
            Typically computed as log(R(x) ** beta) where R(x) is the raw reward

        Returns
        -------
        batch : gd.Batch
            A PyTorch Geometric Batch object containing the graph data with
            relevant training attributes (conditions, rewards, etc.) added
        """
        raise NotImplementedError()

    def get_random_action_prob(self, it: int):
        """
        Get the probability of taking random actions at a given training iteration.
        
        This method implements an exploration schedule that can change throughout training.
        During evaluation, it uses a different (typically lower) random action probability.
        
        Parameters
        ----------
        it : int
            Current training iteration number
            
        Returns
        -------
        float
            Probability of taking a random action (between 0 and 1)
        """
        # Use evaluation random action probability during validation/testing
        if self.is_eval:
            return self.global_cfg.algo.valid_random_action_prob
        
        # Check if we should switch to deterministic training after a certain iteration
        if self.global_cfg.algo.train_det_after is None or it < self.global_cfg.algo.train_det_after:
            return self.global_cfg.algo.train_random_action_prob
        
        # Switch to deterministic (no random actions) after the specified iteration
        return 0


class GFNTask:
    """
    Abstract base class for all GFlowNet tasks and environments.
    
    This class defines the interface for task-specific functionality including:
    - Computing object properties and rewards
    - Handling conditional information (e.g., temperature, preferences)
    - Sampling conditioning information during training
    
    Each specific task (e.g., molecular generation, protein design) should inherit
    from this class and implement the required methods.
    """
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        """
        Convert conditional information and object properties into scalar log-rewards.
        
        This method combines the raw object properties (e.g., molecular descriptors)
        with conditional information (e.g., temperature, preferences) to produce
        the final reward signal used for training.

        Parameters
        ----------
        cond_info : Dict[str, Tensor]
            Dictionary containing conditional information such as:
            - temperature: Controls exploration vs exploitation
            - preferences: Multi-objective optimization weights  
            - focus_region: Regional preferences for certain property ranges
        obj_props : ObjectProperties
            2D tensor where each row represents a vector of properties for one object
            (e.g., for molecules: logP, QED, synthetic accessibility, etc.)

        Returns
        -------
        reward : LogScalar
            1D tensor of scalar log-rewards, one for each object in the batch
            Typically computed as log((sum_i w_i * prop_i)^beta) where w_i are weights
        """
        raise NotImplementedError()

    def compute_obj_properties(self, objs: List[Any]) -> Tuple[ObjectProperties, Tensor]:
        """
        Compute numerical properties for a list of objects using task-specific proxy models.
        
        This method evaluates the quality/properties of generated objects (e.g., molecules)
        using pre-trained proxy models or analytical functions. It handles validation
        and computes the raw property vectors that will later be converted to rewards.

        Parameters
        ----------
        objs : List[Any]
            List of n generated objects (e.g., RDKit molecules, protein structures)
            The specific type depends on the task implementation

        Returns
        -------
        obj_props : ObjectProperties  
            2D tensor of shape (m, p) containing property vectors for the m valid objects
            Each row contains p numerical properties (e.g., logP, QED, SA score for molecules)
        is_valid : Tensor
            1D boolean tensor of shape (n,) indicating which objects are valid
            Invalid objects are filtered out from obj_props
        """
        raise NotImplementedError()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        """
        Sample conditional information for training batches.
        
        This method generates the conditioning information used to control the behavior
        of the GFlowNet during training. The conditioning can evolve over training iterations
        to implement curriculum learning or other training strategies.

        Parameters
        ----------
        n : int
            Number of objects to sample conditional information for
            Must match the batch size for training
        train_it : int
            Current training iteration number
            Can be used to implement curriculum learning or annealing schedules

        Returns
        -------
        cond_info : Dict[str, Tensor]
            Dictionary containing conditional information tensors such as:
            - 'temperature': Temperature values for exploration control
            - 'preferences': Multi-objective optimization weights
            - 'encoding': Encoded conditioning information for the model
            All tensors should have batch dimension n as the first dimension
        """
        raise NotImplementedError()
