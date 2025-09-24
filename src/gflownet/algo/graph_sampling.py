"""
Graph sampling utilities for GFlowNet trajectory generation.

This module provides the core sampling functionality for generating trajectories
from trained GFlowNet models. It handles the step-by-step construction of graphs
by sampling actions from the model's policy and maintaining trajectory statistics.

The main component is GraphSampler, which manages the sampling process including:
- Action sampling with optional temperature scaling
- Trajectory statistics tracking (forward/backward log probabilities)
- Graph validation and terminal state detection
- Support for conditional generation with context information
"""

# Standard library imports
import copy
import warnings
from typing import List, Optional

# PyTorch imports for neural networks and tensors
import torch
import torch.nn as nn
from torch import Tensor

# GFlowNet environment imports for graph building
from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    action_type_to_mask,
)
# Model and utility imports
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.utils.misc import get_worker_device, get_worker_rng


def relabel(g: Graph, ga: GraphAction):
    """
    Relabel graph nodes to ensure 0-N consecutive numbering and update corresponding actions.
    
    This function is essential for maintaining compatibility with torch_geometric and
    EnvironmentContext classes which expect nodes to be labeled consecutively from 0 to N-1.
    When GraphBuildingEnv operations create gaps in node numbering (e.g., after node removal),
    this function renumbers nodes and updates action references accordingly.
    
    Parameters
    ----------
    g : Graph
        Input graph that may have non-consecutive node labels
        Nodes may be labeled with arbitrary integers due to previous operations
    ga : GraphAction  
        Graph action that references nodes in the original labeling
        May contain source/target node references that need updating
        
    Returns
    -------
    tuple[Graph, GraphAction]
        - Relabeled graph with nodes numbered 0 to len(nodes)-1
        - Updated graph action with corrected node references
        
    Notes
    -----
    Special handling for empty graphs with AddNode action: the source remains 0
    since AddNode can be applied to empty graphs where node 0 is the implicit root.
    """
    # Create mapping from old node labels to new consecutive labels 0, 1, 2, ...
    rmap = dict(zip(g.nodes, range(len(g.nodes))))
    
    # Special case: AddNode on empty graph keeps source as 0
    if not len(g) and ga.action == GraphActionType.AddNode:
        rmap[0] = 0  # AddNode can add to the empty graph, the source is still 0
    
    # Apply relabeling to the graph using the mapping
    g = g.relabel_nodes(rmap)
    
    # Update action source node reference if it exists
    if ga.source is not None:
        ga.source = rmap[ga.source]
    
    # Update action target node reference if it exists    
    if ga.target is not None:
        ga.target = rmap[ga.target]
        
    return g, ga


class GraphSampler:
    """
    A helper class for sampling graph trajectories from GFlowNet models.
    
    This class orchestrates the sampling process for generating graph trajectories
    by repeatedly querying a GFlowNet model and applying the sampled actions to
    build graphs step by step. It handles trajectory tracking, validation, and
    supports various experimental features for improved sampling.
    
    The sampler maintains trajectory statistics including forward and backward
    log probabilities, handles terminal state detection, and can optionally
    apply temperature scaling for exploration control.
    """

    def __init__(
        self, ctx, env, max_len, max_nodes, sample_temp=1, correct_idempotent=False, pad_with_terminal_state=False
    ):
        """
        Initialize the GraphSampler with environment and sampling parameters.
        
        Parameters
        ----------
        env : GraphBuildingEnv
            The graph building environment that defines valid actions and transitions
            Handles graph state management and action application
        ctx : GraphBuildingEnvContext
            Environment context providing graph representation and action space details
            Defines how graphs are encoded and what actions are available
        max_len : int
            Maximum number of steps allowed in a trajectory before forced termination
            Prevents infinite loops and controls computational cost
        max_nodes : int  
            Maximum number of nodes allowed in generated graphs before forced termination
            Prevents memory issues with extremely large graphs
        sample_temp : float, default=1
            Softmax temperature for action sampling (experimental feature)
            Values < 1 make sampling more deterministic, > 1 more exploratory
        correct_idempotent : bool, default=False
            Whether to apply corrections for idempotent actions in probability calculations
            Experimental feature for handling actions that don't change the graph state
        pad_with_terminal_state : bool, default=False
            Whether to pad trajectories with explicit terminal states for consistent length
            Experimental feature for batch processing of variable-length trajectories
        """
        # Store core environment components
        self.ctx = ctx     # Environment context for graph operations
        self.env = env     # Graph building environment for state transitions
        
        # Set trajectory length limits with defaults to prevent runaway generation
        self.max_len = max_len if max_len is not None else 128
        self.max_nodes = max_nodes if max_nodes is not None else 128
        
        # Experimental sampling parameters
        self.sample_temp = sample_temp                      # Temperature for action sampling
        self.sanitize_samples = True                        # Whether to validate generated samples
        self.correct_idempotent = correct_idempotent        # Idempotent action correction
        self.pad_with_terminal_state = pad_with_terminal_state  # Terminal state padding

    def sample_from_model(self, model: nn.Module, n: int, cond_info: Optional[Tensor], random_action_prob: float = 0.0):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action at each step

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        # This will be returned
        data = [{"traj": [], "reward_pred": None, "is_valid": True, "is_sink": []} for i in range(n)]
        # Let's also keep track of trajectory statistics according to the model
        fwd_logprob: List[List[Tensor]] = [[] for _ in range(n)]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.new() for _ in range(n)]
        done = [False for _ in range(n)]
        # TODO: instead of padding with Stop, we could have a virtual action whose probability
        # always evaluates to 1. Presently, Stop should convert to a (0,0,0) ActionIndex, which should
        # always be at least a valid index, and will be masked out anyways -- but this isn't ideal.
        # Here we have to pad the backward actions with something, since the backward actions are
        # evaluated at s_{t+1} not s_t.
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]

        rng = get_worker_rng()

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            # Construct graphs for the trajectories that aren't yet done
            torch_graphs = [self.ctx.graph_to_Data(i) for i in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            # Forward pass to get GraphActionCategorical
            # Note about `*_`, the model may be outputting its own bck_cat, but we ignore it if it does.
            # TODO: compute bck_cat.log_prob(bck_a) when relevant
            ci = cond_info[not_done_mask] if cond_info is not None else None
            fwd_cat, *_, log_reward_preds = model(self.ctx.collate(torch_graphs).to(dev), ci)
            if random_action_prob > 0:
                # Device which graphs in the minibatch will get their action randomized
                is_random_action = torch.tensor(
                    rng.uniform(size=len(torch_graphs)) < random_action_prob, device=dev
                ).float()
                # Set the logits to some large value to have a uniform distribution
                fwd_cat.logits = [
                    is_random_action[b][:, None] * torch.ones_like(i) * 100 + i * (1 - is_random_action[b][:, None])
                    for i, b in zip(fwd_cat.logits, fwd_cat.batch)
                ]
            if self.sample_temp != 1:
                sample_cat = copy.copy(fwd_cat)
                sample_cat.logits = [i / self.sample_temp for i in fwd_cat.logits]
                actions = sample_cat.sample()
            else:
                actions = fwd_cat.sample()
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a) for g, a in zip(torch_graphs, actions)]
            log_probs = fwd_cat.log_prob(actions)
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                fwd_logprob[i].append(log_probs[j].unsqueeze(0))
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                bck_a[i].append(self.env.reverse(graphs[i], graph_actions[j]))
                # Check if we're done
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                else:  # If not done, try to step the self.environment
                    gp = graphs[i]
                    try:
                        # self.env.step can raise AssertionError if the action is illegal
                        gp = self.env.step(graphs[i], graph_actions[j])
                        assert len(gp.nodes) <= self.max_nodes
                    except AssertionError:
                        done[i] = True
                        data[i]["is_valid"] = False
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    if t == self.max_len - 1:
                        done[i] = True
                    # If no error, add to the trajectory
                    # P_B = uniform backward
                    n_back = self.env.count_backward_transitions(gp, check_idempotent=self.correct_idempotent)
                    bck_logprob[i].append(torch.tensor([1 / n_back], device=dev).log())
                    data[i]["is_sink"].append(0)
                    graphs[i] = gp
                if done[i] and self.sanitize_samples and not self.ctx.is_sane(graphs[i]):
                    # check if the graph is sane (e.g. RDKit can  construct a molecule from it) otherwise
                    # treat the done action as illegal
                    data[i]["is_valid"] = False
            if all(done):
                break

        # is_sink indicates to a GFN algorithm that P_B(s) must be 1

        # There are 3 types of possible trajectories
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop)], P_B = [..., bck(gp), 1]
        #  B - ends with an invalid action.  = [..., (g, a)],                 = [..., 1]
        #  C - ends at max_len.              = [..., (g, a)],                 = [..., bck(gp)]

        # Let's say we pad terminal states, then:
        #  A - ends with a stop action. traj = [..., (g, a), (gp, Stop), (gp, None)], P_B = [..., bck(gp), 1, 1]
        #  B - ends with an invalid action.  = [..., (g, a), (g, None)],                  = [..., 1, 1]
        #  C - ends at max_len.              = [..., (g, a), (gp, None)],                 = [..., bck(gp), 1]
        # and then P_F(terminal) "must" be 1

        for i in range(n):
            # If we're not bootstrapping, we could query the reward
            # model here, but this is expensive/impractical.  Instead
            # just report forward and backward logprobs
            data[i]["fwd_logprob"] = sum(fwd_logprob[i])
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                # TODO: instead of padding with Stop, we could have a virtual action whose
                # probability always evaluates to 1.
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
    ):
        """Sample a model's P_B starting from a list of graphs, or if the model is None, use a uniform distribution
        over legal actions.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns GraphActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        dev: torch.device
            Device on which data is manipulated
        random_action_prob: float
            Probability of taking a random action (only used if model parameterizes P_B)

        """
        dev = get_worker_device()
        n = len(graphs)
        done = [False] * n
        data = [
            {
                "traj": [(graphs[i], GraphAction(GraphActionType.Stop))],
                "is_valid": True,
                "is_sink": [1],
                "bck_a": [GraphAction(GraphActionType.Stop)],
                "bck_logprobs": [0.0],
                "result": graphs[i],
            }
            for i in range(n)
        ]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        # TODO: This should be doable.
        if random_action_prob > 0:
            warnings.warn("Random action not implemented for backward sampling")

        while sum(done) < n:
            torch_graphs = [self.ctx.graph_to_Data(graphs[i]) for i in not_done(range(n))]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            if model is not None:
                ci = cond_info[not_done_mask] if cond_info is not None else None
                _, bck_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
            else:
                gbatch = self.ctx.collate(torch_graphs)
                action_types = self.ctx.bck_action_type_order
                action_masks = [action_type_to_mask(t, gbatch, assert_mask_exists=True) for t in action_types]
                bck_cat = GraphActionCategorical(
                    gbatch,
                    raw_logits=[torch.ones_like(m) for m in action_masks],
                    keys=[GraphTransformerGFN.action_type_to_key(t) for t in action_types],
                    action_masks=action_masks,
                    types=action_types,
                )
            bck_actions = bck_cat.sample()
            graph_bck_actions = [
                self.ctx.ActionIndex_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)
            ]
            bck_logprobs = bck_cat.log_prob(bck_actions)

            for i, j in zip(not_done(range(n)), range(n)):
                if not done[i]:
                    g = graphs[i]
                    b_a = graph_bck_actions[j]
                    gp = self.env.step(g, b_a)
                    f_a = self.env.reverse(g, b_a)
                    graphs[i], f_a = relabel(gp, f_a)
                    data[i]["traj"].append((graphs[i], f_a))
                    data[i]["bck_a"].append(b_a)
                    data[i]["is_sink"].append(0)
                    data[i]["bck_logprobs"].append(bck_logprobs[j].item())
                    if len(graphs[i]) == 0:
                        done[i] = True

        for i in range(n):
            # See comments in sample_from_model
            data[i]["traj"] = data[i]["traj"][::-1]
            data[i]["bck_a"] = [GraphAction(GraphActionType.Stop)] + data[i]["bck_a"][::-1]
            data[i]["is_sink"] = data[i]["is_sink"][::-1]
            data[i]["bck_logprobs"] = torch.tensor(data[i]["bck_logprobs"][::-1], device=dev).reshape(-1)
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data
