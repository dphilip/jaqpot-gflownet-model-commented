"""
Toy sequence generation task for GFlowNet testing and development.

This module implements a simple sequence generation task that serves as a lightweight
test case for GFlowNet algorithms. The task generates sequences of characters and
rewards them based on the occurrence count of specific target patterns.

Key features:
- Simple character-based sequence generation
- Reward based on pattern matching with target sequences
- Temperature-based conditional generation
- Useful for algorithm development and debugging

The task is designed to be:
1. Fast to run for quick iteration during development
2. Easy to understand and interpret results
3. Suitable for testing new algorithms and model architectures
4. Providing clear signal for learning (pattern-based rewards)

This makes it ideal for validating that GFlowNet implementations are working
correctly before moving to more complex tasks like molecular generation.
"""

import socket
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.seq_building_env import AutoregressiveSeqBuildingContext, SeqBuildingEnv
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class ToySeqTask(GFNTask):
    """
    Simple sequence generation task with pattern-based rewards.
    
    This task generates sequences of characters and rewards them based on how many
    times they contain specific target patterns. The reward is normalized to [0,1]
    based on sequence length and pattern frequency.
    
    The task is useful for:
    - Testing GFlowNet algorithms on a simple, interpretable problem
    - Debugging sequence generation environments and models
    - Rapid prototyping of new conditioning strategies
    - Educational purposes to understand GFlowNet behavior
    
    Attributes:
        seqs (List[str]): Target sequences/patterns to search for in generated sequences
        temperature_conditional (TemperatureConditional): Temperature-based conditioning
        num_cond_dim (int): Dimensionality of conditioning encoding
        norm (float): Normalization factor for rewards based on sequence lengths
    """

    def __init__(
        self,
        seqs: List[str],
        cfg: Config,
    ) -> None:
        """
        Initialize the toy sequence task.
        
        Parameters
        ----------
        seqs : List[str]
            List of target sequences/patterns to reward in generated sequences
            Generated sequences get higher rewards for containing these patterns
        cfg : Config
            Configuration object containing task and conditioning parameters
        """
        self.seqs = seqs
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        # Normalize rewards based on max sequence length and shortest target pattern
        self.norm = cfg.algo.max_len / min(map(len, seqs))

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        """
        Sample temperature conditioning information for sequence generation.
        
        Parameters
        ----------
        n : int
            Number of conditioning samples to generate
        train_it : int
            Current training iteration (unused in this simple task)
            
        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing temperature conditioning information
        """
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        """
        Convert object properties and conditioning info into log-rewards.
        
        Applies temperature scaling to the raw pattern-matching rewards to control
        exploration vs exploitation during training.
        
        Parameters
        ----------
        cond_info : Dict[str, Tensor]
            Dictionary containing temperature conditioning information
        obj_props : ObjectProperties
            Raw reward values based on pattern matching
            
        Returns
        -------
        LogScalar
            Temperature-scaled log-rewards for training
        """
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))

    def compute_obj_properties(self, objs: List[str]) -> Tuple[ObjectProperties, Tensor]:
        """
        Compute pattern-matching rewards for generated sequences.
        
        For each generated sequence, counts how many times any of the target
        patterns appear in it. The counts are normalized by the normalization
        factor to keep rewards in a reasonable range.
        
        Parameters
        ----------
        objs : List[str]
            List of generated sequences to evaluate
            
        Returns
        -------
        ObjectProperties
            2D tensor of rewards (one per sequence) based on pattern matching
        Tensor
            Boolean tensor indicating validity (all sequences are valid in this task)
        """
        # Count occurrences of all target patterns in each generated sequence
        rs = torch.tensor([sum([s.count(p) for p in self.seqs]) for s in objs]).float() / self.norm
        # Return rewards as 2D tensor and mark all sequences as valid
        return ObjectProperties(rs[:, None]), torch.ones(len(objs), dtype=torch.bool)


class ToySeqTrainer(StandardOnlineTrainer):
    task: ToySeqTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.num_validation_gen_steps = 1
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 10
        cfg.algo.max_len = 10
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

    def setup_model(self):
        self.model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
        )

    def setup_task(self):
        self.task = ToySeqTask(
            ["aa", "bb", "cc"],
            cfg=self.cfg,
        )

    def setup_env_context(self):
        self.env = SeqBuildingEnv(None)
        self.ctx = AutoregressiveSeqBuildingContext(
            "abc",
            self.task.num_cond_dim,
        )

    def setup_algo(self):
        super().setup_algo()
        # If the algo implements it, avoid giving, ["A", "AB", "ABC", ...] as a sequence of inputs, and instead give
        # "ABC...Z" as a single input, but grab the logits at every timestep. Only works if using a transformer with
        # causal self-attention.
        self.algo.model_is_autoregressive = True


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.log_dir = "./logs/debug_run_toy_seq"
    config.device = "cuda"
    config.overwrite_existing_exp = True
    config.num_training_steps = 2_000
    config.checkpoint_every = 200
    config.num_workers = 4
    config.print_every = 1
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [2.0]
    config.cond.temperature.num_thermometer_dim = 1
    config.algo.train_random_action_prob = 0.05

    trial = ToySeqTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
