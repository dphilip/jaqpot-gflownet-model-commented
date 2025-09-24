"""
LogP-based molecular generation task using fragment assembly and MPNN proxy models.

This module implements a GFlowNet task for generating molecules optimized for logP
(partition coefficient) values. It uses a pre-trained Message Passing Neural Network (MPNN)
as a proxy model to predict logP values for generated molecules, enabling efficient
optimization without expensive quantum chemical calculations.

The task combines:
- Fragment-based molecular construction using predefined building blocks
- MPNN-based property prediction for fast evaluation
- Temperature-based conditional generation for exploration control
- Reward transformation to map logP values to appropriate ranges

This setup is particularly useful for drug discovery applications where optimizing
logP is important for ADMET properties (Absorption, Distribution, Metabolism,
Excretion, Toxicity).
"""

# Standard library imports
import os
import yaml
import socket
from typing import Callable, Dict, List, Optional, Tuple, Union

# PyTorch and scientific computing imports
import torch
import torch.nn as nn
import numpy as np
from lightning import pytorch as pl

# Chemistry and molecular handling imports
from rdkit.Chem.rdchem import Mol as RDMol
from rdkit import Chem
from torch import Tensor

# GFlowNet framework imports
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.utils.yaml_utils import yml2cfg
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward

# Proxy model imports for logP prediction
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop import data

# Default path to the pre-trained MPNN checkpoint for logP prediction
CKP_PATH = "../proxy_chemprop/checkpoints/best-epoch=84-val_loss=0.06.ckpt"


class LogPTask(GFNTask):
    """
    GFlowNet task for logP-optimized molecular generation using fragment assembly.
    
    This class implements a complete task setup for generating molecules with optimized
    logP values. It uses a fragment-based approach where molecules are built by
    assembling predefined molecular fragments, with rewards computed using a fast
    MPNN proxy model instead of expensive DFT calculations.
    
    The task handles:
    - Loading and managing the MPNN proxy model for logP prediction
    - Converting molecular graphs to appropriate input formats
    - Computing rewards with proper transformation and scaling
    - Conditional generation with temperature-based exploration
    - Integration with the GFlowNet training framework
    """

    def __init__(
        self,
        cfg: Config,
        rew_tran: str = "0-10",
        y_min: float = -5.08,
        y_max: float = 11.29,
        wrap_model: Optional[Callable[[nn.Module], nn.Module]] = None,
        mpnn_ckp_path: str = CKP_PATH,
    ) -> None:
        """
        Initialize the LogP optimization task with proxy model and parameters.
        
        Parameters
        ----------
        cfg : Config
            Global configuration object containing all hyperparameters and settings
            Includes model architecture, training parameters, and task-specific options
        rew_tran : str, default="0-10"
            Reward transformation specification for mapping logP values to rewards
            Format: "{min}-{max}" defining the target reward range
        y_min : float, default=-5.08
            Minimum expected logP value in the dataset used for normalization
            This value is used to scale predictions to the reward range
        y_max : float, default=11.29
            Maximum expected logP value in the dataset used for normalization
            This value is used to scale predictions to the reward range
        wrap_model : Optional[Callable], default=None
            Optional model wrapper function for adding layers or transformations
            If None, no additional wrapping is applied to the proxy model
        mpnn_ckp_path : str, default=CKP_PATH
            Path to the pre-trained MPNN checkpoint file for logP prediction
            Should point to a valid ChemProp model checkpoint
        """
        # Store configuration and task parameters
        self.cfg = cfg                          # Global configuration object
        self.rew_tran = rew_tran               # Reward transformation specification
        self.y_min = y_min                     # Minimum logP value for normalization
        self.y_max = y_max                     # Maximum logP value for normalization
        self.width = y_max - y_min             # Range width for scaling calculations
        
        # Set up model wrapping function (identity if none provided)
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        
        # Store path to MPNN checkpoint
        self.mpnn_ckp_path = mpnn_ckp_path
        
        # Load the pre-trained models for property prediction
        self.models = self._load_task_models()
        
        # Initialize temperature-based conditional generation
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        
        # Initialize molecular featurizer for converting molecules to model inputs
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
        if self.rew_tran == "exp":
            flat_r = np.exp(-(y - self.y_min) / self.width)
        elif self.rew_tran == "unit":
            flat_r = (y - self.y_min) / self.width
        elif self.rew_tran == "0-10":
            flat_r = ((y - self.y_min) / self.width) * 10
        return flat_r

    def inv_transform(self, rp: Tensor) -> float:
        if self.rew_tran == "exp":
            inv_r = -np.log(rp) * self.width + self.y_min
        elif self.rew_tran == "unit":
            inv_r = (1 - rp) * self.width + self.y_min
        elif self.rew_tran == "0-10":
            inv_r = (rp / 10) * self.width + self.y_min
        return inv_r

    def _load_task_models(self):
        # Load MPNN Model from Chemprop with best ckp
        model = load_model(self.mpnn_ckp_path)  # .to(get_worker_device())
        model = self._wrap_model(model)
        return {"logp": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, loader) -> Tensor:
        trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
        preds = trainer.predict(self.models["logp"], loader)[0].reshape((-1)).data.cpu()
        preds[preds.isnan()] = 0
        preds = (
            self.reward_transform(preds)
            .clip(1e-4, 20)
            .reshape(
                -1,
            )
        )
        return preds

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        ####
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles]
        test_dset = data.MoleculeDataset(test_data, featurizer=self.featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=self.cfg.num_workers, pin_memory=True)
        ###
        is_valid = torch.tensor([i is not None for i in test_dset]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid
        preds = self.compute_reward_from_graph(test_loader).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


class LogPTrainer(StandardOnlineTrainer):
    task: LogPTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()

    def setup_task(self):
        self.task = LogPTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()


def main():
    """Example of how this model can be run."""
    import wandb

    # Need to init and GFNTrainer will automatically log to wandb
    wandb.login()
    wandb.init(project="gflow_test")
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logp.yaml")
    config = yml2cfg(file_path)
    trial = LogPTrainer(config)
    trial.run()
    # Read SQL
    # from gflownet.utils.sqlite_log import read_all_results

    # results = read_all_results(config.log_dir + "/valid")
    wandb.finish()
    # Need to decide what to log in wandb
    # 1) online_loss which is the same as tb_loss
    # 2) sampled_reward_avg
    # 3) smiles and reward on every validation. Eventually we want to see the distribution over time


if __name__ == "__main__":
    main()
