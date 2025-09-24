"""
Inference script for GFlowNet models.

This script demonstrates how to load a trained GFlowNet model and use it to generate
molecular graphs. It performs the following steps:

1. Load configuration from YAML file
2. Set up the molecular building environment with fragment context
3. Load pre-trained GFlowNet model weights
4. Generate sample molecules using the trained model
5. Evaluate the generated molecules using a proxy ChemProp model
6. Visualize one of the generated molecules

The script is designed to work with fragment-based molecular generation models
trained using the GFlowNet framework. It demonstrates the complete inference
pipeline from model loading to molecular visualization.

Usage:
    python inference.py

Requirements:
    - Trained GFlowNet model checkpoint
    - Configuration YAML file
    - Proxy model for molecular property evaluation
    - RDKit for molecular manipulation and visualization
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from gflownet.proxy_chemprop.mpnn_pipeline import load_model
from omegaconf import OmegaConf
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.utils.conditioning import TemperatureConditional
from pathlib import Path
from lightning import pytorch as pl
from gflownet.models import bengio2021flow
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop import data

# Define paths for model artifacts and configuration
current_dir = Path(".").resolve()
run_name = "run"  # Directory name containing the trained model
proxy_ckp = "best-epoch=84-val_loss=0.06.ckpt"  # Proxy model checkpoint filename

# Construct paths to required files
yaml_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / run_name / "config.yaml"
model_dir = current_dir / "src" / "gflownet" / "tasks" / "logs" / run_name / "model_state.pt"  
proxy_dir = current_dir / "src" / "gflownet" / "proxy_chemprop" / "checkpoints" / proxy_ckp

# Load configuration from YAML file
cfg = OmegaConf.load(yaml_dir)
# Initialize the graph building environment and temperature conditioning
env = GraphBuildingEnv()  # Core environment for graph construction
temp_cond = TemperatureConditional(cfg)  # Temperature-based conditioning
num_cond_dim = temp_cond.encoding_size()  # Get dimensionality of conditioning

# Set up the fragment-based molecular building context
ctx = FragMolBuildingEnvContext(
    max_frags=cfg.algo.max_nodes,  # Maximum number of fragments allowed
    num_cond_dim=num_cond_dim,     # Conditioning dimension size
    fragments=bengio2021flow.FRAGMENTS,  # Available molecular fragments
)
# Initialize the GFlowNet model architecture
model = GraphTransformerGFN(
    env_ctx=ctx,  # Environment context for molecular building
    cfg=cfg,      # Configuration parameters
    num_graph_out=cfg.algo.tb.do_predict_n + 1,  # Number of graph-level outputs
    do_bck=cfg.algo.tb.do_parameterize_p_b,      # Whether to parameterize backward policy
)

# Load pre-trained model weights from checkpoint
model.load_state_dict((torch.load(model_dir)["sampling_model_state_dict"][0]))
model.eval()  # Set model to evaluation mode
# Initialize the trajectory balance algorithm for sampling
algo = TrajectoryBalance(env, ctx, cfg)

# Set random seeds for reproducible results
np.random.seed(42)
torch.manual_seed(42)

# Sample conditioning information (temperature settings)
cond_info = temp_cond.sample(10)["encoding"]

# Generate molecular samples using the trained model
samples = algo.create_training_data_from_own_samples(model=model, n=10, cond_info=cond_info)

# Extract trajectories and convert to RDKit molecules
trajectories = [sample["traj"] for sample in samples]
# valid = [sample["is_valid"] for sample in samples]  # Validity flags (commented out)
rdkit_mols = [ctx.graph_to_obj(traj[-1][0]) for traj in trajectories]
# Evaluate generated molecules using proxy ChemProp model
# TODO: Automate this process for better integration

# Load the pre-trained proxy model for property prediction
model = load_model(proxy_dir)
featurizer = SimpleMoleculeMolGraphFeaturizer()  # Molecular graph featurizer

# Convert molecules to SMILES strings for processing
smiles = [Chem.MolToSmiles(mol) for mol in rdkit_mols]

# Prepare data for prediction
test_data = [data.MoleculeDatapoint.from_smi(smile) for smile in smiles]
test_dset = data.MoleculeDataset(test_data, featurizer=featurizer)
test_loader = data.build_dataloader(test_dset, shuffle=False)

# Run inference using PyTorch Lightning trainer
trainer = pl.Trainer(logger=None, enable_progress_bar=True, accelerator="cpu", devices=1)
preds = trainer.predict(model, test_loader)[0]
print(preds)  # Print predicted properties

# Generate and save visualization of one molecule
img = Draw.MolToImageFile(rdkit_mols[1], "test.png")

# Alternative grid visualization (commented out for single molecule display)
# img = Draw.MolsToGridImage(
#     [Chem.MolFromSmiles(mol) for mol in results["smiles"]],
#     molsPerRow=5,  # Number of molecules per row
#     legends=[rew for rew in results["r"]],  # Property values as legends
#     subImgSie=(200, 200),  # Size of each molecule image
# )
