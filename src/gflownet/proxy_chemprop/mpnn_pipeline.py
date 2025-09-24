"""
ChemProp-based MPNN training pipeline for molecular property prediction.

This module implements a complete training pipeline for Message Passing Neural Networks (MPNNs)
using the ChemProp library. It provides functionality for training molecular property predictors
that can serve as proxy models in GFlowNet training, enabling efficient reward computation
without expensive quantum chemical calculations.

The pipeline includes:
- Data loading and preprocessing for molecular datasets
- MPNN model configuration and training
- Weights & Biases integration for experiment tracking
- Model checkpointing and evaluation
- Integration with GFlowNet reward computation

This is particularly useful for drug discovery applications where fast property
prediction is needed for molecular optimization tasks.

TODO:
- Integrate with GFlowNet MPNN inference pipeline  
- Add support for multi-GPU training configurations
- Create submodule integration for forked repositories
"""

# Standard library imports for randomization and data handling
import random
import numpy as np

# PyTorch imports for neural network training
import torch
import wandb
from lightning import pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd

# ChemProp imports for molecular property prediction
from chemprop import data, featurizers, models, nn


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments across all libraries.
    
    This function ensures reproducible results by setting seeds for all major
    random number generators used in the training pipeline. This is crucial
    for scientific reproducibility and debugging.
    
    Parameters
    ----------
    seed : int, default=42
        Random seed value to use across all libraries
        Using the same seed should produce identical results
        
    Notes
    -----
    Setting torch.backends.cudnn.deterministic=True may reduce performance
    but ensures reproducibility when using CUDA operations.
    """
    # Set Python's built-in random seed
    random.seed(seed)
    # Set NumPy random seed for array operations
    np.random.seed(seed)
    # Set PyTorch CPU random seed
    torch.manual_seed(seed)
    # Set PyTorch GPU random seed for current device
    torch.cuda.manual_seed(seed)
    # Set PyTorch GPU random seed for all devices
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic CUDA operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    # Disable CUDA optimization heuristics for reproducibility
    torch.backends.cudnn.benchmark = False


def setup_run(args):
    """
    Initialize Weights & Biases experiment tracking with hyperparameters.
    
    This function sets up comprehensive experiment logging by connecting to
    Weights & Biases and initializing a new run with all relevant hyperparameters.
    This enables tracking of training progress, hyperparameter comparisons,
    and result visualization.
    
    Parameters
    ----------
    args : argparse.Namespace or object
        Command line arguments or configuration object containing:
        - project_name: W&B project name for organizing experiments
        - MPNN hyperparameters: architecture and training parameters
        - Optimization parameters: learning rates and schedules
        
    Notes
    -----
    Requires W&B authentication to be set up prior to calling this function.
    The logged hyperparameters should match the actual training configuration.
    """
    # Authenticate with Weights & Biases
    wandb.login()
    
    # Initialize new experiment run with comprehensive hyperparameter logging
    wandb.init(
        project=args.project_name,  # Project name for organizing related experiments
        config={
            # Data preprocessing parameters
            "scaling": args.scaling,                        # Feature scaling method
            "batch_size": args.batch_size,                  # Training batch size
            
            # MPNN architecture parameters
            "message_hidden_dim": args.message_hidden_dim,  # Hidden dimension for message passing
            "depth": args.depth,                            # Number of message passing layers
            "dropout": args.dropout,                        # Dropout rate in MPNN layers
            "activation_mpnn": args.activation_mpnn,        # Activation function for MPNN
            "aggregation": args.aggregation,                # Message aggregation method
            
            # Readout network parameters  
            "hidden_dim_readout": args.hidden_dim_readout,  # Hidden dimension for readout MLP
            "hidden_layers_readout": args.hidden_layers_readout,  # Number of readout layers
            "dropout_readout": args.dropout_readout,        # Dropout rate in readout layers
            "batch_norm": args.batch_norm,                  # Whether to use batch normalization
            
            # Optimization parameters
            "init_lr": args.init_lr,                        # Initial learning rate
            "max_lr": args.max_lr,                          # Maximum learning rate (for schedules)
            "final_lr": args.final_lr,                      # Final learning rate (for schedules)
        },
    )


class WandbLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    # Only log the losses. Nothing else
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = {
            "train/loss": trainer.callback_metrics.get("train_loss"),
            "val/loss": trainer.callback_metrics.get("val_loss"),
        }
        wandb.log(metrics)


def load_data(input_path, smiles_col, target_col):
    df = pd.read_csv(input_path)
    smiles = df.loc[:, smiles_col].values
    target = df.loc[:, target_col].values

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, target)]

    return all_data


def get_activation(fn: str):
    if fn == "relu":
        return nn.Activation.RELU
    elif fn == "leaky_relu":
        return nn.Activation.LEAKYRELU
    elif fn == "prelu":
        return nn.Activation.PRELU
    elif fn == "selu":
        return nn.Activation.SELU
    elif fn == "elu":
        return nn.Activation.ELU
    else:
        raise ValueError(f"Activation function {fn} not supported")


def get_split(all_data: list, split_size: tuple[float, float, float]):
    mols = [d.mol for d in all_data]
    train_indices, val_indices, test_indices = data.make_split_indices(
        mols, "SCAFFOLD_BALANCED", split_size
    )  # unpack the tuple into three separate lists
    train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

    return train_data, val_data, test_data


def create_datasets(train_data: list, val_data: list, test_data: list, scaling: bool = False):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dataset = data.MoleculeDataset(train_data[0], featurizer)
    val_dataset = data.MoleculeDataset(val_data[0], featurizer)
    if scaling:
        scaler = train_dataset.normalize_targets()
        val_dataset.normalize_targets(scaler)
    else:
        scaler = None
    test_dataset = data.MoleculeDataset(test_data[0], featurizer)

    return train_dataset, val_dataset, test_dataset, scaler


def seed_worker(worker_id):
    # worker_id is automatically provided by PyTorch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = data.build_dataloader(
        train_dataset, batch_size, num_workers, seed=42, worker_init_fn=seed_worker, generator=g
    )

    val_loader = data.build_dataloader(
        val_dataset, batch_size, num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g
    )

    test_loader = data.build_dataloader(
        test_dataset, batch_size, num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g
    )

    return train_loader, val_loader, test_loader


def recreate_train_loader(train_dataset, batch_size, num_workers):
    g = torch.Generator()
    g.manual_seed(42)
    return data.build_dataloader(
        train_dataset, batch_size, num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g
    )


def message_passing(args):
    activation = get_activation(args.activation_mpnn)
    return nn.AtomMessagePassing(
        d_h=args.message_hidden_dim,
        depth=args.depth,
        undirected=args.undirected,
        dropout=args.dropout,
        activation=activation,
    )


def get_aggregation(aggr: str):
    if aggr == "mean":
        return nn.MeanAggregation()
    elif aggr == "sum":
        return nn.SumAggregation()
    else:
        raise ValueError(f"Aggregation function {aggr} not supported")


def readout_mlp(args, scaler):
    if args.scaling:
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    else:
        output_transform = scaler
    return nn.RegressionFFN(
        input_dim=args.message_hidden_dim,
        hidden_dim=args.hidden_dim_readout,
        n_layers=args.hidden_layers_readout,
        dropout=args.dropout_readout,
        output_transform=output_transform,
    )


def build_mpnn(mp, agg, ffn, args):
    return models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=args.batch_norm,
        metrics=[
            nn.metrics.RMSE(),
            nn.metrics.MAE(),
            nn.metrics.MSE(),
            nn.metrics.R2Score(),
        ],
        init_lr=args.init_lr,
        max_lr=args.max_lr,
        final_lr=args.final_lr,
    )


def ready_trainer(max_epochs: int = 100):
    checkpointing = ModelCheckpoint(
        dirpath="checkpoints",  # Directory where model checkpoints will be saved
        filename="best-{epoch}-{val_loss:.2f}",  # Filename format for checkpoints, including epoch and validation loss
        monitor="val_loss",  # Metric used to select the best checkpoint (based on validation loss)
        mode="min",  # Save the checkpoint with the lowest validation loss (minimization objective)
        save_last=False,  # Always save the most recent checkpoint, even if it's not the best
    )
    wandb_logger = WandbLoggingCallback()
    # TODO: Check optimizer, learning rate, weight decay, etc...
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=True,  # Use `True` if you want to save model checkpoints. The checkpoints will be saved in the `checkpoints` folder.
        enable_progress_bar=True,
        accelerator="gpu",
        devices=-1,
        deterministic=True,
        max_epochs=max_epochs,  # number of epochs to train for
        callbacks=[
            checkpointing,
            wandb_logger,
        ],  # Use the configured checkpoint callback
    )
    return trainer


def get_best_checkpoint(trainer: pl.Trainer):
    return trainer.checkpoint_callback.best_model_path


def load_model(checkpoint: str):
    # Needs change for gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    hp = checkpoint_data["hyper_parameters"]
    print(checkpoint_data["state_dict"].keys())
    # Message passing configs
    mp_hp = hp["message_passing"]
    del mp_hp["cls"]
    mp = nn.AtomMessagePassing(**mp_hp)
    # Aggregation configs
    agg = nn.SumAggregation()
    # Readout MLP configs
    ffn_hp = hp["predictor"]
    del ffn_hp["cls"]
    predictor = nn.RegressionFFN(**ffn_hp)
    mpnn = models.MPNN(
        mp,
        agg,
        predictor,
        hp["batch_norm"],
        metrics=[
            nn.metrics.RMSE(),
            nn.metrics.MAE(),
            nn.metrics.MSE(),
            nn.metrics.R2Score(),
        ],
        init_lr=5e-4,
        max_lr=5e-3,
        final_lr=5e-5,
    )
    mpnn.load_state_dict(checkpoint_data["state_dict"])
    return mpnn
    # return models.MPNN.load_from_checkpoint(checkpoint, map_location=torch.device("cpu"))


def plot(real, preds):
    plt.figure(figsize=(12, 8))
    plt.scatter(
        real["train"],
        preds["train"],
        color="blue",
        alpha=0.7,
        label="Train",
        marker="o",
    )
    plt.scatter(
        real["val"],
        preds["val"],
        color="green",
        alpha=0.7,
        label="Validation",
        marker="^",
    )
    plt.scatter(real["test"], preds["test"], color="red", alpha=0.7, label="Test", marker="s")

    all_values = np.concatenate(
        [
            real["train"],
            preds["train"],
            real["val"],
            preds["val"],
            real["test"],
            preds["test"],
        ]
    )
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        linestyle="--",
        label="Perfect Prediction",
    )

    # Labeling
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")
    plt.xlim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
    plt.ylim(min_val - 0.1 * (max_val - min_val), max_val + 0.1 * (max_val - min_val))
    plt.tight_layout()
    wandb.log({"plot": plt})


def calc_metrics(predictions, real, split: str):
    if len(predictions) != len(real):
        raise ValueError("Predictions and real values have different lengths")

    y_pred = np.array(predictions)
    y_true = np.array(real)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    metrics = {
        split + "/mae": mae,
        split + "/rmse": rmse,
        split + "/mse": mse,
        split + "/r2": r2,
    }
    wandb.log(metrics)


def predict(train_loader, val_loader, test_loader, model: models.MPNN):
    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices="auto",  # 1 for cpu , 'auto' for everything else
        )
        train_preds = [el[0] for batch in trainer.predict(model, train_loader) for el in batch]
        val_preds = [el[0] for batch in trainer.predict(model, val_loader) for el in batch]
        test_preds = [el[0] for batch in trainer.predict(model, test_loader) for el in batch]

    return train_preds, val_preds, test_preds
