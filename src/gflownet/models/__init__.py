"""
Neural network model architectures for GFlowNets.

This package contains implementations of different neural network architectures
used as function approximators in GFlowNet algorithms. The models learn to
predict action probabilities and state values from graph representations.

Key model types:
- GraphTransformerGFN: Graph transformer architecture with attention mechanisms
- Bengio2021Flow: The original GFlowNet architecture from Bengio et al. 2021
- SeqTransformerGFN: Transformer architecture for sequence generation tasks
- MXMNet: Message-passing networks for molecular property prediction

All models implement a common interface that outputs:
- Per-node action logits (for node-level actions like adding atoms/bonds)
- Per-graph action logits (for graph-level actions like termination)
- Optional backward policy predictions for trajectory balance training

The models use PyTorch Geometric for efficient graph neural network operations
and support various types of conditioning information (temperature, preferences).
"""
