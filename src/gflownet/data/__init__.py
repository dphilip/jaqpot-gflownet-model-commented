"""
Data handling and batch construction for GFlowNet training.

This package provides utilities for managing training data in GFlowNet experiments.
It handles both online data generation (sampling new trajectories from the current
policy) and offline data usage (replay buffers, fixed datasets).

Key components:
- DataSource: Main interface for providing training batches to algorithms
- ReplayBuffer: Storage and sampling of previously generated trajectories
- QM9Dataset: Specialized dataset handling for QM9 molecular data
- Batch construction utilities for converting trajectories to PyTorch Geometric batches

The data system supports:
- Mixed online/offline training strategies
- Efficient replay buffer management with priority sampling
- Multi-objective data handling with conditioning information
- Validation set management for model evaluation
- Parallel data generation with multiple worker processes

Data sources provide the flexibility to combine different data generation
strategies within a single training run, enabling sophisticated training
curricula and data augmentation techniques.
"""
