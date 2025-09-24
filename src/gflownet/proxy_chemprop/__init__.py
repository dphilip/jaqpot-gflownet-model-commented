"""
ChemProp proxy model integration for molecular property prediction.

This package provides integration with ChemProp (Chemical Property Prediction)
models for use as proxy functions in GFlowNet training. Proxy models enable
fast evaluation of molecular properties during training, avoiding expensive
quantum chemical calculations or experimental measurements.

Key components:
- mpnn_pipeline: Main interface for loading and using ChemProp models
- arg_parser: Command-line argument parsing for ChemProp model training
- run: Training scripts for ChemProp models on molecular datasets
- test_inference: Testing utilities for validating proxy model predictions

Features:
- Fast molecular property prediction using message-passing neural networks
- Integration with RDKit for molecular representation and featurization
- Support for various molecular descriptors and targets (LogP, QED, etc.)
- Batch prediction capabilities for efficient GFlowNet training
- Model checkpointing and state management

The proxy models are essential for scalable GFlowNet training on molecular
generation tasks, providing orders-of-magnitude speedup compared to
physics-based property calculations while maintaining reasonable accuracy
for guiding the generation process.

Usage patterns:
1. Train ChemProp models on molecular property datasets
2. Load trained models as proxy functions in GFlowNet tasks
3. Use for rapid property evaluation during trajectory generation
4. Validate proxy predictions against ground truth when available
"""
