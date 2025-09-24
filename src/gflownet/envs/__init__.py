"""
Environment implementations for different generation tasks.

This package contains environment classes that define the state spaces and
action spaces for different generative modeling tasks. Environments handle
the mechanics of object construction (molecules, sequences, etc.) and provide
the interface between algorithms and domain-specific representations.

Key environment types:
- GraphBuildingEnv: Base class for graph construction environments
- FragMolBuildingEnvContext: Fragment-based molecular generation
- MolBuildingEnvContext: Atom-by-atom molecular generation
- SeqBuildingEnv: Sequence generation for language/protein tasks

Each environment defines:
- Valid actions at each state (add atom, add bond, stop, etc.)
- State transitions when actions are taken
- Conversion between internal graph representation and domain objects
- Graph-to-data conversion for neural network processing

The environments follow a common interface that allows algorithms to work
across different domains without modification.
"""
