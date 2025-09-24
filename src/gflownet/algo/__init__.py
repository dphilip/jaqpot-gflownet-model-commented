"""
GFlowNet algorithm implementations.

This package contains implementations of different GFlowNet training algorithms
and sampling strategies. Each algorithm defines how to:
1. Sample trajectories from the current policy
2. Compute training losses from trajectory data
3. Handle different types of conditioning and objectives

Available algorithms:
- TrajectoryBalance: The main GFlowNet training algorithm
- SubTrajectoryBalance: A variant that uses sub-trajectory objectives  
- FlowMatching: An alternative training approach using flow matching
- GraphSampling: Utilities for sampling from graph building environments
- Various baseline methods for comparison

The algorithms are designed to work with different types of environments
(molecular, protein, sequence generation) through the common GraphBuildingEnv
interface.
"""
