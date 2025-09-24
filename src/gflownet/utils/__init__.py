"""
Utility functions and helper classes for GFlowNet implementation.

This package contains various utility modules that support the core GFlowNet
functionality across different components:

Core utilities:
- conditioning: Temperature, preferences, and multi-objective conditioning
- metrics: Evaluation metrics including Pareto frontiers and hypervolume
- transforms: Data transformations like thermometer encoding and log-reward conversion
- misc: Miscellaneous helpers for device management, worker processes, etc.

Specialized utilities:
- graphs: Graph manipulation and analysis functions
- focus_model: Models for focusing generation on specific regions
- multiprocessing_proxy: Safe multiprocessing with complex objects
- multiobjective_hooks: Hooks for multi-objective optimization evaluation
- sqlite_log: Database logging for experiment results
- sascore: Synthetic accessibility scoring for molecules
- yaml_utils: YAML configuration file handling

These utilities are designed to be modular and reusable across different
GFlowNet tasks and environments, providing common functionality that would
otherwise need to be reimplemented in each specific application.
"""
