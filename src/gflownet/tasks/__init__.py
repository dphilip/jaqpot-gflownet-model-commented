"""
Task implementations for specific application domains.

This package contains concrete implementations of GFlowNet tasks for different
application areas. Each task defines the reward function, conditioning setup,
and training loop for a specific generative modeling problem.

Available tasks:
- seh_frag: Fragment-based drug discovery targeting sEH protein
- seh_frag_moo: Multi-objective version with QED, SA, and other properties
- qm9: Molecule generation conditioned on HOMO-LUMO gap from QM9 dataset
- logp_frag: LogP-targeted molecular generation using proxy models
- toy_seq: Simple sequence generation task for testing and development
- make_rings: Ring-finding task for testing graph algorithms

Each task inherits from the GFNTask base class and implements:
- compute_obj_properties(): Evaluate generated objects using proxy models
- cond_info_to_logreward(): Convert conditioning info and properties to rewards
- sample_conditional_information(): Sample conditioning for training batches

Tasks also include their own trainer classes that set up the complete training
loop with appropriate data sources, validation, and logging.
"""
