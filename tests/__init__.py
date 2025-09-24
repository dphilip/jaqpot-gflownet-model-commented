"""
Test suite for GFlowNet implementation.

This package contains unit tests and integration tests for the GFlowNet
framework, ensuring correctness and reliability of the implementation
across different components and use cases.

Test categories:
- test_envs: Environment functionality and state transitions
- test_graph_building_env: Graph construction and action validation
- test_subtb: Sub-trajectory balance algorithm correctness

Test coverage includes:
- Core algorithm implementations (TB, SubTB, etc.)
- Environment state transitions and action masking
- Model forward passes and gradient computation
- Data loading and batch construction
- Utility functions and helper methods
- Multi-objective optimization components

The tests use pytest framework and include:
- Unit tests for individual functions and classes
- Integration tests for complete training loops
- Property-based tests for invariant checking
- Performance regression tests
- Reproducibility tests with fixed random seeds

Running tests:
    pytest tests/                    # Run all tests
    pytest tests/test_envs.py       # Run specific test file
    tox run                         # Run tests in isolated environment

Tests are run automatically in CI/CD pipelines to ensure code quality
and prevent regressions when making changes to the codebase.
"""
