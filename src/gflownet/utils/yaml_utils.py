"""
YAML configuration utilities for GFlowNet dataclass initialization.

This module provides utilities for loading YAML configuration files and converting
them into properly typed dataclass instances. It handles nested dataclass conversion
and ensures type safety when loading configuration from external files.

The main use case is loading experiment configurations from YAML files into the
structured Config dataclass hierarchy used throughout the GFlowNet codebase.
"""

# Standard library imports for YAML parsing and dataclass introspection
import yaml
from dataclasses import is_dataclass, fields
from typing import Type, Any

# GFlowNet imports for configuration types
from gflownet.config import Config


def dict2cls(data_class: Type, data: dict) -> Any:
    """
    Recursively initialize a dataclass instance from a nested dictionary.
    
    This function converts a nested dictionary (typically loaded from YAML/JSON)
    into a properly typed dataclass instance. It handles nested dataclasses by
    recursively calling itself, ensuring the entire configuration hierarchy is
    properly constructed with correct types.
    
    The function only populates fields that exist in both the dictionary and
    the dataclass definition, ignoring extra keys in the dictionary and leaving
    missing fields to use their default values.
    
    Parameters
    ----------
    data_class : Type
        The dataclass type to instantiate
        Must be a dataclass type, not an instance
    data : dict
        Dictionary containing the field values to populate
        Can contain nested dictionaries for nested dataclass fields
        
    Returns
    -------
    Any
        An instance of data_class with fields populated from the dictionary
        Nested dataclass fields are also properly instantiated
        
    Raises
    ------
    TypeError
        If data_class is not a dataclass type
        
    Examples
    --------
    >>> @dataclass
    ... class NestedConfig:
    ...     value: int = 10
    >>> @dataclass  
    ... class MainConfig:
    ...     name: str = "default"
    ...     nested: NestedConfig = NestedConfig()
    >>> data = {"name": "test", "nested": {"value": 42}}
    >>> config = dict2cls(MainConfig, data)
    >>> config.name  # "test"
    >>> config.nested.value  # 42
    """
    # Verify that the provided type is actually a dataclass
    if not is_dataclass(data_class):
        raise TypeError(f"{data_class} is not a dataclass")

    # Initialize dictionary to hold field values for dataclass construction
    field_values = {}
    
    # Iterate through all fields defined in the dataclass
    for field in fields(data_class):
        field_name = field.name  # Name of the field in the dataclass
        field_type = field.type  # Type annotation of the field
        
        # Check if this field has a corresponding value in the input dictionary
        if field_name in data:
            # Extract the value from the dictionary
            value = data[field_name]
            
            # Check if this field is itself a dataclass that needs recursive conversion
            if is_dataclass(field_type):
                # Recursively convert nested dictionary to nested dataclass
                field_values[field_name] = dict2cls(field_type, value)
            else:
                # Use the value directly for primitive types
                field_values[field_name] = value

    # Instantiate the dataclass with the populated field values
    return data_class(**field_values)


def yml2cfg(file_path: str) -> Config:
    """
    Load a YAML configuration file and convert it to a Config dataclass instance.
    
    This function provides a convenient interface for loading GFlowNet configurations
    from YAML files. It combines YAML parsing with dataclass conversion to provide
    a type-safe way to load experimental configurations.
    
    The YAML file should have a structure that matches the Config dataclass hierarchy,
    with nested sections for different configuration components (algo, model, task, etc.).
    
    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file to load
        File must exist and contain valid YAML syntax
        
    Returns
    -------
    Config
        Fully instantiated Config dataclass with nested configuration objects
        All fields are properly typed according to the dataclass definitions
        
    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist
    yaml.YAMLError
        If the file contains invalid YAML syntax
    TypeError
        If the YAML structure doesn't match the expected dataclass hierarchy
        
    Examples
    --------
    >>> config = yml2cfg("experiments/my_experiment.yaml")
    >>> config.num_training_steps  # Access top-level config
    >>> config.algo.method        # Access nested config
    >>> config.model.num_layers   # Access deeply nested config
    """
    # Open and parse the YAML file using safe loading to prevent code execution
    with open(file_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Convert the loaded dictionary to a properly typed Config dataclass instance
    return dict2cls(Config, config_dict)
