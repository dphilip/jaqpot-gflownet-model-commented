"""
Configuration classes for GFlowNet neural network models.

This module defines the configuration dataclasses used to specify the architecture
and hyperparameters of various neural network models in the GFlowNet framework.
It provides structured, type-safe configuration for different model types including
graph transformers and sequence transformers.

The configuration system uses dataclasses with default values and factory methods
to ensure consistent initialization and easy customization of model architectures.
"""

# Standard library imports for dataclass definition and enums
from dataclasses import dataclass, field
from enum import Enum

# GFlowNet imports for strict dataclass functionality
from gflownet.utils.misc import StrictDataClass


@dataclass
class GraphTransformerConfig(StrictDataClass):
    """
    Configuration for Graph Transformer model architecture.
    
    This class defines the hyperparameters specific to graph transformer models
    used for processing graph-structured data in GFlowNet. Graph transformers
    apply attention mechanisms to nodes and edges in graphs, enabling the model
    to capture complex structural relationships.
    
    Attributes
    ----------
    num_heads : int, default=2
        Number of attention heads in the multi-head attention mechanism
        More heads allow the model to attend to different types of relationships
        simultaneously, but increase computational cost
    ln_type : str, default="pre"
        Type of layer normalization to apply ("pre" or "post")
        - "pre": Layer norm applied before attention (Pre-LN transformer)  
        - "post": Layer norm applied after attention (Post-LN transformer)
        Pre-LN typically provides more stable training
    num_mlp_layers : int, default=0
        Number of additional MLP layers in the feed-forward network
        0 means only the standard transformer feed-forward layer
        Additional layers can increase model expressiveness
    concat_heads : bool, default=True
        Whether to concatenate attention heads or average them
        - True: Concatenate outputs from different heads (standard approach)
        - False: Average outputs from different heads (reduces dimensionality)
    """
    num_heads: int = 2                    # Multi-head attention heads count
    ln_type: str = "pre"                  # Layer normalization placement  
    num_mlp_layers: int = 0               # Extra MLP layers in feed-forward
    concat_heads: bool = True             # Head combination strategy


class SeqPosEnc(int, Enum):
    """
    Enumeration of positional encoding types for sequence transformers.
    
    Positional encodings are crucial for transformer models to understand the
    sequential nature of input data, since attention mechanisms are inherently
    permutation-invariant. Different encoding schemes have different properties:
    
    - Pos (0): Absolute positional encoding using sine/cosine functions
    - Rotary (1): Rotary Position Embedding (RoPE) which encodes relative positions
    """
    Pos = 0      # Absolute sinusoidal positional encoding
    Rotary = 1   # Rotary positional embedding (relative positions)


@dataclass
class SeqTransformerConfig(StrictDataClass):
    """
    Configuration for Sequence Transformer model architecture.
    
    This class defines the hyperparameters specific to sequence transformer models
    used for processing sequential data (e.g., SMILES strings, amino acid sequences).
    Sequence transformers are adapted for variable-length sequences with appropriate
    positional encodings.
    
    Attributes
    ----------
    num_heads : int, default=2
        Number of attention heads in the multi-head attention mechanism
        Similar to graph transformers, more heads capture different sequence patterns
    posenc : SeqPosEnc, default=SeqPosEnc.Rotary
        Type of positional encoding to use for sequences
        Rotary encoding often performs better for variable-length sequences
    """
    num_heads: int = 2                    # Multi-head attention heads count
    posenc: SeqPosEnc = SeqPosEnc.Rotary  # Positional encoding strategy


@dataclass
class ModelConfig(StrictDataClass):
    """
    Generic configuration class for all GFlowNet neural network models.
    
    This is the main configuration class that encompasses common hyperparameters
    shared across different model architectures, as well as specific configurations
    for specialized model types. It serves as the central configuration hub for
    model architecture specification.
    
    The class follows a hierarchical structure where general parameters are defined
    at the top level, while architecture-specific parameters are nested in
    specialized configuration objects.
    
    Attributes
    ----------
    num_layers : int, default=3
        The number of transformer/network layers in the model
        More layers increase model capacity but also computational cost
        Typical values range from 2-12 depending on problem complexity
    num_emb : int, default=128
        The number of dimensions in the embedding space
        This determines the hidden dimension size throughout the model
        Common values: 64, 128, 256, 512 (powers of 2 for efficiency)
    dropout : float, default=0
        Dropout probability for regularization during training
        Range: [0.0, 1.0] where 0 means no dropout
        Typical values: 0.1-0.3 for regularization without over-suppression
    graph_transformer : GraphTransformerConfig
        Nested configuration for graph transformer specific parameters
        Factory method ensures each instance gets its own config object
    seq_transformer : SeqTransformerConfig  
        Nested configuration for sequence transformer specific parameters
        Factory method ensures each instance gets its own config object
    """

    num_layers: int = 3                   # Number of network layers
    num_emb: int = 128                    # Embedding dimension size
    dropout: float = 0                    # Dropout rate for regularization
    
    # Nested configurations for specific model types (use factories for independence)
    graph_transformer: GraphTransformerConfig = field(default_factory=GraphTransformerConfig)
    seq_transformer: SeqTransformerConfig = field(default_factory=SeqTransformerConfig)
