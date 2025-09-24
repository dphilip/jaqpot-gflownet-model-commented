"""
Transformation utilities for rewards and encoding in GFlowNet training.

This module provides functions for converting between different representations
of rewards and values used in GFlowNet training, including logarithmic transforms
and thermometer encodings for continuous values.
"""

# PyTorch imports for tensor operations
import torch
from torch import Tensor

# GFlowNet type imports
from gflownet import LogScalar


def to_logreward(reward: Tensor) -> LogScalar:
    """
    Convert linear reward values to logarithmic reward representation.
    
    This function transforms reward values from linear space to log space,
    which is the standard representation used in GFlowNet algorithms for
    numerical stability and mathematical convenience.
    
    The transformation includes:
    1. Squeeze to remove singleton dimensions
    2. Clamp to avoid log(0) which would give -inf
    3. Apply natural logarithm
    
    Parameters
    ----------
    reward : Tensor
        Linear reward values, must be positive
        Can have any shape but typically (batch_size,) or (batch_size, 1)
        
    Returns
    -------
    LogScalar
        Log-transformed reward values with same batch dimensions as input
        Values are clamped to avoid numerical issues with log(0)
    """
    # Remove singleton dimensions, clamp to avoid log(0), then take natural log
    return LogScalar(reward.squeeze().clamp(min=1e-30).log())


def thermometer(v: Tensor, n_bins: int = 50, vmin: float = 0, vmax: float = 1) -> Tensor:
    """
    Apply thermometer encoding to scalar values for neural network input.
    
    Thermometer encoding represents a continuous scalar value as a binary vector
    where elements are 1 up to a threshold determined by the input value, and 0 beyond.
    This encoding is useful for representing continuous values in a way that's
    easier for neural networks to process, especially for conditional generation.
    
    The encoding creates a "temperature gauge" effect:
    - Low values result in few 1s at the beginning
    - High values result in many 1s extending further
    - Values between bins get partial activations via linear interpolation
    
    Parameters
    ----------
    v : Tensor
        Value(s) to encode. Can be any shape - encoding is applied element-wise
        Values outside [vmin, vmax] are clipped to the range boundaries
    n_bins : int, default=50
        The number of dimensions in the thermometer encoding
        Higher values provide finer resolution but increase dimensionality
    vmin : float, default=0
        The minimum value in the encoding range
        Values below this map to all-zeros encoding
    vmax : float, default=1
        The maximum value in the encoding range  
        Values above this map to all-ones encoding
        
    Returns
    -------
    encoding : Tensor
        Thermometer-encoded values with shape: input.shape + (n_bins,)
        Each value becomes a vector of length n_bins with thermometer pattern
        
    Examples
    --------
    >>> v = torch.tensor([0.0, 0.5, 1.0])
    >>> encoded = thermometer(v, n_bins=4, vmin=0, vmax=1)
    >>> # encoded[0] ≈ [0, 0, 0, 0]  (minimum value)
    >>> # encoded[1] ≈ [1, 1, 0, 0]  (middle value) 
    >>> # encoded[2] ≈ [1, 1, 1, 1]  (maximum value)
    """
    # Create evenly spaced bin boundaries from vmin to vmax
    bins = torch.linspace(vmin, vmax, n_bins)
    # Calculate the gap between adjacent bins for normalization
    gap = bins[1] - bins[0]
    # Ensure vmin and vmax are different to avoid division by zero
    assert gap > 0, "vmin and vmax must be different"
    
    # Compute thermometer encoding:
    # 1. v[..., None] adds new dimension for broadcasting with bins
    # 2. bins.reshape broadcasts bins to match v's dimensions plus n_bins
    # 3. Subtract bins from values to get "distance above each bin threshold"
    # 4. Clamp to [0, gap] so values are either 0 (below threshold) or proportional
    # 5. Divide by gap to normalize to [0, 1] range
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap
