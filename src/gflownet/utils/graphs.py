"""
Graph utility functions for structural analysis and feature computation.

This module provides utility functions for analyzing graph structures and computing
graph-theoretic features. The functions are designed to work with PyTorch Geometric
Data objects and support batched operations for efficient processing.

The main focus is on random walk analysis, which is useful for:
- Capturing local graph structure around nodes  
- Computing node similarities and embeddings
- Analyzing graph connectivity patterns
- Feature engineering for graph neural networks
"""

# PyTorch imports for tensor operations
import torch
# PyTorch Geometric imports for graph data structures
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
# Scatter operations for efficient graph computations
from torch_scatter import scatter_add


def random_walk_probs(g: Data, k: int, skip_odd=False):
    """
    Compute random walk return probabilities for each node in a graph.
    
    This function calculates the probability of returning to each starting node
    after exactly k steps of random walk. Random walk probabilities capture
    important structural properties of graphs and can be used as node features
    for downstream tasks.
    
    The random walk transition matrix P is defined as P = D^(-1) * A, where:
    - D is the degree matrix (diagonal matrix of node degrees)
    - A is the adjacency matrix
    - P[i,j] represents the probability of transitioning from node i to node j
    
    The function computes the diagonal elements of P^k for k=1,2,...,k, which
    represent the probability of returning to the starting node after exactly
    k steps.
    
    Parameters
    ----------
    g : Data
        PyTorch Geometric Data object representing the graph
        Must contain edge_index and num_nodes attributes
        Can be a single graph or a batched graph
    k : int
        Maximum number of random walk steps to compute
        Higher values capture longer-range structural information
        Computational complexity grows linearly with k
    skip_odd : bool, default=False
        Whether to skip odd-length walks and only compute even-length walks
        When True, computes P^2, P^4, P^6, ... instead of P^1, P^2, P^3, ...
        Useful for bipartite graphs where odd walks don't return to start
        
    Returns
    -------
    torch.Tensor
        Tensor of shape (num_nodes, k) containing return probabilities
        Element [i, j] is the probability of returning to node i after j+1 steps
        (or 2*(j+1) steps if skip_odd=True)
        
    Examples
    --------
    >>> # Simple triangle graph: 0-1-2-0
    >>> edge_index = torch.tensor([[0,1,1,2,2,0], [1,0,2,1,0,2]])
    >>> g = Data(edge_index=edge_index, num_nodes=3)
    >>> probs = random_walk_probs(g, k=3)
    >>> # probs[0, 1] = probability of returning to node 0 after 2 steps
    
    Notes
    -----
    - For disconnected graphs, isolated nodes have return probability 1 at step 0
    - For graphs with self-loops, the computation includes self-transitions
    - The function handles empty graphs (no edges) gracefully
    - Memory usage grows as O(num_nodes^2) due to dense matrix operations
    """
    # Extract source nodes from edge index for degree computation
    source, _ = g.edge_index[0], g.edge_index[1]
    
    # Compute node degrees by counting outgoing edges
    deg = scatter_add(torch.ones_like(source), source, dim=0, dim_size=g.num_nodes)
    
    # Compute inverse degrees for transition matrix (D^-1)
    deg_inv = deg.pow(-1.0)
    # Handle isolated nodes (degree 0) by setting their inverse degree to 0
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    # Handle empty graph case (no edges)
    if g.edge_index.shape[1] == 0:
        # Create zero transition matrix for empty graph
        P = g.edge_index.new_zeros((1, g.num_nodes, g.num_nodes))
    else:
        # Compute transition matrix P = D^-1 * A
        # Convert sparse adjacency to dense and multiply by degree matrix
        P = torch.diag(deg_inv) @ to_dense_adj(g.edge_index, max_num_nodes=g.num_nodes)  # (1, num_nodes, num_nodes)
    
    # Initialize list to store diagonal elements (return probabilities)
    diags = []
    
    # Set up for odd/even step computation
    if skip_odd:
        # For even steps only: start with P^2
        Pmult = P @ P  # P^2 for computing P^2, P^4, P^6, ...
    else:
        # For all steps: start with P
        Pmult = P      # P for computing P^1, P^2, P^3, ...
    
    # Initialize current power of transition matrix
    Pk = Pmult
    
    # Compute return probabilities for k steps
    for _ in range(k):
        # Extract diagonal elements (return probabilities)
        diags.append(torch.diagonal(Pk, dim1=-2, dim2=-1))
        # Compute next power: P^(n+step) = P^n @ P^step
        Pk = Pk @ Pmult
    
    # Stack diagonals and transpose to get (num_nodes, k) shape
    p = torch.cat(diags, dim=0).transpose(0, 1)  # (num_nodes, k)
    
    return p
