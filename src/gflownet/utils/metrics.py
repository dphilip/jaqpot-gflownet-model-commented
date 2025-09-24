"""
Multi-objective optimization metrics and utilities for GFlowNet evaluation.

This module provides a comprehensive collection of metrics for evaluating the
performance of multi-objective optimization algorithms, particularly in the context
of GFlowNet training. It includes various measures of diversity, convergence,
and solution quality commonly used in multi-objective evolutionary algorithms.

Key metrics included:
- Hypervolume: Volume of objective space dominated by solutions
- IGD: Inverse Generational Distance to reference Pareto front
- Focus coefficients: Measures of alignment with preferred regions
- Pareto frontier analysis: Identification and evaluation of non-dominated solutions
- Diversity measures: Tanimoto similarity, clustering-based diversity
- HSR indicator: Hypervolume-based Sharpe Ratio for portfolio optimization
- Entropy measures: Distribution analysis of solutions in objective space

The module is designed to handle both synthetic benchmark problems and 
real-world applications like molecular generation and financial portfolio optimization.
"""

# Standard library imports for mathematical operations
import math
from copy import deepcopy
from itertools import product

# Numerical computing and machine learning imports
import numpy as np
import torch
import torch.nn as nn

# Multi-objective optimization specific imports
from botorch.utils.multi_objective import infer_reference_point, pareto
from botorch.utils.multi_objective.hypervolume import Hypervolume

# Chemistry and molecular similarity imports  
from rdkit import Chem, DataStructs
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def compute_focus_coef(
    flat_rewards: torch.Tensor, focus_dirs: torch.Tensor, focus_cosim: float, focus_limit_coef: float = 1.0
):
    """
    Compute focus coefficients for reward scaling based on directional preferences.
    
    This function implements a focusing mechanism that scales rewards based on their
    alignment with specified preference directions in the objective space. It creates
    a hypercone-shaped focus region where rewards are scaled according to their
    cosine similarity to the focus direction.
    
    The focus coefficient follows an exponential decay pattern:
    - 1.0 when exactly aligned with focus direction (cosim = 1.0)  
    - focus_limit_coef at the boundary of the focus region (cosim = focus_cosim)
    - 0.0 outside the focus region (cosim < focus_cosim)
    
    This is particularly useful for multi-objective optimization where you want to
    emphasize solutions in specific regions of the objective space while still
    maintaining some exploration elsewhere.
    
    Parameters
    ----------
    flat_rewards : torch.Tensor
        Reward vectors for each sample, shape (batch_size, num_objectives)
        Each row represents a point in the objective space
    focus_dirs : torch.Tensor  
        Focus direction vectors, shape (batch_size, num_objectives)
        Each row specifies the preferred direction for the corresponding sample
    focus_cosim : float
        Cosine similarity threshold defining the boundary of the focus region
        Must be in [0, 1] where higher values create narrower focus regions
    focus_limit_coef : float, default=1.0
        Focus coefficient value at the boundary of the focus region
        Must be in (0, 1] where lower values create steeper decay
        
    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - focus_coef: Focus coefficients for each sample, shape (batch_size,)
        - in_focus_mask: Boolean mask indicating samples within focus region
        
    Examples
    --------
    >>> rewards = torch.tensor([[0.8, 0.6], [0.3, 0.9]])  
    >>> directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> coef, mask = compute_focus_coef(rewards, directions, 0.7, 0.5)
    """
    # Validate input parameters to ensure they are in valid ranges
    assert focus_cosim >= 0.0 and focus_cosim <= 1.0, f"focus_cosim must be in [0, 1], now {focus_cosim}"
    assert (
        focus_limit_coef > 0.0 and focus_limit_coef <= 1.0
    ), f"focus_limit_coef must be in (0, 1], now {focus_limit_coef}"
    
    # Calculate the exponential decay parameter for the focus coefficient
    # This ensures that cosim^gamma = focus_limit_coef when cosim = focus_cosim
    focus_gamma_param = torch.tensor(np.log(focus_limit_coef) / np.log(focus_cosim)).float()
    
    # Compute cosine similarity between rewards and focus directions
    cosim = nn.functional.cosine_similarity(flat_rewards, focus_dirs, dim=1)
    
    # Create boolean mask for samples within the focus region
    in_focus_mask = cosim >= focus_cosim
    
    # Compute focus coefficients: exponential decay inside focus region, 0 outside
    focus_coef = torch.where(in_focus_mask, cosim**focus_gamma_param, 0.0)
    
    return focus_coef, in_focus_mask


def get_focus_accuracy(flat_rewards, focus_dirs, focus_cosim):
    """
    Calculate the fraction of samples that fall within the specified focus region.
    
    This function provides a simple metric for evaluating how well a set of solutions
    aligns with desired preference directions. It computes the percentage of samples
    that have sufficient cosine similarity to their corresponding focus directions.
    
    Parameters
    ----------
    flat_rewards : torch.Tensor
        Reward vectors for each sample, shape (batch_size, num_objectives)
    focus_dirs : torch.Tensor
        Focus direction vectors, shape (batch_size, num_objectives)  
    focus_cosim : float
        Cosine similarity threshold for the focus region boundary
        
    Returns
    -------
    torch.Tensor
        Scalar value representing the fraction of samples in focus (0.0 to 1.0)
    """
    # Use the focus coefficient computation to get the focus mask
    _, in_focus_mask = compute_focus_coef(focus_dirs, flat_rewards, focus_cosim, focus_limit_coef=1.0)
    
    # Return the fraction of samples that are within the focus region
    return in_focus_mask.float().sum() / len(flat_rewards)


def get_limits_of_hypercube(n_dims, n_points_per_dim=10):
    """
    Generate reference points on the faces of a unit hypercube for benchmark testing.
    
    This function creates a set of points that lie on the boundary faces of a unit
    hypercube in n-dimensional space. These points are useful as reference sets
    for computing performance metrics like IGD (Inverse Generational Distance)
    when no true Pareto front is known.
    
    The function discretizes each dimension into equally spaced points and then
    selects only those points that lie on at least one face of the hypercube
    (i.e., have at least one coordinate equal to 1.0).
    
    Parameters
    ----------
    n_dims : int
        Number of dimensions (objectives) in the hypercube
        Determines the dimensionality of the generated reference points
    n_points_per_dim : int, default=10
        Number of discretization points along each dimension
        Higher values provide denser coverage but exponentially more points
        
    Returns
    -------
    np.ndarray
        Array of shape (n_extreme_points, n_dims) containing the boundary points
        Each row is a point on the hypercube boundary with at least one coordinate = 1.0
        
    Examples
    --------  
    >>> points = get_limits_of_hypercube(2, 3)  # 2D square with 3 points per side
    >>> # Returns points like [[0, 1], [0.5, 1], [1, 0], [1, 0.5], [1, 1]]
    """
    # Create linear spaces for each dimension from 0 to 1
    linear_spaces = [np.linspace(0.0, 1.0, n_points_per_dim) for _ in range(n_dims)]
    
    # Generate all combinations of points in the hypercube (Cartesian product)
    grid = np.array(list(product(*linear_spaces)))
    
    # Select only points that lie on the boundary (at least one coordinate equals 1.0)
    # This gives us the "extreme" points on the faces of the hypercube
    extreme_points = grid[np.any(grid == 1, axis=1)]
    
    return extreme_points


def get_IGD(samples, ref_front: np.ndarray = None):
    """
    Compute the Inverse Generational Distance (IGD) of samples relative to a reference front.
    
    IGD measures the average distance from each point on a reference Pareto front to the
    closest point in the sample set. It provides a measure of both convergence (how close
    samples are to the true front) and coverage (how well samples cover the front).
    
    Lower IGD values indicate better performance, with 0 being perfect (samples exactly
    match the reference front). IGD is widely used in multi-objective optimization for
    comparing algorithm performance.
    
    The metric is computed as:
    IGD = (1/|P*|) * Σ_{p ∈ P*} min_{s ∈ S} d(p, s)
    where P* is the reference front, S is the sample set, and d is Euclidean distance.
    
    Parameters
    ----------
    samples : np.ndarray
        Generated samples to evaluate, shape (n_samples, n_objectives)
        Each row represents a point in the objective space
    ref_front : np.ndarray, optional
        Reference Pareto front, shape (n_ref_points, n_objectives)
        If None, uses the hypercube boundary as reference (for unit cube problems)
        
    Returns
    -------
    float
        IGD value (lower is better, 0 is perfect)
        
    References
    ----------
    https://www.sciencedirect.com/science/article/abs/pii/S0377221720309620
    
    Examples
    --------
    >>> samples = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.5]])
    >>> ref_front = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    >>> igd = get_IGD(samples, ref_front)
    """
    # Get the number of objectives from the sample shape
    n_objectives = samples.shape[1]
    
    # Use hypercube boundary as default reference if none provided
    if ref_front is None:
        ref_front = get_limits_of_hypercube(n_dims=n_objectives)

    # Compute pairwise distances between reference points and samples
    # distances[i, j] = distance from ref_point[i] to sample[j]
    distances = cdist(samples, ref_front).T

    # For each reference point, find the minimum distance to any sample
    min_distances = np.min(distances, axis=1)

    # IGD is the average of these minimum distances
    igd = np.mean(min_distances, axis=0)

    return float(igd)


def get_PC_entropy(samples, ref_front=None):
    """
    Computes entropy of the Pareto-Clustered (PC) distribution of the samples.

    For each point in the samples, the closest point in the reference front is
    found. We then compute the entropy of the empirical distribution of the
    samples in the clusters defined by the reference front.

    Parameters
    ----------
        Args:
        front (ndarray): A numpy array containing the coordinates of the points
            on the Pareto front. The tensor should have shape (n_points, n_objectives).
        ref_front (ndarray): A numpy array containing the coordinates of the points
            on the true Pareto front. The tensor should have shape (n_true_points, n_objectives).

    Returns:
        float: The IGD value.
    """
    n_objectives = samples.shape[1]
    if ref_front is None:
        ref_front = get_limits_of_hypercube(n_dims=n_objectives)

    # Compute the distances between each generated sample and each reference point.
    distances = cdist(samples, ref_front).T

    # Find the closest reference point for each generated sample.
    closest_point = np.argmin(distances, axis=0)

    # Construct a categorical distribution from the closest reference points.
    # by counting the number of samples in each category.
    pc_dist = np.bincount(closest_point, minlength=ref_front.shape[0])
    pc_dist = pc_dist / pc_dist.sum()

    # Compute its entropy.
    pc_ent = -np.sum(pc_dist * np.log(pc_dist + 1e-8))

    return float(pc_ent)


def sample_positiveQuadrant_ndim_sphere(n=10, d=2, normalisation="l2"):
    points = np.random.randn(n, d)
    points = np.abs(points)  # positive quadrant
    if normalisation == "l2":
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    elif normalisation == "l1":
        points /= np.sum(points, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalisation {normalisation}")
    return points


def partition_hypersphere(k: int, d: int, n_samples: int = 10000, normalisation: str = "l2"):
    """
    Partition a hypersphere into k clusters.
    ----------
    Parameters
        k: int
            Number of clusters
        d: int
            Dimensionality of the hypersphere
        n_samples: int
            Number of samples to use for clustering
        normalisation: str
            Normalisation to use for the samples and the cluster centers.
            Either 'l1' or 'l2'
    Returns
    -------
        v: np.ndarray
            Array of shape (k, d) containing the cluster centers
    """
    points = sample_positiveQuadrant_ndim_sphere(n_samples, d, normalisation)
    v = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(points).cluster_centers_
    if normalisation == "l2":
        v /= np.linalg.norm(v, axis=1, keepdims=True)
    elif normalisation == "l1":
        v /= np.sum(v, 1, keepdims=True)
    else:
        raise ValueError(f"Unknown normalisation {normalisation}")

    return v


def generate_simplex(dims, n_per_dim):
    spaces = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(dims)]
    return np.array([comb for comb in product(*spaces) if np.allclose(sum(comb), 1.0)])


def pareto_frontier(obj_vals, maximize=True):
    """
    Compute the Pareto frontier of a set of candidate solutions.
    ----------
    Parameters
        candidate_pool: NumPy array of candidate objects
        obj_vals: NumPy array of objective values
    ----------
    """
    # pareto utility assumes maximization
    if maximize:
        pareto_mask = pareto.is_non_dominated(torch.tensor(obj_vals))
    else:
        pareto_mask = pareto.is_non_dominated(-torch.tensor(obj_vals))
    return obj_vals[pareto_mask]


# From https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def get_hypervolume(flat_rewards: torch.Tensor, zero_ref=True) -> float:
    """Compute the hypervolume of a set of trajectories.
    Parameters
    ----------
    flat_rewards: torch.Tensor
      A tensor of shape (num_trajs, num_of_objectives) containing the rewards of each trajectory.
    """
    # Compute the reference point
    if zero_ref:
        reference_point = torch.zeros_like(flat_rewards[0])
    else:
        reference_point = infer_reference_point(flat_rewards)
    # Compute the hypervolume
    hv_indicator = Hypervolume(reference_point)  # Difference
    return hv_indicator.compute(flat_rewards)


def uniform_reference_points(nobj, p=4, scaling=None):
    """Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.
    """

    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points


def r2_indicator_set(reference_points, solutions, utopian_point):
    """Computer R2 indicator value of a set of solutions (*solutions*) given a set of
    reference points (*reference_points) and a utopian_point (*utopian_point).
        :param reference_points: An array of reference points from a uniform distribution.
        :param solutions: the multi-objective solutions (fitness values).
        :param utopian_point: utopian point that represents best possible solution
        :returns: r2 value (float).
    """

    min_list = []
    for v in reference_points:
        max_list = []
        for a in solutions:
            max_list.append(np.max(v * np.abs(utopian_point - a)))

        min_list.append(np.min(max_list))

    v_norm = np.linalg.norm(reference_points)
    r2 = np.sum(min_list) / v_norm

    return r2


def sharpeRatio(p, Q, x, rf):
    """Compute the Sharpe ratio.
    Returns the Sharpe ratio given the expected return vector, p,
    the covariance matrix, Q, the investment column vector, x, and
    the return of the riskless asset, rf.
    Parameters
    ----------
    p : ndarray
        Expected return vector (of size n).
    Q : ndarray
        Covariance (n,n)-matrix.
    x : ndarray
        Investment vector of size (n,1). The sum of which should be 1.
    rf : float
        Return of a riskless asset.
    Returns
    -------
    sr : float
        The HSR value.
    """
    return (x.T.dot(p) - rf) / math.sqrt(x.T.dot(Q).dot(x))


def _sharpeRatioQPMax(p, Q, rf):
    """Sharpe ratio maximization problem - QP formulation"""

    # intentional non-top-level imports to avoid
    # cvxopt dependency for M1 chip users
    from cvxopt import matrix, solvers

    solvers.options["abstol"] = 1e-15
    solvers.options["reltol"] = 1e-15
    solvers.options["feastol"] = 1e-15
    solvers.options["maxiters"] = 1000
    solvers.options["show_progress"] = False
    n = len(p)

    # inequality constraints (investment in assets is higher or equal to 0)
    C = np.diag(np.ones(n))
    d = np.zeros((n, 1), dtype=np.double)

    # equality constraints (just one)
    A = np.zeros((1, n), dtype=np.double)
    b = np.zeros((1, 1), dtype=np.double)
    A[0, :] = p - rf
    b[0, 0] = 1

    # convert numpy matrix to cvxopt matrix
    G, c, A, b, C, d = (
        matrix(Q, tc="d"),
        matrix(np.zeros(n), tc="d"),
        matrix(A, tc="d"),
        matrix(b, tc="d"),
        matrix(C, tc="d"),
        matrix(d, tc="d"),
    )

    sol = solvers.coneqp(G, c, -C, -d, None, A, b, kktsolver="ldl")  # , initvals=self.initGuess)
    y = np.array(sol["x"])

    return y


def sharpeRatioMax(p, Q, rf):
    """Compute the Sharpe ratio and investment of an optimal portfolio.
    Parameters
    ----------
    p : ndarray
        Expected return vector (of size n).
    Q : ndarray
        Covariance (n,n)-matrix.
    rf : float
        Return of a riskless asset.
    Returns
    -------
    sr : float
        The HSR value.
    x : ndarray
        Investment vector of size (n,1).
    """
    y = _sharpeRatioQPMax(p, Q, rf)
    x = y / y.sum()
    x = np.where(x > 1e-9, x, 0)
    sr = sharpeRatio(p, Q, x, rf)
    return sr, x


# Assumes that l <= A << u
# Assumes A, l, u are numpy arrays
def _expectedReturn(A, low, up):
    """
    Returns the expected return (computed as defined by the HSR indicator), as a
    column vector.
    """
    A = np.array(A, dtype=np.double)  # because of division operator in python 2.7
    return ((up - A).prod(axis=-1)) / ((up - low).prod())


def _covariance(A, low, up, p=None):
    """Returns the covariance matrix (computed as defined by the HSR indicator)."""
    p = _expectedReturn(A, low, up) if p is None else p
    Pmax = np.maximum(A[:, np.newaxis, :], A[np.newaxis, ...])
    P = _expectedReturn(Pmax, low, up)

    Q = P - p[:, np.newaxis] * p[np.newaxis, :]
    return Q


def _argunique(pts):
    """Find the unique points of a matrix. Returns their indexes."""
    ix = np.lexsort(pts.T)
    diff = (pts[ix][1:] != pts[ix][:-1]).any(axis=1)
    un = np.ones(len(pts), dtype=bool)
    un[ix[1:]] = diff
    return un


def HSRindicator(A, low, up, managedup=False):
    """
    Compute the HSR indicator of the point set A given reference points l and u.
    Returns the HSR value of A given l and u, and returns the optimal investment.
    By default, points in A are assumed to be unique.
    Tip: Either ensure that A does not contain duplicated points
        (for example, remove them previously and then split the
        investment between the copies as you wish), or set the flag
        'managedup' to True.
    Parameters
    ----------
    A : ndarray
        Input matrix (n,d) with n points and d dimensions.
    low : array_like
        Lower reference point.
    up : array_like
        Upper reference point.
    managedup : bool, optional
        If A contains duplicated points and 'managedup' is set to True, only the
        first copy may be assigned positive investment, all other copies are
        assigned zero investment. Otherwise, no special treatment is given to
        duplicate points.
    Returns
    -------
    hsri : float
        The HSR value.
       x : ndarray
        The optimal investment as a column vector array (n,1).
    """
    n = len(A)
    x = np.zeros((n, 1), dtype=float)

    # if u is not strongly dominated by l or A is the empty set
    if (up <= low).any():
        raise ValueError("The lower reference point does not strongly dominate the upper reference point!")

    if len(A) == 0:
        return 0, x

    valid = (A < up).all(axis=1)
    validix = np.where(valid)[0]

    # if A is the empty set
    if valid.sum() == 0:
        return 0, x
    A = A[valid]  # A only contains points that strongly dominate u
    A = np.maximum(A, low)
    m = len(A)  # new size (m <= n)

    # manage duplicate points
    ix = _argunique(A) if managedup else np.ones(m).astype(bool)
    p = _expectedReturn(A[ix], low, up)
    Q = _covariance(A[ix], low, up, p)

    hsri, x[validix[ix]] = sharpeRatioMax(p, Q, 0)

    return hsri, x


class HSR_Calculator:
    def __init__(self, lower_bound, upper_bound, max_obj_bool=None):
        """
        Class to calculate HSR Indicator with assumption that assumes a maximization on all objectives.
         Parameters
        ----------
        lower_bound : array_like
            Lower reference point.
        upper_bound : array_like
            Upper reference point.
        max_obj_bool : bool, optional
            Details of the objectives for which dimension maximization is not the case.
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_obj_bool = None

        if max_obj_bool is not None:
            self.max_obj_bool = max_obj_bool

    def reset_lower_bound(self, lower_bound):
        self.lower_bound = lower_bound

    def reset_upper_bound(self, upper_bound):
        self.upper_bound = upper_bound

    def make_max_problem(self, matrix):
        if self.max_obj_bool is None:
            return matrix

        max_matrix = deepcopy(matrix)

        for dim in self.max_obj_bool:
            max_matrix[:, dim] = max_matrix**-1

        return max_matrix

    def calculate_hsr(self, solutions):
        max_solutions = self.make_max_problem(solutions)

        hsr_indicator, hsr_invest = HSRindicator(A=max_solutions, low=self.lower_bound, up=self.upper_bound)

        return hsr_indicator, hsr_invest


class Normalizer(object):
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = np.where(scale != 0, scale, 1.0)

    def __call__(self, arr):
        min_val = self.loc - 4 * self.scale
        max_val = self.loc + 4 * self.scale
        clipped_arr = np.clip(arr, a_min=min_val, a_max=max_val)
        norm_arr = (clipped_arr - self.loc) / self.scale

        return norm_arr

    def inv_transform(self, arr):
        return self.scale * arr + self.loc


def all_are_tanimoto_different(thresh, fp, mode_fps, delta=16):
    """
    Equivalent to `all(DataStructs.BulkTanimotoSimilarity(fp, mode_fps) < thresh)` but much faster.
    """
    assert delta > 0
    s = 0
    n = len(mode_fps)
    while s < n:
        e = min(s + delta, n)
        for i in DataStructs.BulkTanimotoSimilarity(fp, mode_fps[s:e]):
            if i >= thresh:
                return False
        s = e
    return True


# Should be calculated per preference
def compute_diverse_top_k(smiles, rewards, k, thresh=0.7):
    # mols is a list of (reward, mol)
    mols = []
    for i in range(len(smiles)):
        mols.append([rewards[i].item(), smiles[i]])
    mols = sorted(mols, key=lambda m: m[0], reverse=True)
    modes = [mols[0]]
    mode_fps = [Chem.RDKFingerprint(mols[0][1])]
    for i in range(1, len(mols)):
        fp = Chem.RDKFingerprint(mols[i][1])
        sim = DataStructs.BulkTanimotoSimilarity(fp, mode_fps)
        if max(sim) < thresh:
            modes.append(mols[i])
            mode_fps.append(fp)
        if len(modes) >= k:
            # last_idx = i
            break
    return np.mean([i[0] for i in modes])  # return sim


def get_topk(rewards, k):
    """
     Parameters
    ----------
    rewards : array_like
        Rewards obtained after taking the convex combination.
        Shape: number_of_preferences x number_of_samples
    k : int
        Top k value

    Returns
    ----------
    avergae Topk rewards across all preferences
    """
    if len(rewards.shape) < 2:
        rewards = torch.unsqueeze(rewards, -1)
    sorted_rewards = torch.sort(rewards, 1).values
    topk_rewards = sorted_rewards[range(rewards.shape[0]), :k]
    mean_topk = torch.mean(topk_rewards.mean(-1))
    return mean_topk


def top_k_diversity(fps, r, K):
    x = []
    for i in np.argsort(r)[::-1]:
        y = fps[i]
        if y is None:
            continue
        x.append(y)
        if len(x) >= K:
            break
    s = np.array([DataStructs.BulkTanimotoSimilarity(i, x) for i in x])
    return (np.sum(s) - len(x)) / (len(x) * len(x) - len(x))  # substract the diagonal


if __name__ == "__main__":
    # Example for 2 dimensions
    # Point set: {(1,3), (2,2), (3,1)},  l = (0,0), u = (4,4)
    A = np.array([[1, 3], [2, 2], [3, 1]])  # matrix with dimensions n x d (n points, d dimensions)
    low = np.zeros(2)  # l must weakly dominate every point in A
    up = np.array([4, 4])  # u must be strongly dominated by every point in A

    # A = np.array([[3.41e-01, 9.72e-01, 2.47e-01],
    #              [9.30e-01, 1.53e-01, 4.72e-01],
    #              [4.56e-01, 1.71e-01, 8.68e-01],
    #              [8.70e-02, 5.94e-01, 9.50e-01],
    #              [5.31e-01, 6.35e-01, 1.95e-01],
    #              [3.12e-01, 3.37e-01, 7.01e-01],
    #              [3.05e-02, 9.10e-01, 7.71e-01],
    #              [8.89e-01, 8.29e-01, 2.07e-02],
    #              [6.92e-01, 3.62e-01, 2.93e-01],
    #              [2.33e-01, 4.55e-01, 6.60e-01]])
    #
    # l = np.zeros(3)  # l must weakly dominate every point in A
    # u = np.array([1, 1, 1])

    hsr_class = HSR_Calculator(lower_bound=low, upper_bound=up)
    hsri, x = hsr_class.calculate_hsr(A)  # compute HSR indicator

    print("Optimal investment:")
    print("%s" % "\n".join(map(str, x[:, 0])))
    print("HSR indicator value: %f" % hsri)
