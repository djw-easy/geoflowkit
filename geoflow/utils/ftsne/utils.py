import numpy as np
from numba import njit
from typing import Optional, Tuple
from scipy.spatial.distance import pdist, squareform



def entropy(p: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """
    Calculates the Shannon entropy for every row in a conditional probability matrix.
    
    Args:
        p (np.ndarray): Conditional probability matrix, where every row sums up to 1
        
    Return:
        1D array of entropies, (n_points,)
    """
    return -(p * np.log2(p + eps)).sum(axis=1)

def get_p_cond(distances: np.ndarray, sigmas_sq: np.ndarray, mask: np.ndarray, eps: float) -> np.ndarray:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    
    Args:
        distances (torch.Tensor): Matrix of squared distances ||x_i - x_j||^2
        sigmas_sq (torch.Tensor): Row vector of squared sigma for each row in distances
        mask (torch.Tensor): A mask tensor to set diagonal elements to zero
        eps (float or torch.Tensor): A small value to avoid division by zero.
    Return:
        Conditional probability matrix
    """
    sigmas_sq_clipped = np.maximum(sigmas_sq, eps)
    logits = -distances / (2 * sigmas_sq_clipped.reshape(-1, 1))
    exp_logits = np.exp(logits)
    masked_exp_logits = exp_logits * mask
    normalization = np.maximum(masked_exp_logits.sum(axis=1, keepdims=True), eps)
    return masked_exp_logits / normalization + eps

def make_joint(distr_cond: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Makes a joint probability distribution out of conditional distribution
    
    Atgs:
        distr_cond (torch.Tensor): Conditional distribution matrix
        eps (float or torch.Tensor): A small value to avoid division by zero
    
    Return:
        Joint distribution matrix. All values in it sum up to 1.
        Too small values are set to fixed epsilon
    """
    n_points = distr_cond.shape[0]
    diag_mask = 1 - np.eye(n_points)
    distr_joint = (distr_cond + distr_cond.T) / (2 * n_points)
    return np.maximum(distr_joint, eps) * diag_mask

@njit  # Use Numba to accelerate the critical loop
def _binary_search_step(
    distances: np.ndarray,
    diag_mask: np.ndarray,
    min_sigma_sq: np.ndarray,
    max_sigma_sq: np.ndarray,
    sq_sigmas: np.ndarray,
    target_entropy: float,
    eps: float,
    tol: float,
    max_iter: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba implementation of a binary search loop for optimization.

    Args:
        distances (np.ndarray): Pairwise distances between points
        diag_mask (np.ndarray): Mask to exclude diagonal elements
        min_sigma_sq (np.ndarray): Minimum squared sigma values
        max_sigma_sq (np.ndarray): Maximum squared sigma values
        sq_sigmas (np.ndarray): Current squared sigma values
        target_entropy (float): Target entropy value
        eps (float): Small value to avoid division by zero
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations

    Returns:
        Tuple[np.ndarray, np.ndarray]: Optimized squared sigma values and conditional probabilities
    """
    finished = np.zeros_like(sq_sigmas, dtype=np.bool_)
    curr_iter = 0
    
    while not np.all(finished) and curr_iter < max_iter:
        # Calculate current conditional probabilities and entropy difference
        p_cond = np.empty_like(distances)
        for i in range(distances.shape[0]):
            sig = sq_sigmas[i]
            sig_clipped = max(sig, eps)
            row = -distances[i] / (2 * sig_clipped)
            exp_row = np.exp(row)
            masked_row = exp_row * diag_mask[i]
            norm = max(masked_row.sum(), eps)
            p_cond[i] = masked_row / norm + eps
        
        ent = -(p_cond * np.log2(p_cond)).sum(axis=1)
        ent_diff = ent - target_entropy
        
        # Update search boundaries
        pos_mask = ent_diff > 0
        neg_mask = ent_diff <= 0
        
        # Update maximum and minimum variances
        new_max_sigma_sq = np.where(pos_mask, sq_sigmas, max_sigma_sq)
        new_min_sigma_sq = np.where(neg_mask, sq_sigmas, min_sigma_sq)
        
        # Update current variances
        not_finished = np.abs(ent_diff) >= tol
        delta = (new_min_sigma_sq + new_max_sigma_sq) / 2
        sq_sigmas = np.where(not_finished, delta, sq_sigmas)
        
        # Update loop variables
        min_sigma_sq = new_min_sigma_sq
        max_sigma_sq = new_max_sigma_sq
        finished = np.abs(ent_diff) < tol
        curr_iter += 1
        
    return sq_sigmas, p_cond

def calc_optimized_p_cond(
    distances: np.ndarray,
    perplexity: float,
    eps: float = 1e-10,
    tol: float = 1e-4,
    max_iter: int = 200,
    min_allowed_sig_sq: float = 0,
    max_allowed_sig_sq: float = 100000,
    joint: bool = True
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    NumPy implementation using a Numba-accelerated binary search.
    
    Args:
        distances (np.ndarray): Input distance matrix (N, N)
        perplexity (float): Target perplexity
        eps (float): Small value to avoid division by zero
        tol (float): Tolerance for binary search
        max_iter (int): Maximum number of iterations
        min_allowed_sig_sq (float): Minimum allowed squared sigma
        max_allowed_sig_sq (float): Maximum allowed squared sigma
        joint (bool): Whether to return joint probability matrix
        
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Joint probability matrix (and/or optimized variances)
    """
    n_points = distances.shape[0]
    target_entropy = np.log2(perplexity)
    diag_mask = 1 - np.eye(n_points)

    # Initialize variance search range
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * np.ones(n_points)
    max_sigma_sq = max_allowed_sig_sq * np.ones(n_points)
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2

    # Use Numba-accelerated core loop
    sq_sigmas, p_cond = _binary_search_step(
        distances, diag_mask, min_sigma_sq, max_sigma_sq, sq_sigmas,
        target_entropy, eps, tol, max_iter
    )

    # Check numerical stability
    if np.isnan(sq_sigmas).any():
        print("Warning! NaN detected in sigmas. Discarding batch.")
        return None

    # Generate final probability matrix
    if joint:
        p_cond = make_joint(p_cond, eps=eps) * diag_mask
    
    return p_cond, sq_sigmas


def get_multivariate_p_cond(distances_list, sigmas_list, combination='intersection', eps: float=1e-10) -> np.array:
    """
    Calculates conditional probability distribution given distances and squared sigmas
    
    Args:
        distances_list (List[np.array]): A list with arrays of shape (N, N) containing the pairwise distances between N points.
        sigmas_list (List[np.array]): A list with row vector of squared sigma for each row in distances
        combination (str): The combination method to use. Can be 'union' or 'intersection'.
        eps (float or np.array): A small value to avoid division by zero.

    Return:
        Conditional probability matrix
    """
    n_points = distances_list[0].shape[0]
    diag_mask = 1 - np.eye(n_points, dtype=bool)

    if combination == 'union':
        logits1 = sum(np.exp(-distances / (2 * np.maximum(sigmas, eps).reshape(-1, 1))) for distances, sigmas in zip(distances_list, sigmas_list))
        l1_1 = np.exp(-distances_list[0] / (2 * np.maximum(sigmas_list[0], eps).reshape(-1, 1)))
        l1_2 = np.exp(-distances_list[1] / (2 * np.maximum(sigmas_list[1], eps).reshape(-1, 1)))

        logits2 = sum(distances / np.maximum(sigmas, eps).reshape(-1, 1) for distances, sigmas in zip(distances_list, sigmas_list))
        logits2 = np.exp(-0.5 * logits2)
        masked_exp_logits = (logits1 - logits2) * diag_mask
    elif combination == 'intersection':
        logits = sum(distances / np.maximum(sigmas, eps).reshape(-1, 1) for distances, sigmas in zip(distances_list, sigmas_list))
        logits = np.exp(-0.5 * logits)
        masked_exp_logits = logits * diag_mask
    else:
        raise ValueError(f'Unknown combination: {combination}, must be "union" or "intersection"')
    normalization = np.maximum(masked_exp_logits.sum(1), eps).reshape(-1, 1)
    distr_cond = masked_exp_logits / normalization
    distr_joint = (distr_cond + distr_cond.T) / (2 * n_points)
    return distr_joint * diag_mask


def inv_sd(embedding, degrees_of_freedom=1.0):
    """
    Computes the pairwise inverse square law distances for the given embedding. 

    Parameters
    ----------
    embedding : array, shape (n_samples, dim)
        Embedding (coordinates in low-dimensional map).

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    Returns
        dist: pairwise inverse square law distances, as a condensed 1D array.
    """
    distances = pdist(embedding, metric='sqeuclidean')
    distances /= degrees_of_freedom
    distances += 1.0
    distances **= (degrees_of_freedom + 1.0) / -2.0
    return distances   


def d2q(distances, degrees_of_freedom=1.0):
    """
    Parameters
    ----------
    distances : array
        pairwise distances of the low-dimension embeddings.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    """
    MACHINE_EPSILON = np.finfo(np.double).eps
    distances /= degrees_of_freedom
    distances += 1.0
    distances **= (degrees_of_freedom + 1.0) / -2.0
    # Q is a heavy-tailed distribution: Student's t-distribution
    if distances.ndim==1:
        constant = 2.0
    elif distances.ndim==2:
        constant = 1.0
    else:
        raise ValueError("Unsupported dist.ndim: {}".format(distances.ndim))
    Q = np.maximum(distances / (constant * np.sum(distances)), MACHINE_EPSILON)
    return Q


def inv_d2q(distances):
    """
    Parameters
    ----------
    distances : array
        pairwise inverse square law distances of the low-dimension embeddings.
    """
    MACHINE_EPSILON = np.finfo(np.double).eps
    # Q is a heavy-tailed distribution: Student's t-distribution
    if distances.ndim==1:
        constant = 2.0
    elif distances.ndim==2:
        constant = 1.0
    else:
        raise ValueError("Unsupported dist.ndim: {}".format(distances.ndim))
    Q = np.maximum(distances / (constant * np.sum(distances)), MACHINE_EPSILON)
    return Q


def kl_divergence(P, Q):
    """
    Parameters
    ----------
    P : array
        Conditional probability matrix.
    Q : array
        Joint probability matrix.
    """
    MACHINE_EPSILON = np.finfo(np.double).eps
    if P.ndim == 1:
        kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        diag_mask = np.eye(P.shape[0]).astype(bool)
        P = P[~diag_mask]
        Q = Q[~diag_mask]
        kl_divergence = 1.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    return kl_divergence


def kl_grad(embedding, P, compute_error=True, degrees_of_freedom=1.0, **kwargs):
    """
    Parameters
    ----------
    embedding : array, shape (n_samples, dim)
        Embedding (coordinates in low-dimensional map).
    P : array
        Conditional probability matrix. 
    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    """
    if P.ndim == 1:
        inv_distances = inv_sd(embedding, degrees_of_freedom)
        Q = inv_d2q(inv_distances)
        if compute_error:
            error = kl_divergence(P, Q)
        else:
            error = np.nan
        PQd = squareform((P - Q) * inv_distances)
    else:
        inv_distances = inv_sd(embedding, degrees_of_freedom)
        Q = inv_d2q(inv_distances)
        inv_distances = squareform(inv_distances)
        Q = squareform(Q)
        if compute_error:
            error = kl_divergence(P, Q)
        else:
            error = np.nan
        PQd = (P - Q) * inv_distances
    
    grad = np.ndarray(embedding.shape)
    for j in range(len(embedding)):
        grad[j] = np.dot(np.ravel(PQd[j], order='K'), embedding[j]-embedding)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    grad *= 4
    return grad, error


def hl_distance(P, Q):
    """
    Parameters
    ----------
    P : array
        Conditional probability matrix.
    Q : array
        Joint probability matrix.
    """
    if P.ndim == 1:
        hl_distance = np.square(np.sqrt(P) - np.sqrt(Q))
    else:
        diag_mask = np.eye(P.shape[0]).astype(bool)
        P = P[~diag_mask]
        Q = Q[~diag_mask]
        hl_distance = np.square(np.sqrt(P) - np.sqrt(Q)) / 2.0
    hl_distance = np.sum(hl_distance)
    return hl_distance


def hl_grad(embedding, P, compute_error=True, **kwargs):
    """
    Parameters
    ----------
    embedding : array, shape (n_samples, dim)
        Embedding (coordinates in low-dimensional map).
    P : array
        Conditional probability matrix.
    compute_error: bool, default=True
        If False, the hl_distance is not computed and returns NaN.
    """
    # TODO


class GDOptimizer:
    epoch = 0
    change = 0
    def __init__(self, lr, momentum, lr_scheduler=None):
        self.lr = lr
        self.momentum = momentum
        # assert isinstance(lr_scheduler, callable) or lr_scheduler is None
        self.lr_scheduler = lr_scheduler

    def __call__(self, obj_func, embedding, pijs, projections, degrees_of_freedom=None):
        """
        Parameters
        ----------
        obj_func : function
            Objective function, ex. kl_grad.
        embedding : array
            Embedding (coordinates in low-dimensional map).
        pijs : List[array, shape (n_samples, n_samples)]
            Conditional probability matrixs of different features of flow.
        projections : list
            Projections of different features of flow.
        degrees_of_freedom : int
            Degrees of freedom of the Student's-t distribution.
        """
        total_error = 0.0
        grads = []
        kwargs = {'degrees_of_freedom': degrees_of_freedom}
        proj_matrix = np.eye(embedding.shape[1])
        for pij, proj in zip(pijs, projections):
            if isinstance(proj, int):
                proj = proj_matrix[proj: proj+1, :]
                proj_grad, error = obj_func(embedding @ proj.T, pij, **kwargs)
                grads += [proj_grad @ proj]
            else:
                proj = proj_matrix[proj, :]
                proj_grad, error = obj_func(embedding @ proj.T, pij, **kwargs)
                grads += [proj_grad @ proj]
            total_error += error
        grad = sum(grads) / len(projections)
        embedding -= self.lr * grad + self.momentum * self.change
        self.change = grad
        self.epoch += 1
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler(self.epoch) * self.lr

        return embedding, total_error
