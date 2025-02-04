import numpy as np
from numba import njit, prange
from typing import Optional, Tuple
from scipy.spatial.distance import pdist, squareform


EPSILON_DBL = 1e-8
PERPLEXITY_TOLERANCE = 1e-5


def make_joint(distr_cond: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Makes a joint probability distribution out of conditional distribution.

    Parameters:
        distr_cond (np.ndarray): Conditional distribution matrix.
        eps (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: Joint distribution matrix. All values in it sum up to 1.
                    Too small values are set to fixed epsilon.
    """
    n_points = distr_cond.shape[0]
    diag_mask = 1 - np.eye(n_points)
    distr_joint = (distr_cond + distr_cond.T) / (2 * n_points)
    return np.maximum(distr_joint, eps) * diag_mask

@njit(nogil=True, parallel=True)
def _binary_search_perplexity_numba(
    sqdistances: np.ndarray,
    desired_perplexity: float,
    steps: int
) -> np.ndarray:
    """
    Binary search for sigmas of conditional Gaussians using Numba acceleration.

    Parameters:
        sqdistances (np.ndarray): Squared distances between samples and their neighbors.
                                  Shape (n_samples, n_neighbors), dtype=np.float32.
        desired_perplexity (float): Target perplexity of the conditional distributions.
        steps (int): Number of binary search steps.

    Returns:
        np.ndarray: Conditional probabilities matrix.
                    Shape (n_samples, n_neighbors), dtype=np.float64.
    """
    n_samples, n_neighbors = sqdistances.shape
    using_neighbors = n_neighbors < n_samples
    desired_entropy = np.log(desired_perplexity)
    
    # Initialize output array and beta storage
    P = np.zeros((n_samples, n_neighbors), dtype=np.float64)
    beta_arr = np.zeros(n_samples, dtype=np.float64)

    # Parallel processing over samples
    for i in prange(n_samples):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0

        # Binary search for optimal beta
        for _ in range(steps):
            sum_Pi = 0.0
            # Compute unnormalized probabilities
            for j in range(n_neighbors):
                # Skip diagonal when using full matrix
                if not using_neighbors and j == i:
                    continue
                val = np.exp(-sqdistances[i, j] * beta)
                P[i, j] = val
                sum_Pi += val

            # Handle zero sum case
            if sum_Pi <= 0.0:
                sum_Pi = EPSILON_DBL

            # Normalize and compute entropy
            sum_disti_Pi = 0.0
            for j in range(n_neighbors):
                if not using_neighbors and j == i:
                    continue
                P[i, j] /= sum_Pi
                sum_disti_Pi += sqdistances[i, j] * P[i, j]

            entropy = np.log(sum_Pi) + beta * sum_disti_Pi
            entropy_diff = entropy - desired_entropy

            # Check convergence
            if abs(entropy_diff) <= PERPLEXITY_TOLERANCE:
                break

            # Update beta boundaries
            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

        beta_arr[i] = beta
    
    beta_arr = 1 / (beta_arr * 2)
    return P, beta_arr

def calc_optimized_p_cond(
    distances: np.ndarray,
    perplexity: float,
    steps: int = 100,
    joint: bool = True
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    NumPy implementation using a Numba-accelerated binary search.

    Parameters:
        distances (np.ndarray): Input distance matrix (N, N).
        perplexity (float): Target perplexity.
        steps (int): Number of binary search steps.
        joint (bool): Whether to return joint probability matrix.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: Joint probability matrix (and/or optimized variances).
    """
    p_cond, sq_sigmas = _binary_search_perplexity_numba(
        distances, perplexity, steps
    )

    # Generate final probability matrix
    if joint:
        p_cond = make_joint(p_cond)
    
    return p_cond, sq_sigmas


def get_multivariate_p_cond(distances_list, sigmas_list, combination='intersection', eps: float=1e-10) -> np.array:
    """
    Calculates conditional probability distribution given distances and squared sigmas.

    Parameters:
        distances_list (List[np.array]): A list with arrays of shape (N, N) containing the pairwise distances between N points.
        sigmas_list (List[np.array]): A list with row vector of squared sigma for each row in distances.
        combination (str): The combination method to use. Can be 'union' or 'intersection'.
        eps (float): A small value to avoid division by zero.

    Returns:
        np.array: Conditional probability matrix.
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

    Parameters:
        embedding (np.ndarray): Embedding (coordinates in low-dimensional map).
                                Shape (n_samples, dim).
        degrees_of_freedom (float): Degrees of freedom of the Student's-t distribution.

    Returns:
        np.ndarray: Pairwise inverse square law distances, as a condensed 1D array.
    """
    distances = pdist(embedding, metric='sqeuclidean')
    distances /= degrees_of_freedom
    distances += 1.0
    distances **= (degrees_of_freedom + 1.0) / -2.0
    return distances   


def d2q(distances, degrees_of_freedom=1.0):
    """
    Converts distances to Q distribution.

    Parameters:
        distances (np.ndarray): Pairwise distances of the low-dimension embeddings.
        degrees_of_freedom (float): Degrees of freedom of the Student's-t distribution.

    Returns:
        np.ndarray: Q distribution.
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
    Converts inverse square law distances to Q distribution.

    Parameters:
        distances (np.ndarray): Pairwise inverse square law distances of the low-dimension embeddings.

    Returns:
        np.ndarray: Q distribution.
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
    Calculates the Kullback-Leibler divergence between P and Q.

    Parameters:
        P (np.ndarray): Conditional probability matrix.
        Q (np.ndarray): Joint probability matrix.

    Returns:
        float: KL divergence value.
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
    Computes the gradient of the KL divergence.

    Parameters:
        embedding (np.ndarray): Embedding (coordinates in low-dimensional map).
                                Shape (n_samples, dim).
        P (np.ndarray): Conditional probability matrix.
        compute_error (bool): If False, the kl_divergence is not computed and returns NaN.
        degrees_of_freedom (float): Degrees of freedom of the Student's-t distribution.

    Returns:
        Tuple[np.ndarray, float]: Gradient and error (KL divergence).
    """
    if P.ndim==2:
        P = squareform(P)
    
    inv_distances = inv_sd(embedding, degrees_of_freedom)
    Q = inv_d2q(inv_distances)
    if compute_error:
        error = kl_divergence(P, Q)
    else:
        error = np.nan
    PQd = squareform((P - Q) * inv_distances)
    
    grad = np.ndarray(embedding.shape)
    for j in range(len(embedding)):
        grad[j] = np.dot(np.ravel(PQd[j], order='K'), embedding[j]-embedding)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    grad *= 4
    return grad, error


def hd_distance(P, Q):
    """
    Calculates the Hellinger distance between P and Q.

    Parameters:
        P (np.ndarray): Conditional probability matrix.
        Q (np.ndarray): Joint probability matrix.

    Returns:
        float: Hellinger distance value.
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


def hd_grad(embedding, P, compute_error=True, degrees_of_freedom=1.0, **kwargs):
    """
    Computes the gradient of the Hellinger distance.

    Parameters:
        embedding (np.ndarray): Embedding (coordinates in low-dimensional map).
                                Shape (n_samples, dim).
        P (np.ndarray): Conditional probability matrix.
        compute_error (bool): If False, the hl_distance is not computed and returns NaN.
        degrees_of_freedom (float): Degrees of freedom of the Student's-t distribution.

    Returns:
        Tuple[np.ndarray, float]: Gradient and error (KL divergence).
    """
    if P.ndim==2:
        P = squareform(P)
    
    inv_distances = inv_sd(embedding, degrees_of_freedom)
    Q = inv_d2q(inv_distances)
    if compute_error:
        # using KL divergence instead of HD distance
        error = kl_divergence(P, Q)
    else:
        error = np.nan
    sPQ = np.sqrt(P*Q)
    s = np.sum(sPQ) * 2
    sPQ = squareform((sPQ - s * Q) * inv_distances)
    
    grad = np.ndarray(embedding.shape)
    for j in range(len(embedding)):
        grad[j] = np.dot(np.ravel(sPQ[j], order='K'), embedding[j]-embedding)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    grad *= 4
    return grad, error


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
        Performs gradient descent optimization.

        Parameters:
            obj_func (callable): Objective function, e.g., kl_grad.
            embedding (np.ndarray): Embedding (coordinates in low-dimensional map).
            pijs (List[np.ndarray]): Conditional probability matrices of different features of flow.
                                     Shape (n_samples, n_samples) for each array.
            projections (list): Projections of different features of flow.
            degrees_of_freedom (float): Degrees of freedom of the Student's-t distribution.

        Returns:
            Tuple[np.ndarray, float]: Updated embedding and total error.
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
