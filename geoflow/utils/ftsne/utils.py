import numba
import numpy as np
from scipy.spatial.distance import pdist, squareform



# @numba.jit(nopython=True, error_model='numpy')          # https://github.com/numba/numba/issues/4360
def Hbeta(D, beta):

    P = np.exp(-D * beta)
    sumP = np.sum(P)
    P = P / sumP
    H = np.log(sumP) + beta * np.sum(D * P)
    
    return H, P

# @numba.jit(nopython=True)
def d2p_job(Di, logU, max_iteration=200, tol=1e-4):
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    
    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    tries = 0
    while tries < max_iteration and np.abs(Hdiff) > tol:
        # If not, increase or decrease precision
        if Hdiff > 0:
            beta_min = beta
            if np.isinf(beta_max):      # Numba compatibility: isposinf --> isinf
                beta *= 2.
            else:
                beta = (beta + beta_max) / 2.
        else:
            beta_max = beta
            if np.isinf(beta_min):      # Numba compatibility: isneginf --> isinf
                beta /= 2. 
            else:
                beta = (beta + beta_min) / 2.

        H, thisP = Hbeta(Di, beta)
        Hdiff = H - logU
        tries += 1

    return thisP, beta

def d2p(D, perplexity):
    n = D.shape[0]
    logU = np.log(perplexity)

    idx = (1 - np.eye(n)).astype(bool)
    D = D[idx].reshape((n, -1))

    betas = np.zeros(n)
    P = np.zeros([n, n])
    for i in range(n):
        P[i, idx[i]], betas[i] = d2p_job(D[i], logU)

    P[np.isnan(P)] = 0
    P = P + P.T
    P = P / P.sum()
    P = np.maximum(P, 1e-12)

    return P, betas



def binary_search_perplexity(distances, perplexity):
    """
    Computes the sigmas from given distances with a certain perplexity.

    Parameters
    ----------
    distances : array, shape (n_samples*n_samples)
        Pairwise distances.

    perpelxity : float, >0
        Desired perplexity of the joint probability distribution.
    
    Returns
    -------
    sigma : array, shape (n_samples,)
        The sigmas for each sample.
    """
    n_samples = distances.shape[0]
    target_entropy = np.log(perplexity)


def get_multivariate_p_cond(distances_list, sigmas_list, combination='intersection', eps: float=1e-10):
    """
    Calculates conditional probability distribution given distances and squared sigmas
    
    Args:
        distances_list (List[np.ndarray]): A list with arrays of shape (N, N) containing the pairwise distances between N points.
        sigmas_list (List[np.ndarray]): A list with row vector of squared sigma for each row in distances.
        combination (str): The combination method to use. Can be 'union' or 'intersection'.
        eps (float or np.ndarray): A small value to avoid division by zero.

    Return:
        Conditional probability matrix
    """
    n_points = distances_list[0].shape[0]
    diag_mask = 1 - np.eye(n_points)

    if combination == 'union':
        logits1 = sum(np.exp(-distances / (2 * np.maximum(sigmas, eps).reshape(-1, 1))) for distances, sigmas in zip(distances_list, sigmas_list))
        logits2 = sum(distances / np.maximum(sigmas, eps).reshape(-1, 1) for distances, sigmas in zip(distances_list, sigmas_list))
        logits2 = np.exp(-logits2)
        masked_exp_logits = (logits1 - logits2) * diag_mask
    elif combination == 'intersection':
        logits = sum(distances / np.maximum(sigmas, eps).reshape(-1, 1) for distances, sigmas in zip(distances_list, sigmas_list))
        logits = np.exp(-logits)
        masked_exp_logits = logits * diag_mask
    else:
        raise ValueError(f'Unknown combination: {combination}, must be "union" or "intersection"')
    
    normalization = np.maximum(masked_exp_logits.sum(1), eps).reshape(-1, 1)
    P = masked_exp_logits / normalization

    n_points = P.shape[0]
    P = (P + P.T) / (2 * n_points)
    return squareform(P)


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


def kl_grad(embedding, P, degrees_of_freedom=1.0, compute_error=True):
    """
    Parameters
    ----------
    embedding : array, shape (n_samples, dim)
        Embedding (coordinates in low-dimensional map).

    P : array
        Conditional probability matrix. 

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.
    """
    if P.ndim == 1:
        inv_distances = inv_sd(embedding, degrees_of_freedom)
        Q = inv_d2q(inv_distances)
        if compute_error:
            error = kl_divergence(P, Q)
        else:
            error = np.nan
        grad = np.ndarray(embedding.shape)
        PQd = squareform((P-Q)*inv_distances)
    else:
        inv_distances = inv_sd(embedding, degrees_of_freedom)
        Q = inv_d2q(inv_distances)
        inv_distances = squareform(inv_distances)
        Q = squareform(Q)
        if compute_error:
            error = kl_divergence(P, Q)
        else:
            error = np.nan
        grad = np.ndarray(embedding.shape)
        PQd = (P - Q) * inv_distances
    
    for i in range(len(embedding)):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'), embedding[i]-embedding)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    grad *= 4
    return grad, error





