import numpy as np
from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.flowmetrics import pairwise_distances


def nth_largest(arr, n, axis):
    """Calculate the Nth largest number along the specified axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array of numbers
    n : int
        The Nth largest number to find (starting from 1)
    axis : int
        The axis along which to calculate the Nth largest number

    Returns
    -------
    np.ndarray
        The Nth largest number along the specified axis

    Raises
    ------
    ValueError
        If n is greater than the length of the specified axis
    """
    if n > arr.shape[axis]:
        raise ValueError("n cannot be greater than the length of the specified axis")
    
    # Use np.partition to find the Nth largest number
    kth = -n  # Index of the Nth largest number
    partitioned = np.partition(arr, kth=kth, axis=axis)
    
    # Extract the Nth largest number
    result = np.take_along_axis(partitioned, np.array([kth]), axis=axis)
    return np.squeeze(result, axis=axis)


def _second_order_density(dis_matrix, distance='max', k=1, mask=None):
    """Calculate the second-order density of flows.

    Parameters
    ----------
    dis_matrix : np.ndarray
        Distance matrix with shape (N, N) between flows
    distance : str, optional
        The distance metric type used, by default 'max'
    k : int, optional
        The k-th nearest neighbor to consider, by default 1
    mask : np.ndarray, optional
        Boolean mask to filter flows (1-D array). If None, all flows are used

    Returns
    -------
    float
        The calculated second-order density

    Raises
    ------
    AssertionError
        If distance matrix is not square
    NotImplementedError
        If distance metric is not 'max'
    """
    assert dis_matrix.ndim==2 and dis_matrix.shape[0] == dis_matrix.shape[1], "The distance matrix must be square."
    flow_num = dis_matrix.shape[0]
    if mask is not None:
        assert isinstance(mask, np.ndarray), "The mask must be a numpy array."
        assert mask.ndim == 1 and mask.size <= flow_num, "The mask must be 1-D and smaller or equal than the number of flows."
        assert np.any(mask), "The mask must contain at least one True value."
    else:
        mask = np.ones(flow_num, dtype=bool)
    available_flow_num = np.count_nonzero(mask)
    
    diagonal_mask = np.ones((flow_num, flow_num), dtype=bool)
    np.fill_diagonal(diagonal_mask, False)
    dis_matrix = dis_matrix[diagonal_mask].reshape(flow_num, flow_num-1)
    dis_matrix = dis_matrix[mask, :]
    
    if distance == 'max':
        if k == 1:
            volume = (np.square(np.pi) * np.sum(np.min(dis_matrix, axis=1)**4))
        else:
            volume = (np.square(np.pi) * np.sum(nth_largest(dis_matrix, k, axis=1)**4))
    else:
        # TODO
        NotImplementedError("The second-order density is only implemented for max distance.")
    
    return available_flow_num / volume


def second_order_density(fdf: FlowDataFrame=None, dis_matrix=None, distance='max', k=1, mask=None, **kwargs):
    """Calculate the second-order density for a FlowDataFrame.

    Parameters
    ----------
    fdf : FlowDataFrame, optional
        The input flow dataframe
    dis_matrix : np.ndarray, optional
        Pre-computed distance matrix with shape (N, N). If None, it will be calculated
    distance : str, optional
        The distance metric type used, by default 'max'
    k : int, optional
        The k-th nearest neighbor to consider, by default 1
    mask : GeoSeries or geometric object, optional
        The GeoSeries (elementwise) or geometric object to test if each flow is within.
        If None, all flows are used
    **kwargs
        Additional keyword arguments for pairwise_distances function

    Returns
    -------
    float
        The calculated second-order density

    Raises
    ------
    AssertionError
        If distance matrix is not square
    """
    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance, **kwargs)
    else:
        assert dis_matrix.ndim==2 and dis_matrix.shape[0] == dis_matrix.shape[1], "The distance matrix must be square."
    if mask is not None:
        assert fdf is not None, "The fdf must be provided when mask is not None."
        mask = fdf.within(mask).values
    
    return _second_order_density(dis_matrix, distance=distance, k=k, mask=mask)

