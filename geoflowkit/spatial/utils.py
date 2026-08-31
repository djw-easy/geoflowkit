from typing import Any, Union

import numpy as np

from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.flowmetrics import pairwise_distances


def nth_largest(
    arr: np.ndarray,
    n: int,
    axis: int,
) -> np.ndarray:
    """Find the Nth largest number along the specified axis.

    Parameters
    ----------
    arr : np.ndarray
        Input array of numbers.
    n : int
        The Nth largest number to find (1 = largest, 2 = second largest, etc.).
    axis : int
        The axis along which to find the Nth largest number.

    Returns
    -------
    np.ndarray
        The Nth largest number(s) along the specified axis.

    Raises
    ------
    ValueError
        If ``n`` is greater than the length of the specified axis.
    """
    if n < 1 or n > arr.shape[axis]:
        raise ValueError(
            "n must be between 1 and the length of the specified axis"
        )
    
    # Use np.partition to find the Nth largest number
    kth = -n  # Index of the Nth largest number
    partitioned = np.partition(arr, kth=kth, axis=axis)
    
    # Build index shape: all 1s except along target axis
    idx_shape = tuple(1 if i == axis else d for i, d in enumerate(arr.shape))
    idx_shape = tuple(s if s > 0 else 1 for s in idx_shape)
    indices = np.full(idx_shape, kth, dtype=int)
    
    # Extract the Nth largest number
    result = np.take_along_axis(partitioned, indices, axis=axis)
    return np.squeeze(result, axis=axis)


def _second_order_density(
    dis_matrix: np.ndarray,
    distance: str = "max",
    k: int = 1,
    mask: Union[np.ndarray, None] = None,
) -> float:
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
    ValueError
        If the distance matrix or mask has an invalid shape.
    TypeError
        If mask is not a NumPy array.
    NotImplementedError
        If distance metric is not 'max'
    """
    if dis_matrix.ndim != 2 or dis_matrix.shape[0] != dis_matrix.shape[1]:
        raise ValueError("The distance matrix must be square")
    flow_num = dis_matrix.shape[0]
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("The mask must be a NumPy array")
        if mask.ndim != 1 or mask.size != flow_num:
            raise ValueError(
                "The mask must be 1-D with one value per flow"
            )
        if not np.any(mask):
            raise ValueError("The mask must contain at least one True value")
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
        raise NotImplementedError("The second-order density is only implemented for max distance.")
    
    return available_flow_num / volume


def second_order_density(
    fdf: Union[FlowDataFrame, None] = None,
    dis_matrix: Union[np.ndarray, None] = None,
    distance: str = "max",
    k: int = 1,
    mask: Union[Any, None] = None,
    **kwargs: Any,
) -> float:
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
    ValueError
        If the distance matrix is not square.
    """
    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance, **kwargs)
    else:
        if dis_matrix.ndim != 2 or dis_matrix.shape[0] != dis_matrix.shape[1]:
            raise ValueError("The distance matrix must be square")
    if mask is not None:
        if fdf is None:
            raise ValueError("fdf must be provided when mask is not None")
        mask = fdf.within(mask).values
    
    return _second_order_density(dis_matrix, distance=distance, k=k, mask=mask)

