import numpy as np
from geoflow.flowdataframe import FlowDataFrame
from geoflow.flowprocess import pairwise_distances
from shapely.geometry import LineString, Point, Polygon


def nth_largest(arr, n, axis):
    """
    Calculate the Nth largest number along the specified axis.

    Parameters:
        arr (np.ndarray): Input array.
        n (int): The Nth largest number (starting from 1).
        axis (int): The axis along which to calculate.

    Returns:
        np.ndarray: The Nth largest number along the specified axis.
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
    """
    Calculate the second-order density of flows.

    Parameters:
        dis_matrix (np.ndarray): Distance matrix with shape (N, N) between flows.
        distance (str, optional): The distance metric type used. Default is 'max'.
        k (int, optional): The k-th nearest neighbor to consider. Default is 1.
        mask (np.ndarray, optional): Boolean mask to filter flows, 1-D np.ndarray. If None, all flows are used.

    Returns:
        float: The calculated second-order density.

    Raises:
        NotImplementedError: If distance is not 'max'.
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
    """
    Calculate the second-order density for a FlowDataFrame.

    Parameters:
        fdf (FlowDataFrame, optional): The input flow dataframe.
        dis_matrix (np.ndarray, optional): Pre-computed distance matrix with shape (N, N). If None, it will be calculated.
        distance (str, optional): The distance metric type used. Default is 'max'.
        k (int, optional): The k-th nearest neighbor to consider. Default is 1.
        mask (GeoSeries or geometric object, optional): The GeoSeries (elementwise) or geometric object to test if each flow is within. Default is None, which means all flows are used.
        **kwargs: Additional keyword arguments for pairwise_distances function.

    Returns:
        float: The calculated second-order density.
    """
    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance, **kwargs)
    else:
        assert dis_matrix.ndim==2 and dis_matrix.shape[0] == dis_matrix.shape[1], "The distance matrix must be square."
    if mask is not None:
        assert fdf is not None, "The fdf must be provided when mask is not None."
        mask = fdf.within(mask).values
    
    return _second_order_density(dis_matrix, distance=distance, k=k, mask=mask)


def _params_for_kl_func(fdf: FlowDataFrame, dr, k=1, distance='max', dis_matrix=None, mask=None, **kwargs):
    """
    Calculate parameters for K and L functions.

    Parameters:
        fdf (FlowDataFrame): The input flow dataframe.
        dr (float): The step size for distance intervals.
        k (int, optional): The Nth nearest neighbors for each flow to calculate the density. Default is 1.
        distance (str): The distance metric to use. Default is 'max'.
        dis_matrix (np.ndarray, optional): Pre-computed distance matrix. If None, it will be calculated.
        mask (GeoSeries or geometric object, optional): The GeoSeries (elementwise) or geometric object to test if each flow is within. Default is None, which means all flows are used.
        **kwargs: Additional keyword arguments for pairwise_distances function.

    Returns:
        tuple: A tuple containing:
            - flow_num (int): Number of flows.
            - density (float): Second-order density of flows.
            - interval_num (int): Number of distance intervals.
            - mask (np.ndarray): Boolean mask for flows.
            - dis_matrix (np.ndarray): Distance matrix between flows.
    """
    flow_num = fdf.shape[0]
    bounds = fdf.total_bounds
    diagonal_length = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    interval_num = int(diagonal_length // dr)
    
    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance, **kwargs)
        
    if mask is not None:
        mask = fdf.within(mask).values
        density = _second_order_density(dis_matrix=dis_matrix, distance=distance, k=k, mask=mask)
    else:
        mask = np.ones(flow_num, dtype=bool)
        density = _second_order_density(dis_matrix=dis_matrix, distance=distance, k=k)
    
    available_flow_num = np.count_nonzero(mask)
    return available_flow_num, density, interval_num, mask, dis_matrix


def k_func(fdf: FlowDataFrame, dr, k=1, distance='max', dis_matrix: np.ndarray=None, mask=None, **kwargs):
    """
    Calculate the K function for flow data.

    The K function is a measure of spatial clustering or dispersion in point patterns,
    adapted here for flow data.

    Parameters:
        fdf (FlowDataFrame): The input flow dataframe.
        dr (float): The step size for distance intervals.
        k (int, optional): The Nth nearest neighbors for each flow to calculate the density. Default is 1.
        distance (str, optional): The distance metric to use. Default is 'max'.
        dis_matrix (np.ndarray, optional): Pre-computed distance matrix. If None, it will be calculated.
        mask (GeoSeries or geometric object, optional): The GeoSeries (elementwise) or geometric object to test if each flow is within. Default is None, which means all flows are used.
        **kwargs: Additional keyword arguments for pairwise_distances function.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - r_list (np.ndarray): Array of distance values.
            - kr_list (np.ndarray): Array of K function values corresponding to each distance.

    Note:
        The K function for flows is calculated as K(r) = (number of flows within distance r) / (flow density),
        where flow density is the second-order density of flows in the study area.
        
    Reference:
        Tao R, Thill J C. Spatial cluster detection in spatial flow data[J]. Geographical Analysis, 2016,48(4):355-372.
    """
    flow_num, density, interval_num, mask, dis_matrix = _params_for_kl_func(fdf, dr, k=k,  
                                                                            distance=distance, 
                                                                            dis_matrix=dis_matrix, 
                                                                            mask=mask, **kwargs)

    r_list, kr_list = np.zeros(interval_num), np.zeros(interval_num)
    for i in range(1, interval_num+1):
        flow_within_circle = dis_matrix[mask] <= (dr * i)
        flow_within_circle_num = np.count_nonzero(flow_within_circle) - flow_num
        kr = flow_within_circle_num / (flow_num * density)
        kr_list[i-1] = kr
        r_list[i-1] = dr * i
    return r_list, kr_list


def l_func(fdf: FlowDataFrame, dr, k=1, distance='max', dis_matrix: np.ndarray=None, mask=None, **kwargs):
    """
    Calculate the L function for flow data.

    The L function is a variant of Ripley's K function, transformed to have a 
    constant variance and expectation of zero under complete spatial randomness.

    Parameters:
        fdf (FlowDataFrame): The input flow dataframe.
        dr (float): The step size for distance intervals.
        k (int, optional): The Nth nearest neighbors for each flow to calculate the density. Default is 1.
        distance (str, optional): The distance metric to use. Default is 'max'.
        dis_matrix (np.ndarray, optional): Pre-computed distance matrix. If None, it will be calculated.
        mask (GeoSeries or geometric object, optional): The GeoSeries (elementwise) or geometric object to test if each flow is within. Default is None, which means all flows are used.
        **kwargs: Additional keyword arguments for pairwise_distances function.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - r_list (np.ndarray): Array of distance values.
            - lr_list (np.ndarray): Array of L function values corresponding to each distance.

    Note:
        The L function is defined as L(r) = sqrt(K(r) / pi) - r, where K(r) is the K function.
        Positive values of L(r) indicate clustering, while negative values indicate dispersion.
    
    Reference:
        Shu, H., et al., 2021. L-function of geographical flows, International Journal of Geographical Information Science, 35 (4), 689–716.
    """
    flow_num, density, interval_num, mask, dis_matrix = _params_for_kl_func(fdf, dr, k=k,  
                                                                            distance=distance, 
                                                                            dis_matrix=dis_matrix, 
                                                                            mask=mask, **kwargs)
    
    r_list, lr_list = np.zeros(interval_num), np.zeros(interval_num)
    for i in range(1, interval_num+1):
        flow_within_circle = dis_matrix[mask] <= (dr * i)
        flow_within_circle_num = np.count_nonzero(flow_within_circle)
        kr = flow_within_circle_num / (flow_num * density)
        lr = np.power(kr / np.square(np.pi), 1/4) - (dr * i)
        lr_list[i-1] = lr
        r_list[i-1] = dr * i
    return r_list, lr_list


def local_l_func(fdf, r, distance='max', dis_matrix: np.ndarray=None, **kwargs):
    """
    Calculate the local L function for flow data.

    The local L function is a localized version of the L function, computed for each individual flow.

    Parameters:
        fdf (FlowDataFrame): The input flow dataframe.
        r (float): The radius for which to calculate the local L function.
        distance (str, optional): The distance metric to use. Default is 'max'.
        dis_matrix (np.ndarray, optional): Pre-computed distance matrix. If None, it will be calculated.
        **kwargs: Additional keyword arguments for pairwise_distances function.

    Returns:
        np.ndarray: An array of local L function values for each flow.

    Note:
        The local L function helps identify local clustering or dispersion patterns around individual flows.
        Positive values indicate local clustering, while negative values indicate local dispersion.
    
    Reference:
        Shu, H., et al., 2021. L-function of geographical flows, International Journal of Geographical Information Science, 35 (4), 689–716.
    """
    flow_num = fdf.shape[0]
    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance, **kwargs)
    density = second_order_density(fdf, distance=distance, dis_matrix=dis_matrix)

    llrs = np.zeros(flow_num)
    for i in range(flow_num):
        flow_within_circle = dis_matrix[i] <= r
        flow_within_circle_num = np.count_nonzero(flow_within_circle)
        llr = np.power(flow_within_circle_num / (np.square(np.pi)*density), 1/4) - r
        llrs[i-1] = llr
    return llrs


