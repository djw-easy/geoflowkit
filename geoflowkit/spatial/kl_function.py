import numpy as np
from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.flowprocess import pairwise_distances
from geoflowkit.spatial.utils import _second_order_density, second_order_density


def _params_for_kl_func(fdf: FlowDataFrame, dr, k=1, distance='max', dis_matrix=None, mask=None, **kwargs):
    """Calculate parameters for K and L functions.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe
    dr : float
        The step size for distance intervals
    k : int, optional
        The Nth nearest neighbors for each flow to calculate the density, by default 1
    distance : str, optional
        The distance metric to use, by default 'max'
    dis_matrix : np.ndarray, optional
        Pre-computed distance matrix. If None, it will be calculated
    mask : GeoSeries or geometric object, optional
        The GeoSeries (elementwise) or geometric object to test if each flow is within.
        If None, all flows are used
    **kwargs
        Additional keyword arguments for pairwise_distances function

    Returns
    -------
    tuple
        A tuple containing:
        - flow_num (int): Number of flows
        - density (float): Second-order density of flows
        - interval_num (int): Number of distance intervals
        - mask (np.ndarray): Boolean mask for flows
        - dis_matrix (np.ndarray): Distance matrix between flows
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
    """Calculate the K function for flow data.

    The K function is a measure of spatial clustering or dispersion in point patterns,
    adapted here for flow data.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe
    dr : float
        The step size for distance intervals
    k : int, optional
        The Nth nearest neighbors for each flow to calculate the density, by default 1
    distance : str, optional
        The distance metric to use, by default 'max'
    dis_matrix : np.ndarray, optional
        Pre-computed distance matrix. If None, it will be calculated
    mask : GeoSeries or geometric object, optional
        The GeoSeries (elementwise) or geometric object to test if each flow is within.
        If None, all flows are used
    **kwargs
        Additional keyword arguments for pairwise_distances function

    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - r_list (np.ndarray): Array of distance values
        - kr_list (np.ndarray): Array of K function values corresponding to each distance

    Notes
    -----
    The K function for flows is calculated as:
    K(r) = (number of flows within distance r) / (flow density)
    where flow density is the second-order density of flows in the study area

    References
    ----------
    Tao R, Thill J C. Spatial cluster detection in spatial flow data, 
    Geographical Analysis, 2016, 48 (4): 355-372.
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
    """Calculate the L function for flow data.

    The L function is a variant of Ripley's K function, transformed to have a
    constant variance and expectation of zero under complete spatial randomness.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe
    dr : float
        The step size for distance intervals
    k : int, optional
        The Nth nearest neighbors for each flow to calculate the density, by default 1
    distance : str, optional
        The distance metric to use, by default 'max'
    dis_matrix : np.ndarray, optional
        Pre-computed distance matrix. If None, it will be calculated
    mask : GeoSeries or geometric object, optional
        The GeoSeries (elementwise) or geometric object to test if each flow is within.
        If None, all flows are used
    **kwargs
        Additional keyword arguments for pairwise_distances function

    Returns
    -------
    tuple
        A tuple containing two numpy arrays:
        - r_list (np.ndarray): Array of distance values
        - lr_list (np.ndarray): Array of L function values corresponding to each distance

    Notes
    -----
    The L function is defined as:
    L(r) = sqrt(K(r) / pi) - r
    where K(r) is the K function.
    Positive values of L(r) indicate clustering, while negative values indicate dispersion.

    References
    ----------
    Shu, H., et al., 2021. L-function of geographical flows,
    International Journal of Geographical Information Science, 35 (4), 689–716.
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
    """Calculate the local L function for flow data.

    The local L function is a localized version of the L function, computed for each individual flow.

    Parameters
    ----------
    fdf : FlowDataFrame
        The input flow dataframe
    r : float
        The radius for which to calculate the local L function
    distance : str, optional
        The distance metric to use, by default 'max'
    dis_matrix : np.ndarray, optional
        Pre-computed distance matrix. If None, it will be calculated
    **kwargs
        Additional keyword arguments for pairwise_distances function

    Returns
    -------
    np.ndarray
        An array of local L function values for each flow

    Notes
    -----
    The local L function helps identify local clustering or dispersion patterns around individual flows.
    Positive values indicate local clustering, while negative values indicate local dispersion.

    References
    ----------
    Shu, H., et al., 2021. L-function of geographical flows,
    International Journal of Geographical Information Science, 35 (4), 689–716.
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


