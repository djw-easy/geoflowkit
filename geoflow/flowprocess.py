import shapely
import warnings
import numpy as np
from shapely.geometry import LineString
from geoflow.flowdataframe import FlowDataFrame
from scipy.spatial.distance import pdist, squareform



def pairwise_distances(fdf: FlowDataFrame, distance='max', metric='euclidean', w1=1, w2=1, length=True):
    """
    Calculate the pairwise distance matrix for flows based on their origin and destination points.

    Parameters:
        fdf (FlowDataFrame): A FlowDataFrame containing the flow data.
        distance (str): The type of distance calculation. Options are 'max', 'min', 'sum', 'mean', or 'weighted'.
        metric (str or function): The distance metric to use. Default is 'euclidean'.
        w1 (float): Weight for origin distances when distance is 'weighted'. Default is 1.
        w2 (float): Weight for destination distances when distance is 'weighted'. Default is 1.
        length (bool): Whether to use flow lengths for weighted distance calculation. Default is True.

    Returns:
        np.ndarray: A square distance matrix of shape [N, N], where N is the number of flows.

    Raises:
        ValueError: If an invalid 'distance' is specified.
    """

    origins = shapely.get_coordinates(fdf.origin_points)
    destinations = shapely.get_coordinates(fdf.dest_points)

    o_dis_matrix = squareform(pdist(origins, metric=metric))
    d_dis_matrix = squareform(pdist(destinations, metric=metric))

    dis_matrix = np.stack([o_dis_matrix, d_dis_matrix], axis=-1)  # Shape: [N, N, 2]

    if distance == 'max':
        return np.max(dis_matrix, axis=-1)
    elif distance =='sum':
        return np.sum(dis_matrix, axis=-1)
    elif distance == 'min':
        return np.min(dis_matrix, axis=-1)
    elif distance == 'mean':
        return np.mean(dis_matrix, axis=-1)
    elif distance == 'weighted':
        if metric != 'euclidean':
            warnings.warn("Weighted distances may be inaccurate when metric is not euclidean.")
        if not length:
            return np.sqrt(w1*o_dis_matrix**2 + w2*d_dis_matrix**2)
        else:
            length = fdf.length.values.reshape(-1, 1)
            return np.sqrt(w1*o_dis_matrix**2 + w2*d_dis_matrix**2 / (length @ length.T))
    else:
        raise ValueError("distance must be 'max', 'sum', 'min', 'mean', or 'weighted'")
    

    
    