import shapely
import warnings
import numpy as np
from geoflowkit.flowdataframe import FlowDataFrame
from scipy.spatial.distance import pdist, squareform


def _is_geographic_crs(crs) -> bool:
    """Check if CRS is geographic (lat/lon).

    Parameters
    ----------
    crs : CRS or None
        Coordinate reference system.

    Returns
    -------
    bool
        True if CRS is geographic, False otherwise.
    """
    if crs is None:
        return False
    try:
        return crs.is_geographic
    except AttributeError:
        return False


def _haversine_pdist(coords: np.ndarray) -> np.ndarray:
    """Calculate pairwise haversine distances for geographic coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with [longitude, latitude] in degrees.

    Returns
    -------
    np.ndarray
        Condensed pairwise distance matrix (N*(N-1)/2,).
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert to radians
    coords_rad = np.radians(coords)

    # Extract lon and lat
    lon1 = coords_rad[:, 0]
    lat1 = coords_rad[:, 1]

    n = len(coords)
    n_pairs = n * (n - 1) // 2
    distances = np.empty(n_pairs)

    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dlon = lon1[j] - lon1[i]
            dlat = lat1[j] - lat1[i]
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1[i]) * np.cos(lat1[j]) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            distances[idx] = R * c
            idx += 1

    return distances


def pairwise_distances(fdf: FlowDataFrame, distance='max', metric='euclidean',
                      w1=1, w2=1, length=True, handle_geographic='warn'):
    """Calculate the pairwise distance matrix for flows based on their origin and destination points.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    distance : str
        The type of distance calculation. Options are:
        - 'max': Maximum of origin and destination distances
        - 'min': Minimum of origin and destination distances
        - 'sum': Sum of origin and destination distances
        - 'mean': Average of origin and destination distances
        - 'weighted': Weighted combination of origin and destination distances
    metric : str or function, optional
        The distance metric to use, by default 'euclidean'.
        Ignored if handle_geographic is not 'euclidean'.
    w1 : float, optional
        Weight for origin distances when distance is 'weighted', by default 1
    w2 : float, optional
        Weight for destination distances when distance is 'weighted', by default 1
    length : bool, optional
        Whether to use flow lengths for weighted distance calculation, by default True
    handle_geographic : str, optional
        How to handle geographic (lat/lon) CRS. Options are:
        - 'warn': Show warning and use haversine distance (default)
        - 'error': Raise ValueError if geographic CRS detected
        - 'euclidean': Use euclidean distance without warning (user responsibility)

    Returns
    -------
    np.ndarray
        A square distance matrix of shape [N, N], where N is the number of flows

    Raises
    ------
    ValueError
        If an invalid 'distance' type is specified or if handle_geographic='error'
        and a geographic CRS is detected.
    """
    origins = shapely.get_coordinates(fdf.o)
    destinations = shapely.get_coordinates(fdf.d)

    # Check CRS
    crs = getattr(fdf, 'crs', None)
    is_geographic = _is_geographic_crs(crs)

    if is_geographic:
        if handle_geographic == 'error':
            raise ValueError(
                "Geographic CRS detected. "
                "Use a projected CRS (e.g., EPSG:3857) for accurate distances, "
                "or set handle_geographic='warn' or 'euclidean'."
            )
        elif handle_geographic == 'warn':
            warnings.warn(
                "Input CRS is geographic (lat/lon). "
                "Using haversine distance for accurate great-circle calculation. "
                "For faster euclidean distance on projected coordinates, "
                "consider converting to a projected CRS (e.g., EPSG:3857)."
            )
            # Use haversine for geographic coordinates
            o_dis = _haversine_pdist(origins)
            d_dis = _haversine_pdist(destinations)
        else:  # 'euclidean'
            o_dis = pdist(origins, metric=metric)
            d_dis = pdist(destinations, metric=metric)
    else:
        o_dis = pdist(origins, metric=metric)
        d_dis = pdist(destinations, metric=metric)

    # Combine origin and destination distances in compressed form, then expand
    if distance == 'max':
        return squareform(np.maximum(o_dis, d_dis))
    elif distance == 'sum':
        return squareform(o_dis + d_dis)
    elif distance == 'min':
        return squareform(np.minimum(o_dis, d_dis))
    elif distance == 'mean':
        return squareform((o_dis + d_dis) / 2)
    elif distance == 'weighted':
        if is_geographic:
            warnings.warn("Weighted distances with haversine may be inaccurate.")
        if not length:
            return np.sqrt(w1 * o_dis ** 2 + w2 * d_dis ** 2)
        else:
            length = fdf.length.values.reshape(-1, 1)
            return np.sqrt(w1 * o_dis ** 2 + w2 * d_dis ** 2 / (length @ length.T))
    else:
        raise ValueError("distance must be 'max', 'sum', 'min', 'mean', or 'weighted'")
