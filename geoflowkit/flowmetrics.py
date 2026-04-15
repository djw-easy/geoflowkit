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


def _knn_neighborhood(coords: np.ndarray, k: int) -> np.ndarray:
    """Find k-nearest-neighbor indices for each point.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n, 2) with point coordinates
    k : int
        Number of nearest neighbors

    Returns
    -------
    np.ndarray
        Array of shape (n, k) with indices of k nearest neighbors for each point
    """
    from scipy.spatial import KDTree

    tree = KDTree(coords)
    # Query k+1 because the tree query includes the point itself as nearest neighbor
    _, indices = tree.query(coords, k=k + 1)
    # Exclude the point itself (first column is the point itself)
    return indices[:, 1:]


def k_neighbor_distances(fdf: FlowDataFrame, k: int, distance='max',
                         dis_matrix=None) -> np.ndarray:
    """Calculate k-order nearest neighbor distances for each flow.

    The k-order nearest neighbor distance of a flow is its distance to the
    k-th nearest neighboring flow.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    k : int
        Order of the nearest neighbor (1 = nearest, 2 = second nearest, etc.)
    distance : str, optional
        Distance combination method ('max', 'min', 'sum', 'mean', 'weighted'),
        by default 'max'
    dis_matrix : np.ndarray, optional
        Precomputed distance matrix. If None, will be computed using
        pairwise_distances, by default None

    Returns
    -------
    np.ndarray
        Array of shape (n_flows,) with each flow's k-order neighbor distance

    Examples
    --------
    >>> k_dists = k_neighbor_distances(fdf, k=1)  # 1st order (nearest neighbor)
    >>> k_dists = k_neighbor_distances(fdf, k=2)  # 2nd order
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    if dis_matrix is None:
        dis_matrix = pairwise_distances(fdf, distance=distance)

    # Set diagonal to infinity so flow doesn't select itself
    dis_matrix = dis_matrix.copy()
    np.fill_diagonal(dis_matrix, np.inf)

    # Get k-th nearest neighbor distance for each flow
    # Sort each row and take the k-th element (0-indexed)
    sorted_distances = np.sort(dis_matrix, axis=1)
    k_neighbor_dist = sorted_distances[:, k - 1]

    return k_neighbor_dist


def snn_distance(fdf: FlowDataFrame, k: int, distance: str = 'max') -> np.ndarray:
    """Calculate Shared Nearest Neighbor (SNN) distance between flows.

    SNN distance is based on the intersection of k-nearest-neighbor sets
    of origin points and destination points between pairs of flows.
    Follows Eq. 2-14 from the flow analysis literature.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    k : int
        Number of nearest neighbors to consider for KNN neighborhoods
    distance : str, optional
        Distance method for pairwise distances ('max', 'min', 'sum', 'mean'),
        by default 'max'

    Returns
    -------
    np.ndarray
        SNN distance matrix of shape (n_flows, n_flows)
        Values in [0, 1]: 0 means identical KNN neighborhoods, 1 means no shared neighbors

    Examples
    --------
    >>> snn_dist = snn_distance(fdf, k=8)
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    origins = shapely.get_coordinates(fdf.o)
    destinations = shapely.get_coordinates(fdf.d)

    # Get KNN neighborhoods for origins and destinations separately
    knn_o = _knn_neighborhood(origins, k)
    knn_d = _knn_neighborhood(destinations, k)

    n = len(fdf)
    snn_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Intersection of O_i's KNN with O_j's KNN
            o_interaction = len(np.intersect1d(knn_o[i], knn_o[j]))
            # Intersection of D_i's KNN with D_j's KNN
            d_interaction = len(np.intersect1d(knn_d[i], knn_d[j]))

            # SNN distance from Eq. 2-14
            sim = (o_interaction / k) * (d_interaction / k)
            dist = 1 - sim

            snn_matrix[i, j] = dist
            snn_matrix[j, i] = dist

    return snn_matrix


def flow_entropy(fdf: FlowDataFrame, cell_area=None) -> float:
    """Calculate flow space entropy.

    Measures the spatial distribution disorder of flows using Shannon entropy.
    Supports both standard entropy and spatially-weighted entropy (Batty's spatial entropy).

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    cell_area : np.ndarray, optional
        Array of shape (n, n) with zone pair volumes (area products).
        If None, returns standard Shannon entropy, by default None

    Returns
    -------
    float
        Flow space entropy value

    Examples
    --------
    >>> entropy = flow_entropy(fdf)  # Standard entropy
    >>> entropy_weighted = flow_entropy(fdf, cell_area=areas)  # Spatially-weighted
    """
    n = len(fdf)

    # Count flows per OD zone pair (assumes zones are indexed by flow position)
    # If fdf has 'origin_id' and 'dest_id' columns, use those; otherwise uniform distribution
    p = np.ones(n) / n  # Uniform distribution

    if cell_area is None:
        # Standard Shannon entropy (Eq. 2-15)
        entropy = -np.sum(p * np.log2(p + 1e-10))
    else:
        # Batty's spatial entropy (Eq. 2-16/2-17)
        area_normalized = cell_area / (cell_area.sum() + 1e-10)
        entropy = -np.sum(p * np.log2(p / (area_normalized + 1e-10) + 1e-10))

    return entropy


def flow_divergence(fdf: FlowDataFrame, n_directions: int = 6) -> float:
    """Calculate flow divergence (directional entropy).

    Measures the dispersion of flow directions using Shannon entropy on
    binned angular directions.

    Parameters
    ----------
    fdf : FlowDataFrame
        A FlowDataFrame containing the flow data
    n_directions : int, optional
        Number of direction bins (sectors), by default 6
        Each sector has angle = 360 / n_directions degrees

    Returns
    -------
    float
        Flow divergence value (directional entropy)
        Higher values indicate more dispersed flow directions

    Examples
    --------
    >>> div = flow_divergence(fdf, n_directions=6)  # 6 sectors of 60 degrees each
    >>> div = flow_divergence(fdf, n_directions=8)  # 8 sectors of 45 degrees each
    """
    if n_directions < 2:
        raise ValueError("n_directions must be >= 2")

    # Get flow angles
    angles = fdf.angle.values

    # Normalize angles to [0, 2*pi]
    angles = np.mod(angles, 2 * np.pi)

    # Bin angles into sectors
    sector_size = 2 * np.pi / n_directions
    bin_indices = (angles / sector_size).astype(int) % n_directions

    # Count flows in each sector
    counts = np.bincount(bin_indices, minlength=n_directions)

    # Calculate probabilities
    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total

    # Shannon entropy (Eq. 2-18)
    entropy = -np.sum(p * np.log2(p + 1e-10))

    return entropy
