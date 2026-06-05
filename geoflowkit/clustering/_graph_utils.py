"""Utilities for converting FlowDataFrame to networkx graph."""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import shapely

from geoflowkit.flowdataframe import FlowDataFrame


def _assign_zones_grid(fdf, cell_size):
    """Assign flows to grid-based zones.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    cell_size : float
        Grid cell size in CRS units.

    Returns
    -------
    o_zones : np.ndarray
        Origin zone IDs for each flow.
    d_zones : np.ndarray
        Destination zone IDs for each flow.
    zone_centroids : dict
        Mapping from zone ID to (x, y) centroid.
    """
    if len(fdf) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    bounds = fdf.total_bounds
    min_x, min_y, max_x, max_y = bounds

    n_cols = max(1, int(np.ceil((max_x - min_x) / cell_size)))
    n_rows = max(1, int(np.ceil((max_y - min_y) / cell_size)))

    origins = shapely.get_coordinates(fdf.o)
    destinations = shapely.get_coordinates(fdf.d)

    o_cols = np.clip(((origins[:, 0] - min_x) / cell_size).astype(int), 0, n_cols - 1)
    o_rows = np.clip(((origins[:, 1] - min_y) / cell_size).astype(int), 0, n_rows - 1)
    o_zones = o_rows * n_cols + o_cols

    d_cols = np.clip(((destinations[:, 0] - min_x) / cell_size).astype(int), 0, n_cols - 1)
    d_rows = np.clip(((destinations[:, 1] - min_y) / cell_size).astype(int), 0, n_rows - 1)
    d_zones = d_rows * n_cols + d_cols

    zone_centroids = {}
    for z in set(o_zones) | set(d_zones):
        r, c = divmod(z, n_cols)
        cx = min_x + (c + 0.5) * cell_size
        cy = min_y + (r + 0.5) * cell_size
        zone_centroids[z] = (cx, cy)

    return o_zones, d_zones, zone_centroids


def _assign_zones_aggregate(fdf):
    """Assign zones by aggregating unique origin/destination coordinates.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.

    Returns
    -------
    o_zones : np.ndarray
        Origin zone IDs for each flow.
    d_zones : np.ndarray
        Destination zone IDs for each flow.
    zone_centroids : dict
        Mapping from zone ID to (x, y) centroid.
    """
    if len(fdf) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), {}

    origins = shapely.get_coordinates(fdf.o)
    destinations = shapely.get_coordinates(fdf.d)

    all_points = np.vstack([origins, destinations])

    unique_points, inverse = np.unique(
        np.round(all_points, decimals=8), axis=0, return_inverse=True
    )

    n_origins = len(origins)
    o_zones = inverse[:n_origins]
    d_zones = inverse[n_origins:]

    zone_centroids = {i: (p[0], p[1]) for i, p in enumerate(unique_points)}

    return o_zones, d_zones, zone_centroids


def _assign_zones_custom(fdf, zone_func):
    """Assign zones using a custom user-provided function.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zone_func : callable
        A function with signature:
            zone_func(fdf) -> (o_zones, d_zones, zone_centroids)
        where:
            - o_zones: np.ndarray of origin zone IDs
            - d_zones: np.ndarray of destination zone IDs
            - zone_centroids: dict mapping zone ID to (x, y) centroid

    Returns
    -------
    o_zones : np.ndarray
        Origin zone IDs for each flow.
    d_zones : np.ndarray
        Destination zone IDs for each flow.
    zone_centroids : dict
        Mapping from zone ID to (x, y) centroid.
    """
    return zone_func(fdf)


def _assign_zones_gdf(fdf, zones):
    """Assign zones using an external GeoDataFrame of zone polygons.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zones : gpd.GeoDataFrame
        Zone polygons with a 'zone_id' column.

    Returns
    -------
    o_zones : np.ndarray
        Origin zone IDs for each flow.
    d_zones : np.ndarray
        Destination zone IDs for each flow.
    zone_centroids : dict
        Mapping from zone ID to (x, y) centroid.
    """
    if fdf.crs != zones.crs:
        zones = zones.to_crs(fdf.crs)

    if 'zone_id' not in zones.columns:
        raise ValueError("zones GeoDataFrame must have a 'zone_id' column")

    origins = fdf.o
    destinations = fdf.d

    o_gdf = gpd.GeoDataFrame(geometry=origins, crs=fdf.crs)
    d_gdf = gpd.GeoDataFrame(geometry=destinations, crs=fdf.crs)

    o_joined = gpd.sjoin(o_gdf, zones[['zone_id', 'geometry']], how='inner', predicate='within')
    d_joined = gpd.sjoin(d_gdf, zones[['zone_id', 'geometry']], how='inner', predicate='within')

    o_zones = o_joined['zone_id'].values
    d_zones = d_joined['zone_id'].values

    if len(o_zones) != len(fdf) or len(d_zones) != len(fdf):
        raise ValueError(
            "Some flows could not be assigned to zones. "
            "Ensure all origin and destination points fall within zone polygons."
        )

    zone_centroids = {}
    for _, row in zones.iterrows():
        zid = row['zone_id']
        centroid = row['geometry'].centroid
        zone_centroids[zid] = (centroid.x, centroid.y)

    return o_zones, d_zones, zone_centroids


def flows_to_graph(fdf, zone_method='grid', cell_size=None,
                   weight='count', zones=None, zone_func=None):
    """Convert a FlowDataFrame to a networkx graph.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zone_method : str, default='grid'
        Zone assignment method:
        - 'grid': Divide space into a regular grid
        - 'aggregate': Aggregate unique OD coordinate pairs
        - 'gdf': Use an external GeoDataFrame of zone polygons
        - 'custom': Use a custom zone assignment function
    cell_size : float, optional
        Grid cell size in CRS units (required when zone_method='grid').
    weight : str, default='count'
        Edge weight type:
        - 'count': Number of flows between zones
        - 'volume': Sum of a column named 'volume' (or flow count if absent)
        - 'length': Sum of flow lengths between zones
    zones : gpd.GeoDataFrame, optional
        Zone polygons (required when zone_method='gdf').
    zone_func : callable, optional
        Custom zone assignment function (required when zone_method='custom').
        Signature: zone_func(fdf) -> (o_zones, d_zones, zone_centroids)

    Returns
    -------
    G : nx.Graph
        An undirected graph where nodes are zones and edges are flows.
    o_zones : np.ndarray
        Origin zone ID for each flow (same order as fdf).
    d_zones : np.ndarray
        Destination zone ID for each flow (same order as fdf).
    zone_centroids : dict
        Mapping from zone ID to (x, y) centroid.

    Raises
    ------
    ValueError
        If required parameters are missing or zone_method is invalid.
    """
    if zone_method == 'grid':
        if cell_size is None:
            raise ValueError("cell_size is required when zone_method='grid'")
        o_zones, d_zones, zone_centroids = _assign_zones_grid(fdf, cell_size)
    elif zone_method == 'aggregate':
        o_zones, d_zones, zone_centroids = _assign_zones_aggregate(fdf)
    elif zone_method == 'gdf':
        if zones is None:
            raise ValueError("zones GeoDataFrame is required when zone_method='gdf'")
        o_zones, d_zones, zone_centroids = _assign_zones_gdf(fdf, zones)
    elif zone_method == 'custom':
        if zone_func is None:
            raise ValueError("zone_func is required when zone_method='custom'")
        o_zones, d_zones, zone_centroids = _assign_zones_custom(fdf, zone_func)
    else:
        raise ValueError(
            f"Invalid zone_method: {zone_method}. "
            f"Must be 'grid', 'aggregate', 'gdf', or 'custom'."
        )

    G = nx.Graph()

    for z, centroid in zone_centroids.items():
        G.add_node(z, pos=centroid)

    if weight == 'volume' and 'volume' in fdf.columns:
        w_values = fdf['volume'].values
    elif weight == 'length':
        w_values = fdf.length.values
    else:
        w_values = np.ones(len(fdf))

    edge_weights = {}
    for i in range(len(fdf)):
        o_z = int(o_zones[i])
        d_z = int(d_zones[i])
        key = (min(o_z, d_z), max(o_z, d_z))
        if key not in edge_weights:
            edge_weights[key] = 0.0
        edge_weights[key] += w_values[i]

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    return G, o_zones, d_zones, zone_centroids


def assign_flow_labels(fdf, o_zones, d_zones, zone_communities):
    """Assign community labels to each flow based on zone communities.

    Flows whose origin and destination belong to the same community get
    that community's label. Cross-community flows get label -1.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    o_zones : np.ndarray
        Origin zone ID for each flow.
    d_zones : np.ndarray
        Destination zone ID for each flow.
    zone_communities : dict
        Mapping from zone ID to community label.

    Returns
    -------
    labels : np.ndarray
        Community label for each flow. -1 indicates cross-community flow.
    """
    labels = np.full(len(fdf), -1, dtype=int)

    for i in range(len(fdf)):
        o_comm = zone_communities.get(int(o_zones[i]), -1)
        d_comm = zone_communities.get(int(d_zones[i]), -1)
        if o_comm == d_comm and o_comm != -1:
            labels[i] = o_comm

    return labels
