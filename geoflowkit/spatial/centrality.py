"""Centrality metrics for flow data to quantify location irreplaceability."""

import numpy as np
import pandas as pd
import geopandas as gpd

from geoflowkit import FlowDataFrame


def _i_index_func(flow_lengths: np.ndarray, alpha: float) -> dict:
    """Calculate I-index for a single zone.

    Parameters
    ----------
    flow_lengths : np.ndarray
        Array of flow lengths (in meters) within the zone.
    alpha : float
        Conversion factor.

    Returns
    -------
    dict
        Dictionary containing I_index, flow_count, and total_length (in km).
    """
    if len(flow_lengths) == 0:
        return {'I_index': 0, 'flow_count': 0, 'total_length': 0.0}

    total_length = np.sum(flow_lengths) / 1000  # Convert to km
    flow_count = len(flow_lengths)

    # Sort in descending order
    sorted_lengths = np.sort(flow_lengths)[::-1]

    # Find maximum i where at least i flows have length >= alpha * i
    i_idx = 0
    for rank, length in enumerate(sorted_lengths, start=1):
        if rank <= length / alpha:
            i_idx = rank
        else:
            break

    return {
        'I_index': i_idx,
        'flow_count': flow_count,
        'total_length': total_length
    }


def i_index(fdf: FlowDataFrame, zones: gpd.GeoDataFrame,
            alpha: float = None, od_type: str = 'd') -> gpd.GeoDataFrame:
    """Calculate I-index (irreplaceability index) for each zone.

    The I-index quantifies the irreplaceability of a location based on flows,
    combining flow volume and flow length into a single metric following
    the H-index principle.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zones : gpd.GeoDataFrame
        Zone polygons with geometry column and a zone_id column.
        The zone_id column should uniquely identify each zone.
    alpha : float, optional
        Conversion factor. If None, it is automatically calculated as:
        alpha = median(median_flow_length_per_zone) / median(flow_count_per_zone)
    od_type : str, default='d'
        Which point to use for spatial join:
        - 'd': Use destination points (flows ending in a zone)
        - 'o': Use origin points (flows starting from a zone)

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with each zone's I-index result, containing:
        - zone_id: Zone identifier
        - I_index: The irreplaceability index
        - flow_count: Number of flows in the zone
        - total_length: Total flow length in km
        - alpha: The conversion factor used
        - geometry: Zone polygon geometry

    Notes
    -----
    I-index definition:
    The I-index of a location is the maximum value of i such that at least i flows
    with a length of at least α * i meters have reached this location, where α is
    the conversion factor.

    The higher the I-index, the more irreplaceable the location - it attracts
    many flows with long distances.

    References
    ----------
    [1] Wang, X., Chen, J., Pei, T.*, Song, C., Liu, Y., Shu, H., … Chen, X. (2021). 
        I-index for quantifying an urban location’s irreplaceability. 
        Computers, Environment and Urban Systems, 90, Article 101711.
    [2] Original implementation: https://github.com/Lreis-GeoFlow-Lab/Flow_I_index

    Examples
    --------
    >>> from geoflowkit import read_file
    >>> from geoflowkit.spatial import i_index
    >>> import geopandas as gpd

    >>> fdf = read_file('flows.gpkg')
    >>> zones = gpd.read_file('zones.gpkg')
    >>> result = i_index(fdf, zones)

    >>> # With custom alpha
    >>> result = i_index(fdf, zones, alpha=1000.0)

    >>> # Using origin points instead of destination
    >>> result = i_index(fdf, zones, od_type='o')
    """
    # Ensure CRS consistency
    if fdf.crs != zones.crs:
        zones = zones.to_crs(fdf.crs)

    # Check for zone_id column
    if 'zone_id' not in zones.columns:
        raise ValueError("zones must have a 'zone_id' column")

    # Calculate flow lengths
    fdf = fdf.copy()
    fdf['flow_length'] = fdf.length

    # Create point GeoDataFrame based on od_type
    if od_type.lower() == 'd':
        points = fdf.d  # Destination points
    elif od_type.lower() == 'o':
        points = fdf.o  # Origin points
    else:
        raise ValueError(f"od_type must be 'd' or 'o', got {od_type}")

    points_gdf = gpd.GeoDataFrame(
        fdf[['flow_length']].copy(),
        geometry=points,
        crs=fdf.crs
    )

    # Spatial join: assign flows to zones
    flows_with_zones = gpd.sjoin(
        points_gdf, zones[['zone_id', 'geometry']],
        how='inner',
        predicate='within'
    )

    # Auto-calculate alpha if not provided
    if alpha is None:
        grouped = flows_with_zones.groupby('zone_id')
        medians = grouped['flow_length'].median()
        counts = grouped.size()
        alpha = np.median(medians.values) / np.median(counts.values)

    # Calculate I-index for each zone
    results = []
    for zone_id in zones['zone_id']:
        zone_flows = flows_with_zones[flows_with_zones['zone_id'] == zone_id]
        if len(zone_flows) > 0:
            lengths = zone_flows['flow_length'].values
        else:
            lengths = np.array([])
        result = _i_index_func(lengths, alpha)
        result['zone_id'] = zone_id
        results.append(result)

    # Build result DataFrame with I-index results
    i_index_df = pd.DataFrame(results)

    # Merge with original zones to preserve all zone attributes
    result_df = zones.merge(i_index_df, on='zone_id', how='left')

    # Add alpha column
    result_df['alpha'] = alpha

    # Fill NaN values for zones with no flows
    result_df['I_index'] = result_df['I_index'].fillna(0).astype(int)
    result_df['flow_count'] = result_df['flow_count'].fillna(0).astype(int)
    result_df['total_length'] = result_df['total_length'].fillna(0.0)

    # Reorder columns: put I-index related columns after zone_id
    id_cols = ['zone_id', 'I_index', 'flow_count', 'total_length', 'alpha']
    other_cols = [c for c in result_df.columns if c not in id_cols and c != 'geometry']
    result_df = result_df[id_cols + other_cols + ['geometry']]

    return result_df
