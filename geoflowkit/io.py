import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, MultiPoint

from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries
from geoflowkit.flowdataframe import FlowDataFrame


def flows_from_od(o, d, crs=None):
    """Create a FlowSeries from origin-destination point pairs.

    Parameters
    ----------
    o : array-like
        Origin points as a 2D array of coordinates (n_points x 2)
    d : array-like
        Destination points as a 2D array of coordinates (n_points x 2)
    crs : str, optional
        Coordinate reference system identifier

    Returns
    -------
    FlowSeries
        A series of Flow objects connecting each origin-destination pair

    Raises
    ------
    ValueError
        If input arrays are not 2D or have mismatched lengths
    """
    o_points = np.asarray(o, dtype=np.float64)
    d_points = np.asarray(d, dtype=np.float64)

    if o_points.ndim != 2:
        raise ValueError("Origin points must be a 2D array")
    if d_points.ndim != 2:
        raise ValueError("Destination points must be a 2D array")
    if len(o_points) != len(d_points):
        raise ValueError("Origin and destination arrays must have the same length")

    # Vectorized flow creation - create Flow objects
    flows = [Flow([o_points[i], d_points[i]]) for i in range(len(o_points))]

    return FlowSeries(flows, crs=crs)


def flows_from_geometry(geometry, crs=None):
    """Create a FlowSeries from geometric objects.

    Parameters
    ----------
    geometry : array-like
        Input geometries that can be:
        - Flow objects
        - LineString objects (uses first and last points)
        - MultiPoint objects (uses first and last points)
    crs : str, optional
        Coordinate reference system identifier

    Returns
    -------
    FlowSeries
        A series of Flow objects created from the input geometries

    Raises
    ------
    ValueError
        If there is a CRS mismatch between input and specified CRS
    TypeError
        If input contains unsupported geometry types or is not array-like
    """
    if (
        hasattr(geometry, "crs")
        or (isinstance(geometry, pd.Series) and hasattr(geometry.array, "crs"))
    ) and crs:
        data_crs = geometry.crs if hasattr(geometry, "crs") else geometry.array.crs
        if not data_crs == crs:
            raise ValueError(
                "CRS mismatch between CRS of the passed geometries "
                "and 'crs'. Use 'GeoSeries.set_crs(crs, "
                "allow_override=True)' to overwrite CRS or "
                "'GeoSeries.to_crs(crs)' to reproject geometries. "
            )
    else:
        data_crs = None

    if pd.api.types.is_list_like(geometry):
        geoms = []
        ls_geoms = []
        mp_geoms = []

        # Separate geometries by type for batch processing
        for geom in geometry:
            if isinstance(geom, Flow):
                geoms.append(geom)
            elif isinstance(geom, LineString):
                ls_geoms.append(geom)
            elif isinstance(geom, MultiPoint):
                mp_geoms.append(geom)
            else:
                raise TypeError(f"Geometry type {type(geom)} can't convert to flow. ")

        # Process LineStrings vectorized
        if ls_geoms:
            n_coords_per_line = np.array([len(g.coords) for g in ls_geoms])
            if np.any(n_coords_per_line > 2):
                warnings.warn(
                    "Some LineString have more than two points. "
                    "Only the first and last points will be kept.",
                    UserWarning
                )
            # Vectorized coordinate extraction
            all_coords = shapely.get_coordinates(ls_geoms)
            cumsum = np.concatenate([[0], np.cumsum(n_coords_per_line)])
            first_points = all_coords[cumsum[:-1]]
            last_points = all_coords[cumsum[1:] - 1]
            ls_flows = [Flow([first_points[i], last_points[i]]) for i in range(len(first_points))]
            geoms = geoms + ls_flows if geoms else ls_flows

        # Process MultiPoints vectorized
        if mp_geoms:
            n_points_per_mp = np.array([len(g.geoms) for g in mp_geoms])
            if np.any(n_points_per_mp > 2):
                warnings.warn(
                    "Some MultiPoint have more than two points. "
                    "Only the first and last points will be kept.",
                    UserWarning
                )
            # Vectorized coordinate extraction
            all_coords = shapely.get_coordinates(mp_geoms)
            cumsum = np.concatenate([[0], np.cumsum(n_points_per_mp)])
            first_points = all_coords[cumsum[:-1]]
            last_points = all_coords[cumsum[1:] - 1]
            mp_flows = [Flow([first_points[i], last_points[i]]) for i in range(len(first_points))]
            geoms = geoms + mp_flows if geoms else mp_flows

        geometry = FlowSeries(geoms, crs=data_crs)
        return geometry
    else:
        raise TypeError("Input geometry must be an array-like object. ")


def read_csv(file_path, use_cols, crs=None, **kwargs) -> FlowDataFrame:
    """
    Read GeoFlow data from a csv file.

    Parameters:
    -----------
    file_path (str): The path to the csv file.
    use_cols (list): The columns to use, which are the columns of the X and Y coordinates of the origin point of the flow,
        and the columns of the X and Y coordinates of the destination point, respectively.
    crs (str or dict, optional): The coordinate reference system of the GeoFlow data.
    **kwargs: Additional arguments passed to pandas.read_csv.

    Returns:
    --------
    FlowDataFrame: The GeoFlow data.
    """
    if len(use_cols) != 4:
        raise ValueError("Invalid columns, should be four columns, like [origin_x, origin_y, destination_x, destination_y]")
    df = pd.read_csv(file_path, **kwargs)

    geometry = flows_from_od(df[use_cols[:2]].values, df[use_cols[2:]].values, crs=crs)

    return FlowDataFrame(df, geometry=geometry, crs=crs)


def read_file(file_path, **kwargs) -> FlowDataFrame:
    """
    Read GeoFlow data from a file.
    
    Parameters:
    -----------
    file_path (str): The path to the file.
    
    Returns:
    --------
    FlowDataFrame: The GeoFlow data.
    """
    gdf = gpd.read_file(file_path, **kwargs)
    gdf['geometry'] = flows_from_geometry(gdf['geometry'])
    
    return FlowDataFrame(gdf, geometry='geometry')


