import warnings

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from shapely.geometry import LineString, MultiPoint


from geoflow.flow import Flow
from geoflow.flowseries import FlowSeries


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
    AssertionError
        If input arrays are not 2D
    """
    o_points = np.asarray(o, dtype=np.float64)
    assert o_points.ndim == 2, "Origin points must be a 2D array"
    d_points = np.asarray(d, dtype=np.float64)
    assert d_points.ndim == 2, "Destination points must be a 2D array"
    
    flows = [Flow([o, d]) for o, d in zip(o_points, d_points)]

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
        ls_warning_issued = False
        mp_warning_issued = False
        
        geoms = []
        for geom in geometry:
            if isinstance(geom, Flow):
                geoms.append(geom)
            elif isinstance(geom, LineString):
                if len(geom.coords) > 2:
                    if not ls_warning_issued:
                        warnings.warn("Some LineString have more than two points. Only the first and last points will be kept.", UserWarning)
                        ls_warning_issued = True
                geoms.append(Flow([geom.coords[0], geom.coords[-1]]))
            elif isinstance(geom, MultiPoint):
                coordinates = [point.coords[0] for point in geom.geoms]
                if len(coordinates) > 2:
                    if not mp_warning_issued:
                        warnings.warn("Some MultiPoint have more than two points. Only the first and last points will be kept.", UserWarning)
                        mp_warning_issued = True
                geoms.append(Flow([coordinates[0], coordinates[-1]]))
            else:
                raise TypeError(f"Geometry type {type(geom)} can't convert to flow. ")
        geometry = FlowSeries(geoms, crs=data_crs)
        return geometry
    else:
        raise TypeError("Input geometry must be an array-like object. ")



