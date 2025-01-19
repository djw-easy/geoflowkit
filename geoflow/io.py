import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def geometry_to_flow(geometry):
    """
    Convert a GeoDataFrame's geometry column to a flow-compatible format.

    This function checks if all geometries are LineString type and processes any LineStrings 
    with more than two points by keeping only the first and last points.

    Parameters:
    -----------
    geometry : GeoSeries
        The geometry column of a GeoDataFrame containing flow data.

    Returns:
    --------
    GeoSeries
        A GeoSeries with all geometries as LineStrings with exactly two points.

    Raises:
    -------
    ValueError
        If any geometry is not a LineString.

    Warns:
    ------
    UserWarning
        If any LineString has more than two points.
    """
    if not all(isinstance(geom, LineString) for geom in geometry):
        raise ValueError("All geometries must be LineString type")

    warning_issued = False
    for idx, geom in enumerate(geometry):
        if len(geom.coords) > 2:
            if not warning_issued:
                import warnings
                warnings.warn("Some LineStrings have more than two points. Only the first and last points will be kept.", UserWarning)
                warning_issued = True
            geometry.iloc[idx] = LineString([geom.coords[0], geom.coords[-1]])

    return geometry


def read_csv(file_path, use_cols, crs=None, **kwargs):
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
    from geoflow.flowdataframe import FlowDataFrame
    assert len(use_cols) == 4, "Invalid columns, should be four columns, like [origin_x, origin_y, destination_x, destination_y]"
    df = pd.read_csv(file_path, **kwargs)
    geometry = [LineString([o, d]) for o, d in zip(df[use_cols[:2]].values, df[use_cols[2:]].values)]
    
    return FlowDataFrame(df, geometry=geometry, crs=crs)


def read_file(file_path):
    """
    Read GeoFlow data from a file.
    
    Parameters:
    -----------
    file_path (str): The path to the file.
    
    Returns:
    --------
    FlowDataFrame: The GeoFlow data.
    """
    from geoflow.flowdataframe import FlowDataFrame
    gdf = gpd.read_file(file_path)
    gdf['geometry'] = geometry_to_flow(gdf['geometry'])
    
    return FlowDataFrame(gdf)


