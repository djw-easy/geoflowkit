import pandas as pd
import geopandas as gpd


from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.array import flows_from_od, flows_from_geometry



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
    assert len(use_cols) == 4, "Invalid columns, should be four columns, like [origin_x, origin_y, destination_x, destination_y]"
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


