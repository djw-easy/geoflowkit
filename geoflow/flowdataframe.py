import json
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import DataFrame
from geopandas import GeoDataFrame
from geopandas.geodataframe import _ensure_geometry

import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point

from geoflow.io import geometry_to_flow


class FlowDataFrame(GeoDataFrame):
    """
    A DataFrame with a geometry column.

    Parameters
    ----------
    data : array-like, dict, DataFrame or GeoDataFrame, optional
        The data to use.
    use_cols (list): The columns to use, which are the columns of the X and Y coordinates of the origin point of the flow, 
        and the columns of the X and Y coordinates of the destination point, respectively.
    crs (str or dict, optional): The coordinate reference system of the GeoFlow data.
    """

    def __init__(self, data=None, *args, geometry=None, use_cols=None, crs=None, **kwargs):
        # Handle internal pandas operations where data is already a GeoDataFrame or has geometry
        if isinstance(data, (GeoDataFrame, FlowDataFrame)) or (
            isinstance(data, DataFrame) and 'geometry' in data.columns
        ):
            geometry = getattr(data, 'geometry', data.get('geometry'))
            geometry = _ensure_geometry(geometry, crs=crs or getattr(data, 'crs', None))
            geometry = geometry_to_flow(geometry)
            kwargs.pop('geometry', None)
            super().__init__(data, *args, geometry=geometry, crs=crs or getattr(data, 'crs', None), **kwargs)
            return

        # Handle normal DataFrame initialization
        if (
            kwargs.get("copy") is None
            and isinstance(data, DataFrame)
            and not isinstance(data, GeoDataFrame)
        ):
            kwargs.update(copy=True)

        # Create geometry from use_cols if provided
        if use_cols is not None:
            if not isinstance(data, DataFrame):
                data = DataFrame(data, *args, **kwargs)
            if not all(col in data.columns for col in use_cols):
                raise ValueError(f"Not all columns in use_cols {use_cols} found in data")
            origin_points = data[use_cols[:2]].values
            dest_points = data[use_cols[2:]].values
            geometry = [LineString([o, d]) for o, d in zip(origin_points, dest_points)]
            super().__init__(data, *args, geometry=geometry, crs=crs, **kwargs)
        super().__init__(data, *args, geometry=geometry, crs=crs, **kwargs)

    def __getitem__(self, key):
        """Override to ensure type preservation during indexing"""
        # Handle column selection
        if isinstance(key, list):
            if 'geometry' not in key:
                key = key + ['geometry']
            result = super().__getitem__(key)
            if isinstance(result, GeoDataFrame):
                result.__class__ = FlowDataFrame
            return result
        return super().__getitem__(key)

    @property
    def _constructor(self):
        return FlowDataFrame

    @property
    def _constructor_sliced(self):
        from pandas import Series
        return Series

    def check_geographic_crs(self, stacklevel):
        """Check CRS and warn if the planar operation is done in a geographic CRS"""
        if self.crs and self.crs.is_geographic:
            warnings.warn(
                "Geometry is in a geographic CRS. Results from are likely incorrect. "
                "Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before this operation.\n",
                UserWarning,
                stacklevel=stacklevel,
            )

    @property
    def volume(self):
        """
        Calculate the volume of the flow space.

        Returns
        -------
        volume : float
            The volume of the flow space.
        """
        self.check_geographic_crs(3)
        bounds = self.total_bounds
        return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

    @property
    def density(self):
        """
        Calculate the density of the flow space.

        Returns
        -------
        density : float
            The density of the flow space.
        """
        return len(self) / self.volume

    @property
    def origin_points(self):
        """
        Get the origin points of the flow.

        Returns
        -------
        origin_points : array-like
            The origin points of the flow.
        """
        return np.array(self.geometry.apply(lambda x: x.coords[0]).tolist())

    @property
    def dest_points(self):
        """
        Get the destination points of the flow.

        Returns
        -------
        dest_points : array-like
            The destination points of the flow.
        """
        return np.array(self.geometry.apply(lambda x: x.coords[1]).tolist())

    def plot(self, kind='arrow', ax=None, column=None, figsize=None, **kwargs):
        """
        Plot the flow data.

        Parameters
        ----------
        kind : str, default 'arrow'
            The type of plot to draw. Options include 'arrow' for flow arrows,
            or any other plot type supported by GeoPandas.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot. If None, a new figure and axes will be created.
        column : str, optional
            The name of the column to be used for coloring the arrows when kind='arrow'.
        figsize : tuple, optional
            The size of the figure to create in inches (width, height).
        **kwargs : dict
            Additional keyword arguments to be passed to the plotting function.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes on which the plot was drawn.

        Raises
        ------
        ValueError
            If the specified column is not found in the FlowDataFrame when kind='arrow'.

        Notes
        -----
        When kind='arrow', this method uses matplotlib's quiver plot to draw flow arrows.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        if kind == 'arrow':
            if column is not None:
                if column not in self.columns:
                    raise ValueError(f"Column '{column}' not found in the FlowDataFrame. ")
                C = self[column].values
            else:
                C = None
            origins = self.origin_points
            destinations = self.dest_points
            u = destinations[:, 0] - origins[:, 0]
            v = destinations[:, 1] - origins[:, 1]

            quiver_kwargs = {
                'angles': 'xy',
                'scale_units': 'xy',
                'scale': 1
            }
            # 使用kwargs更新quiver_kwargs，如果有重复的键，kwargs中的值会覆盖quiver_kwargs中的值
            quiver_kwargs.update(kwargs)

            if C is not None:
                ax.quiver(origins[:, 0], origins[:, 1], u, v, C, **quiver_kwargs)
            else:
                ax.quiver(origins[:, 0], origins[:, 1], u, v, **quiver_kwargs)
            return ax
        else:
            return super().plot(kind=kind, ax=ax, column=column, figsize=figsize, **kwargs)
        
    def to_grid(self, delta_x=None, delta_y=None, x_size=None, y_size=None):
        """Divide the study area into a grid and calculate the grid of the origin and destination points of each flow.

        Parameters:
            delta_x (float): The length of a single grid cell, defaults to None.
            delta_y (float): The width of a single grid cell, defaults to None.
            x_size (int): The number of grids in the x direction, defaults to None.
            y_size (int): The number of grids in the y direction, defaults to None.

        Returns:
            FlowDataFrame: The gridded flow dataset.
        """
        if delta_x is None and delta_y is None:
            if x_size is None or y_size is None:
                raise ValueError("Must specify either delta_x and delta_y, or x_size and y_size.")
            delta_x = (self.bounds.maxx - self.bounds.minx) / x_size
            delta_y = (self.bounds.maxy - self.bounds.miny) / y_size
        elif x_size is None or y_size is None:
            if delta_x is None or delta_y is None:
                raise ValueError("Must specify either delta_x and delta_y, or x_size and y_size.")
            x_size = int((self.bounds.maxx - self.bounds.minx) / delta_x)
            y_size = int((self.bounds.maxy - self.bounds.miny) / delta_y)
        else:
            raise ValueError("Must specify either delta_x and delta_y, or x_size and y_size.")
        
        # Calculate the grid for origin and destination points
        origins = self.origin_points
        destinations = self.dest_points
        self.loc[:, 'o_grid_x'] = ((origins[:, 0] - self.bounds.minx) / delta_x).fillna(-1).astype(int)
        self.loc[:, 'o_grid_y'] = ((origins[:, 1] - self.bounds.miny) / delta_y).fillna(-1).astype(int)
        self.loc[:, 'd_grid_x'] = ((destinations[:, 0] - self.bounds.minx) / delta_x).fillna(-1).astype(int)
        self.loc[:, 'd_grid_y'] = ((destinations[:, 1] - self.bounds.miny) / delta_y).fillna(-1).astype(int)
        
        # Create grid IDs
        self.loc[:, 'o_grid_id'] = self['o_grid_y'] * x_size + self['o_grid_x']
        self.loc[:, 'd_grid_id'] = self['d_grid_y'] * x_size + self['d_grid_x']
        
        return self
    
    def within(self, mask):
        """
        Select the flow data within the given mask.

        Parameters
        ----------
        mask (GeoSeries or geometric object): The GeoSeries (elementwise) or geometric object to test if each flow is within.

        Returns
        -------
        mask (pd.Series): A boolean Series indicating whether each flow is within the mask.
        """
        origins = self.geometry.apply(lambda line: Point(line.coords[0]))
        destinations = self.geometry.apply(lambda line: Point(line.coords[-1]))
        is_start_within = origins.within(mask)
        is_end_within = destinations.within(mask)
        mask = is_start_within & is_end_within
        
        return mask



