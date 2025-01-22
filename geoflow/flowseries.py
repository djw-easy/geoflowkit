import warnings
from typing import Any, Union


import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import Series
from geopandas import GeoSeries
import matplotlib.pyplot as plt
from geopandas.base import GeoPandasBase
from geopandas.array import GeometryArray
from geopandas.array import GeometryDtype
from pandas.core.internals import SingleBlockManager


from geoflow.flow import Flow
from geoflow.base import FlowBase


class FlowSeries(FlowBase, GeoPandasBase, Series):
    """
    A Series object designed to store Flow objects exclusively.
    All data in this series must be instances of Flow class.
    """

    def __init__(self, data=None, index=None, crs=None, **kwargs):
        name = kwargs.pop("name", None)
        if data is not None:
            # Skip validation if input is already a validated FlowSeries
            if isinstance(data, FlowSeries):
                raise ValueError("Input data is already a validated FlowSeries")
            
            if (
                hasattr(data, "crs")
                or (isinstance(data, pd.Series) and hasattr(data.array, "crs"))
            ) and crs:
                data_crs = data.crs if hasattr(data, "crs") else data.array.crs
                if not data_crs:
                    # make a copy to avoid setting CRS to passed GeometryArray
                    data = data.copy()
                else:
                    if not data_crs == crs:
                        raise ValueError(
                            "CRS mismatch between CRS of the passed geometries "
                            "and 'crs'. Use 'GeoSeries.set_crs(crs, "
                            "allow_override=True)' to overwrite CRS or "
                            "'GeoSeries.to_crs(crs)' to reproject geometries. "
                        )
            # Validate data
            data = self._validate_data(data)
            
            if isinstance(data, Flow):
                # fix problem for scalar geometries passed, ensure the list of
                # scalars is of correct length if index is specified
                n = len(index) if index is not None else 1
                data = [data] * n
            
            s = pd.Series(data, index=index, name=name, **kwargs)
            
            index = s.index
            name = s.name
            geometry_array = GeometryArray(np.asarray(data), crs=crs)
        
        # Initialize parent Series with geometry array
        super().__init__(data=geometry_array, index=index, name=name, **kwargs)
        if not self.crs:
            self.crs = crs

    @GeoPandasBase.crs.setter
    def crs(self, value):
        """Set the Coordinate Reference System (CRS) of the FlowSeries."""
        if self.crs is not None:
            warnings.warn(
                "Overriding the CRS of a GeoSeries that already has CRS. "
                "This unsafe behavior will be deprecated in future versions. "
                "Use GeoSeries.set_crs method instead.",
                stacklevel=2,
                category=DeprecationWarning,
            )
        self.geometry.values.crs = value

    @property
    def geometry(self) -> 'FlowSeries':
        return self
    
    @property
    def _constructor(self):
        return FlowSeries

    def _constructor_from_mgr(self, mgr, axes):
        assert isinstance(mgr, SingleBlockManager)

        if not isinstance(mgr.blocks[0].dtype, GeometryDtype):
            raise TypeError("All elements must be Flow objects")
        
        return FlowSeries._from_mgr(mgr, axes)
    
    def _wrapped_pandas_method(self, mtd, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        print('warp')
        val = getattr(super(), mtd)(*args, **kwargs)
        if isinstance(val, (Series, GeoSeries)):
            val.__class__ = FlowSeries
            val.crs = self.crs
        return val
    
    def _validate_data(self, data):
        """Validate that all elements in data are Flow objects"""
        if isinstance(data, pd.Series):
            if not all(isinstance(item, Flow) for item in data):
                raise TypeError("All elements must be Flow objects")
            return data
        elif isinstance(data, (list, np.ndarray, tuple, GeometryArray)):
            if not all(isinstance(item, Flow) for item in data):
                raise TypeError("All elements must be Flow objects")
            return data
        elif isinstance(data, Flow):
            return data
        else:
            raise TypeError("Data must be Flow object(s)")
    
    def __setitem__(self, key, value):
        """Override to ensure only Flow objects can be set"""
        if not isinstance(value, Flow):
            raise TypeError("Can only set Flow objects")
        super().__setitem__(key, value)
    
    def astype(self, dtype, copy=True):
        """Override astype to prevent type conversion"""
        if dtype != object:
            raise TypeError("FlowSeries can only have dtype object")
        return self.copy() if copy else self
    
    def set_crs(
        self,
        crs: Union[Any, None] = None,
        epsg: Union[int, None] = None,
        inplace: bool = False,
        allow_override: bool = False,
    ) -> 'FlowSeries':
        return GeoSeries.set_crs(
            self,
            crs=crs,
            epsg=epsg,
            inplace=inplace,
            allow_override=allow_override,
        )
        
    def to_crs(self, crs: Union[Any, None] = None, epsg: Union[int, None] = None) -> 'FlowSeries':
        gs = GeoSeries.to_crs(self, crs=crs, epsg=epsg)
        return FlowSeries(
            gs, crs=gs.crs
        )
    
    def plot(self, ax=None, C=None, figsize=None, **kwargs) -> plt.Axes:
        """
        Plot the flow data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot. If None, a new figure and axes will be created.
        C : 1D or 2D array-like, optional
            Numeric data that defines the arrow colors by colormapping via norm and cmap.
            This does not support explicit colors. If you want to set colors directly, use color instead. 
            The size of C must match the number of arrow locations.
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
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        from shapely import get_coordinates
        origins = get_coordinates(self.o)
        destinations = get_coordinates(self.d)
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



