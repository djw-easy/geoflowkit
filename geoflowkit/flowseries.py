import warnings
from typing import Any, Union, overload

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

from geoflowkit.flow import Flow
from geoflowkit.base import FlowBase


class FlowSeries(FlowBase, GeoPandasBase, Series):
    """A Series object that stores only Flow geometry objects.

    FlowSeries extends pandas Series and GeoPandasBase to provide specialized
    functionality for working with flow data (origin-destination pairs).
    All elements in the series must be Flow geometry objects.

    Parameters
    ----------
    data : array-like, optional
        Flow objects or array-like containing Flow objects.
    index : array-like, optional
        Index to assign to the data.
    crs : str or dict, optional
        Coordinate reference system identifier.

    Examples
    --------
    >>> from geoflowkit import Flow, FlowSeries
    >>> flows = [Flow([[0, 0], [1, 1]]), Flow([[1, 2], [3, 4]])]
    >>> fs = FlowSeries(flows)
    >>> len(fs)
    2
    >>> fs.o
    0    POINT (0 0)
    1    POINT (1 2)
    dtype: geometry
    """

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        crs: Union[str, int, None] = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.pop("name", None)
        if data is not None:
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
            # Validate data (skip if already a FlowSeries with matching CRS)
            if not isinstance(data, FlowSeries):
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
    def crs(self, value: Union[str, int, None]) -> None:
        """Set the Coordinate Reference System (CRS) of the FlowSeries.

        Parameters
        ----------
        value : str, int, or None
            The CRS to assign. Can be an EPSG code, WKT string, or PROJ string.

        Raises
        ------
        DeprecationWarning
            If the GeoSeries already has a CRS.
        """
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
    def geometry(self) -> "FlowSeries":
        """Alias for the FlowSeries itself.

        Provided for compatibility with GeoPandasBase.

        Returns
        -------
        FlowSeries
            self.
        """
        return self

    @property
    def _constructor(self) -> type["FlowSeries"]:
        return FlowSeries

    def _constructor_from_mgr(
        self, mgr: SingleBlockManager, axes: Any
    ) -> "FlowSeries":
        """Create a FlowSeries from a pandas SingleBlockManager."""
        assert isinstance(mgr, SingleBlockManager)

        if not isinstance(mgr.blocks[0].dtype, GeometryDtype):
            raise TypeError("All elements must be Flow objects")

        return FlowSeries._from_mgr(mgr, axes)

    def _wrapped_pandas_method(self, mtd: str, *args: Any, **kwargs: Any) -> Any:
        """Wrap a generic pandas method to ensure it returns a FlowSeries."""
        val = getattr(super(), mtd)(*args, **kwargs)
        if isinstance(val, (Series, GeoSeries)):
            val.__class__ = FlowSeries
            val.crs = self.crs
        return val
    
    def _validate_data(self, data: Any) -> Any:
        """Validate that all elements in data are Flow objects.

        Parameters
        ----------
        data : array-like
            Data to validate.

        Returns
        -------
        Any
            The validated data.

        Raises
        ------
        TypeError
            If any element is not a Flow object.
        """
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
    
    def __setitem__(self, key: Any, value: Flow) -> None:
        """Set a value in the FlowSeries.

        Parameters
        ----------
        key : Any
            Index key.
        value : Flow
            The Flow geometry to set.

        Raises
        ------
        TypeError
            If the value is not a Flow object.
        """
        if not isinstance(value, Flow):
            raise TypeError("Can only set Flow objects")
        super().__setitem__(key, value)
    
    def astype(self, dtype: Any, copy: bool = True) -> "FlowSeries":
        """Cast to object dtype only.

        Parameters
        ----------
        dtype : type
            Must be ``object``. FlowSeries cannot be cast to other types.
        copy : bool, default True
            Whether to return a copy.

        Returns
        -------
        FlowSeries
            A copy of the FlowSeries (if ``copy=True``).
        """
        if dtype != object:
            raise TypeError("FlowSeries can only have dtype object")
        return self.copy() if copy else self
    
    def set_crs(
        self,
        crs: Union[Any, None] = None,
        epsg: Union[int, None] = None,
        inplace: bool = False,
        allow_override: bool = False,
    ) -> "FlowSeries":
        """Set the CRS of the FlowSeries.

        Parameters
        ----------
        crs : str, dict, or None, optional
            The CRS to set.
        epsg : int, optional
            EPSG code for the CRS.
        inplace : bool, default False
            If True, modify in place.
        allow_override : bool, default False
            If True, allow overriding an existing CRS.

        Returns
        -------
        FlowSeries
            The FlowSeries with the new CRS, or None if inplace.
        """
        if inplace:
            GeoSeries.set_crs(
                self, crs=crs, epsg=epsg, inplace=True, allow_override=allow_override
            )
            return None
        gs = GeoSeries.set_crs(
            self, crs=crs, epsg=epsg, inplace=False, allow_override=allow_override
        )
        return FlowSeries(gs.values, crs=gs.crs)

    def to_crs(
        self,
        crs: Union[Any, None] = None,
        epsg: Union[int, None] = None,
    ) -> "FlowSeries":
        """Transform flows to a new coordinate reference system.

        Parameters
        ----------
        crs : str, dict, or None, optional
            The target CRS.
        epsg : int, optional
            EPSG code for the target CRS.

        Returns
        -------
        FlowSeries
            A new FlowSeries with transformed geometries.
        """
        import shapely

        gs = GeoSeries.to_crs(self, crs=crs, epsg=epsg)
        flows = [Flow(shapely.get_coordinates(geom)) for geom in gs]
        return FlowSeries(flows, crs=gs.crs)

    def plot(
        self,
        ax: Union[plt.Axes, None] = None,
        C: Union[np.ndarray, None] = None,
        figsize: Union[tuple, None] = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot flows as arrows using matplotlib quiver.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the plot. If None, a new figure
            and axes will be created.
        C : array-like, optional
            Numeric data that defines the arrow colors by colormapping
            via norm and cmap. Does not support explicit colors.
            Must match the number of arrow locations.
        figsize : tuple, optional
            The size of the figure to create in inches as ``(width, height)``.
        **kwargs : dict
            Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.quiver`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes on which the plot was drawn.
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



