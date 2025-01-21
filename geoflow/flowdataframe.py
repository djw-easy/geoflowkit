import json
import warnings
from functools import wraps

import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from geopandas.base import GeoPandasBase
from geopandas.base import is_geometry_type
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryArray, GeometryDtype


from geoflow.flow import Flow
from geoflow.flowseries import FlowSeries


from packaging.version import Version
PANDAS_GE_30 = Version(pd.__version__) >= Version("3.0.0.dev0")
crs_mismatch_error = (
    "CRS mismatch between CRS of the passed geometries "
    "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
    "allow_override=True)' to overwrite CRS or "
    "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
)


def _ensure_geometry(data, crs=None):
    """
    Ensure the data is of geometry dtype or converted to it.

    If input is a (Geo)Series, output is a GeoSeries, otherwise output
    is GeometryArray.

    If the input is a GeometryDtype with a set CRS, `crs` is ignored.
    """
    if is_geometry_type(data):
        if isinstance(data, FlowSeries):
            return data
        if isinstance(data, (Series, GeoSeries)):
            data = FlowSeries(data)
        if data.crs is None and crs is not None:
            # Avoids caching issues/crs sharing issues
            data = data.copy()
            if isinstance(data, GeometryArray):
                data.crs = crs
            else:
                data.array.crs = crs
            data = FlowSeries(data)
        return data
    else:
        if isinstance(data, FlowSeries):
            return data
        elif isinstance(data, (Series, GeoSeries)):
            data = FlowSeries(data, crs=crs)
            return data
        elif isinstance(data, (list, tuple, np.ndarray)):
            data = FlowSeries(data, crs=crs)
            return data
        elif isinstance(data, Flow):
            return FlowSeries([data], crs=crs)
        else:
            raise TypeError("All elements must be Flow objects")


class FlowDataFrame(DataFrame, GeoPandasBase):
    """
    A DataFrame subclass for handling flow data with geographic operations.

    FlowDataFrame extends pandas DataFrame and GeoPandasBase to provide specialized
    functionality for working with flow data that has geographic components.
    Each row represents a flow with origin and destination geometries.

    Parameters
    ----------
    data : dict, list, pandas.DataFrame, optional
        Input data containing flow information
    geometry : str, array-like, optional
        Column name or array containing geometry data
    crs : pyproj.CRS, optional
        Coordinate reference system for the geometries
    *args, **kwargs
        Additional arguments passed to pandas.DataFrame constructor

    Notes
    -----
    - Most functionality is adapted from GeoDataFrame
    - Direct inheritance from GeoDataFrame is avoided due to unique properties
    - Supports all standard pandas DataFrame operations
    - Adds geographic operations specific to flow data
    """
    
    _metadata = ["_geometry_column_name"]

    _internal_names = DataFrame._internal_names + ["geometry"]
    _internal_names_set = set(_internal_names)

    _geometry_column_name = None
    
    def __init__(self, data=None, *args, geometry=None, crs=None, **kwargs):
        """
        Initialize a FlowDataFrame with flow data and geometries.

        Parameters
        ----------
        data : dict, list, pandas.DataFrame, optional
            Input data containing flow information
        *args : tuple
            Additional positional arguments passed to pandas.DataFrame
        geometry : str, array-like, optional
            Column name or array containing geometry data
        crs : pyproj.CRS, optional
            Coordinate reference system for the geometries
        **kwargs : dict
            Additional keyword arguments passed to pandas.DataFrame

        Notes
        -----
        - If geometry is not specified, looks for 'geometry' column
        - Validates CRS consistency between input and specified CRS
        - Handles both single geometry and flow (origin-destination) geometries
        """
        
        if (
            kwargs.get("copy") is None
            and isinstance(data, DataFrame)
            and not isinstance(data, GeoDataFrame)
        ):
            kwargs.update(copy=True)
        super().__init__(data, *args, **kwargs)
        
        # if gdf passed in and geo_col is set, we use that for geometry
        if geometry is None and isinstance(data, GeoDataFrame):
            self._geometry_column_name = data._geometry_column_name
            if crs is not None and data.crs != crs:
                raise ValueError(crs_mismatch_error)
        
        if (
            geometry is None
            and self.columns.nlevels == 1
            and "geometry" in self.columns
        ):
            # Check for multiple columns with name "geometry". If there are,
            # self["geometry"] is a gdf and constructor gets recursively recalled
            # by pandas internals trying to access this
            if (self.columns == "geometry").sum() > 1:
                raise ValueError(
                    "GeoDataFrame does not support multiple columns "
                    "using the geometry column name 'geometry'."
                )

            # only if we have actual geometry values -> call set_geometry
            if (
                hasattr(self["geometry"].values, "crs")
                and self["geometry"].values.crs
                and crs
                and not self["geometry"].values.crs == crs
            ):
                raise ValueError(crs_mismatch_error)
            self["geometry"] = _ensure_geometry(self["geometry"].values, crs)
            geometry = "geometry"
        
        if geometry is not None:
            if (
                hasattr(geometry, "crs")
                and geometry.crs
                and crs
                and not geometry.crs == crs
            ):
                raise ValueError(crs_mismatch_error)

            if isinstance(geometry, (Series, GeoSeries, FlowSeries)) and geometry.name not in (
                "geometry",
                None,
            ):
                # __init__ always creates geometry col named "geometry"
                # rename as `set_geometry` respects the given series name
                geometry = geometry.rename("geometry")
                geometry = _ensure_geometry(geometry, crs)

            self.set_geometry(geometry, inplace=True, crs=crs)

        if geometry is None and crs:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Supply geometry using the 'geometry=' keyword argument, "
                "or by providing a DataFrame with column name 'geometry'",
            )
        
    def __setattr__(self, attr, val):
        # have to special case geometry b/c pandas tries to use as column...
        if attr == "geometry":
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)

    def _get_geometry(self):
        if self._geometry_column_name not in self:
            if self._geometry_column_name is None:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    "but the active geometry column to use has not been set. "
                )
            else:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    f"but the active geometry column ('{self._geometry_column_name}') "
                    "is not present. "
                )
            geo_cols = list(self.columns[self.dtypes == "geometry"])
            if len(geo_cols) > 0:
                msg += (
                    f"\nThere are columns with geometry data type ({geo_cols}), and "
                    "you can either set one as the active geometry with "
                    'df.set_geometry("name") or access the column as a '
                    'GeoSeries (df["name"]) and call the method directly on it.'
                )
            else:
                msg += (
                    "\nThere are no existing columns with geometry data type. You can "
                    "add a geometry column as the active geometry column with "
                    "df.set_geometry. "
                )

            raise AttributeError(msg)
        return self[self._geometry_column_name]

    def _set_geometry(self, col):
        if not pd.api.types.is_list_like(col):
            raise ValueError("Must use a list-like to set the geometry property")
        self.set_geometry(col, inplace=True)

    geometry = property(
        fget=_get_geometry, fset=_set_geometry, doc="Geometry data for GeoDataFrame"
    )
    
    def set_geometry(self, col, inplace=False, crs=None):
        """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : column label or array-like
            An existing column name or values to set as the new geometry column.
            If values (array-like, (Geo)Series) are passed, then if they are named
            (Series) the new geometry column will have the corresponding name,
            otherwise the existing geometry column will be replaced. If there is
            no existing geometry column, the new geometry column will use the
            default name "geometry".
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)
        crs : pyproj.CRS, optional
            Coordinate system to use. The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
            If passed, overrides both DataFrame and col's crs.
            Otherwise, tries to get crs from passed col values or DataFrame.

        Returns
        -------
        FlowDataFrame
        """
        # Most of the code here is taken from GeoDataFrame.set_geometry()
        if inplace:
            frame = self
        else:
            if PANDAS_GE_30:
                frame = self.copy(deep=False)
            else:
                frame = self.copy()

        geo_column_name = self._geometry_column_name

        if geo_column_name is None:
            geo_column_name = "geometry"
        
        if isinstance(col, FlowSeries):
            level = col
            geo_column_name = col.name if col.name else geo_column_name
        elif isinstance(col, str):
            level = frame[col]
            geo_column_name = col
            if not isinstance(level, FlowSeries):
                try:
                    level = FlowSeries(level, crs=crs)
                except Exception as e:
                    raise TypeError(f"Find error: {e} when converting {col} to FlowSeries")
        else:
            try:
                level = FlowSeries(col, crs=crs)
                geo_column_name = col.name if col.name else geo_column_name
            except Exception as e:
                raise TypeError(f"Find error: {e} when converting {col} to FlowSeries")

        if not crs:
            crs = getattr(level, "crs", None)

        # Check that we are using a listlike of geometries
        level = _ensure_geometry(level, crs=crs)
        # ensure_geometry only sets crs on level if it has crs==None
        if isinstance(level, FlowSeries):
            level.array.crs = crs
        else:
            level.crs = crs
        # update _geometry_column_name prior to assignment
        # to avoid default is None warning
        frame._geometry_column_name = geo_column_name
        frame[geo_column_name] = level

        if not inplace:
            return frame

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
    
    def __setitem__(self, key, value):
        """
        Overwritten to preserve CRS of GeometryArray in cases like
        df['geometry'] = [geom... for geom in df.geometry]
        """

        if not pd.api.types.is_list_like(key) and (
            key == self._geometry_column_name
            or 
            (key == "geometry" and self._geometry_column_name is None)
        ):
            if pd.api.types.is_scalar(value) and isinstance(value, Flow):
                value = [value] * self.shape[0]
            if self._geometry_column_name is not None:
                crs = getattr(self, "crs", None)
            else:  # don't use getattr, because a col "crs" might exist
                crs = None
            value = _ensure_geometry(value, crs=crs)
            
        super().__setitem__(key, value)

    @property
    def _constructor(self):
        """Override pandas/geopandas internal constructor"""
        return FlowDataFrame
    
    def _constructor_from_mgr(self, mgr, axes):
        # replicate _geodataframe_constructor_with_fallback behaviour
        # unless safe to skip
        if not any(isinstance(block.dtype, GeometryDtype) for block in mgr.blocks):
            return FlowDataFrame(
                pd.DataFrame._from_mgr(mgr, axes)
            )
        fdf = FlowDataFrame._from_mgr(mgr, axes)
        # _from_mgr doesn't preserve metadata (expect __finalize__ to be called)
        # still need to mimic __init__ behaviour with geometry=None
        if (fdf.columns == "geometry").sum() == 1:  # only if "geometry" is single col
            fdf._geometry_column_name = "geometry"
        return fdf
    
    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """
            A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries:

            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """
            srs = pd.Series(*args, **kwargs)
            is_row_proxy = srs.index.is_(self.columns)
            if is_geometry_type(srs) and not is_row_proxy:
                try:
                    srs = FlowSeries(srs)
                except:
                    srs = GeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced
    
    def _constructor_sliced_from_mgr(self, mgr, axes):
        is_row_proxy = mgr.index.is_(self.columns)

        if isinstance(mgr.blocks[0].dtype, GeometryDtype) and not is_row_proxy:
            try:
                return FlowSeries._from_mgr(mgr, axes)
            except:
                return GeoSeries._from_mgr(mgr, axes)
        return Series._from_mgr(mgr, axes)

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
            
    @GeoPandasBase.crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        if self._geometry_column_name is None:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Use GeoDataFrame.set_geometry to set the active "
                "geometry column.",
            )

        if hasattr(self.geometry.values, "crs"):
            if self.crs is not None:
                warnings.warn(
                    "Overriding the CRS of a GeoDataFrame that already has CRS. "
                    "This unsafe behavior will be deprecated in future versions. "
                    "Use GeoDataFrame.set_crs method instead",
                    stacklevel=2,
                    category=DeprecationWarning,
                )
            self.geometry.values.crs = value
        else:
            # column called 'geometry' without geometry
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without an active geometry "
                "column is not supported. Use GeoDataFrame.set_geometry to set "
                "the active geometry column.",
            )
            
    def set_crs(self, crs=None, epsg=None, inplace=False, allow_override=False):
        """
        Set the Coordinate Reference System (CRS) of the ``GeoDataFrame``.

        If there are multiple geometry columns within the GeoDataFrame, only
        the CRS of the active geometry column is set.

        Pass ``None`` to remove CRS from the active geometry column.

        Notes
        -----
        The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS | None, optional
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional
            EPSG code specifying the projection.
        inplace : bool, default False
            If True, the CRS of the GeoDataFrame will be changed in place
            (while still returning the result) instead of making a copy of
            the GeoDataFrame.
        allow_override : bool, default False
            If the the GeoDataFrame already has a CRS, allow to replace the
            existing CRS, even when both are not equal.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        Setting CRS to a GeoDataFrame without one:

        >>> gdf.crs is None
        True

        >>> gdf = gdf.set_crs('epsg:3857')
        >>> gdf.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        Overriding existing CRS:

        >>> gdf = gdf.set_crs(4326, allow_override=True)

        Without ``allow_override=True``, ``set_crs`` returns an error if you try to
        override CRS.

        See also
        --------
        GeoDataFrame.to_crs : re-project to another CRS

        """
        if not inplace:
            df = self.copy()
        else:
            df = self
        df.geometry = df.geometry.set_crs(
            crs=crs, epsg=epsg, allow_override=allow_override, inplace=True
        )
        return df

    def to_crs(self, crs=None, epsg=None, inplace=False):
        """Transform geometries to a new coordinate reference system.

        Transform all geometries in an active geometry column to a different coordinate
        reference system.  The ``crs`` attribute on the current GeoSeries must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        of projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics. Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        inplace : bool, optional, default: False
            Whether to return a new GeoDataFrame or do the transformation in
            place.

        Returns
        -------
        GeoDataFrame

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)
        >>> gdf.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        >>> gdf = gdf.to_crs(3857)
        >>> gdf
            col1                       geometry
        0  name1  POINT (111319.491 222684.209)
        1  name2  POINT (222638.982 111325.143)
        >>> gdf.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        See also
        --------
        GeoDataFrame.set_crs : assign CRS without re-projection
        """
        if inplace:
            df = self
        else:
            df = self.copy()
        geom = df.geometry.to_crs(crs=crs, epsg=epsg)
        df.geometry = geom
        if not inplace:
            return df

    def within(self, mask) -> pd.Series:
        """
        Select the flow data within the given mask.

        Parameters
        ----------
        mask (GeoSeries or geometric object): The GeoSeries (elementwise) or geometric object to test if each flow is within.

        Returns
        -------
        mask (pd.Series): A boolean Series indicating whether each flow is within the mask.
        """
        is_start_within = self.o.within(mask)
        is_end_within = self.d.within(mask)
        mask = is_start_within & is_end_within
        
        return mask
    
    @property
    def volume(self) -> float:
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
    def density(self) -> float:
        """
        Calculate the density of the flow space.

        Returns
        -------
        density : float
            The density of the flow space.
        """
        return len(self) / self.volume
    
    @property
    def o(self) -> GeoSeries:
        """
        Get the origin points of the flow.

        Returns
        -------
        origin_points : GeoSeries
            The origin points of the flow.
        """
        return self.geometry.get_geometry(0)
    
    @property
    def d(self) -> GeoSeries:
        """
        Get the destination points of the flow.

        Returns
        -------
        destination_points : GeoSeries
            The destination points of the flow.
        """
        return self.geometry.get_geometry(1)

    def plot(self, kind='arrow', ax=None, column=None, figsize=None, **kwargs) -> plt.Axes:
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
        else:
            return super().plot(kind=kind, ax=ax, figsize=figsize, **kwargs)
        
    def to_grid(self, delta_x=None, delta_y=None, x_size=None, y_size=None) -> 'FlowDataFrame':
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
        origins = shapely.get_coordinates(self.origin_points)
        destinations = shapely.get_coordinates(self.dest_points)
        self.loc[:, 'o_grid_x'] = ((origins[:, 0] - self.bounds.minx) / delta_x).fillna(-1).astype(int)
        self.loc[:, 'o_grid_y'] = ((origins[:, 1] - self.bounds.miny) / delta_y).fillna(-1).astype(int)
        self.loc[:, 'd_grid_x'] = ((destinations[:, 0] - self.bounds.minx) / delta_x).fillna(-1).astype(int)
        self.loc[:, 'd_grid_y'] = ((destinations[:, 1] - self.bounds.miny) / delta_y).fillna(-1).astype(int)
        
        # Create grid IDs
        self.loc[:, 'o_grid_id'] = self['o_grid_y'] * x_size + self['o_grid_x']
        self.loc[:, 'd_grid_id'] = self['d_grid_y'] * x_size + self['d_grid_x']
        
        return self



