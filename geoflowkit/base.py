from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from geopandas import GeoSeries, GeoDataFrame

from geoflowkit.flow import Flow


class FlowBase:

    @property
    def o(self) -> GeoSeries:
        """Origin points of the flows.

        Returns
        -------
        GeoSeries
            A GeoSeries containing the origin point of each flow.
        """
        return self.geometry.get_geometry(0)
    
    @property
    def d(self) -> GeoSeries:
        """Destination points of the flows.

        Returns
        -------
        GeoSeries
            A GeoSeries containing the destination point of each flow.
        """
        return self.geometry.get_geometry(1)
    
    def check_geographic_crs(self, stacklevel: int) -> None:
        """Warn if a geographic CRS is set for planar operations.

        Parameters
        ----------
        stacklevel : int
            The stack level offset for the warning.
        """
        if self.crs and self.crs.is_geographic:
            warnings.warn(
                "Geometry is in a geographic CRS. Results from are likely incorrect. "
                "Use 'GeoSeries.to_crs()' to re-project geometries to a "
                "projected CRS before this operation.\n",
                UserWarning,
                stacklevel=stacklevel,
            )
    
    @property
    def volume(self) -> float:
        """Area of the bounding box of the flows.

        Returns
        -------
        float
            The area of the bounding box (width × height).
        """
        self.check_geographic_crs(3)
        bounds = self.total_bounds
        return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

    @property
    def density(self) -> float:
        """Number of flows per unit area of the bounding box.

        Returns
        -------
        float
            The number of flows divided by the bounding box area.
        """
        return len(self) / self.volume
    
    @property
    def length(self) -> Series:
        """Straight-line distance from origin to destination for each flow.

        Returns
        -------
        Series
            A Series of distances (one per flow).
        """
        return self.o.distance(self.d)
    
    @property
    def angle(self) -> Series:
        """Direction of each flow in radians, measured counterclockwise from east.

        Returns
        -------
        Series
            A Series of angles in radians in the range [-π, π].
        """
        self.check_geographic_crs(3)
        o = self.o
        d = self.d
        dx = d.x - o.x
        dy = d.y - o.y
        return np.arctan2(dy, dx)
    
    def within(
        self,
        mask: Union["GeoDataFrame", "GeoSeries", Any],
        align: Union[bool, None] = None,
    ) -> Series:
        """Select flows whose origin and destination both fall inside mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip flows.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoSeries.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.
        align : bool | None, default None
            If True, automatically aligns GeoSeries based on their indices.
            If False, the order of elements is preserved. None defaults to True.

        Returns
        -------
        Series
            A boolean Series indicating whether each flow is within the mask.
        """
        is_start_within = self.o.within(mask, align=align)
        is_end_within = self.d.within(mask, align=align)
        mask = is_start_within & is_end_within
        
        return mask
    
    def clip(
        self,
        mask: Union["GeoDataFrame", "GeoSeries", Any],
    ) -> "FlowBase":
        """Select flows whose origin and destination both fall inside mask.

        Both layers must be in the same Coordinate Reference System (CRS).

        If there are multiple polygons in mask, data from the FlowSeries or FlowDataFrame will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip flows.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoSeries.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.

        Returns
        -------
        FlowBase
            A FlowSeries or FlowDataFrame containing only the flows that fall
            completely within the mask.
        """
        mask = self.within(mask)
        return self.loc[mask]

    def distance(
        self,
        other: Union[Flow, "FlowSeries"],
        distance: str = "max",
        align: Union[bool, None] = None,
        w1: float = 1,
        w2: float = 1,
        length: bool = False,
    ) -> Series:
        """Calculate the distance between this and another flow series.

        Computes the distance between each flow in this series and the
        corresponding flow in ``other`` on a 1-to-1 row-wise basis.

        Parameters
        ----------
        other : Flow or FlowSeries
            The Flow or FlowSeries to calculate the distance to.
        distance : str, optional
            The method to calculate the distance. Options are:

            - ``'max'`` : Maximum of origin and destination distances (default)
            - ``'sum'`` : Sum of origin and destination distances
            - ``'min'`` : Minimum of origin and destination distances
            - ``'mean'`` : Average of origin and destination distances
            - ``'weighted'`` : Weighted combination of origin and destination distances
        align : bool | None, optional
            If True, automatically aligns FlowSeries based on their indices.
            If False, preserves the order of elements.
            If None, defaults to True.
        w1 : float, optional
            Weight for origin distances when ``distance='weighted'``. Default is 1.
        w2 : float, optional
            Weight for destination distances when ``distance='weighted'``. Default is 1.
        length : bool, optional
            If True, uses flow lengths for weighted distance calculation.
            Default is False.

        Returns
        -------
        Series
            A Series containing the calculated distances between flows.

        Raises
        ------
        ValueError
            If an invalid ``distance`` method is specified.

        Notes
        -----
        The ``'weighted'`` distance option provides a way to combine origin and
        destination distances with custom weights. When ``length`` is True, it
        incorporates the lengths of the flows in the calculation.
        """
        from geoflowkit.flow import Flow
        from geoflowkit.flowseries import FlowSeries
        assert isinstance(other, (Flow, FlowSeries)), "other must be a Flow or FlowSeries"
        
        o_dis = np.asarray(self.o.distance(other.o, align=align))
        d_dis = np.asarray(self.d.distance(other.d, align=align))
        od_dis = np.vstack([o_dis, d_dis])
        
        if distance == 'max':
            return pd.Series(np.max(od_dis, axis=0), index=self.index)
        elif distance =='sum':
            return pd.Series(np.sum(od_dis, axis=0), index=self.index)
        elif distance == 'min':
            return pd.Series(np.min(od_dis, axis=0), index=self.index)
        elif distance == 'mean':
            return pd.Series(np.mean(od_dis, axis=0), index=self.index)
        elif distance == 'weighted':
            if not length:
                return pd.Series(w1*o_dis + w2*d_dis, index=self.index)
            else:
                up = w1*o_dis**2 + w2*d_dis**2
                down = self.length.values * other.length.values
                return pd.Series(np.sqrt(up / down), index=self.index)
        else:
            raise ValueError("distance must be 'max', 'sum', 'min', 'mean', or 'weighted'")


