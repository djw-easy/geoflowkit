import warnings

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from geopandas import GeoSeries, GeoDataFrame


class FlowBase:

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
    def length(self) -> Series:
        """
        Calculate the length of each flow.

        Returns
        -------
        length : Series
            The length of each flow.
        """
        return self.o.distance(self.d)
    
    @property
    def angle(self) -> Series:
        """
        Calculate the angle of each flow.

        Returns
        -------
        angle : Series
            The angle of each flow.
        """
        self.check_geographic_crs(3)
        o = self.o
        d = self.d
        dx = d.x - o.x
        dy = d.y - o.y
        return np.arctan2(dy, dx)
    
    def within(self, mask, align=None) -> Series:
        """
        Select the flow data within the given mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip `fdf` or `flowseries`.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoSeries.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.
        align : bool | None (default None)
            If True, automatically aligns GeoSeries based on their indices. 
            If False, the order of elements is preserved. None defaults to True.

        Returns
        -------
        mask (pd.Series): A boolean Series indicating whether each flow is within the mask.
        """
        is_start_within = self.o.within(mask, align=align)
        is_end_within = self.d.within(mask, align=align)
        mask = is_start_within & is_end_within
        
        return mask
    
    def clip(self, mask):
        """Clip flow to the mask extent.

        Both layers must be in the same Coordinate Reference System (CRS).

        If there are multiple polygons in mask, data from the FlowSeries or FlowDataFrame will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip `fdf` or `flowseries`.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoSeries.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.

        Returns
        -------
        FlowSeries or FlowDataFrame
            Flow data clipped to the polygon boundary from mask.
        """
        mask = self.within(mask)
        return self.loc[mask]

    def distance(self, other, distance='max', align=None, w1=1, w2=1, length=False):
        """Calculate the distance between this FlowSeries and another FlowSeries.

        This method computes the distance between each flow in this FlowSeries and the corresponding flow in the 'other' FlowSeries.
        The calculation is performed on a 1-to-1 row-wise basis.

        Parameters
        ----------
        other : Flow or FlowSeries
            The Flow or FlowSeries to calculate the distance to.
        distance : str, optional
            The method to calculate the distance. Options are:
            - 'max': Maximum of origin and destination distances (default)
            - 'sum': Sum of origin and destination distances
            - 'min': Minimum of origin and destination distances
            - 'mean': Average of origin and destination distances
            - 'weighted': Weighted combination of origin and destination distances
        align : bool or None, optional
            If True, automatically aligns FlowSeries based on their indices.
            If False, preserves the order of elements. 
            If None, defaults to True.
        w1 : float, optional
            Weight for origin distances when distance is 'weighted'. Default is 1.
        w2 : float, optional
            Weight for destination distances when distance is 'weighted'. Default is 1.
        length : bool, optional
            If True, uses flow lengths for weighted distance calculation. Default is False.

        Returns
        -------
        pd.Series
            A Series containing the calculated distances between flows.

        Raises
        ------
        ValueError
            If an invalid 'distance' method is specified.

        Notes
        -----
        The 'weighted' distance option provides a way to combine origin and destination distances
        with custom weights. When 'length' is True, it incorporates the lengths of the flows
        in the calculation, which can be useful for certain spatial analyses.
        """
        from geoflow.flow import Flow
        from geoflow.flowseries import FlowSeries
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


