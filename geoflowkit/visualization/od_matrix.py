"""Origin-Destination matrix heatmap visualization for flow data.

Provides :class:`ODMatrixVisualizer` for building and plotting OD matrices
from :class:`~geoflowkit.flowdataframe.FlowDataFrame` instances, and the
:func:`od_matrix` convenience function.
"""

import numpy as np
import matplotlib.pyplot as plt

from geoflowkit.visualization._utils import (
    _assign_zones,
    _build_od_matrix,
)


class ODMatrixVisualizer:
    """Build and display an OD matrix heatmap from flow data.

    Flows are mapped to spatial zones defined by a user-supplied
    GeoDataFrame of polygons, then aggregated into an origin-destination
    matrix.  Zones with zero outflow are removed from the origin (row)
    axis, and zones with zero inflow are removed from the destination
    (column) axis.

    Parameters
    ----------
    zones : GeoDataFrame
        Polygon geometries defining the spatial zones.  Must contain a
        column that uniquely identifies each zone (see *zone_id_col*).
    zone_id_col : str, optional
        Column in *zones* to use as the zone identifier.  ``None``
        (default) uses the GeoDataFrame index.
    weight : str, default='count'
        Aggregation weight: ``'count'``, ``'volume'``, or ``'length'``.
    cmap : str or Colormap, default='OrRd'
        Colormap for the heatmap.
    vmin, vmax : float, optional
        Colormap range.  When ``None`` the range is inferred from data.
    show_labels : bool, default=True
        Whether to show zone ID labels on the axes.
    label_fontsize : int, default=10
        Font size for axis labels.
    include_self_flows : bool, default=True
        Whether to include flows where the origin and destination zone
        are the same (diagonal entries).  When ``False`` the diagonal
        is zeroed out.

    Attributes
    ----------
    matrix_ : np.ndarray
        The ``(n_origin_zones, n_dest_zones)`` OD matrix (set after
        :meth:`fit`).
    o_ids_ : np.ndarray
        Zone IDs for origin (row) axis, including only zones with
        nonzero outflow.
    d_ids_ : np.ndarray
        Zone IDs for destination (column) axis, including only zones
        with nonzero inflow.
    zone_centroids_ : dict
        Mapping from zone ID to ``(x, y)`` centroid.
    o_zones_, d_zones_ : np.ndarray
        Per-flow zone assignments (set after :meth:`fit`).
    outflows_, inflows_ : dict
        Per-zone total outflow / inflow volumes.

    References
    ----------
    .. [1] Voorhees, A. M. (2013). *Shopping Centre Location and Consumer
           Behaviour*. In Regional Science Association: Papers (Vol. 11).
    .. [2] Guo, D., & Gahegan, M. (2006). *Spatial ordering and encoding
           for geographic data mining and visualization*. Journal of
           Intelligent Information Systems, 27(3), 243-266.
    """

    def __init__(self, zones, zone_id_col=None, weight='count',
                 cmap='OrRd', vmin=None, vmax=None,
                 show_labels=True, label_fontsize=10,
                 include_self_flows=True):
        self.zones = zones
        self.zone_id_col = zone_id_col
        self.weight = weight
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.show_labels = show_labels
        self.label_fontsize = label_fontsize
        self.include_self_flows = include_self_flows

        # Fit-time attributes
        self.matrix_ = None
        self.o_ids_ = None
        self.d_ids_ = None
        self.zone_centroids_ = None
        self.o_zones_ = None
        self.d_zones_ = None
        self._outflows = None
        self._inflows = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, fdf):
        """Aggregate flows into an OD matrix.

        Parameters
        ----------
        fdf : FlowDataFrame
            Input flow data.

        Returns
        -------
        self : ODMatrixVisualizer
            Fitted visualizer.
        """
        self.o_zones_, self.d_zones_, self.zone_centroids_ = _assign_zones(
            fdf, self.zones, zone_id_col=self.zone_id_col,
        )

        # All zone IDs from the GeoDataFrame (includes zones with 0 flows)
        all_zone_ids = list(self.zone_centroids_.keys())
        n_all = len(all_zone_ids)

        # Build raw OD matrix (only zones with at least one flow)
        raw_matrix, raw_zone_ids = _build_od_matrix(
            fdf, self.o_zones_, self.d_zones_, weight=self.weight,
        )

        # Expand to full N×N matrix (zero-pad for zones with no flows)
        full_matrix = np.zeros((n_all, n_all))
        raw_idx = {z: i for i, z in enumerate(raw_zone_ids)}
        for ri, rz in enumerate(all_zone_ids):
            for ci, cz in enumerate(all_zone_ids):
                if rz in raw_idx and cz in raw_idx:
                    full_matrix[ri, ci] = raw_matrix[raw_idx[rz], raw_idx[cz]]

        # Optionally zero out diagonal (self-flows).
        if not self.include_self_flows:
            np.fill_diagonal(full_matrix, 0.0)

        # Compute per-zone outflow / inflow.
        if self.weight == 'volume' and 'volume' in fdf.columns:
            w_values = fdf['volume'].values
        elif self.weight == 'length':
            w_values = fdf.length.values
        else:
            w_values = np.ones(len(fdf))

        self._outflows = {z: 0.0 for z in all_zone_ids}
        self._inflows = {z: 0.0 for z in all_zone_ids}
        for i in range(len(fdf)):
            self._outflows[self.o_zones_[i]] += w_values[i]
            self._inflows[self.d_zones_[i]] += w_values[i]

        # Exclude zones with zero outflow from rows and zero inflow from
        # columns.  At least one zone is kept in each axis when non-empty.
        o_ids = [z for z in all_zone_ids if self._outflows[z] > 0]
        d_ids = [z for z in all_zone_ids if self._inflows[z] > 0]
        if not o_ids and all_zone_ids:
            o_ids = all_zone_ids
        if not d_ids and all_zone_ids:
            d_ids = all_zone_ids

        self.o_ids_ = np.array(o_ids)
        self.d_ids_ = np.array(d_ids)

        # Map original all-zone indices to filtered row / column indices.
        all_idx = {z: i for i, z in enumerate(all_zone_ids)}
        o_pos = [all_idx[z] for z in o_ids]
        d_pos = [all_idx[z] for z in d_ids]

        self.matrix_ = full_matrix[np.ix_(o_pos, d_pos)]
        return self

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, ax=None, figsize=None, colorbar=True):
        """Draw the OD matrix as a heatmap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on.  A new figure and axes are created when
            ``None``.
        figsize : tuple of (float, float), optional
            Figure size in inches (only used when *ax* is ``None``).
        colorbar : bool, default=True
            Whether to add a colour bar.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the heatmap.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before plot().")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            self.matrix_,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            origin='upper',
            interpolation='nearest',
            aspect='auto',
        )

        # --- axis labels ---
        if self.show_labels:
            if len(self.d_ids_) > 0:
                tick_labels = [str(z) for z in self.d_ids_]
                ax.set_xticks(range(len(self.d_ids_)))
                ax.set_xticklabels(tick_labels, fontsize=self.label_fontsize,
                                   rotation=45, ha='right')
                ax.set_xlabel('Destination Zone',
                              fontsize=self.label_fontsize + 1)

            if len(self.o_ids_) > 0:
                tick_labels = [str(z) for z in self.o_ids_]
                ax.set_yticks(range(len(self.o_ids_)))
                ax.set_yticklabels(tick_labels, fontsize=self.label_fontsize)
                ax.set_ylabel('Origin Zone',
                              fontsize=self.label_fontsize + 1)

        # --- colorbar ---
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, shrink=0.9)
            cbar.ax.tick_params(labelsize=self.label_fontsize)

        return ax

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit_plot(self, fdf, ax=None, figsize=None, colorbar=True):
        """Fit and plot in a single call.

        Parameters
        ----------
        fdf : FlowDataFrame
            Input flow data.
        ax : matplotlib.axes.Axes, optional
        figsize : tuple, optional
        colorbar : bool, default=True

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        self.fit(fdf)
        return self.plot(ax=ax, figsize=figsize, colorbar=colorbar)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def od_matrix(fdf, zones, zone_id_col=None, weight='count',
              cmap='OrRd', vmin=None, vmax=None,
              show_labels=True, label_fontsize=10,
              include_self_flows=True,
              ax=None, figsize=None, colorbar=True):
    """Build and display an OD matrix heatmap for flow data.

    This is a convenience wrapper around :class:`ODMatrixVisualizer`.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zones : GeoDataFrame
        Zone polygon geometries.
    zone_id_col : str, default='zone_id'
        Column in *zones* identifying each zone.
    weight : str, default='count'
        ``'count'``, ``'volume'``, or ``'length'``.
    cmap : str or Colormap, default='OrRd'
        Heatmap colormap.
    vmin, vmax : float, optional
        Colormap range.
    show_labels : bool, default=True
        Show zone ID axis labels.
    label_fontsize : int, default=10
        Font size for labels.
    include_self_flows : bool, default=True
        Include flows where origin and destination zones are the same.
        When ``False`` the matrix diagonal is zeroed out.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    figsize : tuple, optional
        Figure size when *ax* is ``None``.
    colorbar : bool, default=True
        Add a colour bar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the heatmap.
    """
    vis = ODMatrixVisualizer(
        zones=zones,
        zone_id_col=zone_id_col,
        weight=weight,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_labels=show_labels,
        label_fontsize=label_fontsize,
        include_self_flows=include_self_flows,
    )
    return vis.fit_plot(fdf, ax=ax, figsize=figsize, colorbar=colorbar)
