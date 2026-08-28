"""Origin-Destination matrix heatmap visualization for flow data.

Provides :class:`ODMatrixVisualizer` for building and plotting OD matrices
from :class:`~geoflowkit.flowdataframe.FlowDataFrame` instances, and the
:func:`od_matrix` convenience function.
"""

from collections.abc import Callable

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from geoflowkit.visualization._utils import (
    _assign_zones,
    _draw_size_overlay,
    _get_weights,
    _prepare_zones,
)


class ODMatrixVisualizer:
    """Build and display an OD matrix heatmap from flow data.

    Flows are mapped to spatial zones defined by a user-supplied
    GeoDataFrame of polygons, then aggregated into an origin-destination
    matrix.  Zones with zero outflow are removed from the origin (row)
    axis, and zones with zero inflow are removed from the destination
    (column) axis.

    When *dest_zones* is provided separately, the matrix may be
    non-square (M origins × N destinations).

    Parameters
    ----------
    origin_zones : GeoDataFrame
        Zone polygons for the origin (row) axis.  Must contain a
        column that uniquely identifies each zone (see *zone_id_col*).
    dest_zones : GeoDataFrame, keyword-only, optional
        Zone polygons for the destination (column) axis.  When
        ``None`` (default) the same zones are used for both axes.
    zone_id_col : str, optional
        Column in *origin_zones* to use as the zone identifier.
        ``None`` uses the GeoDataFrame index.
    dest_zone_id_col : str, optional
        Column in *dest_zones* for zone identifier.  ``None`` uses
        *zone_id_col*.
    weight : str or callable, default='count'
        Aggregation weight: ``'count'``, ``'length'``,
        ``'divergence'``, ``'volume'``, or a numeric column name.  A
        callable is shorthand for count aggregation followed by that
        callable as *weight_transform*.
    weight_transform : callable, optional
        NumPy-compatible function applied to the complete OD matrix after
        aggregation, for example ``np.log1p``.  It must return a finite
        array of the same shape and preserve zero values.
    cbar_label : str, optional
        Colorbar title.  A readable label is inferred from *weight* and
        *weight_transform* when omitted.
    size_weight : str, optional
        When set, overlay proportional circles whose area reflects
        this metric (same options as *weight*).  Color remains
        controlled by *weight*.
    cmap : str or Colormap, default='OrRd'
        Colormap for the heatmap.
    vmin, vmax : float, optional
        Colormap range in transformed display units.  When ``None`` the
        range is inferred from data.
    show_labels : bool, default=True
        Whether to show zone ID labels on the axes.
    label_fontsize : int, default=10
        Font size for axis labels.
    include_self_flows : bool, default=True
        Whether to include flows where the origin and destination zone
        are the same (diagonal entries).  When ``False`` the diagonal
        is zeroed out.  Only effective when both axes use the same
        zone set.

    Attributes
    ----------
    matrix_ : np.ndarray
        The transformed ``(n_origin_zones, n_dest_zones)`` display matrix
        (set after :meth:`fit`).
    raw_matrix_ : np.ndarray
        The corresponding matrix before *weight_transform* is applied.
    size_matrix_ : np.ndarray or None
        Size values for each cell (same shape as *matrix_*), or
        ``None`` when *size_weight* is not set.
    o_ids_ : np.ndarray
        Zone IDs for origin (row) axis, including only zones with
        nonzero outflow.
    d_ids_ : np.ndarray
        Zone IDs for destination (column) axis, including only zones
        with nonzero inflow.
    o_centroids_, d_centroids_ : dict
        Mapping from zone ID to ``(x, y)`` centroid for origins / destinations.
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

    def __init__(
        self,
        origin_zones: gpd.GeoDataFrame,
        *,
        dest_zones: gpd.GeoDataFrame | None = None,
        zone_id_col: str | None = None,
        dest_zone_id_col: str | None = None,
        weight: str | Callable = 'count',
        weight_transform: Callable | None = None,
        cbar_label: str | None = None,
        size_weight: str | None = None,
        cmap: str | plt.Colormap = 'OrRd',
        vmin: float | None = None,
        vmax: float | None = None,
        show_labels: bool = True,
        label_fontsize: int = 10,
        include_self_flows: bool = True,
    ):
        self.origin_zones = origin_zones

        self.dest_zones = dest_zones if dest_zones is not None else self.origin_zones
        self._asymmetric = self.dest_zones is not self.origin_zones
        self.zone_id_col = zone_id_col
        self.dest_zone_id_col = dest_zone_id_col if dest_zone_id_col is not None else zone_id_col

        if callable(weight):
            if weight_transform is not None:
                raise ValueError(
                    "A callable weight cannot be combined with "
                    "weight_transform"
                )
            weight_transform = weight
            weight = 'count'
        elif not isinstance(weight, str):
            raise TypeError("weight must be a string or callable")
        if weight_transform is not None and not callable(weight_transform):
            raise TypeError("weight_transform must be callable or None")
        if cbar_label is not None and not isinstance(cbar_label, str):
            raise TypeError("cbar_label must be a string or None")

        self.weight = weight
        self.weight_transform = weight_transform
        self.cbar_label = (
            cbar_label
            if cbar_label is not None
            else self._default_cbar_label(weight, weight_transform)
        )
        self.size_weight = size_weight
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.show_labels = show_labels
        self.label_fontsize = label_fontsize
        self.include_self_flows = include_self_flows

        self.matrix_ = None
        self.raw_matrix_ = None
        self.size_matrix_ = None
        self.o_ids_ = None
        self.d_ids_ = None
        self.o_centroids_ = None
        self.d_centroids_ = None
        self.o_zones_ = None
        self.d_zones_ = None
        self._outflows = None
        self._inflows = None
        self._raw_matrix = None
        self._display_matrix = None
        self._raw_size_matrix = None
        self._o_all_ids = None
        self._d_all_ids = None

    @staticmethod
    def _default_cbar_label(weight, weight_transform):
        if weight_transform is None:
            return str(weight).capitalize()
        transform_name = getattr(weight_transform, '__name__', '')
        if transform_name and transform_name != '<lambda>':
            return f"{transform_name}({weight})"
        return f"Transformed {weight}"

    def _apply_weight_transform(self, matrix):
        """Apply and validate the post-aggregation display transform."""
        if self.weight_transform is None:
            return matrix.copy()
        try:
            transformed = np.asarray(
                self.weight_transform(matrix.copy()),
                dtype=float,
            )
        except Exception as exc:
            raise ValueError("weight_transform failed") from exc
        if transformed.shape != matrix.shape:
            raise ValueError(
                "weight_transform must return an array with the same shape"
            )
        if not np.all(np.isfinite(transformed)):
            raise ValueError(
                "weight_transform must return only finite values"
            )
        zero_mask = np.isclose(matrix, 0.0)
        if zero_mask.any() and not np.allclose(
            transformed[zero_mask], 0.0,
        ):
            raise ValueError("weight_transform must preserve zero values")
        return transformed

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
        self.o_zones_, self.d_zones_, self.o_centroids_, self.d_centroids_ = _assign_zones(
            fdf, self.origin_zones, zone_id_col=self.zone_id_col,
            dest_zones=None if not self._asymmetric else self.dest_zones,
            dest_zone_id_col=None if not self._asymmetric else self.dest_zone_id_col,
        )

        if self._asymmetric:
            o_prepared = _prepare_zones(self.origin_zones, zone_id_col=self.zone_id_col)
            d_prepared = _prepare_zones(self.dest_zones, zone_id_col=self.dest_zone_id_col)
            o_all_ids = o_prepared['zone_id'].tolist()
            d_all_ids = d_prepared['zone_id'].tolist()
        else:
            o_all_ids = list(self.o_centroids_.keys())
            d_all_ids = o_all_ids

        self._o_all_ids = o_all_ids
        self._d_all_ids = d_all_ids

        w_values = _get_weights(fdf, self.weight)
        if self.size_weight is not None:
            size_values = _get_weights(fdf, self.size_weight)
        else:
            size_values = None

        o_to_idx = {z: i for i, z in enumerate(o_all_ids)}
        d_to_idx = {z: i for i, z in enumerate(d_all_ids)}
        full_matrix = np.zeros((len(o_all_ids), len(d_all_ids)))
        if size_values is not None:
            size_matrix = np.zeros((len(o_all_ids), len(d_all_ids)))

        self._outflows = {z: 0.0 for z in o_all_ids}
        self._inflows = {z: 0.0 for z in d_all_ids}
        for i in range(len(fdf)):
            oz = self.o_zones_[i]
            dz = self.d_zones_[i]
            if oz in o_to_idx and dz in d_to_idx:
                oi = o_to_idx[oz]
                di = d_to_idx[dz]
                full_matrix[oi, di] += w_values[i]
                if size_values is not None:
                    size_matrix[oi, di] += size_values[i]
            if oz in self._outflows:
                self._outflows[oz] += w_values[i]
            if dz in self._inflows:
                self._inflows[dz] += w_values[i]

        if not self.include_self_flows and not self._asymmetric:
            np.fill_diagonal(full_matrix, 0.0)

        self._raw_matrix = full_matrix
        self._display_matrix = self._apply_weight_transform(full_matrix)
        self._raw_size_matrix = size_matrix if size_values is not None else None

        o_ids = [z for z in o_all_ids if self._outflows[z] > 0]
        d_ids = [z for z in d_all_ids if self._inflows[z] > 0]
        if not o_ids and o_all_ids:
            o_ids = o_all_ids
        if not d_ids and d_all_ids:
            d_ids = d_all_ids

        self.o_ids_ = np.array(o_ids)
        self.d_ids_ = np.array(d_ids)

        o_pos = [o_to_idx[z] for z in o_ids]
        d_pos = [d_to_idx[z] for z in d_ids]
        selected = np.ix_(o_pos, d_pos)
        self.raw_matrix_ = full_matrix[selected].copy()
        self.matrix_ = self._display_matrix[selected].copy()
        self.size_matrix_ = (
            size_matrix[np.ix_(o_pos, d_pos)]
            if self._raw_size_matrix is not None else None
        )
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

        # --- size circles ---
        if self.size_matrix_ is not None:
            ny, nx = self.matrix_.shape
            j_idx, i_idx = np.meshgrid(np.arange(nx), np.arange(ny))
            xs = j_idx.ravel()
            ys = (ny - 1 - i_idx).ravel()
            sizes = self.size_matrix_.ravel()
            vals = self.matrix_.ravel()
            mask = ~np.isnan(vals) & (vals != 0)
            _draw_size_overlay(ax, xs[mask], ys[mask], sizes[mask])

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
            cbar.set_label(self.cbar_label, fontsize=self.label_fontsize)

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

