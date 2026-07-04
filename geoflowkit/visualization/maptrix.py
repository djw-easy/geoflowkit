"""MapTrix visualisation for geographical flow data.

Implements the MapTrix method [1]_ which combines origin and destination
maps with a rotated OD matrix connected by guide lines, allowing
simultaneous display of spatial interaction and geographic context.

References
----------
.. [1] Yang, X., Zhu, D., Guo, D., Liu, C., & Ye, X. (2016).
   *MapTrix: A spatial flow data exploration system using matrix-based
   visual analytics*. International Journal of Geographical Information
   Science, 31(4), 710–733.
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from geoflowkit.visualization.od_matrix import ODMatrixVisualizer
from geoflowkit.visualization._utils import (
    _ax_to_fig,
    _calculate_matrix_anchor_point,
    _calculate_rotated_point,
    _compute_representative_points,
    _draw_size_overlay,
    _linear_scaling,
    _plot_centroids,
    _plot_labels,
    _rotate_matrix,
)

class MapTrixVisualizer(ODMatrixVisualizer):
    """MapTrix layout: origin/destination maps + rotated OD matrix + guide lines.

    Parameters
    ----------
    origin_zones : GeoDataFrame
        Zone polygons for the origin map.
    dest_zones : GeoDataFrame, keyword-only, optional
        Zone polygons for the destination map.  When
        ``None`` (default) the same zones are used for both maps.
    zone_id_col : str, optional
        Column in *origin_zones* used as the zone identifier.
        ``None`` uses the GeoDataFrame index.
    dest_zone_id_col : str, optional
        Column in *dest_zones* for zone identifier.  ``None`` uses
        *zone_id_col*.
    weight : str, default='count'
        Aggregation weight: ``'count'``, ``'length'``,
        ``'divergence'``, or ``'volume'``.
    size_weight : str, optional
        When set, overlay proportional circles whose area reflects
        this metric (same options as *weight*).  Color remains
        controlled by *weight*.
    matrix_cmap : str or Colormap, default='OrRd'
        Colormap for the OD matrix heatmap.
    vmin, vmax : float, optional
        Colormap range; inferred from data when ``None``.
    map_cmap : str or Colormap, optional
        Colormap for centroid circles (defaults to *matrix_cmap*).
    max_centroid_size : float, default=300.0
        Maximum marker size for proportional circles.
    line_color : str, default='black'
        Guide line colour.
    line_alpha : float, default=0.6
        Guide line transparency.
    show_labels : bool, default=True
        Show zone ID labels on the maps.
    label_fontsize : int, default=10
        Font size for map labels.
    out_title : str, default='Outflow'
        Title for the origin map.
    in_title : str, default='Inflow'
        Title for the destination map.
    title_fontsize : int, default=16
        Font size for map titles.
    include_self_flows : bool, default=True
        Include flows where origin == destination (matrix diagonal).
        When ``False`` the diagonal is zeroed.  Only effective when
        both axes use the same zone set.
    height_ratios : list of float, optional
        Height ratios for the two map subplot rows
        (default ``[0.5, 0.5]``).
    width_ratios : list of float, optional
        Width ratios for left (maps) vs right (matrix) columns
        (default ``[1, 2]``).
    cbar_kwds : dict, optional
        Extra keyword arguments forwarded to
        :meth:`matplotlib.figure.Figure.colorbar` (e.g. ``shrink``,
        ``aspect``, ``pad``).  Defaults control vertical orientation
        with ``shrink=0.9``, ``aspect=30``, ``pad=0.01``.

    Attributes
    ----------
    matrix_ : np.ndarray
        OD matrix after :meth:`fit`.
    size_matrix_ : np.ndarray or None
        Size values per cell, or ``None`` when *size_weight* is not set.
    zone_ids_ : np.ndarray
        All zone IDs.
    o_centroids_, d_centroids_ : dict
        Mapping zone ID → ``(x, y)`` representative point for origins / destinations.
    o_order_, d_order_ : list
        Zone ordering for origin columns / destination rows.
    """

    def __init__(
        self,
        origin_zones: gpd.GeoDataFrame,
        *,
        dest_zones: gpd.GeoDataFrame | None = None,
        zone_id_col: str | None = None,
        dest_zone_id_col: str | None = None,
        weight: str = 'count',
        size_weight: str | None = None,
        matrix_cmap: str | plt.Colormap = 'OrRd',
        vmin: float | None = None,
        vmax: float | None = None,
        map_cmap: str | plt.Colormap | None = None,
        max_centroid_size: float = 300.0,
        line_color: str = 'black',
        line_alpha: float = 0.6,
        show_labels: bool = True,
        label_fontsize: int = 10,
        out_title: str = 'Outflow',
        in_title: str = 'Inflow',
        title_fontsize: int = 16,
        include_self_flows: bool = True,
        height_ratios: list | None = None,
        width_ratios: list | None = None,
        cbar_kwds: dict | None = None,
    ):
        super().__init__(
            origin_zones=origin_zones, dest_zones=dest_zones,
            zone_id_col=zone_id_col, dest_zone_id_col=dest_zone_id_col,
            weight=weight, size_weight=size_weight,
            cmap=matrix_cmap, vmin=vmin, vmax=vmax,
            show_labels=show_labels, label_fontsize=label_fontsize,
            include_self_flows=include_self_flows,
        )

        self.matrix_cmap = matrix_cmap
        self.map_cmap = map_cmap if map_cmap is not None else matrix_cmap
        self.max_centroid_size = max_centroid_size
        self.line_color = line_color
        self.line_alpha = line_alpha
        self.out_title = out_title
        self.in_title = in_title
        self.title_fontsize = title_fontsize

        self.height_ratios = height_ratios if height_ratios is not None else [0.5, 0.5]
        self.width_ratios = width_ratios if width_ratios is not None else [1, 2]
        self.cbar_kwds = {} if cbar_kwds is None else dict(cbar_kwds)

        # fit-time
        self._full_matrix = None
        self._full_size_matrix = None
        self.o_order_ = None
        self.d_order_ = None
        self._bounds = None
        self._o_zone_to_idx = None
        self._d_zone_to_idx = None

        # plot-time
        self._im = None
        self._transform = None

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(self, fdf):
        """Aggregate flows and prepare layout."""
        super().fit(fdf)

        self._full_matrix = self._raw_matrix.copy()
        self._full_size_matrix = (
            self._raw_size_matrix.copy()
            if self._raw_size_matrix is not None else None
        )
        o_all_ids = list(self._o_all_ids)
        d_all_ids = list(self._d_all_ids) if self._asymmetric else o_all_ids

        # representative points (override parent's centroid-based coords)
        rep_o = _compute_representative_points(
            self.origin_zones, zone_id_col=self.zone_id_col,
        )
        for zid in self.o_centroids_:
            if zid in rep_o:
                self.o_centroids_[zid] = rep_o[zid]

        rep_d = (
            _compute_representative_points(
                self.dest_zones, zone_id_col=self.dest_zone_id_col,
            ) if self._asymmetric else rep_o
        )
        for zid in self.d_centroids_:
            if zid in rep_d:
                self.d_centroids_[zid] = rep_d[zid]

        # index maps
        self.zone_ids_ = np.array(o_all_ids)
        self._o_zone_to_idx = {z: i for i, z in enumerate(o_all_ids)}
        self._d_zone_to_idx = (
            {z: i for i, z in enumerate(d_all_ids)}
            if self._asymmetric else self._o_zone_to_idx
        )

        # initial ordering by centroid Y
        if self._asymmetric:
            self.o_order_ = sorted(
                [z for z in o_all_ids if z in self.o_centroids_],
                key=lambda z: self.o_centroids_[z][1],
            )
            self.d_order_ = sorted(
                [z for z in d_all_ids if z in self.d_centroids_],
                key=lambda z: self.d_centroids_[z][1],
            )
        else:
            self.o_order_ = sorted(o_all_ids, key=lambda z: self.o_centroids_[z][1])
            self.d_order_ = sorted(o_all_ids, key=lambda z: self.d_centroids_[z][1])

        # exclude zero-flow zones
        for attr, flow_dict in [('o_order_', self._outflows),
                                 ('d_order_', self._inflows)]:
            order = getattr(self, attr)
            if len(order) > 1:
                nonzero = [z for z in order if flow_dict[z] > 0]
                if nonzero:
                    setattr(self, attr, nonzero)

        self._apply_ordering()
        self._bounds = fdf.total_bounds if len(fdf) > 0 else None
        return self

    def _apply_ordering(self):
        """Recompute ``matrix_`` from ``_full_matrix`` using current ordering.
        Origins → columns (top edge), destinations → rows (left edge).
        """
        if self._full_matrix is None or self.o_order_ is None or self.d_order_ is None:
            return
        o_pos = [self._o_zone_to_idx[z] for z in self.o_order_]
        d_pos = [self._d_zone_to_idx[z] for z in self.d_order_]
        self.matrix_ = self._full_matrix[np.ix_(o_pos, d_pos)].T
        if self._full_size_matrix is not None:
            self.size_matrix_ = self._full_size_matrix[np.ix_(o_pos, d_pos)].T
        else:
            self.size_matrix_ = None
        if not self.include_self_flows and not self._asymmetric:
            for j, d_zid in enumerate(self.d_order_):
                for i, o_zid in enumerate(self.o_order_):
                    if o_zid == d_zid:
                        self.matrix_[j, i] = np.nan

    # ==================================================================
    # Plot
    # ==================================================================

    def plot(self, fig=None, figsize=(14, 8)):
        """Create the full MapTrix figure."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before plot().")
        if fig is None:
            fig = plt.figure(figsize=figsize)

        ax_map_o, ax_map_d, ax_matrix, im, transform = self._build_figure(fig)
        plt.subplots_adjust(hspace=0.01, wspace=-0.15, left=0.0001)
        self._im = im
        self._transform = transform
        self._draw_colorbar(fig, ax_matrix)
        fig.canvas.draw()

        self._draw_guide_lines(fig, ax_map_o, ax_map_d, ax_matrix)
        self._draw_matrix_anchors(fig, ax_matrix)
        return fig

    def fit_plot(self, fdf, fig=None, figsize=(14, 8)):
        self.fit(fdf)
        return self.plot(fig=fig, figsize=figsize)

    def _draw_guide_lines(self, fig, ax_map_o, ax_map_d, ax_matrix):
        """Draw straight guide lines from map centroids to matrix anchors."""
        n_rows, n_cols = self.matrix_.shape

        def _width_map(flow_dict):
            if not flow_dict:
                return {}
            vals = np.array(list(flow_dict.values()))
            widths = _linear_scaling(vals, (0.5, 2.5))
            return {z: w for z, w in zip(flow_dict.keys(), widths)}

        outflow_w_map = _width_map(self._outflows)
        inflow_w_map = _width_map(self._inflows)

        centroids_map = {
            'origin': self.o_centroids_,
            'dest': self.d_centroids_,
        }

        for side, ax_map, zid_order, flow_dict, width_map, matrix_side in [
            ('origin', ax_map_o, self.o_order_, self._outflows, outflow_w_map, 'left'),
            ('dest',   ax_map_d, self.d_order_, self._inflows,  inflow_w_map, 'bottom'),
        ]:
            centroids = centroids_map[side]
            for idx, zid in enumerate(zid_order):
                if zid not in centroids:
                    continue

                cx, cy = centroids[zid]
                fig_cx, fig_cy = _ax_to_fig(ax_map, fig, cx, cy)

                anchor_x, anchor_y = _calculate_matrix_anchor_point(
                    self._transform, n_rows, n_cols, matrix_side, idx,
                )
                fig_ax, fig_ay = _ax_to_fig(ax_matrix, fig, anchor_x, anchor_y)

                fig.add_artist(Line2D(
                    [fig_cx, fig_ax], [fig_cy, fig_ay],
                    transform=fig.transFigure,
                    color=self.line_color,
                    linewidth=width_map.get(zid, 1.0),
                    alpha=self.line_alpha,
                    solid_capstyle='round',
                    clip_on=False,
                    zorder=10,
                ))

    # ==================================================================
    # Figure construction
    # ==================================================================

    def _build_figure(self, fig):
        gs = fig.add_gridspec(2, 2,
                              height_ratios=self.height_ratios,
                              width_ratios=self.width_ratios)
        ax_map_o = fig.add_subplot(gs[0, 0])
        ax_map_d = fig.add_subplot(gs[1, 0])
        ax_matrix = fig.add_subplot(gs[:, 1])
        self._draw_map(ax_map_o, self.o_order_, self._outflows,
                       self.origin_zones, self.o_centroids_)
        self._draw_map(ax_map_d, self.d_order_, self._inflows,
                       self.dest_zones, self.d_centroids_)
        im, transform = self._draw_matrix(ax_matrix)
        self._draw_titles(ax_map_o, ax_map_d)
        return ax_map_o, ax_map_d, ax_matrix, im, transform

    def _draw_map(self, ax, zid_order, flow_dict, zones_gdf, centroids):
        """Draw one map with zone boundaries, centroids, and labels."""
        centroids = {z: centroids[z] for z in zid_order if z in centroids}
        flows = np.array([flow_dict[z] for z in zid_order])
        sizes = self._scale_sizes(flows)
        colors = flows if np.ptp(flows) > 0 else 'k'

        self._draw_map_base(ax, zones_gdf)
        _plot_centroids(
            ax, centroids, sizes=sizes, colors=colors,
            cmap=self.map_cmap, vmin=self.vmin, vmax=self.vmax,
        )
        if self.show_labels:
            lbls = {z: str(z) for z in centroids}
            _plot_labels(ax, centroids, lbls, fontsize=self.label_fontsize)
        ax.axis('off')

    def _draw_map_base(self, ax, zones_gdf=None):
        if zones_gdf is None:
            zones_gdf = self.origin_zones
        if zones_gdf is not None:
            zones_gdf.boundary.plot(
                ax=ax, edgecolor='gray', linewidth=0.5, facecolor='none',
            )
        if zones_gdf is not None and len(zones_gdf) > 0:
            mnx, mny, mxx, mxy = zones_gdf.total_bounds
            padx = max((mxx - mnx) * 0.05, 0.01)
            pady = max((mxy - mny) * 0.05, 0.01)
            ax.set_xlim(mnx - padx, mxx + padx)
            ax.set_ylim(mny - pady, mxy + pady)

    def _draw_matrix(self, ax):
        cmap = plt.get_cmap(self.matrix_cmap)
        cmap.set_bad('white')
        im, transform = _rotate_matrix(
            ax, self.matrix_,
            cmap=cmap, vmin=self.vmin, vmax=self.vmax,
        )

        # size circles overlay
        if self.size_matrix_ is not None:
            n_rows, n_cols = self.matrix_.shape
            cx_list, cy_list, sizes_list = [], [], []
            for i in range(n_rows):
                for j in range(n_cols):
                    val = self.matrix_[i, j]
                    if np.isnan(val) or val == 0:
                        continue
                    cx, cy = _calculate_rotated_point(
                        transform, n_rows, i, j,
                    )
                    cx_list.append(cx)
                    cy_list.append(cy)
                    sizes_list.append(self.size_matrix_[i, j])
            _draw_size_overlay(ax, cx_list, cy_list, sizes_list)

        for spine in ax.spines.values():
            spine.set_visible(False)
        return im, transform

    def _draw_titles(self, ax_map_o, ax_map_d):
        for ax, title in [(ax_map_o, self.out_title), (ax_map_d, self.in_title)]:
            ax.annotate(title, xy=(0, 0.5), xycoords='axes fraction',
                        rotation=90, va='center', ha='right',
                        xytext=(15, 0), textcoords='offset points',
                        fontsize=self.title_fontsize, weight='bold')

    def _draw_matrix_anchors(self, fig, ax_matrix):
        if self._transform is None or self.matrix_ is None:
            return
        n_rows, n_cols = self.matrix_.shape
        o_xs, o_ys = [], []
        for idx in range(n_cols):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "left", idx,
            )
            o_xs.append(x)
            o_ys.append(y)
        d_xs, d_ys = [], []
        for idx in range(n_rows):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "bottom", idx,
            )
            d_xs.append(x)
            d_ys.append(y)
        ax_matrix.scatter(o_xs, o_ys, s=30, c="cyan", edgecolor="black", zorder=20)
        ax_matrix.scatter(d_xs, d_ys, s=30, c="magenta", edgecolor="black", zorder=20)

    # ==================================================================
    # Colorbar & scale
    # ==================================================================

    def _draw_colorbar(self, fig, ax_matrix):
        if self._im is None:
            return
        kwds = dict(orientation='vertical', aspect=30, shrink=0.9,
                    pad=0.01, location='right')
        kwds.update(self.cbar_kwds)
        cbar = fig.colorbar(self._im, ax=ax_matrix, **kwds)
        cbar.ax.tick_params(labelsize=self.label_fontsize + 2)

    def _scale_sizes(self, values):
        if len(values) == 0 or np.ptp(values) == 0:
            return np.full_like(values, self.max_centroid_size / 2.0, dtype=float)
        return _linear_scaling(values, (max(self.max_centroid_size / 10.0, 20.0),
                                        self.max_centroid_size))


