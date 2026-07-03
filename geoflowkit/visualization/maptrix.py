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
from shapely.geometry import Point

from geoflowkit.visualization.od_matrix import ODMatrixVisualizer
from geoflowkit.visualization._utils import (
    _rotate_matrix,
    _calculate_matrix_anchor_point,
    _calculate_rotated_point,
    _ax_to_fig,
    _linear_scaling,
    _plot_centroids,
    _plot_labels,
    _compute_representative_points,
    _prepare_zones,
)

class MapTrixVisualizer:
    """MapTrix layout: origin/destination maps + rotated OD matrix + guide lines.

    Parameters
    ----------
    zones : GeoDataFrame, optional
        Polygon geometries defining the spatial zones.  Used as a
        backward-compatible alias when *origin_zones* is not given.
    origin_zones : GeoDataFrame, keyword-only, optional
        Zone polygons for the origin map.  Takes precedence over
        *zones*.
    dest_zones : GeoDataFrame, keyword-only, optional
        Zone polygons for the destination map.  When ``None`` (default)
        the same zones are used for both maps.
    zone_id_col : str, optional
        Column in *origin_zones* (or *zones*) used as the zone
        identifier.  ``None`` uses the GeoDataFrame index.
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

    Attributes
    ----------
    matrix_ : np.ndarray
        OD matrix after :meth:`fit`.
    size_matrix_ : np.ndarray or None
        Size values per cell, or ``None`` when *size_weight* is not set.
    zone_ids_ : np.ndarray
        All zone IDs.
    zone_centroids_ : dict
        Mapping zone ID → ``(x, y)`` representative point.
    o_order_, d_order_ : list
        Zone ordering for origin columns / destination rows.
    """

    def __init__(
        self,
        zones: gpd.GeoDataFrame | None = None,
        *,
        origin_zones: gpd.GeoDataFrame | None = None,
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
    ):
        self._matrix_vis = ODMatrixVisualizer(
            zones=zones, origin_zones=origin_zones, dest_zones=dest_zones,
            zone_id_col=zone_id_col, dest_zone_id_col=dest_zone_id_col,
            weight=weight, size_weight=size_weight,
            cmap=matrix_cmap, vmin=vmin, vmax=vmax,
            show_labels=show_labels, label_fontsize=label_fontsize,
            include_self_flows=include_self_flows,
        )

        # convenience references
        self.origin_zones = self._matrix_vis.origin_zones
        self.dest_zones = self._matrix_vis.dest_zones
        self._asymmetric = self._matrix_vis._asymmetric
        self.zone_id_col = zone_id_col
        self.dest_zone_id_col = dest_zone_id_col if dest_zone_id_col is not None else zone_id_col
        self.weight = weight
        self.size_weight = size_weight
        self.matrix_cmap = matrix_cmap
        self.vmin = vmin
        self.vmax = vmax
        self.map_cmap = map_cmap if map_cmap is not None else matrix_cmap
        self.max_centroid_size = max_centroid_size
        self.line_color = line_color
        self.line_alpha = line_alpha
        self.show_labels = show_labels
        self.label_fontsize = label_fontsize
        self.out_title = out_title
        self.in_title = in_title
        self.title_fontsize = title_fontsize
        self.include_self_flows = include_self_flows

        # fit-time
        self.matrix_ = None
        self.size_matrix_ = None
        self._full_matrix = None
        self._full_size_matrix = None
        self.zone_ids_ = None
        self.zone_centroids_ = None
        self.o_zones_ = None
        self.d_zones_ = None
        self._outflows = None
        self._inflows = None
        self.o_order_ = None
        self.d_order_ = None
        self._bounds = None
        self._zone_to_idx = None
        self._d_zone_to_idx = None
        self._o_zone_geometries_ = None
        self._d_zone_geometries_ = None
        self._o_all_ids = None
        self._d_all_ids = None

        # plot-time
        self._im = None
        self._transform = None
        self._o_scatter = None
        self._d_scatter = None
        self._o_labels = []
        self._d_labels = []

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(self, fdf):
        """Aggregate flows and prepare layout."""
        self._matrix_vis.fit(fdf)

        # copy computed data
        self.o_zones_ = self._matrix_vis.o_zones_
        self.d_zones_ = self._matrix_vis.d_zones_
        self.zone_centroids_ = dict(self._matrix_vis.zone_centroids_)
        self._outflows = dict(self._matrix_vis._outflows)
        self._inflows = dict(self._matrix_vis._inflows)
        self._full_matrix = self._matrix_vis._raw_matrix.copy()
        self._full_size_matrix = (
            self._matrix_vis._raw_size_matrix.copy()
            if self._matrix_vis._raw_size_matrix is not None else None
        )
        o_all_ids = list(self._matrix_vis._o_all_ids)
        d_all_ids = list(self._matrix_vis._d_all_ids)
        self._o_all_ids = o_all_ids
        self._d_all_ids = d_all_ids

        # NaN diagonal for MapTrix visual transparency
        if not self.include_self_flows and not self._asymmetric:
            np.fill_diagonal(self._full_matrix, np.nan)

        # zone geometries
        o_prepared = _prepare_zones(self.origin_zones, zone_id_col=self.zone_id_col)
        self._o_zone_geometries_ = {
            row["zone_id"]: row["geometry"]
            for _, row in o_prepared.iterrows()
        }
        if self._asymmetric:
            d_prepared = _prepare_zones(self.dest_zones, zone_id_col=self.dest_zone_id_col)
            self._d_zone_geometries_ = {
                row["zone_id"]: row["geometry"]
                for _, row in d_prepared.iterrows()
            }
        else:
            self._d_zone_geometries_ = self._o_zone_geometries_

        # representative points
        rep_points = _compute_representative_points(
            self.origin_zones, zone_id_col=self.zone_id_col,
        )
        if self._asymmetric:
            rep_points.update(_compute_representative_points(
                self.dest_zones, zone_id_col=self.dest_zone_id_col,
            ))
        for zid in self.zone_centroids_:
            if zid in rep_points:
                self.zone_centroids_[zid] = rep_points[zid]

        # index maps
        self.zone_ids_ = np.array(o_all_ids)
        self._zone_to_idx = {z: i for i, z in enumerate(o_all_ids)}
        self._d_zone_to_idx = (
            {z: i for i, z in enumerate(d_all_ids)}
            if self._asymmetric else self._zone_to_idx
        )

        # initial ordering by centroid Y
        centroids = self.zone_centroids_
        if self._asymmetric:
            self.o_order_ = sorted(
                [z for z in o_all_ids if z in centroids],
                key=lambda z: centroids[z][1],
            )
            self.d_order_ = sorted(
                [z for z in d_all_ids if z in centroids],
                key=lambda z: centroids[z][1],
            )
        else:
            self.o_order_ = sorted(o_all_ids, key=lambda z: centroids[z][1])
            self.d_order_ = sorted(o_all_ids, key=lambda z: centroids[z][1])

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
        o_pos = [self._zone_to_idx[z] for z in self.o_order_]
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

        for side, ax_map, zid_order, flow_dict, width_map, matrix_side in [
            ('origin', ax_map_o, self.o_order_, self._outflows, outflow_w_map, 'left'),
            ('dest',   ax_map_d, self.d_order_, self._inflows,  inflow_w_map, 'bottom'),
        ]:
            for idx, zid in enumerate(zid_order):
                if zid not in self.zone_centroids_:
                    continue

                cx, cy = self.zone_centroids_[zid]
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
        gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 0.5])
        ax_map_o = fig.add_subplot(gs[0, 0])
        ax_map_d = fig.add_subplot(gs[1, 0])
        ax_matrix = fig.add_subplot(gs[:, 1])
        self._draw_map(ax_map_o, self.o_order_, self._outflows,
                       self.origin_zones,
                       scatter='_o_scatter', labels='_o_labels')
        self._draw_map(ax_map_d, self.d_order_, self._inflows,
                       self.dest_zones,
                       scatter='_d_scatter', labels='_d_labels')
        im, transform = self._draw_matrix(ax_matrix)
        self._draw_titles(ax_map_o, ax_map_d)
        return ax_map_o, ax_map_d, ax_matrix, im, transform

    def _draw_map(self, ax, zid_order, flow_dict, zones_gdf, scatter, labels):
        """Draw one map with zone boundaries, centroids, and labels."""
        centroids = {z: self.zone_centroids_[z] for z in zid_order}
        flows = np.array([flow_dict[z] for z in zid_order])
        sizes = self._scale_sizes(flows)
        colors = flows if np.ptp(flows) > 0 else 'k'

        self._draw_map_base(ax, zones_gdf)
        setattr(self, scatter, _plot_centroids(
            ax, centroids, sizes=sizes, colors=colors,
            cmap=self.map_cmap, vmin=self.vmin, vmax=self.vmax,
        ))
        setattr(self, labels, [])
        if self.show_labels:
            lbls = {z: str(z) for z in centroids}
            setattr(self, labels,
                    _plot_labels(ax, centroids, lbls, fontsize=self.label_fontsize))
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
            if sizes_list:
                s_arr = np.array(sizes_list)
                if np.ptp(s_arr) > 0:
                    scaled = _linear_scaling(s_arr, (20, 800))
                else:
                    scaled = np.full_like(s_arr, 200.0)
                ax.scatter(cx_list, cy_list, s=scaled,
                           facecolors='none', edgecolors='gray',
                           linewidths=0.5, alpha=0.7, zorder=5)

        for spine in ax.spines.values():
            spine.set_visible(False)
        return im, transform

    def _draw_titles(self, ax_map_o, ax_map_d):
        for ax, title in [(ax_map_o, self.out_title), (ax_map_d, self.in_title)]:
            ax.annotate(title, xy=(0, 0.5), xycoords='axes fraction',
                        rotation=90, va='center', ha='right',
                        xytext=(15, 0), textcoords='offset points',
                        fontsize=self.title_fontsize, weight='bold')

    def _redraw_centroids(self, ax, positions, zid_order, flow_dict,
                          scatter, labels):
        """Remove old scatter/labels and redraw all centroids at final positions."""
        old_scatter = getattr(self, scatter, None)
        if old_scatter is not None:
            old_scatter.remove()
            setattr(self, scatter, None)
        for t in getattr(self, labels, []):
            t.remove()
        setattr(self, labels, [])

        flows = np.array([flow_dict[z] for z in zid_order])
        sizes = self._scale_sizes(flows)
        kwargs = dict(colors=flows, cmap=self.map_cmap, vmin=self.vmin, vmax=self.vmax) \
            if np.ptp(flows) > 0 else dict(colors='k')
        setattr(self, scatter, _plot_centroids(
            ax, positions, sizes=sizes, zorder=5, **kwargs,
        ))
        if self.show_labels:
            lbls = {z: str(z) for z in zid_order}
            setattr(self, labels, _plot_labels(
                ax, positions, lbls, fontsize=self.label_fontsize,
            ))

    def _sample_zone_candidate_points(self, zid, ax_map, fig, grid_size=17):
        if self._o_zone_geometries_ is None:
            return []
        geom = self._o_zone_geometries_.get(zid)
        if geom is None or geom.is_empty:
            return []
        mnx, mny, mxx, mxy = geom.bounds
        xs = np.linspace(mnx, mxx, grid_size)
        ys = np.linspace(mny, mxy, grid_size)

        candidates = []
        rp = geom.representative_point()
        candidates.append(_ax_to_fig(ax_map, fig, rp.x, rp.y))
        c = geom.centroid
        if geom.contains(c) or geom.touches(c):
            candidates.append(_ax_to_fig(ax_map, fig, c.x, c.y))
        for x in xs:
            for y in ys:
                pt = Point(float(x), float(y))
                if geom.contains(pt) or geom.touches(pt):
                    candidates.append(_ax_to_fig(ax_map, fig, x, y))

        unique, seen = [], set()
        for x, y in candidates:
            key = (round(float(x), 8), round(float(y), 8))
            if key not in seen:
                seen.add(key)
                unique.append((float(x), float(y)))
        return unique

    def _draw_matrix_anchors(self, fig, ax_matrix):
        """Debug helper: draw matrix anchor points for origin and destination."""
        if self._transform is None or self.matrix_ is None:
            return

        n_rows, n_cols = self.matrix_.shape

        # origin anchors: left edge (upper-left diamond edge with CW rotation)
        o_xs, o_ys = [], []
        for idx in range(n_cols):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "left", idx,
            )
            o_xs.append(x)
            o_ys.append(y)

        # destination anchors: bottom edge (lower-left diamond edge with CW rotation)
        d_xs, d_ys = [], []
        for idx in range(n_rows):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "bottom", idx,
            )
            d_xs.append(x)
            d_ys.append(y)

        ax_matrix.scatter(
            o_xs, o_ys,
            s=30, c="cyan", edgecolor="black", zorder=20,
            label="origin anchors",
        )

        ax_matrix.scatter(
            d_xs, d_ys,
            s=30, c="magenta", edgecolor="black", zorder=20,
            label="destination anchors",
        )

    # ==================================================================
    # Colorbar & scale
    # ==================================================================

    def _draw_colorbar(self, fig, ax_matrix):
        if self._im is None:
            return
        cbar = fig.colorbar(self._im, ax=ax_matrix, pad=0.01,
                            orientation='vertical', aspect=30, shrink=0.9,
                            location='right')
        cbar.ax.tick_params(labelsize=self.label_fontsize + 2)

    def _scale_sizes(self, values):
        if len(values) == 0 or np.ptp(values) == 0:
            return np.full_like(values, self.max_centroid_size / 2.0, dtype=float)
        return _linear_scaling(values, (max(self.max_centroid_size / 10.0, 20.0),
                                        self.max_centroid_size))


