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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from geoflowkit.visualization._utils import (
    _assign_zones,
    _build_od_matrix,
    _rotate_matrix,
    _calculate_matrix_anchor_point,
    _ax_to_fig,
    _fig_to_ax,
    _linear_scaling,
    _plot_centroids,
    _plot_labels,
    _compute_representative_points,
    _prepare_zones,
)
from geoflowkit.visualization._gp_optimizer import GPLayoutOptimizer

class MapTrixVisualizer:
    """MapTrix layout: origin/destination maps + rotated OD matrix + guide lines.

    Parameters
    ----------
    zones : GeoDataFrame
        Polygon geometries defining the spatial zones.
    zone_id_col : str, optional
        Column in *zones* used as the zone identifier.  ``None`` uses
        the GeoDataFrame index.
    weight : str, default='count'
        Aggregation weight: ``'count'``, ``'length'``,
        ``'divergence'``, or ``'entropy'``.
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
        When ``False`` the diagonal is zeroed.
    leader_sep_weight : float, default=8.0
        DP weight for uniform guide-line spacing.
    leader_center_weight : float, default=1.0
        DP penalty for moving connection points from the original centroid.
    leader_min_c_gap : float, default=0.003
        Minimum gap between neighbouring characteristic values *C*.
    leader_max_candidates : int, default=45
        Max candidate points per zone during polygon sampling.
    leader_split_candidates : int, default=25
        Number of candidate split-line positions to enumerate during
        joint layout optimisation.
    leader_split_balance_weight : float, default=0.05
        Penalty weight for unbalanced split-lines (ratio of upper/lower
        zone counts) in layout scoring.
    show_split_lines : bool, default=False
        Draw horizontal split lines on the maps.

    Attributes
    ----------
    matrix_ : np.ndarray
        OD matrix after :meth:`fit`.
    zone_ids_ : np.ndarray
        All zone IDs.
    zone_centroids_ : dict
        Mapping zone ID → ``(x, y)`` representative point.
    o_order_, d_order_ : list
        Zone ordering for origin columns / destination rows.
    origin_split_y_, dest_split_y_ : float or None
        Split-line y (figure coords); set during ordering optimisation.
    """

    def __init__(self, zones, zone_id_col=None, weight='count',
                 matrix_cmap='OrRd', vmin=None, vmax=None,
                 map_cmap=None, max_centroid_size=300.0,
                 line_color='black', line_alpha=0.6,
                 show_labels=True, label_fontsize=10,
                 out_title='Outflow', in_title='Inflow',
                 title_fontsize=16, include_self_flows=True,
                 spacing_weight=8.0, center_weight=1.0):
        self.zones = zones
        self.zone_id_col = zone_id_col
        self.weight = weight
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
        self._full_matrix = None
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
        self._zone_geometries_ = None

        # plot-time
        self._im = None
        self._transform = None
        self._o_scatter = None
        self._d_scatter = None
        self._o_labels = []
        self._d_labels = []

        # GP result cache (populated by plot() when use_gp=True)
        self._gp_origin_result = None
        self._gp_dest_result = None

        # guide-line geometry
        self._angle_deg = 45.0
        self._spacing_weight = spacing_weight
        self._center_weight = center_weight

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(self, fdf):
        """Aggregate flows and prepare layout."""
        # Zone assignment
        self.o_zones_, self.d_zones_, self.zone_centroids_ = _assign_zones(
            fdf, self.zones, zone_id_col=self.zone_id_col,
        )
        rep_points = _compute_representative_points(
            self.zones, zone_id_col=self.zone_id_col,
        )
        for zid in self.zone_centroids_:
            if zid in rep_points:
                self.zone_centroids_[zid] = rep_points[zid]

        zones_prepared = _prepare_zones(self.zones, zone_id_col=self.zone_id_col)
        self._zone_geometries_ = {
            row["zone_id"]: row["geometry"]
            for _, row in zones_prepared.iterrows()
        }

        all_zone_ids = list(self.zone_centroids_.keys())
        self.zone_ids_ = np.array(all_zone_ids)
        self._zone_to_idx = {z: i for i, z in enumerate(all_zone_ids)}

        # Build OD matrix
        raw_matrix, raw_zone_ids = _build_od_matrix(
            fdf, self.o_zones_, self.d_zones_, weight=self.weight,
        )
        n_all = len(all_zone_ids)
        full_matrix = np.zeros((n_all, n_all))
        raw_idx = {z: i for i, z in enumerate(raw_zone_ids)}
        for ri, rz in enumerate(all_zone_ids):
            for ci, cz in enumerate(all_zone_ids):
                if rz in raw_idx and cz in raw_idx:
                    full_matrix[ri, ci] = raw_matrix[raw_idx[rz], raw_idx[cz]]
        if not self.include_self_flows:
            np.fill_diagonal(full_matrix, np.nan)
        self._full_matrix = full_matrix

        # Per-zone flows
        if self.weight == 'length':
            w_values = fdf.length.values
        elif self.weight == 'divergence':
            w_values = fdf.angle.values
        else:
            w_values = np.ones(len(fdf))
        self._outflows = {z: 0.0 for z in all_zone_ids}
        self._inflows = {z: 0.0 for z in all_zone_ids}
        for i in range(len(fdf)):
            self._outflows[self.o_zones_[i]] += w_values[i]
            self._inflows[self.d_zones_[i]] += w_values[i]

        # Initial ordering by centroid Y
        centroids = self.zone_centroids_
        self.o_order_ = sorted(all_zone_ids, key=lambda z: centroids[z][1])
        self.d_order_ = sorted(all_zone_ids, key=lambda z: centroids[z][1])

        # Exclude zero-flow zones (keep at least one when non-empty)
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
        d_pos = [self._zone_to_idx[z] for z in self.d_order_]
        self.matrix_ = self._full_matrix[np.ix_(o_pos, d_pos)].T

    # ==================================================================
    # Plot
    # ==================================================================

    def plot(self, fig=None, figsize=(14, 8)):
        """Create the full MapTrix figure."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before plot().")
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # Build figure
        ax_map_o, ax_map_d, ax_matrix, im, transform = self._build_figure(fig)
        plt.subplots_adjust(hspace=0.01, wspace=-0.15, left=0.0001)
        self._draw_colorbar(fig, ax_matrix)
        fig.canvas.draw()
        self._im = im
        self._transform = transform

        # GP layout optimisation
        if len(self.o_order_) > 1 or len(self.d_order_) > 1:
            ax_map_o, ax_map_d, ax_matrix = self._run_gp_optimisation(
                fig, ax_map_o, ax_map_d, ax_matrix)

        self._debug_draw_matrix_anchors(fig, ax_matrix)

        # Draw GP guide lines
        if self._gp_origin_result is not None:
            self._draw_gp_guide_lines(fig, ax_map_o, ax_map_d)
        return fig

    def fit_plot(self, fdf, fig=None, figsize=(14, 8)):
        self.fit(fdf)
        return self.plot(fig=fig, figsize=figsize)

    # ==================================================================
    # GP optimisation
    # ==================================================================

    def _run_gp_optimisation(self, fig, ax_map_o, ax_map_d, ax_matrix):
        """Run GP layout optimisation for both sides.

        Runs GP using the *current* figure axes.  After optimisation the
        ordering and matrix data are updated in-place on the existing
        axes — the figure is **not** cleared or rebuilt, so all
        figure-coordinate positions computed by GP remain valid.
        """
        n_rows, n_cols = self.matrix_.shape
        transform = self._transform

        # ---- Origin side ----
        if len(self.o_order_) > 1:
            origin_ids = list(self.o_order_)
            origin_centroids_fig = {
                z: _ax_to_fig(
                    ax_map_o, fig, *self.zone_centroids_[z])
                for z in origin_ids
            }

            gp_origin = GPLayoutOptimizer(
                angle_deg=self._angle_deg,
                center_weight=self._center_weight,
                spacing_weight=self._spacing_weight,
                random_state=42,
            )
            self._gp_origin_result = gp_origin.optimize(
                zone_ids=origin_ids,
                zone_geometries=self._zone_geometries_,
                zone_centroids_fig=origin_centroids_fig,
                ax_map=ax_map_o,
                ax_matrix=ax_matrix,
                fig=fig,
                matrix_shape=(n_rows, n_cols),
                transform=transform,
                is_origin=True,
            )
            self.o_order_ = self._gp_origin_result["order"]
        else:
            self._gp_origin_result = None

        # ---- Destination side ----
        if len(self.d_order_) > 1:
            dest_ids = list(self.d_order_)
            dest_centroids_fig = {
                z: _ax_to_fig(
                    ax_map_d, fig, *self.zone_centroids_[z])
                for z in dest_ids
            }

            gp_dest = GPLayoutOptimizer(
                angle_deg=self._angle_deg,
                center_weight=self._center_weight,
                spacing_weight=self._spacing_weight,
                random_state=42,
            )
            self._gp_dest_result = gp_dest.optimize(
                zone_ids=dest_ids,
                zone_geometries=self._zone_geometries_,
                zone_centroids_fig=dest_centroids_fig,
                ax_map=ax_map_d,
                ax_matrix=ax_matrix,
                fig=fig,
                matrix_shape=(n_rows, n_cols),
                transform=transform,
                is_origin=False,
            )
            self.d_order_ = self._gp_dest_result["order"]
        else:
            self._gp_dest_result = None

        # Apply ordering and update matrix *without* rebuilding figure.
        # This keeps all figure-coordinate geoms from GP valid.
        changed = (self._gp_origin_result is not None
                   or self._gp_dest_result is not None)
        if changed:
            self._apply_ordering()
            # Update the imshow data in-place (same axes, same transform)
            if self._im is not None:
                self._im.set_data(self.matrix_)
            fig.canvas.draw()

        return ax_map_o, ax_map_d, ax_matrix

    def _draw_gp_guide_lines(self, fig, ax_map_o, ax_map_d):
        """Draw guide lines directly from GP-optimised results."""
        from geoflowkit.visualization._utils import (
            _fig_to_ax, _linear_scaling, _plot_centroids, _plot_labels)

        # Line-width scaling
        def _width_map(flow_dict):
            if not flow_dict:
                return {}
            vals = np.array(list(flow_dict.values()))
            widths = _linear_scaling(vals, (0.5, 2.5))
            return {z: w for z, w in zip(flow_dict.keys(), widths)}

        outflow_w_map = _width_map(self._outflows)
        inflow_w_map = _width_map(self._inflows)

        for side, result, ax, zid_order, flow_dict, scatter, labels, width_map in [
            ('origin', self._gp_origin_result, ax_map_o,
             self.o_order_, self._outflows,
             '_o_scatter', '_o_labels', outflow_w_map),
            ('dest', self._gp_dest_result, ax_map_d,
             self.d_order_, self._inflows,
             '_d_scatter', '_d_labels', inflow_w_map),
        ]:
            if result is None:
                continue

            positions = {}
            for item in result['geoms']:
                zid = item['zid']
                geom = item['geom']
                if geom is None:
                    continue
                p_x, p_y = geom['p']
                q_x, q_y = geom['q']
                m_x, m_y = geom['m']

                fig.add_artist(Line2D(
                    [p_x, q_x, m_x],
                    [p_y, q_y, m_y],
                    transform=fig.transFigure,
                    color=self.line_color,
                    linewidth=width_map.get(zid, 1.0),
                    alpha=self.line_alpha,
                    solid_capstyle='round',
                    clip_on=False,
                    zorder=10,
                ))

                data_xy = _fig_to_ax(ax, fig, p_x, p_y)
                positions[zid] = (data_xy[0], data_xy[1])

            # Draw split line if available
            split_y = result.get('split_y')
            if split_y is not None:
                box = ax.get_position()
                fig.add_artist(Line2D(
                    [box.x0, box.x1], [split_y, split_y],
                    transform=fig.transFigure, color='0.75',
                    linewidth=0.8, linestyle='--', alpha=0.8,
                    clip_on=False, zorder=8))

            self._redraw_centroids(
                ax, positions, zid_order, flow_dict, scatter, labels)

    # ==================================================================
    # Figure construction
    # ==================================================================

    def _build_figure(self, fig):
        gs = fig.add_gridspec(2, 2, height_ratios=[0.5, 0.5])
        ax_map_o = fig.add_subplot(gs[0, 0])
        ax_map_d = fig.add_subplot(gs[1, 0])
        ax_matrix = fig.add_subplot(gs[:, 1])
        self._draw_map(ax_map_o, self.o_order_, self._outflows,
                       scatter='_o_scatter', labels='_o_labels')
        self._draw_map(ax_map_d, self.d_order_, self._inflows,
                       scatter='_d_scatter', labels='_d_labels')
        im, transform = self._draw_matrix(ax_matrix)
        self._draw_titles(ax_map_o, ax_map_d)
        return ax_map_o, ax_map_d, ax_matrix, im, transform

    def _draw_map(self, ax, zid_order, flow_dict, scatter, labels):
        """Draw one map with zone boundaries, centroids, and labels."""
        centroids = {z: self.zone_centroids_[z] for z in zid_order}
        flows = np.array([flow_dict[z] for z in zid_order])
        sizes = self._scale_sizes(flows)
        colors = flows if np.ptp(flows) > 0 else 'k'

        self._draw_map_base(ax)
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

    def _draw_map_base(self, ax):
        if self.zones is not None:
            self.zones.boundary.plot(
                ax=ax, edgecolor='gray', linewidth=0.5, facecolor='none',
            )
        if self.zones is not None and len(self.zones) > 0:
            mnx, mny, mxx, mxy = self.zones.total_bounds
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
        if self._zone_geometries_ is None:
            return []
        geom = self._zone_geometries_.get(zid)
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
    def _debug_draw_matrix_anchors(self, fig, ax_matrix):
        """Debug helper: draw matrix anchor points on top and left edges."""
        if self._transform is None or self.matrix_ is None:
            return

        n_rows, n_cols = self.matrix_.shape

        # origin anchors: top edge
        top_xs, top_ys = [], []
        for j in range(n_cols):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "top", j,
            )
            top_xs.append(x)
            top_ys.append(y)

        # destination anchors: left edge
        left_xs, left_ys = [], []
        for i in range(n_rows):
            x, y = _calculate_matrix_anchor_point(
                self._transform, n_rows, n_cols, "left", i,
            )
            left_xs.append(x)
            left_ys.append(y)

        ax_matrix.scatter(
            top_xs, top_ys,
            s=30,
            c="cyan",
            edgecolor="black",
            zorder=20,
            label="origin anchors",
        )

        ax_matrix.scatter(
            left_xs, left_ys,
            s=30,
            c="magenta",
            edgecolor="black",
            zorder=20,
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

# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def maptrix(fdf, zones, zone_id_col=None, weight='count',
            matrix_cmap='OrRd', vmin=None, vmax=None,
            map_cmap=None, max_centroid_size=300.0,
            line_color='black', line_alpha=0.6,
            show_labels=True, label_fontsize=10,
            out_title='Outflow', in_title='Inflow',
            title_fontsize=16, include_self_flows=True,
            spacing_weight=8.0, center_weight=1.0,
            fig=None, figsize=(14, 8)):
    """Create a MapTrix visualisation for flow data.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zones : GeoDataFrame
        Zone polygon geometries.
    zone_id_col : str, optional
        Zone identifier column.
    weight : str, default='count'
        ``'count'``, ``'length'``, ``'divergence'``, or ``'entropy'``.
    matrix_cmap : str or Colormap, default='OrRd'
        Matrix heatmap colormap.
    vmin, vmax : float, optional
        Colormap range.
    map_cmap : str or Colormap, optional
        Centroid circle colormap (defaults to *matrix_cmap*).
    max_centroid_size : float, default=300.0
        Max marker size.
    line_color : str, default='black'
        Guide line colour.
    line_alpha : float, default=0.6
        Guide line transparency.
    show_labels : bool, default=True
        Show zone ID labels.
    label_fontsize : int, default=10
        Label font size.
    out_title : str, default='Outflow'
        Origin map title.
    in_title : str, default='Inflow'
        Destination map title.
    title_fontsize : int, default=16
        Map title font size.
    include_self_flows : bool, default=True
        Include diagonal entries in the matrix.
    spacing_weight : float, default=8.0
        GP fitness weight for guide-line spacing uniformity.
    center_weight : float, default=1.0
        GP fitness weight for centroid deviation.
    fig : matplotlib.figure.Figure, optional
    figsize : tuple, default=(14, 8)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    vis = MapTrixVisualizer(
        zones=zones, zone_id_col=zone_id_col, weight=weight,
        matrix_cmap=matrix_cmap, vmin=vmin, vmax=vmax,
        map_cmap=map_cmap, max_centroid_size=max_centroid_size,
        line_color=line_color, line_alpha=line_alpha,
        show_labels=show_labels, label_fontsize=label_fontsize,
        out_title=out_title, in_title=in_title,
        title_fontsize=title_fontsize, include_self_flows=include_self_flows,
        spacing_weight=spacing_weight, center_weight=center_weight,
    )
    return vis.fit_plot(fdf, fig=fig, figsize=figsize)
