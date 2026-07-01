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
from shapely.geometry import Point

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
from geoflowkit.visualization._gp_optimizer import (
    GPLayoutOptimizer,
    gp_polyline_geometry,
)


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
                 leader_sep_weight=8.0, leader_center_weight=1.0,
                 leader_min_c_gap=0.003, leader_max_candidates=45,
                 leader_split_candidates=25,
                 leader_split_balance_weight=0.05,
                 show_split_lines=False,
                 use_gp=True, gp_random_state=None):
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
        self.show_split_lines = show_split_lines
        self.use_gp = use_gp
        self._gp_random_state = gp_random_state

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
        self._leader_angle_deg = 45.0
        self.origin_split_y_ = None
        self.dest_split_y_ = None
        self._leader_sep_weight = leader_sep_weight
        self._leader_center_weight = leader_center_weight
        self._leader_min_c_gap = leader_min_c_gap
        self._leader_max_candidates = leader_max_candidates
        self._leader_split_candidates = leader_split_candidates
        self._leader_split_balance_weight = leader_split_balance_weight

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
            np.fill_diagonal(full_matrix, 0.0)
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

    def plot(self, fig=None, figsize=(14, 8), optimize_order=True):
        """Create the full MapTrix figure."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before plot().")
        if fig is None:
            fig = plt.figure(figsize=figsize)

        need_opt = optimize_order and len(self.o_order_) > 1

        # Build figure
        ax_map_o, ax_map_d, ax_matrix, im, transform = self._build_figure(fig)
        plt.subplots_adjust(hspace=0.01, wspace=-0.15, left=0.0001)
        fig.canvas.draw()
        self._im = im
        self._transform = transform

        # Layout optimisation
        if need_opt:
            if self.use_gp:
                # ---- Genetic programming ----
                ax_map_o, ax_map_d, ax_matrix = self._run_gp_optimisation(
                    fig, ax_map_o, ax_map_d, ax_matrix)
            else:
                # ---- DP optimisation (legacy) ----
                o_pos, d_pos = self._compute_guide_positions(
                    fig, ax_map_o, ax_map_d)

                try:
                    new_o, new_d, new_o_split, new_d_split = \
                        self._optimize_layout_joint(
                            fig=fig,
                            ax_map_o=ax_map_o,
                            ax_map_d=ax_map_d,
                            ax_matrix=ax_matrix,
                            origin_positions=o_pos,
                            dest_positions=d_pos,
                        )
                except RuntimeError:
                    new_o, new_d = list(self.o_order_), list(self.d_order_)
                    new_o_split, new_d_split = (
                        self.origin_split_y_, self.dest_split_y_)

                changed = (
                    new_o != list(self.o_order_)
                    or new_d != list(self.d_order_)
                    or self.origin_split_y_ != new_o_split
                    or self.dest_split_y_ != new_d_split
                )

                if changed:
                    self.o_order_ = new_o
                    self.d_order_ = new_d
                    self.origin_split_y_ = new_o_split
                    self.dest_split_y_ = new_d_split

                    self._apply_ordering()

                    fig.clear()
                    ax_map_o, ax_map_d, ax_matrix, im, transform = (
                        self._build_figure(fig))
                    plt.subplots_adjust(
                        hspace=0.01, wspace=-0.15, left=0.0001)
                    fig.canvas.draw()

                    self._im = im
                    self._transform = transform

        self._draw_colorbar(fig, ax_matrix)
        fig.canvas.draw()

        self._debug_draw_matrix_anchors(fig, ax_matrix)

        # Draw guide lines
        if self.use_gp and self._gp_origin_result is not None:
            self._draw_gp_guide_lines(fig, ax_map_o, ax_map_d)
        else:
            self._draw_guide_lines(fig, ax_map_o, ax_map_d, ax_matrix)
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
                angle_deg=self._leader_angle_deg,
                center_weight=self._leader_center_weight,
                spacing_weight=self._leader_sep_weight,
                random_state=self._gp_random_state,
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
                angle_deg=self._leader_angle_deg,
                center_weight=self._leader_center_weight,
                spacing_weight=self._leader_sep_weight,
                random_state=self._gp_random_state,
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
        import numpy as np
        from matplotlib.lines import Line2D
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
        im, transform = _rotate_matrix(
            ax, self.matrix_,
            cmap=self.matrix_cmap, vmin=self.vmin, vmax=self.vmax,
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

    # ==================================================================
    # Guide lines
    # ==================================================================

    def _draw_guide_lines(self, fig, ax_map_o, ax_map_d, ax_matrix):
        """Collect records, optimise connection points, draw guide lines."""
        n_d_rows = len(self.d_order_)
        transform = self._transform

        # Scale line widths by flow volume
        def _width_map(flow_dict):
            if not flow_dict:
                return {}
            vals = np.array(list(flow_dict.values()))
            widths = _linear_scaling(vals, (0.5, 2.5))
            return {z: w for z, w in zip(flow_dict.keys(), widths)}

        outflow_w_map = _width_map(self._outflows)
        inflow_w_map = _width_map(self._inflows)

        # Resolve split lines
        for split_attr, ax in [('origin_split_y_', ax_map_o),
                                ('dest_split_y_', ax_map_d)]:
            if getattr(self, split_attr) is None:
                box = ax.get_position()
                setattr(self, split_attr, 0.5 * (box.y0 + box.y1))

        if self.show_split_lines:
            for split_y, ax in [(self.origin_split_y_, ax_map_o),
                                 (self.dest_split_y_, ax_map_d)]:
                self._draw_horizontal_split_line(fig, ax, split_y)

        # Process origin lines (top edge, c_direction=+1)
        self._process_line_group(
            fig, ax_map_o, ax_matrix, self.o_order_,
            outflow_w_map, self.origin_split_y_, +1,
            scatter='_o_scatter', labels='_o_labels',
            flow_dict=self._outflows, n_d_rows=n_d_rows, transform=transform,
        )

        # Process destination lines (left edge, c_direction=-1)
        self._process_line_group(
            fig, ax_map_d, ax_matrix, self.d_order_,
            inflow_w_map, self.dest_split_y_, -1,
            scatter='_d_scatter', labels='_d_labels',
            flow_dict=self._inflows, n_d_rows=n_d_rows, transform=transform,
        )

    def _process_line_group(self, fig, ax_map, ax_matrix, zid_order,
                            width_map, split_y, c_direction,
                            scatter, labels, flow_dict, n_d_rows, transform):
        """Collect records, run DP optimisation, draw lines, redraw centroids."""
        is_origin = (c_direction == +1)
        records = []

        n_rows, n_cols = self.matrix_.shape

        for idx, zid in enumerate(zid_order):
            if is_origin:
                m_x, m_y = _calculate_matrix_anchor_point(
                    transform=transform,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    side="top",
                    index=idx,
                )
            else:
                m_x, m_y = _calculate_matrix_anchor_point(
                    transform=transform,
                    n_rows=n_rows,
                    n_cols=n_cols,
                    side="left",
                    index=idx,
                )
            m_fig = _ax_to_fig(ax_matrix, fig, m_x, m_y)

            cx, cy = self.zone_centroids_[zid]
            raw_fig = _ax_to_fig(ax_map, fig, cx, cy)

            records.append({
                "zid": zid, "matrix_fig": m_fig, "raw_map_fig": raw_fig,
                "is_upper": raw_fig[1] >= split_y,
                "linewidth": width_map.get(zid, 1.0),
            })

        # DP optimise all connection points
        try:
            optimized = self._optimize_group_connection_points(
                records, ax_map=ax_map, fig=fig,
                split_y=split_y, angle_deg=self._leader_angle_deg,
                c_direction=c_direction,
            )
        except RuntimeError:
            # Optimisation fell back — use raw map points for all zones.
            optimized = {rec["zid"]: rec["raw_map_fig"] for rec in records}

        # Collect final geometries and validate before drawing
        final_items = []
        positions = {}

        for rec in records:
            zid = rec["zid"]

            map_fig = optimized.get(zid, rec["raw_map_fig"])

            geom = self._leader_polyline_geometry(
                map_fig=map_fig,
                matrix_fig=rec["matrix_fig"],
                split_y=split_y,
                angle_deg=self._leader_angle_deg,
                require_feasible=True,
                required_upper=rec["is_upper"],
            )

            if geom is None:
                # Geometry check failed with required_upper — try without.
                geom = self._leader_polyline_geometry(
                    map_fig=map_fig,
                    matrix_fig=rec["matrix_fig"],
                    split_y=split_y,
                    angle_deg=self._leader_angle_deg,
                    require_feasible=True,
                    required_upper=None,
                )

            if geom is None:
                # Still no valid geometry — search the zone polygon for
                # any feasible connection point.
                adj = self._adjust_map_point_for_split_line(
                    rec["zid"], ax_map, fig,
                    rec["raw_map_fig"], rec["matrix_fig"],
                    split_y, self._leader_angle_deg,
                )
                geom = self._leader_polyline_geometry(
                    map_fig=adj,
                    matrix_fig=rec["matrix_fig"],
                    split_y=split_y,
                    angle_deg=self._leader_angle_deg,
                    require_feasible=True,
                    required_upper=None,
                )
                if geom is not None:
                    map_fig = adj

            if geom is not None:
                final_items.append((rec, map_fig, geom))
            else:
                data_xy = _fig_to_ax(ax_map, fig, map_fig[0], map_fig[1])
                positions[zid] = (data_xy[0], data_xy[1])

        # Debug / strict check
        self._check_no_opposite_diagonal_crossings(
            [(rec["zid"], geom) for rec, map_fig, geom in final_items]
        )

        # Only draw after validation
        for rec, map_fig, geom in final_items:
            zid = rec["zid"]

            p_x, p_y = geom["p"]
            q_x, q_y = geom["q"]
            m_x, m_y = geom["m"]

            fig.add_artist(Line2D(
                [p_x, q_x, m_x],
                [p_y, q_y, m_y],
                transform=fig.transFigure,
                color=self.line_color,
                linewidth=rec["linewidth"],
                alpha=self.line_alpha,
                solid_capstyle="round",
                zorder=10,
            ))

            data_xy = _fig_to_ax(ax_map, fig, map_fig[0], map_fig[1])
            positions[zid] = (data_xy[0], data_xy[1])

        # Replace old scatter/labels with final optimised positions
        self._redraw_centroids(ax_map, positions, zid_order, flow_dict,
                               scatter, labels)

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

    # ==================================================================
    # Split-line helpers
    # ==================================================================

    def _draw_horizontal_split_line(self, fig, ax, split_y,
                                     color="0.75", linewidth=0.8,
                                     linestyle="--"):
        box = ax.get_position()
        fig.add_artist(Line2D(
            [box.x0, box.x1], [split_y, split_y],
            transform=fig.transFigure, color=color,
            linewidth=linewidth, linestyle=linestyle, alpha=0.8, zorder=8,
        ))

    def _leader_characteristic(self, map_fig, is_upper, tan_k):
        x, y = map_fig
        return y - tan_k * x if is_upper else y + tan_k * x

    def _leader_polyline_geometry(self, map_fig, matrix_fig, split_y,
                                  angle_deg=45.0, require_feasible=True,
                                  required_upper=None):
        """Return geometry of one MapTrix guide line.

        Parameters
        ----------
        map_fig : tuple
            Map connection point in figure coordinates.
        matrix_fig : tuple
            Matrix anchor point in figure coordinates.
        split_y : float
            Horizontal split line in figure coordinates.
        angle_deg : float
            Leader diagonal angle relative to horizontal.
        require_feasible : bool
            If True, return None when the point/anchor configuration cannot
            produce a valid split L-shaped leader.
        required_upper : bool or None
            If True, the map point must be in the upper group (p_y >= split_y).
            If False, it must be in the lower group (p_y < split_y).
            If None, the side is auto-detected.

        Returns
        -------
        geom : dict or None
        """
        p_x, p_y = map_fig
        m_x, m_y = matrix_fig

        tan_k = np.tan(np.deg2rad(angle_deg))
        eps = 1e-10

        if abs(tan_k) < eps:
            return None

        actual_upper = p_y >= split_y

        if required_upper is not None and actual_upper != required_upper:
            return None

        is_upper = actual_upper

        # upper line: map point below its matrix anchor, diagonal goes upward
        # lower line: map point above its matrix anchor, diagonal goes downward
        if is_upper:
            if require_feasible and not (m_y > split_y and m_y > p_y):
                return None
            gap = m_y - p_y
        else:
            if require_feasible and not (m_y < split_y and m_y < p_y):
                return None
            gap = p_y - m_y

        if require_feasible and gap <= eps:
            return None

        cross_x = p_x + gap / tan_k
        cross_y = m_y

        # Elbow should be between map point and matrix anchor.
        if require_feasible and not (p_x < cross_x < m_x):
            return None

        p = (float(p_x), float(p_y))
        q = (float(cross_x), float(cross_y))
        m = (float(m_x), float(m_y))

        return {
            "p": p,
            "q": q,
            "m": m,
            "is_upper": is_upper,
            "diag": (p, q),
            "horiz": (q, m),
        }

    def _is_split_line_feasible(self, map_fig, matrix_fig, split_y,
                                 angle_deg=45.0):
        return self._leader_polyline_geometry(
            map_fig=map_fig,
            matrix_fig=matrix_fig,
            split_y=split_y,
            angle_deg=angle_deg,
            require_feasible=True,
        ) is not None

    # ==================================================================
    # Candidate sampling & adjustment
    # ==================================================================

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

    def _adjust_map_point_for_split_line(self, zid, ax_map, fig,
                                          raw_map_fig, matrix_fig, split_y,
                                          angle_deg=45.0):
        """Single-zone fallback: search feasible point, prefer nearest raw."""
        if self._is_split_line_feasible(raw_map_fig, matrix_fig, split_y, angle_deg):
            return raw_map_fig
        feasible = self._sample_feasible(
            zid, ax_map, fig, matrix_fig, split_y, angle_deg, grid_size=19,
        ) or self._sample_feasible(
            zid, ax_map, fig, matrix_fig, split_y, angle_deg, grid_size=31,
        )
        if feasible:
            raw_x, raw_y = raw_map_fig
            return min(feasible, key=lambda p: (p[0] - raw_x) ** 2 + (p[1] - raw_y) ** 2)
        return raw_map_fig

    def _sample_feasible(self, zid, ax_map, fig, matrix_fig, split_y,
                         angle_deg, grid_size):
        cands = self._sample_zone_candidate_points(zid, ax_map, fig, grid_size)
        return [p for p in cands if self._is_split_line_feasible(
            p, matrix_fig, split_y, angle_deg)]

    def _build_feasible_candidates_for_zone(
            self, zid, ax_map, fig, raw_map_fig, matrix_fig,
            split_y, angle_deg=45.0, grid_size=31, force_upper=None):
        """Build feasible candidate dicts for DP.

        Returns an empty list if this zone has no legal guide-line candidate
        under the current split-line, matrix anchor, and side constraint.

        The caller must treat an empty result as a hard layout failure.
        """
        tan_k = np.tan(np.deg2rad(angle_deg))
        raw_x, raw_y = raw_map_fig

        def _collect(gs, fu):
            cands = self._sample_zone_candidate_points(zid, ax_map, fig, gs)
            cands.append(raw_map_fig)
            out, seen = [], set()
            for px, py in cands:
                key = (round(float(px), 8), round(float(py), 8))
                if key in seen:
                    continue
                seen.add(key)
                geom = self._leader_polyline_geometry(
                    map_fig=(px, py),
                    matrix_fig=matrix_fig,
                    split_y=split_y,
                    angle_deg=angle_deg,
                    require_feasible=True,
                    required_upper=fu,
                )
                if geom is None:
                    continue
                C = self._leader_characteristic((px, py), geom["is_upper"], tan_k)
                out.append({
                    "point": (px, py),
                    "center_cost": (px - raw_x) ** 2 + (py - raw_y) ** 2,
                    "C": C,
                    "geom": geom,
                })
            out.sort(key=lambda d: d["center_cost"])
            return out[:self._leader_max_candidates]

        out = _collect(grid_size, force_upper)

        if not out:
            out = _collect(max(grid_size + 20, 51), force_upper)

        # Last resort: try adjust, but only if it keeps the required side.
        if not out:
            adj = self._adjust_map_point_for_split_line(
                zid, ax_map, fig, raw_map_fig, matrix_fig, split_y, angle_deg,
            )

            adj_is_upper = adj[1] >= split_y

            if force_upper is not None and adj_is_upper != force_upper:
                return []

            geom = self._leader_polyline_geometry(
                map_fig=adj,
                matrix_fig=matrix_fig,
                split_y=split_y,
                angle_deg=angle_deg,
                require_feasible=True,
            )

            if geom is not None:
                C = self._leader_characteristic(adj, adj_is_upper, tan_k)
                out = [{
                    "point": adj,
                    "center_cost": (adj[0] - raw_x) ** 2 + (adj[1] - raw_y) ** 2,
                    "C": C,
                    "geom": geom,
                }]

        if not out:
            return []

        return out

    # ==================================================================
    # DP optimisation
    # ==================================================================

    def _orient(self, a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def _on_segment(self, a, b, c, eps=1e-10):
        """Return True if point c lies on segment ab."""
        return (
            min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
            and abs(self._orient(a, b, c)) <= eps
        )

    def _segments_intersect(self, s1, s2, eps=1e-10):
        """Robust segment intersection test."""
        a, b = s1
        c, d = s2

        o1 = self._orient(a, b, c)
        o2 = self._orient(a, b, d)
        o3 = self._orient(c, d, a)
        o4 = self._orient(c, d, b)

        # Proper intersection
        if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
           ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
            return True

        # Collinear / endpoint cases
        if abs(o1) <= eps and self._on_segment(a, b, c, eps):
            return True
        if abs(o2) <= eps and self._on_segment(a, b, d, eps):
            return True
        if abs(o3) <= eps and self._on_segment(c, d, a, eps):
            return True
        if abs(o4) <= eps and self._on_segment(c, d, b, eps):
            return True

        return False

    def _same_point(self, a, b, eps=1e-8):
        return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps

    def _leader_geometries_cross(self, g1, g2):
        """Return True if two L-shaped leaders cross.

        Shared endpoints are ignored, though in this MapTrix layout different
        leaders should normally not share endpoints anyway.
        """
        if g1 is None or g2 is None:
            return True

        segs1 = [g1["diag"], g1["horiz"]]
        segs2 = [g2["diag"], g2["horiz"]]

        for s1 in segs1:
            for s2 in segs2:
                if not self._segments_intersect(s1, s2):
                    continue

                # Ignore exact shared endpoints.
                shared_endpoint = False
                for p in s1:
                    for q in s2:
                        if self._same_point(p, q):
                            shared_endpoint = True
                            break
                    if shared_endpoint:
                        break

                if not shared_endpoint:
                    return True

        return False

    def _diagonal_segments_cross(self, g1, g2):
        """Check only diagonal segment intersection."""
        if g1 is None or g2 is None:
            return False

        s1 = g1["diag"]
        s2 = g2["diag"]

        if not self._segments_intersect(s1, s2):
            return False

        # Ignore exact shared endpoints.
        for p in s1:
            for q in s2:
                if self._same_point(p, q):
                    return False

        return True

    def _check_no_opposite_diagonal_crossings(self, final_geoms):
        """Check upward vs downward diagonal segment crossings.

        Parameters
        ----------
        final_geoms : list of tuple
            List of ``(zid, geom)``.
        """
        for i in range(len(final_geoms)):
            zid_i, g_i = final_geoms[i]
            for j in range(i + 1, len(final_geoms)):
                zid_j, g_j = final_geoms[j]

                if g_i["is_upper"] == g_j["is_upper"]:
                    continue

                if self._diagonal_segments_cross(g_i, g_j):
                    raise RuntimeError(
                        "Opposite-direction diagonal guide lines cross: "
                        f"{zid_i!r} and {zid_j!r}"
                    )

    def _optimize_side_connection_points(self, records, ax_map, fig,
                                          split_y, angle_deg, c_direction):
        """DP for one upper/lower band. Returns {zid: map_fig}.

        This version optimises not only C-ordering and spacing, but also
        explicit polyline intersections, especially diagonal-vs-horizontal
        crossings.
        """
        if not records:
            return {}

        # Important: keep the matrix order.
        # Within this order, candidates are chosen to satisfy:
        #   1. monotone characteristic value S
        #   2. near-uniform spacing
        #   3. no real L-polyline intersections
        cand_lists = []

        for rec in records:
            cands = self._build_feasible_candidates_for_zone(
                rec["zid"], ax_map, fig,
                rec["raw_map_fig"], rec["matrix_fig"],
                split_y, angle_deg, 31, rec["is_upper"],
            )

            # If no feasible candidate exists, this is a hard failure.
            if not cands:
                raise RuntimeError(
                    f"MapTrix guide-line optimisation failed: "
                    f"zone {rec['zid']!r} has no feasible connection candidate."
                )

            valid = []
            for c in cands:
                if "geom" not in c or c["geom"] is None:
                    geom = self._leader_polyline_geometry(
                        map_fig=c["point"],
                        matrix_fig=rec["matrix_fig"],
                        split_y=split_y,
                        angle_deg=angle_deg,
                        require_feasible=True,
                    )
                    if geom is None:
                        continue
                    c["geom"] = geom

                c["S"] = c_direction * c["C"]
                valid.append(c)

            if not valid:
                raise RuntimeError(
                    f"MapTrix guide-line optimisation failed: "
                    f"zone {rec['zid']!r} has no valid feasible "
                    f"connection candidate."
                )

            cand_lists.append(valid)

        if len(records) == 1:
            rec = records[0]
            best = min(cand_lists[0], key=lambda d: d["center_cost"])
            return {rec["zid"]: best["point"]}

        all_s = np.array([c["S"] for cl in cand_lists for c in cl], dtype=float)
        s_min, s_max = float(np.min(all_s)), float(np.max(all_s))

        if s_max > s_min:
            targets = np.linspace(s_min, s_max, len(records))
        else:
            targets = np.full(len(records), s_min)

        best_path = None
        best_cost = np.inf

        for gap_scale in [1.0, 0.5, 0.25, 0.1, 0.0]:
            mg = self._leader_min_c_gap * gap_scale

            # For every candidate at current i, store:
            #   cost, path_indices
            states = []

            first_states = []
            for j, c in enumerate(cand_lists[0]):
                unary = (
                    self._leader_center_weight * c["center_cost"]
                    + self._leader_sep_weight * (c["S"] - targets[0]) ** 2
                )
                first_states.append({
                    "cost": unary,
                    "path": [j],
                })
            states.append(first_states)

            ok = True

            for i in range(1, len(records)):
                curr_states = []

                for j, c in enumerate(cand_lists[i]):
                    unary = (
                        self._leader_center_weight * c["center_cost"]
                        + self._leader_sep_weight * (c["S"] - targets[i]) ** 2
                    )

                    best_state_cost = np.inf
                    best_state_path = None

                    for prev_state in states[i - 1]:
                        prev_j = prev_state["path"][-1]
                        prev_c = cand_lists[i - 1][prev_j]

                        # Monotone C/S constraint.
                        if c["S"] - prev_c["S"] < mg:
                            continue

                        # Hard constraint: no crossings with any previous leader.
                        has_cross = False
                        for prev_i, prev_idx in enumerate(prev_state["path"]):
                            prev_geom = cand_lists[prev_i][prev_idx]["geom"]
                            if self._leader_geometries_cross(c["geom"], prev_geom):
                                has_cross = True
                                break

                        if has_cross:
                            continue

                        val = prev_state["cost"] + unary

                        if val < best_state_cost:
                            best_state_cost = val
                            best_state_path = prev_state["path"] + [j]

                    curr_states.append({
                        "cost": best_state_cost,
                        "path": best_state_path,
                    })

                curr_states = [
                    s for s in curr_states
                    if s["path"] is not None and np.isfinite(s["cost"])
                ]

                if not curr_states:
                    ok = False
                    break

                states.append(curr_states)

            if not ok:
                continue

            final_state = min(states[-1], key=lambda s: s["cost"])

            if final_state["cost"] < best_cost:
                best_cost = final_state["cost"]
                best_path = final_state["path"]

            # If there are no crossings, stop early.
            if best_path is not None:
                total_cross = 0
                geoms = [
                    cand_lists[i][idx]["geom"]
                    for i, idx in enumerate(best_path)
                ]
                for a in range(len(geoms)):
                    for b in range(a + 1, len(geoms)):
                        if self._leader_geometries_cross(geoms[a], geoms[b]):
                            total_cross += 1

                if total_cross == 0:
                    break

        if best_path is None:
            raise RuntimeError(
                "MapTrix guide-line optimisation failed: "
                "no crossing-free candidate path exists for this side. "
                "Try increasing leader_max_candidates, increasing grid size, "
                "adjusting split-line candidates, or relaxing leader_min_c_gap."
            )

        return {
            rec["zid"]: cand_lists[i][idx]["point"]
            for i, (rec, idx) in enumerate(zip(records, best_path))
        }

    def _optimize_group_connection_points(self, records, ax_map, fig,
                                           split_y, angle_deg, c_direction):
        """Split records into upper/lower bands, DP each independently."""
        result = {}
        for side in [True, False]:
            subset = [r for r in records if r["is_upper"] == side]
            result.update(self._optimize_side_connection_points(
                subset, ax_map, fig, split_y, angle_deg, c_direction,
            ))
        return result

    # ==================================================================
    # Guide line drawing
    # ==================================================================

    def _draw_split_l_shaped_line(self, fig, matrix_fig, map_fig, linewidth,
                                   color, split_y, angle_deg=45.0):
        """Draw map_point → cross_point → matrix_anchor.

        Matrix anchor is fixed and must never be modified.
        """
        geom = self._leader_polyline_geometry(
            map_fig=map_fig,
            matrix_fig=matrix_fig,
            split_y=split_y,
            angle_deg=angle_deg,
            require_feasible=True,
        )

        # If this happens, the optimisation/fallback produced an invalid point.
        # Do not silently use abs(dy), because that creates visually plausible
        # but topologically wrong leaders that can cross other horizontal segments.
        if geom is None:
            return False

        p_x, p_y = geom["p"]
        q_x, q_y = geom["q"]
        m_x, m_y = geom["m"]

        fig.add_artist(Line2D(
            [p_x, q_x, m_x],
            [p_y, q_y, m_y],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth,
            alpha=self.line_alpha,
            solid_capstyle="round",
            zorder=10,
        ))
        return True

    # ==================================================================
    # Ordering optimisation
    # ==================================================================

    def _compute_guide_positions(self, fig, ax_map_o, ax_map_d):
        """Extract figure-coordinate positions for every zone in current order."""
        def _extract(ax, zids):
            return [dict(zid=z, p_x=x, p_y=y) for z in zids
                    for x, y in [_ax_to_fig(ax, fig, *self.zone_centroids_[z])]]
        return _extract(ax_map_o, self.o_order_), _extract(ax_map_d, self.d_order_)

    def _compute_split_y_from_positions(self, positions):
        """Median-based horizontal split y from map point y coords."""
        ys = np.sort(np.array([p["p_y"] for p in positions], dtype=float))
        n = len(ys)
        if n == 0:
            return 0.5
        if n == 1:
            return float(ys[0])
        mid = n // 2
        return float(0.5 * (ys[mid - 1] + ys[mid])) if n % 2 == 0 else float(ys[mid])

    def _optimize_ordering_fixed_split(self, origin_positions, dest_positions):
        """Re-order zones by split-line characteristic C so guide lines don't cross.

        Origin (top edge, col_idx ↑): lower C = y + t*x ↑, upper C = y - t*x ↑.
        Dest   (left edge, row_idx ↓): upper C = y - t*x ↓, lower C = y + t*x ↓.

        This uses a fixed median split — kept as a fallback; the joint
        optimisation in ``_optimize_layout_joint`` is preferred.
        """
        t = np.tan(np.deg2rad(self._leader_angle_deg))

        def _reorder(positions, row_order_descending, split_attr):
            """Row order: ascending for origin, descending for destination."""
            if len(positions) <= 1:
                if len(positions) == 1:
                    setattr(self, split_attr, positions[0]["p_y"])
                return [p["zid"] for p in positions]

            by_y = sorted(positions, key=lambda p: (p["p_y"], p["p_x"]))
            setattr(self, split_attr, self._compute_split_y_from_positions(by_y))
            spl = getattr(self, split_attr)

            lower = sorted(
                [p for p in positions if p["p_y"] < spl],
                key=lambda p: (p["p_y"] + t * p["p_x"], p["p_y"], p["p_x"]),
                reverse=row_order_descending,
            )
            upper = sorted(
                [p for p in positions if p["p_y"] >= spl],
                key=lambda p: (p["p_y"] - t * p["p_x"], p["p_y"], p["p_x"]),
                reverse=row_order_descending,
            )
            return [p["zid"] for p in lower + upper] if not row_order_descending \
                else [p["zid"] for p in upper + lower]

        return (
            _reorder(origin_positions, False, 'origin_split_y_'),
            _reorder(dest_positions, True, 'dest_split_y_'),
        )

    # ==================================================================
    # Joint layout optimisation (split-lines + matrix ordering)
    # ==================================================================

    def _optimize_layout_joint(self, fig, ax_map_o, ax_map_d, ax_matrix,
                               origin_positions, dest_positions):
        """Jointly optimise split lines and matrix ordering.

        Returns
        -------
        new_o_order : list
        new_d_order : list
        origin_split_y : float
        dest_split_y : float
        """
        new_o_order, origin_split_y = self._optimize_split_and_order_for_side(
            fig=fig,
            ax_map=ax_map_o,
            ax_matrix=ax_matrix,
            positions=origin_positions,
            is_origin=True,
        )

        new_d_order, dest_split_y = self._optimize_split_and_order_for_side(
            fig=fig,
            ax_map=ax_map_d,
            ax_matrix=ax_matrix,
            positions=dest_positions,
            is_origin=False,
        )

        return new_o_order, new_d_order, origin_split_y, dest_split_y

    def _optimize_split_and_order_for_side(self, fig, ax_map, ax_matrix,
                                           positions, is_origin):
        """Optimise one side: either origin map or destination map.

        Parameters
        ----------
        positions : list of dict
            Each item has zid, p_x, p_y in figure coordinates.
        is_origin : bool
            True for origin columns, False for destination rows.

        Returns
        -------
        best_order : list
        best_split_y : float
        """
        if len(positions) <= 1:
            if len(positions) == 1:
                return [positions[0]["zid"]], positions[0]["p_y"]
            return [], 0.5

        candidate_splits = self._candidate_split_ys(
            positions, ax_map, max_candidates=self._leader_split_candidates,
        )

        best_score = np.inf
        best_order = None
        best_split_y = None

        for split_y in candidate_splits:
            ordered_positions = self._order_positions_given_split(
                positions=positions,
                split_y=split_y,
                is_origin=is_origin,
            )

            score = self._score_split_order_layout(
                fig=fig,
                ax_map=ax_map,
                ax_matrix=ax_matrix,
                ordered_positions=ordered_positions,
                split_y=split_y,
                is_origin=is_origin,
            )

            if not np.isfinite(score):
                continue

            if score < best_score:
                best_score = score
                best_order = [p["zid"] for p in ordered_positions]
                best_split_y = split_y

        if best_order is None:
            side_name = "origin" if is_origin else "destination"
            raise RuntimeError(
                f"MapTrix layout failed for {side_name}: "
                "no split-line/order candidate can give every zone a "
                "feasible guide line. Try increasing leader_split_candidates, "
                "leader_max_candidates, using a larger figure, or adjusting "
                "matrix/map spacing."
            )

        return best_order, best_split_y

    def _candidate_split_ys(self, positions, ax_map, max_candidates=25):
        """Generate candidate horizontal split lines in figure coordinates."""
        ys = np.array([p["p_y"] for p in positions], dtype=float)
        ys = np.unique(np.round(ys, 10))

        box = ax_map.get_position()
        y_min = box.y0 + 1e-4
        y_max = box.y1 - 1e-4

        candidates = []

        # 1. map box center
        candidates.append(0.5 * (box.y0 + box.y1))

        # 2. midpoints between neighbouring zone points
        ys_sorted = np.sort(ys)
        for a, b in zip(ys_sorted[:-1], ys_sorted[1:]):
            mid = 0.5 * (a + b)
            if y_min < mid < y_max:
                candidates.append(mid)

        # 3. a few quantile splits for robustness
        for q in np.linspace(0.2, 0.8, 7):
            val = float(np.quantile(ys_sorted, q))
            if y_min < val < y_max:
                candidates.append(val)

        # de-duplicate
        candidates = sorted(set(round(float(c), 10) for c in candidates
                                if y_min < c < y_max))

        # limit candidate count
        if len(candidates) > max_candidates:
            idx = np.linspace(0, len(candidates) - 1, max_candidates).astype(int)
            candidates = [candidates[i] for i in idx]

        return candidates

    def _order_positions_given_split(self, positions, split_y, is_origin):
        """Return ordered positions under a given split line.

        Origin:
            lower group: C = y + t*x ascending
            upper group: C = y - t*x ascending

        Destination:
            upper group: C = y - t*x descending
            lower group: C = y + t*x descending
        """
        t = np.tan(np.deg2rad(self._leader_angle_deg))

        lower = [p for p in positions if p["p_y"] < split_y]
        upper = [p for p in positions if p["p_y"] >= split_y]

        lower_sorted = sorted(
            lower,
            key=lambda p: (
                p["p_y"] + t * p["p_x"],
                p["p_y"],
                p["p_x"],
            ),
            reverse=not is_origin,
        )

        upper_sorted = sorted(
            upper,
            key=lambda p: (
                p["p_y"] - t * p["p_x"],
                p["p_y"],
                p["p_x"],
            ),
            reverse=not is_origin,
        )

        if is_origin:
            # matrix columns from left to right
            return lower_sorted + upper_sorted
        else:
            # matrix rows from top to bottom
            return upper_sorted + lower_sorted

    def _score_split_order_layout(self, fig, ax_map, ax_matrix,
                                  ordered_positions, split_y, is_origin):
        """Score a candidate split + order.

        Lower is better.
        """
        if not ordered_positions:
            return np.inf

        n = len(ordered_positions)

        n_matrix_rows, n_matrix_cols = self.matrix_.shape

        c_direction = +1 if is_origin else -1

        center_costs = []
        s_values = []
        geoms = []

        for idx, p in enumerate(ordered_positions):
            zid = p["zid"]
            raw_map_fig = (p["p_x"], p["p_y"])

            if is_origin:
                # top edge, column idx
                m_x, m_y = _calculate_matrix_anchor_point(
                    transform=self._transform,
                    n_rows=n_matrix_rows,
                    n_cols=n_matrix_cols,
                    side="top",
                    index=idx,
                )
            else:
                # left edge, row idx
                m_x, m_y = _calculate_matrix_anchor_point(
                    transform=self._transform,
                    n_rows=n_matrix_rows,
                    n_cols=n_matrix_cols,
                    side="left",
                    index=idx,
                )

            matrix_fig = _ax_to_fig(ax_matrix, fig, m_x, m_y)

            force_upper = raw_map_fig[1] >= split_y

            # Candidate search for scoring — match DP grid size.
            cands = self._build_feasible_candidates_for_zone(
                zid=zid,
                ax_map=ax_map,
                fig=fig,
                raw_map_fig=raw_map_fig,
                matrix_fig=matrix_fig,
                split_y=split_y,
                angle_deg=self._leader_angle_deg,
                grid_size=31,
                force_upper=force_upper,
            )

            # Hard constraint: every zone must have at least one
            # feasible leader candidate.
            if not cands:
                return np.inf

            best = min(cands, key=lambda d: d["center_cost"])
            center_costs.append(best["center_cost"])
            s_values.append(c_direction * best["C"])

            if "geom" not in best or best["geom"] is None:
                return np.inf

            geoms.append(best["geom"])

        center_cost = float(np.sum(center_costs))

        s_values = np.array(s_values, dtype=float)

        # Penalize non-monotonic or too-close characteristic values.
        if len(s_values) > 1:
            diffs = np.diff(s_values)
            min_gap = self._leader_min_c_gap

            gap_violation = np.sum(np.maximum(0.0, min_gap - diffs) ** 2)

            s_min = float(np.min(s_values))
            s_max = float(np.max(s_values))
            if s_max > s_min:
                targets = np.linspace(s_min, s_max, len(s_values))
                uniform_cost = float(np.sum((s_values - targets) ** 2))
            else:
                uniform_cost = 1.0
        else:
            gap_violation = 0.0
            uniform_cost = 0.0

        # Penalize extremely unbalanced split.
        n_upper = sum(1 for p in ordered_positions if p["p_y"] >= split_y)
        n_lower = n - n_upper
        balance_cost = ((n_upper - n_lower) / max(n, 1)) ** 2

        # Penalize real polyline crossings.
        cross_cost = 0.0
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                if self._leader_geometries_cross(geoms[i], geoms[j]):
                    cross_cost += 1.0

        # Total score.
        score = (
            self._leader_center_weight * center_cost
            + self._leader_sep_weight * uniform_cost
            + 100.0 * gap_violation
            + self._leader_split_balance_weight * balance_cost
            + 1e4 * cross_cost
        )

        return score

    # ==================================================================
    # Debug helper
    # ==================================================================

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
            leader_sep_weight=8.0, leader_center_weight=1.0,
            leader_min_c_gap=0.003, leader_max_candidates=45,
            leader_split_candidates=25,
            leader_split_balance_weight=0.05,
            show_split_lines=False,
            use_gp=True, gp_random_state=None,
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
    leader_sep_weight : float, default=8.0
        DP weight for uniform guide-line spacing.
    leader_center_weight : float, default=1.0
        DP penalty for moving from original centroid.
    leader_min_c_gap : float, default=0.003
        Minimum C-gap between neighbouring lines.
    leader_max_candidates : int, default=45
        Max candidate points per zone.
    leader_split_candidates : int, default=25
        Number of candidate split-line positions to evaluate.
    leader_split_balance_weight : float, default=0.05
        Penalty for unbalanced split-lines in layout scoring.
    show_split_lines : bool, default=False
        Draw horizontal split lines.
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
        leader_sep_weight=leader_sep_weight, leader_center_weight=leader_center_weight,
        leader_min_c_gap=leader_min_c_gap, leader_max_candidates=leader_max_candidates,
        leader_split_candidates=leader_split_candidates,
        leader_split_balance_weight=leader_split_balance_weight,
        show_split_lines=show_split_lines,
        use_gp=use_gp, gp_random_state=gp_random_state,
    )
    return vis.fit_plot(fdf, fig=fig, figsize=figsize)
