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
    _calculate_rotated_point,
    _ax_to_fig,
    _fig_to_ax,
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
    zones : GeoDataFrame
        Polygon geometries defining the spatial zones.
    zone_id_col : str, optional
        Column in *zones* used as the zone identifier.  ``None`` uses
        the GeoDataFrame index.
    weight : str, default='count'
        Aggregation weight: ``'count'``, ``'volume'``, or ``'length'``.
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
                 show_split_lines=False):
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

        # guide-line geometry
        self._leader_angle_deg = 45.0
        self.origin_split_y_ = None
        self.dest_split_y_ = None
        self._leader_sep_weight = leader_sep_weight
        self._leader_center_weight = leader_center_weight
        self._leader_min_c_gap = leader_min_c_gap
        self._leader_max_candidates = leader_max_candidates

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

        # Ordering optimisation — rebuild if order changed
        if need_opt:
            o_pos, d_pos = self._compute_guide_positions(fig, ax_map_o, ax_map_d)
            new_o, new_d = self._optimize_ordering(o_pos, d_pos)
            if new_o != list(self.o_order_) or new_d != list(self.d_order_):
                self.o_order_, self.d_order_ = new_o, new_d
                self._apply_ordering()
                fig.clear()
                ax_map_o, ax_map_d, ax_matrix, im, transform = self._build_figure(fig)
                plt.subplots_adjust(hspace=0.01, wspace=-0.15, left=0.0001)
                fig.canvas.draw()
                self._im = im
                self._transform = transform

        self._draw_colorbar(fig, ax_matrix)
        fig.canvas.draw()
        self._draw_guide_lines(fig, ax_map_o, ax_map_d, ax_matrix)
        return fig

    def fit_plot(self, fdf, fig=None, figsize=(14, 8)):
        self.fit(fdf)
        return self.plot(fig=fig, figsize=figsize)

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
        for idx, zid in enumerate(zid_order):
            if is_origin:
                m_x, m_y = _calculate_rotated_point(transform, n_d_rows, -0.5, idx)
            else:
                m_x, m_y = _calculate_rotated_point(transform, n_d_rows, idx, -0.5)
            m_fig = _ax_to_fig(ax_matrix, fig, m_x, m_y)

            cx, cy = self.zone_centroids_[zid]
            raw_fig = _ax_to_fig(ax_map, fig, cx, cy)

            records.append({
                "zid": zid, "matrix_fig": m_fig, "raw_map_fig": raw_fig,
                "is_upper": raw_fig[1] >= split_y,
                "linewidth": width_map.get(zid, 1.0),
            })

        # DP optimise all connection points
        optimized = self._optimize_group_connection_points(
            records, ax_map=ax_map, fig=fig,
            split_y=split_y, angle_deg=self._leader_angle_deg,
            c_direction=c_direction,
        )

        positions = {}
        for rec in records:
            zid = rec["zid"]
            map_fig = optimized.get(zid, rec["raw_map_fig"])
            self._draw_split_l_shaped_line(
                fig=fig, matrix_fig=rec["matrix_fig"], map_fig=map_fig,
                linewidth=rec["linewidth"], color=self.line_color,
                split_y=split_y, angle_deg=self._leader_angle_deg,
            )
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

    def _is_split_line_feasible(self, map_fig, matrix_fig, split_y,
                                 angle_deg=45.0):
        p_x, p_y = map_fig
        m_x, m_y = matrix_fig
        tan_k = np.tan(np.deg2rad(angle_deg))
        if abs(tan_k) < 1e-6:
            return False
        if p_y >= split_y:
            if not (m_y > split_y and m_y > p_y):
                return False
            gap = m_y - p_y
        else:
            if not (m_y < split_y and m_y < p_y):
                return False
            gap = p_y - m_y
        cross_x = p_x + gap / tan_k
        return p_x < cross_x < m_x

    def _leader_characteristic(self, map_fig, is_upper, tan_k):
        x, y = map_fig
        return y - tan_k * x if is_upper else y + tan_k * x

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
        """Build ≥1 candidate dicts for DP. Retries with relaxed constraints."""
        tan_k = np.tan(np.deg2rad(angle_deg))
        raw_x, raw_y = raw_map_fig
        raw_is_upper = raw_y >= split_y

        def _collect(gs, fu):
            cands = self._sample_zone_candidate_points(zid, ax_map, fig, gs)
            cands.append(raw_map_fig)
            out, seen = [], set()
            for px, py in cands:
                key = (round(float(px), 8), round(float(py), 8))
                if key in seen:
                    continue
                seen.add(key)
                is_up = py >= split_y
                if fu is not None and is_up != fu:
                    continue
                if not self._is_split_line_feasible((px, py), matrix_fig, split_y, angle_deg):
                    continue
                C = self._leader_characteristic((px, py), is_up, tan_k)
                out.append({"point": (px, py), "center_cost": (px - raw_x) ** 2 + (py - raw_y) ** 2, "C": C})
            out.sort(key=lambda d: d["center_cost"])
            return out[:self._leader_max_candidates]

        out = _collect(grid_size, force_upper)
        if not out and force_upper is not None:
            out = _collect(grid_size, None)
        if not out:
            out = _collect(max(grid_size + 20, 51), force_upper)
        if not out and force_upper is not None:
            out = _collect(max(grid_size + 20, 51), None)

        # Last resort: try old adjust, then raw point
        if not out:
            adj = self._adjust_map_point_for_split_line(
                zid, ax_map, fig, raw_map_fig, matrix_fig, split_y, angle_deg,
            )
            C = self._leader_characteristic(adj, raw_is_upper, tan_k)
            out = [{"point": adj, "center_cost": (adj[0] - raw_x) ** 2 + (adj[1] - raw_y) ** 2, "C": C}]
        if not out:
            C = self._leader_characteristic(raw_map_fig, raw_is_upper, tan_k)
            out = [{"point": raw_map_fig, "center_cost": 1e6, "C": C}]
        return out

    # ==================================================================
    # DP optimisation
    # ==================================================================

    def _optimize_side_connection_points(self, records, ax_map, fig,
                                          split_y, angle_deg, c_direction):
        """DP for one upper/lower band. Returns {zid: map_fig}."""
        if not records:
            return {}
        if len(records) == 1:
            rec = records[0]
            cands = self._build_feasible_candidates_for_zone(
                rec["zid"], ax_map, fig, rec["raw_map_fig"], rec["matrix_fig"],
                split_y, angle_deg, 31, rec["is_upper"],
            )
            return {rec["zid"]: min(cands, key=lambda d: d["center_cost"])["point"]}

        tan_k = np.tan(np.deg2rad(angle_deg))
        cand_lists = []
        for rec in records:
            cands = self._build_feasible_candidates_for_zone(
                rec["zid"], ax_map, fig, rec["raw_map_fig"], rec["matrix_fig"],
                split_y, angle_deg, 31, rec["is_upper"],
            )
            for c in cands:
                c["S"] = c_direction * c["C"]
            cand_lists.append(cands)

        all_s = np.array([c["S"] for cl in cand_lists for c in cl], dtype=float)
        s_min, s_max = float(np.min(all_s)), float(np.max(all_s))
        targets = np.linspace(s_min, s_max, len(records)) if s_max > s_min else np.full(len(records), s_min)

        best = None
        for gap_scale in [1.0, 0.5, 0.25, 0.1, 0.0]:
            mg = self._leader_min_c_gap * gap_scale
            dp, parent = [], []
            dp.append(np.array([
                self._leader_center_weight * c["center_cost"]
                + self._leader_sep_weight * (c["S"] - targets[0]) ** 2
                for c in cand_lists[0]
            ], dtype=float))
            parent.append([-1] * len(dp[0]))
            ok = True
            for i in range(1, len(records)):
                cd = np.full(len(cand_lists[i]), np.inf)
                cp = [-1] * len(cand_lists[i])
                for j, c in enumerate(cand_lists[i]):
                    u = (self._leader_center_weight * c["center_cost"]
                         + self._leader_sep_weight * (c["S"] - targets[i]) ** 2)
                    best_prev, best_k = np.inf, -1
                    for k, prev in enumerate(cand_lists[i - 1]):
                        if c["S"] - prev["S"] < mg:
                            continue
                        val = dp[i - 1][k] + u
                        if val < best_prev:
                            best_prev, best_k = val, k
                    cd[j], cp[j] = best_prev, best_k
                if not np.isfinite(cd).any():
                    ok = False; break
                dp.append(cd); parent.append(cp)
            if not ok:
                continue
            li = int(np.nanargmin(dp[-1]))
            if not np.isfinite(dp[-1][li]):
                continue
            idxs = [li]
            for i in range(len(records) - 1, 0, -1):
                idxs.append(parent[i][idxs[-1]])
            best = list(reversed(idxs))
            break

        if best is None:  # fallback: nearest feasible per zone
            result = {}
            for rec, cl in zip(records, cand_lists):
                result[rec["zid"]] = min(cl, key=lambda d: d["center_cost"])["point"]
            return result
        return {rec["zid"]: cl[idx]["point"] for rec, cl, idx in zip(records, cand_lists, best)}

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
        """Draw map_point → cross_point → matrix_point with 90+angle_deg elbow."""
        m_x, m_y = matrix_fig
        p_x, p_y = map_fig
        tan_k = np.tan(np.deg2rad(angle_deg))
        eps = 1e-6
        is_upper = p_y >= split_y

        # Enforce split-line geometry, with fallback m_y adjustment
        if is_upper:
            if not (m_y > split_y and m_y > p_y):
                m_y = max(p_y, split_y) + 1e-4
            gap = m_y - p_y
        else:
            if not (m_y < split_y and m_y < p_y):
                m_y = min(p_y, split_y) - 1e-4
            gap = p_y - m_y

        cross_x = p_x + gap / tan_k
        cross_y = m_y

        # Gentle left-to-right clamp as last resort
        if cross_x <= p_x:
            cross_x = p_x + eps
        if cross_x >= m_x:
            cross_x = m_x - eps
            if cross_x <= p_x:
                cross_x = (p_x + m_x) / 2.0

        fig.add_artist(Line2D(
            [p_x, cross_x, m_x], [p_y, cross_y, m_y],
            transform=fig.transFigure, color=color, linewidth=linewidth,
            alpha=self.line_alpha, solid_capstyle="round", zorder=10,
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

    def _optimize_ordering(self, origin_positions, dest_positions):
        """Re-order zones by split-line characteristic C so guide lines don't cross.

        Origin (top edge, col_idx ↑): lower C = y + t*x ↑, upper C = y - t*x ↑.
        Dest   (left edge, row_idx ↓): upper C = y - t*x ↓, lower C = y + t*x ↓.
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
            show_split_lines=False,
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
        ``'count'``, ``'volume'``, or ``'length'``.
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
        show_split_lines=show_split_lines,
    )
    return vis.fit_plot(fdf, fig=fig, figsize=figsize)
