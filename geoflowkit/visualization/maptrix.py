"""Static MapTrix visualisation for geographical flow data.

The implementation follows the static layout described by Yang et al.:
two maps are connected to distinct sides of a centred, rotated OD matrix
with one index leader per origin and destination zone.  Flow values only
affect the visual encoding; ordering and routing depend on geography.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from shapely.geometry import Point

from geoflowkit.visualization._utils import (
    _ax_to_fig,
    _calculate_matrix_anchor_point,
    _calculate_rotated_point,
    _compute_representative_points,
    _linear_scaling,
    _plot_labels,
    _prepare_zones,
    _rotate_matrix,
)
from geoflowkit.visualization.od_matrix import ODMatrixVisualizer


class MapTrixVisualizer(ODMatrixVisualizer):
    """Draw a static MapTrix layout.

    Origins are matrix rows and destinations are matrix columns.  The
    origin map connects to the rotated left (row) edge; the destination
    map connects to the rotated bottom (column) edge.  When both zone ID
    sets are identical, the row and column orders are also identical.

    Parameters
    ----------
    origin_zones, dest_zones, zone_id_col, dest_zone_id_col, weight,
    size_weight, matrix_cmap, vmin, vmax, show_labels, label_fontsize,
    include_self_flows
        See :class:`~geoflowkit.visualization.ODMatrixVisualizer`.
    map_cmap : str or Colormap, optional
        Colormap for map proportional symbols.  Defaults to
        ``matrix_cmap``.
    max_centroid_size : float, default=260
        Maximum marker area on the maps, in points squared.
    max_matrix_symbol_size : float, default=220
        Maximum area of the optional matrix size overlay.
    line_color : color, optional
        Set one colour for both leader groups.  When omitted, origins and
        destinations use distinct colours.
    origin_line_color, destination_line_color : color
        Colours for row and column leaders.
    line_alpha : float, default=0.72
        Leader opacity.
    leader_linewidth : float, default=1.25
        Fixed leader width.  It deliberately does not depend on flow.
    origin_map_rect, destination_map_rect, matrix_rect, colorbar_rect :
        tuple of four floats, optional
        ``(left, bottom, width, height)`` in figure coordinates.  Defaults
        follow the proportions in ``maptrix-static-layout-spec.md``.
    corridor_gap : float, default=0.018
        Horizontal gap, in figure coordinates, between a map and the
        bend rail for its leaders.
    min_leader_gap : float, default=6
        Desired minimum vertical separation between adjacent leaders, in
        display pixels.  Sites are moved only to sampled points that stay
        inside their zone.
    cbar_kwds : dict, optional
        Extra keyword arguments for the matrix colorbar.

    Attributes
    ----------
    row_order_, column_order_ : list
        Origin row order and destination column order.
    layout_ : dict or None
        Pixel-space static geometry produced by :meth:`plot`.  It contains
        row/column ports, leader paths, map rectangles, and matrix edges.
        Screen coordinates use x-right/y-down.
    axes_ : dict or None
        Axes for the origin map, destination map, matrix, and colorbar.
    """

    _DEFAULT_ORIGIN_RECT = (0.04, 0.56, 0.30, 0.36)
    _DEFAULT_DESTINATION_RECT = (0.04, 0.08, 0.30, 0.36)
    _DEFAULT_MATRIX_RECT = (0.39, 0.08, 0.52, 0.84)
    _DEFAULT_COLORBAR_RECT = (0.94, 0.22, 0.015, 0.56)

    def __init__(
        self,
        origin_zones: gpd.GeoDataFrame,
        *,
        dest_zones: gpd.GeoDataFrame | None = None,
        zone_id_col: str | None = None,
        dest_zone_id_col: str | None = None,
        weight: str = "count",
        size_weight: str | None = None,
        matrix_cmap: str | plt.Colormap = "OrRd",
        vmin: float | None = None,
        vmax: float | None = None,
        map_cmap: str | plt.Colormap | None = None,
        max_centroid_size: float = 260.0,
        max_matrix_symbol_size: float = 220.0,
        line_color=None,
        origin_line_color="#2878B5",
        destination_line_color="#D97706",
        line_alpha: float = 0.72,
        leader_linewidth: float = 1.25,
        show_labels: bool = True,
        label_fontsize: int = 9,
        out_title: str = "Origins · row index",
        in_title: str = "Destinations · column index",
        matrix_title: str = "Origin–destination matrix",
        title_fontsize: int = 13,
        include_self_flows: bool = True,
        origin_map_rect: tuple[float, float, float, float] | None = None,
        destination_map_rect: tuple[float, float, float, float] | None = None,
        matrix_rect: tuple[float, float, float, float] | None = None,
        colorbar_rect: tuple[float, float, float, float] | None = None,
        corridor_gap: float = 0.018,
        min_leader_gap: float = 6.0,
        site_grid_size: int = 7,
        map_facecolor="#F2F4F5",
        map_edgecolor="#8B949E",
        cbar_kwds: dict | None = None,
        height_ratios: list | None = None,
        width_ratios: list | None = None,
    ):
        super().__init__(
            origin_zones=origin_zones,
            dest_zones=dest_zones,
            zone_id_col=zone_id_col,
            dest_zone_id_col=dest_zone_id_col,
            weight=weight,
            size_weight=size_weight,
            cmap=matrix_cmap,
            vmin=vmin,
            vmax=vmax,
            show_labels=show_labels,
            label_fontsize=label_fontsize,
            include_self_flows=include_self_flows,
        )
        self._validate_zones(
            self.origin_zones, self.zone_id_col, "origin_zones",
        )
        self._validate_zones(
            self.dest_zones, self.dest_zone_id_col, "dest_zones",
        )

        if height_ratios is not None or width_ratios is not None:
            warnings.warn(
                "height_ratios and width_ratios are deprecated; use the "
                "*_rect layout arguments instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.matrix_cmap = matrix_cmap
        self.map_cmap = map_cmap if map_cmap is not None else matrix_cmap
        self.max_centroid_size = max_centroid_size
        self.max_matrix_symbol_size = max_matrix_symbol_size
        if line_color is not None:
            origin_line_color = destination_line_color = line_color
        self.origin_line_color = origin_line_color
        self.destination_line_color = destination_line_color
        self.line_alpha = line_alpha
        self.leader_linewidth = leader_linewidth
        self.out_title = out_title
        self.in_title = in_title
        self.matrix_title = matrix_title
        self.title_fontsize = title_fontsize
        self.origin_map_rect = self._validate_rect(
            origin_map_rect or self._DEFAULT_ORIGIN_RECT, "origin_map_rect",
        )
        self.destination_map_rect = self._validate_rect(
            destination_map_rect or self._DEFAULT_DESTINATION_RECT,
            "destination_map_rect",
        )
        self.matrix_rect = self._validate_rect(
            matrix_rect or self._DEFAULT_MATRIX_RECT, "matrix_rect",
        )
        self.colorbar_rect = self._validate_rect(
            colorbar_rect or self._DEFAULT_COLORBAR_RECT, "colorbar_rect",
        )
        map_right = max(
            self.origin_map_rect[0] + self.origin_map_rect[2],
            self.destination_map_rect[0] + self.destination_map_rect[2],
        )
        if map_right >= self.matrix_rect[0]:
            raise ValueError("Map rectangles must remain to the left of matrix_rect")
        destination_top = (
            self.destination_map_rect[1] + self.destination_map_rect[3]
        )
        if destination_top > self.origin_map_rect[1]:
            raise ValueError(
                "Origin and destination map rectangles must use separate "
                "vertical corridors"
            )
        self.corridor_gap = float(corridor_gap)
        self.min_leader_gap = float(min_leader_gap)
        self.site_grid_size = max(int(site_grid_size), 3)
        self.map_facecolor = map_facecolor
        self.map_edgecolor = map_edgecolor
        self.cbar_kwds = {} if cbar_kwds is None else dict(cbar_kwds)

        self._full_matrix = None
        self._full_size_matrix = None
        self._o_zone_to_idx = None
        self._d_zone_to_idx = None
        self._same_entity_set = False
        self.row_order_ = None
        self.column_order_ = None
        # Backwards-compatible aliases.
        self.o_order_ = None
        self.d_order_ = None

        self._im = None
        self._transform = None
        self.axes_ = None
        self.layout_ = None
        self._leader_geometry = None

    @staticmethod
    def _validate_rect(rect, name):
        if len(rect) != 4:
            raise ValueError(f"{name} must be a (left, bottom, width, height) tuple")
        rect = tuple(float(v) for v in rect)
        if rect[2] <= 0 or rect[3] <= 0:
            raise ValueError(f"{name} width and height must be positive")
        if min(rect) < 0 or rect[0] + rect[2] > 1 or rect[1] + rect[3] > 1:
            raise ValueError(f"{name} must fit inside normalized figure coordinates")
        return rect

    @staticmethod
    def _validate_zones(zones, zone_id_col, name):
        prepared = _prepare_zones(zones, zone_id_col=zone_id_col)
        if prepared.empty:
            raise ValueError(f"{name} must contain at least one zone")
        if prepared["zone_id"].isna().any():
            raise ValueError(f"{name} contains a missing zone ID")
        duplicates = prepared.loc[
            prepared["zone_id"].duplicated(), "zone_id"
        ].tolist()
        if duplicates:
            raise ValueError(f"{name} contains duplicate zone IDs: {duplicates}")
        if prepared.geometry.is_empty.any() or prepared.geometry.isna().any():
            raise ValueError(f"{name} contains an empty geometry")

    # ------------------------------------------------------------------
    # Fit and ordering
    # ------------------------------------------------------------------

    def fit(self, fdf):
        """Aggregate flows and establish geography-only row/column order."""
        super().fit(fdf)

        self._full_matrix = self._raw_matrix.copy()
        self._full_size_matrix = (
            self._raw_size_matrix.copy()
            if self._raw_size_matrix is not None
            else None
        )
        origin_ids = list(self._o_all_ids)
        destination_ids = list(self._d_all_ids)

        self.o_centroids_.update(
            _compute_representative_points(
                self.origin_zones, zone_id_col=self.zone_id_col,
            )
        )
        destination_points = _compute_representative_points(
            self.dest_zones, zone_id_col=self.dest_zone_id_col,
        )
        if self.d_centroids_ is self.o_centroids_:
            self.d_centroids_ = dict(destination_points)
        else:
            self.d_centroids_.update(destination_points)

        self._o_zone_to_idx = {zid: i for i, zid in enumerate(origin_ids)}
        self._d_zone_to_idx = {zid: i for i, zid in enumerate(destination_ids)}
        self._same_entity_set = (
            len(origin_ids) == len(destination_ids)
            and set(origin_ids) == set(destination_ids)
        )

        self.row_order_ = self._order_for_side(
            origin_ids, self.o_centroids_, side="row",
        )
        if self._same_entity_set:
            self.column_order_ = list(self.row_order_)
        else:
            self.column_order_ = self._order_for_side(
                destination_ids, self.d_centroids_, side="column",
            )

        self.o_order_ = self.row_order_
        self.d_order_ = self.column_order_
        self.o_ids_ = np.asarray(self.row_order_)
        self.d_ids_ = np.asarray(self.column_order_)
        self.zone_ids_ = np.asarray(origin_ids)
        self._apply_ordering()
        self.layout_ = None
        return self

    @staticmethod
    def _order_for_side(ids, points, side):
        """Order sites along the exposed right boundary of a map.

        The first segment of every leader is horizontal and ends on a
        vertical bend rail.  Projecting sites onto that rail (screen y)
        preserves their order exactly; ports on both selected matrix
        sides are also monotone from top to bottom.  This is the
        one-sided boundary-labelling approximation used by this static
        renderer.
        """
        available = [(zid, points[zid]) for zid in ids if zid in points]
        if not available:
            return list(ids)

        xs = np.asarray([p[0] for _, p in available], dtype=float)
        ys = np.asarray([p[1] for _, p in available], dtype=float)
        y_span = max(float(np.ptp(ys)), 1e-12)
        y0 = float(ys.min())

        scored = []
        for original_index, (zid, (x, y)) in enumerate(available):
            y_screen = 1.0 - (y - y0) / y_span
            x_tie_breaker = float(x) if side == "column" else -float(x)
            scored.append((y_screen, x_tie_breaker, original_index, zid))

        result = [zid for _, _, _, zid in sorted(scored)]
        result.extend(zid for zid in ids if zid not in set(result))
        return result

    def _apply_ordering(self):
        """Apply origin-row and destination-column order to both matrices."""
        if self._full_matrix is None:
            return
        origin_positions = [self._o_zone_to_idx[z] for z in self.row_order_]
        destination_positions = [
            self._d_zone_to_idx[z] for z in self.column_order_
        ]
        self.matrix_ = self._full_matrix[
            np.ix_(origin_positions, destination_positions)
        ].copy()
        self.size_matrix_ = (
            self._full_size_matrix[
                np.ix_(origin_positions, destination_positions)
            ].copy()
            if self._full_size_matrix is not None
            else None
        )
        if not self.include_self_flows and self._same_entity_set:
            for row, origin_id in enumerate(self.row_order_):
                col = self.column_order_.index(origin_id)
                self.matrix_[row, col] = np.nan
                if self.size_matrix_ is not None:
                    self.size_matrix_[row, col] = np.nan

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def fit_plot(self, fdf, fig=None, figsize=(15, 9)):
        """Fit and plot in one call."""
        return self.fit(fdf).plot(fig=fig, figsize=figsize)

    def plot(self, fig=None, figsize=(15, 9)):
        """Render the complete static MapTrix figure."""
        if self.matrix_ is None:
            raise RuntimeError("Call fit() before plot().")
        if fig is None:
            fig = plt.figure(figsize=figsize, facecolor="white")
        else:
            fig.clf()

        ax_origin = fig.add_axes(self.origin_map_rect)
        ax_destination = fig.add_axes(self.destination_map_rect)
        ax_matrix = fig.add_axes(self.matrix_rect)
        ax_colorbar = fig.add_axes(self.colorbar_rect)
        self.axes_ = {
            "origin": ax_origin,
            "destination": ax_destination,
            "matrix": ax_matrix,
            "colorbar": ax_colorbar,
        }

        self._draw_map(
            ax_origin,
            self.row_order_,
            self._outflows,
            self.origin_zones,
            self.o_centroids_,
            self.out_title,
            self.origin_line_color,
        )
        self._draw_map(
            ax_destination,
            self.column_order_,
            self._inflows,
            self.dest_zones,
            self.d_centroids_,
            self.in_title,
            self.destination_line_color,
        )
        self._im, self._transform = self._draw_matrix(ax_matrix)
        self._draw_colorbar(fig, ax_colorbar)

        # The active matrix axes rectangle is only known after equal-aspect
        # adjustment, so leader geometry is resolved after a canvas draw.
        fig.canvas.draw()
        self._leader_geometry = self._draw_leaders(
            fig, ax_origin, ax_destination, ax_matrix,
        )
        self._draw_matrix_ports(ax_matrix)
        fig.canvas.draw()
        self.layout_ = self._export_layout(fig, ax_origin, ax_destination, ax_matrix)
        return fig

    def _draw_map(
        self,
        ax,
        zone_order,
        flow_totals,
        zones,
        points,
        title,
        accent,
    ):
        zones.plot(
            ax=ax,
            facecolor=self.map_facecolor,
            edgecolor=self.map_edgecolor,
            linewidth=0.65,
            zorder=0,
        )
        if len(zones):
            minx, miny, maxx, maxy = zones.total_bounds
            pad_x = max((maxx - minx) * 0.045, 0.01)
            pad_y = max((maxy - miny) * 0.045, 0.01)
            ax.set_xlim(minx - pad_x, maxx + pad_x)
            ax.set_ylim(miny - pad_y, maxy + pad_y)

        active_points = {z: points[z] for z in zone_order if z in points}
        values = np.asarray(
            [flow_totals.get(z, 0.0) for z in active_points], dtype=float,
        )
        sizes = self._scale_sizes(values, self.max_centroid_size)
        coords = list(active_points.values())
        if coords:
            ax.scatter(
                [p[0] for p in coords],
                [p[1] for p in coords],
                c=values if np.ptp(values) > 0 else accent,
                cmap=self.map_cmap if np.ptp(values) > 0 else None,
                s=sizes,
                edgecolors="white",
                linewidths=0.8,
                alpha=0.9,
                zorder=3,
            )
        if self.show_labels:
            _plot_labels(
                ax,
                active_points,
                {z: str(z) for z in active_points},
                fontsize=self.label_fontsize,
            )

        ax.set_title(
            title,
            loc="left",
            fontsize=self.title_fontsize,
            fontweight="semibold",
            color="#24292F",
            pad=8,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def _draw_matrix(self, ax):
        cmap = plt.get_cmap(self.matrix_cmap)
        if hasattr(cmap, "copy"):
            cmap = cmap.copy()
        cmap.set_bad("#FFFFFF")
        im, transform = _rotate_matrix(
            ax,
            self.matrix_,
            cmap=cmap,
            vmin=self.vmin,
            vmax=self.vmax,
        )

        if self.size_matrix_ is not None:
            points, raw_sizes = [], []
            rows, cols = self.matrix_.shape
            for row in range(rows):
                for col in range(cols):
                    value = self.matrix_[row, col]
                    size_value = self.size_matrix_[row, col]
                    if (
                        np.isnan(value)
                        or value == 0
                        or np.isnan(size_value)
                        or size_value <= 0
                    ):
                        continue
                    points.append(
                        _calculate_rotated_point(transform, rows, row, col)
                    )
                    raw_sizes.append(size_value)
            if points:
                scaled = self._scale_sizes(
                    np.asarray(raw_sizes), self.max_matrix_symbol_size,
                )
                ax.scatter(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    s=scaled,
                    facecolors="none",
                    edgecolors="#263238",
                    linewidths=0.65,
                    alpha=0.72,
                    zorder=6,
                )

        ax.set_title(
            self.matrix_title,
            fontsize=self.title_fontsize,
            fontweight="semibold",
            color="#24292F",
            pad=12,
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        return im, transform

    def _draw_colorbar(self, fig, ax_colorbar):
        kwds = {"orientation": "vertical"}
        kwds.update(self.cbar_kwds)
        colorbar = fig.colorbar(self._im, cax=ax_colorbar, **kwds)
        colorbar.ax.tick_params(labelsize=max(self.label_fontsize - 1, 7))
        colorbar.outline.set_linewidth(0.6)
        colorbar.set_label(
            self.weight.capitalize(),
            fontsize=self.label_fontsize,
            color="#57606A",
        )

    def _draw_leaders(self, fig, ax_origin, ax_destination, ax_matrix):
        rows, cols = self.matrix_.shape
        geometry = {"origin": [], "destination": []}
        groups = [
            (
                "origin",
                ax_origin,
                self.row_order_,
                self.o_centroids_,
                self.origin_zones,
                self.zone_id_col,
                "left",
                rows,
                self.origin_line_color,
                0.0,
            ),
            (
                "destination",
                ax_destination,
                self.column_order_,
                self.d_centroids_,
                self.dest_zones,
                self.dest_zone_id_col,
                "bottom",
                cols,
                self.destination_line_color,
                self.corridor_gap * 0.45,
            ),
        ]

        for (
            kind,
            map_ax,
            order,
            sites,
            zones,
            zone_id_col,
            matrix_side,
            expected_count,
            color,
            extra_gap,
        ) in groups:
            if len(order) != expected_count:
                raise RuntimeError("Matrix dimensions and leader order are inconsistent")

            optimized_sites = self._optimize_map_sites(
                fig, map_ax, order, sites, zones, zone_id_col,
            )
            map_ax.scatter(
                [point[0] for point in optimized_sites.values()],
                [point[1] for point in optimized_sites.values()],
                s=10,
                c=color,
                edgecolors="white",
                linewidths=0.35,
                zorder=5,
            )

            map_box = map_ax.get_position()
            port_points = []
            for index in range(len(order)):
                x, y = _calculate_matrix_anchor_point(
                    self._transform, rows, cols, matrix_side, index,
                )
                port_points.append(_ax_to_fig(ax_matrix, fig, x, y))

            min_port_x = min(point[0] for point in port_points)
            bend_x = min(
                map_box.x1 + self.corridor_gap + extra_gap,
                min_port_x - max(self.corridor_gap * 0.35, 0.006),
            )
            bend_x = max(bend_x, map_box.x1 + 0.004)

            for index, (zone_id, port) in enumerate(zip(order, port_points)):
                if zone_id not in optimized_sites:
                    continue
                site = _ax_to_fig(map_ax, fig, *optimized_sites[zone_id])
                bend = np.asarray((bend_x, site[1]), dtype=float)
                path = [np.asarray(site), bend, np.asarray(port)]
                fig.add_artist(
                    Line2D(
                        [p[0] for p in path],
                        [p[1] for p in path],
                        transform=fig.transFigure,
                        color=color,
                        linewidth=self.leader_linewidth,
                        alpha=self.line_alpha,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                        clip_on=False,
                        zorder=2,
                    )
                )
                geometry[kind].append(
                    {
                        "id": zone_id,
                        "kind": (
                            "origin-row"
                            if kind == "origin"
                            else "column-destination"
                        ),
                        "site": tuple(site),
                        "bend": tuple(bend),
                        "port": tuple(port),
                        "path": [tuple(p) for p in path],
                        "order": index,
                    }
                )
        return geometry

    def _optimize_map_sites(
        self, fig, ax, order, initial_sites, zones, zone_id_col,
    ):
        """Spread adjacent sites using in-polygon discrete candidates."""
        prepared = _prepare_zones(zones, zone_id_col=zone_id_col)
        geometries = dict(zip(prepared["zone_id"], prepared.geometry))
        result = {}
        previous_display_y = None

        for zone_id in order:
            if zone_id not in initial_sites or zone_id not in geometries:
                continue
            initial = np.asarray(initial_sites[zone_id], dtype=float)
            geometry = geometries[zone_id]
            minx, miny, maxx, maxy = geometry.bounds
            x_values = np.linspace(minx, maxx, self.site_grid_size + 2)[1:-1]
            y_values = np.linspace(miny, maxy, self.site_grid_size + 2)[1:-1]
            candidates = [tuple(initial)]
            candidates.extend(
                (float(x), float(y))
                for y in y_values
                for x in x_values
                if geometry.covers(Point(float(x), float(y)))
            )

            initial_display = ax.transData.transform(initial)
            scored = []
            for candidate in candidates:
                display = ax.transData.transform(candidate)
                if (
                    previous_display_y is not None
                    and display[1] > previous_display_y - self.min_leader_gap
                ):
                    continue
                displacement = np.sum((display - initial_display) ** 2)
                scored.append((float(displacement), candidate, display[1]))

            if scored:
                _, chosen, chosen_display_y = min(scored, key=lambda item: item[0])
            else:
                chosen = tuple(initial)
                chosen_display_y = initial_display[1]
            result[zone_id] = chosen
            previous_display_y = chosen_display_y
        return result

    def _draw_matrix_ports(self, ax):
        if not self._leader_geometry:
            return
        for kind, color in [
            ("origin", self.origin_line_color),
            ("destination", self.destination_line_color),
        ]:
            figure_ports = [
                leader["port"] for leader in self._leader_geometry[kind]
            ]
            data_ports = [
                ax.transData.inverted().transform(
                    ax.figure.transFigure.transform(point)
                )
                for point in figure_ports
            ]
            ax.scatter(
                [p[0] for p in data_ports],
                [p[1] for p in data_ports],
                s=19,
                facecolors=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=9,
                clip_on=False,
            )

    # ------------------------------------------------------------------
    # Geometry export
    # ------------------------------------------------------------------

    @staticmethod
    def _fig_point_to_screen(fig, point):
        x, y = fig.transFigure.transform(point)
        return {"x": float(x), "y": float(fig.bbox.height - y)}

    @staticmethod
    def _axes_rect_to_screen(fig, ax):
        box = ax.get_window_extent()
        return {
            "x": float(box.x0),
            "y": float(fig.bbox.height - box.y1),
            "w": float(box.width),
            "h": float(box.height),
        }

    def _export_layout(self, fig, ax_origin, ax_destination, ax_matrix):
        rows, cols = self.matrix_.shape
        origin_leaders = []
        destination_leaders = []
        for source, target in [
            (self._leader_geometry["origin"], origin_leaders),
            (self._leader_geometry["destination"], destination_leaders),
        ]:
            for leader in source:
                target.append(
                    {
                        **{k: leader[k] for k in ("id", "kind", "order")},
                        "site": self._fig_point_to_screen(fig, leader["site"]),
                        "bend": self._fig_point_to_screen(fig, leader["bend"]),
                        "port": self._fig_point_to_screen(fig, leader["port"]),
                        "path": [
                            self._fig_point_to_screen(fig, point)
                            for point in leader["path"]
                        ],
                    }
                )

        row_ports = {
            leader["id"]: leader["port"] for leader in origin_leaders
        }
        column_ports = {
            leader["id"]: leader["port"] for leader in destination_leaders
        }

        local_corners = [(0, rows), (cols, rows), (cols, 0), (0, 0)]
        corners = []
        for point in local_corners:
            data_point = self._transform.transform_point(point)
            fig_point = _ax_to_fig(ax_matrix, fig, *data_point)
            corners.append(self._fig_point_to_screen(fig, fig_point))

        return {
            "row_order": list(self.row_order_),
            "column_order": list(self.column_order_),
            "row_ports": row_ports,
            "column_ports": column_ports,
            "origin_leaders": origin_leaders,
            "destination_leaders": destination_leaders,
            "map_rects": {
                "origin": self._axes_rect_to_screen(fig, ax_origin),
                "destination": self._axes_rect_to_screen(fig, ax_destination),
            },
            "matrix": {
                "rows": rows,
                "columns": cols,
                "corners": corners,
                "row_side": {
                    "start": corners[0],
                    "end": corners[3],
                },
                "column_side": {
                    "start": corners[3],
                    "end": corners[2],
                },
            },
            "same_entity_set": self._same_entity_set,
        }

    @staticmethod
    def _scale_sizes(values, maximum):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        minimum = min(maximum * 0.12, 30.0)
        if np.ptp(values) == 0:
            return np.full(values.shape, (minimum + maximum) / 2.0)
        return _linear_scaling(values, (minimum, maximum))
