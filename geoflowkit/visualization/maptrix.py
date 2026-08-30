"""Static MapTrix visualisation for geographical flow data.

The implementation follows the static layout described by Yang et al.:
two maps are connected to distinct sides of a centred, rotated OD matrix
with one index leader per origin and destination zone.  Flow values only
affect the visual encoding; ordering and routing depend on geography.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix
from shapely.geometry import LineString, Point

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
    weight_transform, cbar_label, size_weight, matrix_cmap, vmin, vmax,
    show_labels, label_fontsize, include_self_flows
        See :class:`~geoflowkit.visualization.ODMatrixVisualizer`.
    map_label_fontsize : int, optional
        Font size for zone labels on the maps.  Defaults to
        ``label_fontsize``.
    matrix_label_fontsize : int, optional
        Font size for origin and destination zone labels on the two matrix
        edges without leader lines.  Defaults to ``label_fontsize``.
    origin_matrix_label_rotation : float, default=-45
        Rotation angle, in degrees, for origin labels on the matrix's
        leader-free row edge.
    destination_matrix_label_rotation : float, default=45
        Rotation angle, in degrees, for destination labels on the matrix's
        leader-free column edge.
    cbar_tick_fontsize : int, optional
        Font size for colorbar tick labels.  Defaults to
        ``max(label_fontsize - 1, 7)``.
    cbar_label_fontsize : int, optional
        Font size for the colorbar title.  Defaults to
        ``label_fontsize``.
    map_cmap : str or Colormap, optional
        Colormap for map proportional symbols.  Defaults to ``"viridis"``
        so low-value map symbols remain visible independently of the matrix
        colormap.
    centroid_size_range : tuple of (float, float), default=(30, 260)
        Minimum and maximum marker areas on the maps, in points squared.
        Values must be finite, positive, increasing, and no greater than
        2000.
    max_matrix_symbol_size : float, default=220
        Maximum area of the optional matrix size overlay, in points
        squared.  Must not exceed 1600.
    line_color : color, optional
        Set one colour for both leader groups.  When omitted, origins and
        destinations use distinct colours.
    origin_line_color, destination_line_color : color
        Colours for row and column leaders.
    line_alpha : float, default=0.72
        Leader opacity.
    show_matrix_ports : bool, default=True
        Whether to draw circular markers where leaders meet the matrix edges.
    leader_routing : {'diagonal-horizontal', 'horizontal-diagonal'}
        Segment order from map site to matrix port.  The default follows
        the original MapTrix design: diagonal map-to-bend segment,
        followed by a horizontal bend-to-matrix segment.  The alternative
        retains the previous GeoFlowKit layout.
    allow_leader_crossings : bool, default=False
        Whether leaders may intersect when ``leader_routing`` is
        ``"diagonal-horizontal"``.  Set to ``True`` when a fixed row or
        column order makes a crossing-free fixed-slope layout infeasible.
        Permissive layouts are still optimised lexicographically: first
        minimise crossings, then improve leader clearance, then minimise
        route length and site displacement.  This has no effect for
        ``"horizontal-diagonal"`` routing.
    allow_reorder : bool, default=True
        Whether MapTrix may reorder matrix rows and columns for geographic
        placement and leader routing.  Set to ``False`` to preserve the
        input order of ``origin_zones`` and ``dest_zones``.
    leader_angle : float, default=45
        Absolute diagonal angle in degrees.  This is an implementation
        default, not a value prescribed by the MapTrix paper.  All
        diagonals in an up/down band use the corresponding fixed slopes
        ``-tan(theta)`` and ``tan(theta)`` in screen coordinates.
    leader_width_range : tuple of (float, float), default=(0.8, 4.5)
        Minimum and maximum leader widths.  Origin leaders encode total
        outflow; destination leaders encode total inflow.  Pass ``None``
        to use the fixed ``leader_linewidth`` instead.  Widths are in
        points and cannot exceed 12.
    leader_linewidth : float, default=1.25
        Fallback fixed width when ``leader_width_range=None``.  Must be
        positive and no greater than 12 points.
    map_vertical_gap : float, default=0.14
        Signed vertical gap between the two equal-sized map frames, as a
        fraction of the reference layout height at ``matrix_scale=1``.
        Negative values make the map frames overlap.
    map_matrix_gap : float, default=0.10
        Signed horizontal gap from the maps' common right edge to the
        rotated matrix frame, as a fraction of matrix frame height.
    matrix_colorbar_gap : float, default=0.05
        Signed horizontal gap from the matrix frame to the colorbar frame,
        as a fraction of matrix frame height.
    colorbar_height_ratio : float, default=0.88
        Colorbar frame height divided by matrix frame height.  The colorbar
        is vertically centred on the matrix.
    matrix_scale : float, default=1
        Uniform scale of the rotated matrix frame relative to the map stack,
        between 0.5 and 1.5.  Scaling is vertically symmetric: the matrix's
        top and bottom offsets from the map stack remain equal.
    show_layout_frames : bool, default=False
        Draw pale-grey outlines around the two maps, rotated-matrix bounding
        box, and colorbar.  Intended as a layout debugging aid.
    layout_rect : tuple of four floats, default=(0.04, 0.08, 0.894, 0.84)
        Target ``(left, bottom, width, height)`` region in normalized figure
        coordinates.  The component layout is uniformly fitted and centred
        in this region using display-space dimensions.
    corridor_gap : float, default=0.018
        Horizontal gap, in figure coordinates, between a map and the
        bend rail for its leaders.
    min_leader_gap : float, default=12
        Desired minimum visible gap between parallel diagonal segments,
        in display pixels and excluding their stroke widths.  It is a
        capped max-min objective: crossings remain forbidden even when
        this gap cannot be reached.  For legacy routing it controls the
        vertical separation between adjacent map sites.
    site_grid_size : int, default=7
        Interior sampling density used to find map attachment sites.
    routing_solver : {'auto', 'exact', 'heuristic'}, default='auto'
        Fixed-angle ordering solver. ``'auto'`` keeps the exact MILP for
        small layouts and switches to deterministic local search for
        larger layouts.
    exact_routing_zone_limit : int, default=12
        Largest row or column count solved exactly in ``'auto'`` mode.
    routing_max_iterations : int, default=30
        Maximum improving-swap passes used by the heuristic solver.
    map_title_pad : float, default=0
        Distance, in points, between each map's left border and its
        vertical ``out_title`` or ``in_title`` axis label.
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

    _COLORBAR_WIDTH_RATIO = 0.028
    _LAYOUT_FRAME_COLOR = "#C7CCD1"
    _LAYOUT_FRAME_LINEWIDTH = 0.7
    _MIN_MATRIX_SCALE = 0.5
    _MAX_MATRIX_SCALE = 1.5
    _MAX_CENTROID_SIZE = 2000.0
    _MAX_MATRIX_SYMBOL_SIZE = 1600.0
    _MAX_LEADER_WIDTH = 12.0

    def __init__(
        self,
        origin_zones: gpd.GeoDataFrame,
        *,
        dest_zones: gpd.GeoDataFrame | None = None,
        zone_id_col: str | None = None,
        dest_zone_id_col: str | None = None,
        weight: str | Callable = "count",
        weight_transform: Callable | None = None,
        cbar_label: str | None = None,
        size_weight: str | None = None,
        matrix_cmap: str | plt.Colormap = "OrRd",
        vmin: float | None = None,
        vmax: float | None = None,
        map_cmap: str | plt.Colormap = "viridis",
        centroid_size_range: tuple[float, float] = (30.0, 260.0),
        max_matrix_symbol_size: float = 220.0,
        line_color=None,
        origin_line_color="#2878B5",
        destination_line_color="#D97706",
        line_alpha: float = 0.72,
        show_matrix_ports: bool = True,
        leader_routing: str = "diagonal-horizontal",
        allow_leader_crossings: bool = False,
        allow_reorder: bool = True,
        leader_angle: float = 45.0,
        leader_width_range: tuple[float, float] | None = (0.8, 4.5),
        leader_linewidth: float = 1.25,
        show_labels: bool = True,
        label_fontsize: int = 9,
        map_label_fontsize: int | None = None,
        matrix_label_fontsize: int | None = None,
        origin_matrix_label_rotation: float = -25.0,
        destination_matrix_label_rotation: float = 25.0,
        cbar_tick_fontsize: int | None = None,
        cbar_label_fontsize: int | None = None,
        out_title: str = "Origins · row index",
        in_title: str = "Destinations · column index",
        title_fontsize: int = 13,
        map_title_pad: float = 0.0,
        include_self_flows: bool = True,
        map_vertical_gap: float = 0.14,
        map_matrix_gap: float = 0.10,
        matrix_colorbar_gap: float = 0.05,
        colorbar_height_ratio: float = 0.88,
        matrix_scale: float = 1.0,
        show_layout_frames: bool = False,
        layout_rect: tuple[float, float, float, float] = (
            0.04, 0.08, 0.894, 0.84,
        ),
        corridor_gap: float = 0.018,
        min_leader_gap: float = 12.0,
        site_grid_size: int = 7,
        routing_solver: str = "auto",
        exact_routing_zone_limit: int = 12,
        routing_max_iterations: int = 30,
        map_facecolor="#F2F4F5",
        map_edgecolor="#8B949E",
        cbar_kwds: dict | None = None,
    ):
        super().__init__(
            origin_zones=origin_zones,
            dest_zones=dest_zones,
            zone_id_col=zone_id_col,
            dest_zone_id_col=dest_zone_id_col,
            weight=weight,
            weight_transform=weight_transform,
            cbar_label=cbar_label,
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

        self.matrix_cmap = matrix_cmap
        self.map_cmap = map_cmap
        self.centroid_size_range = self._validate_size_range(
            centroid_size_range,
            "centroid_size_range",
            self._MAX_CENTROID_SIZE,
        )
        self.max_matrix_symbol_size = self._validate_bounded_positive(
            max_matrix_symbol_size,
            "max_matrix_symbol_size",
            self._MAX_MATRIX_SYMBOL_SIZE,
        )
        if line_color is not None:
            origin_line_color = destination_line_color = line_color
        self.origin_line_color = origin_line_color
        self.destination_line_color = destination_line_color
        self.line_alpha = line_alpha
        if not isinstance(show_matrix_ports, (bool, np.bool_)):
            raise TypeError("show_matrix_ports must be a boolean")
        self.show_matrix_ports = bool(show_matrix_ports)
        valid_routing = {"diagonal-horizontal", "horizontal-diagonal"}
        if leader_routing not in valid_routing:
            raise ValueError(
                f"leader_routing must be one of {sorted(valid_routing)}"
            )
        self.leader_routing = leader_routing
        if not isinstance(allow_leader_crossings, (bool, np.bool_)):
            raise TypeError("allow_leader_crossings must be a boolean")
        self.allow_leader_crossings = bool(allow_leader_crossings)
        if not isinstance(allow_reorder, (bool, np.bool_)):
            raise TypeError("allow_reorder must be a boolean")
        self.allow_reorder = bool(allow_reorder)
        self.leader_angle = float(leader_angle)
        if not 0.0 < self.leader_angle < 90.0:
            raise ValueError("leader_angle must be between 0 and 90 degrees")
        self.leader_slope = float(
            np.tan(np.radians(self.leader_angle))
        )
        if leader_width_range is not None:
            if len(leader_width_range) != 2:
                raise ValueError("leader_width_range must contain two values")
            leader_width_range = tuple(float(v) for v in leader_width_range)
            if (
                not np.all(np.isfinite(leader_width_range))
                or leader_width_range[0] <= 0
                or leader_width_range[1] < leader_width_range[0]
            ):
                raise ValueError(
                    "leader_width_range must be finite, positive, and "
                    "increasing"
                )
            if leader_width_range[1] > self._MAX_LEADER_WIDTH:
                raise ValueError(
                    "leader_width_range maximum must not exceed "
                    f"{self._MAX_LEADER_WIDTH:g} points"
                )
        self.leader_width_range = leader_width_range
        self.leader_linewidth = self._validate_bounded_positive(
            leader_linewidth,
            "leader_linewidth",
            self._MAX_LEADER_WIDTH,
        )
        self.out_title = out_title
        self.in_title = in_title
        self.title_fontsize = title_fontsize
        self.map_title_pad = float(map_title_pad)
        if not np.isfinite(self.map_title_pad):
            raise ValueError("map_title_pad must be finite")
        self.map_label_fontsize = (
            map_label_fontsize if map_label_fontsize is not None else label_fontsize
        )
        self.matrix_label_fontsize = (
            matrix_label_fontsize
            if matrix_label_fontsize is not None
            else label_fontsize
        )
        self.origin_matrix_label_rotation = float(
            origin_matrix_label_rotation
        )
        self.destination_matrix_label_rotation = float(
            destination_matrix_label_rotation
        )
        if not np.isfinite(self.origin_matrix_label_rotation):
            raise ValueError(
                "origin_matrix_label_rotation must be finite"
            )
        if not np.isfinite(self.destination_matrix_label_rotation):
            raise ValueError(
                "destination_matrix_label_rotation must be finite"
            )
        self.cbar_tick_fontsize = (
            cbar_tick_fontsize if cbar_tick_fontsize is not None
            else max(label_fontsize - 1, 7)
        )
        self.cbar_label_fontsize = (
            cbar_label_fontsize if cbar_label_fontsize is not None else label_fontsize
        )
        if not isinstance(show_layout_frames, (bool, np.bool_)):
            raise TypeError("show_layout_frames must be a boolean")
        self.show_layout_frames = bool(show_layout_frames)
        self.matrix_scale = self._validate_bounded_range(
            matrix_scale,
            "matrix_scale",
            self._MIN_MATRIX_SCALE,
            self._MAX_MATRIX_SCALE,
        )
        self.layout_rect = self._validate_rect(layout_rect, "layout_rect")
        self.map_vertical_gap = self._validate_gap(
            map_vertical_gap, "map_vertical_gap",
        )
        if not -1.0 < self.map_vertical_gap < 1.0:
            raise ValueError("map_vertical_gap must be between -1 and 1")
        self.map_matrix_gap = self._validate_gap(
            map_matrix_gap, "map_matrix_gap",
        )
        self.matrix_colorbar_gap = self._validate_gap(
            matrix_colorbar_gap, "matrix_colorbar_gap",
        )
        self.colorbar_height_ratio = self._validate_gap(
            colorbar_height_ratio, "colorbar_height_ratio",
        )
        if not 0.0 < self.colorbar_height_ratio <= 1.0:
            raise ValueError(
                "colorbar_height_ratio must be greater than 0 and "
                "no greater than 1"
            )
        self._set_resolved_rects({
            "origin_map": None,
            "destination_map": None,
            "matrix": None,
            "colorbar": None,
        })
        self.corridor_gap = float(corridor_gap)
        if self.corridor_gap < 0:
            raise ValueError("corridor_gap must be non-negative")
        self.min_leader_gap = float(min_leader_gap)
        if self.min_leader_gap < 0:
            raise ValueError("min_leader_gap must be non-negative")
        self.site_grid_size = max(int(site_grid_size), 3)
        valid_solvers = {"auto", "exact", "heuristic"}
        if routing_solver not in valid_solvers:
            raise ValueError(
                f"routing_solver must be one of {sorted(valid_solvers)}"
            )
        self.routing_solver = routing_solver
        if (
            not isinstance(exact_routing_zone_limit, (int, np.integer))
            or exact_routing_zone_limit < 2
        ):
            raise ValueError(
                "exact_routing_zone_limit must be an integer of at least 2"
            )
        self.exact_routing_zone_limit = int(exact_routing_zone_limit)
        if (
            not isinstance(routing_max_iterations, (int, np.integer))
            or routing_max_iterations < 1
        ):
            raise ValueError(
                "routing_max_iterations must be a positive integer"
            )
        self.routing_max_iterations = int(routing_max_iterations)
        self.map_facecolor = map_facecolor
        self.map_edgecolor = map_edgecolor
        self.cbar_kwds = {} if cbar_kwds is None else dict(cbar_kwds)

        self._full_matrix = None
        self._full_raw_matrix = None
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
        self._fixed_layout_solution = None
        self._routing_diagnostics = None
        self._routing_started_at = None
        self._logical_layout = None
        self.o_sites_ = None
        self.d_sites_ = None

    @staticmethod
    def _validate_rect(rect, name):
        if len(rect) != 4:
            raise ValueError(f"{name} must be a (left, bottom, width, height) tuple")
        rect = tuple(float(v) for v in rect)
        if not np.all(np.isfinite(rect)):
            raise ValueError(f"{name} values must be finite")
        if rect[2] <= 0 or rect[3] <= 0:
            raise ValueError(f"{name} width and height must be positive")
        if min(rect) < 0 or rect[0] + rect[2] > 1 or rect[1] + rect[3] > 1:
            raise ValueError(f"{name} must fit inside normalized figure coordinates")
        return rect

    @staticmethod
    def _validate_gap(value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _validate_bounded_range(value, name, minimum, maximum):
        value = float(value)
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum:g} and {maximum:g}"
            )
        return value

    def _set_resolved_rects(self, rects):
        self.origin_map_rect = rects["origin_map"]
        self.destination_map_rect = rects["destination_map"]
        self.matrix_rect = rects["matrix"]
        self.colorbar_rect = rects["colorbar"]

    @staticmethod
    def _zone_display_aspect(zones):
        """Return the natural width/height ratio for a zone layer."""
        if len(zones) == 0:
            return 1.0
        minx, miny, maxx, maxy = zones.total_bounds
        width = float(maxx - minx)
        height = float(maxy - miny)
        if width <= 0.0 or height <= 0.0:
            return 1.0
        data_aspect = 1.0
        if zones.crs is not None and zones.crs.is_geographic:
            mean_latitude = float((miny + maxy) / 2.0)
            cosine = abs(np.cos(np.deg2rad(mean_latitude)))
            data_aspect = 1.0 / max(cosine, 1e-6)
        return width / (height * data_aspect)

    def _common_map_aspect(self):
        """Balance the natural aspects while keeping both map frames equal."""
        origin_aspect = self._zone_display_aspect(self.origin_zones)
        destination_aspect = self._zone_display_aspect(self.dest_zones)
        return float(np.sqrt(origin_aspect * destination_aspect))

    def _matrix_axes_frame_ratio(self):
        """Axes-side / rotated-matrix-frame-side for the current padding."""
        rows, cols = self.matrix_.shape
        frame_side = (rows + cols) / np.sqrt(2.0)
        pad = max(rows, cols) * 0.04
        return float((frame_side + 2.0 * pad) / frame_side)

    def _resolve_automatic_rects(self, fig):
        """Solve semantic layout constraints in display-pixel space."""
        figure_width = float(fig.bbox.width)
        figure_height = float(fig.bbox.height)
        left, bottom, width, height = self.layout_rect
        target_left = left * figure_width
        target_bottom = bottom * figure_height
        target_width = width * figure_width
        target_height = height * figure_height

        reference_height = 1.0
        map_height = (
            reference_height - self.map_vertical_gap
        ) / 2.0
        map_width = self._common_map_aspect() * map_height
        matrix_frame_size = self.matrix_scale
        matrix_bottom = (reference_height - matrix_frame_size) / 2.0
        matrix_left = (
            map_width + self.map_matrix_gap * matrix_frame_size
        )
        colorbar_left = (
            matrix_left + matrix_frame_size
            + self.matrix_colorbar_gap * matrix_frame_size
        )
        colorbar_width = self._COLORBAR_WIDTH_RATIO * matrix_frame_size
        colorbar_height = (
            self.colorbar_height_ratio * matrix_frame_size
        )
        colorbar_bottom = (reference_height - colorbar_height) / 2.0

        logical_rects = {
            "origin_map": (
                0.0,
                map_height + self.map_vertical_gap,
                map_width,
                map_height,
            ),
            "destination_map": (0.0, 0.0, map_width, map_height),
            "matrix_frame": (
                matrix_left,
                matrix_bottom,
                matrix_frame_size,
                matrix_frame_size,
            ),
            "colorbar": (
                colorbar_left,
                colorbar_bottom,
                colorbar_width,
                colorbar_height,
            ),
        }
        logical_left = min(rect[0] for rect in logical_rects.values())
        logical_right = max(
            rect[0] + rect[2] for rect in logical_rects.values()
        )
        logical_width = logical_right - logical_left
        logical_bottom = min(rect[1] for rect in logical_rects.values())
        logical_top = max(
            rect[1] + rect[3] for rect in logical_rects.values()
        )
        logical_height = logical_top - logical_bottom
        scale = min(
            target_height / logical_height,
            target_width / logical_width,
        )
        base_left = (
            target_left + (target_width - logical_width * scale) / 2.0
            - logical_left * scale
        )
        base_bottom = (
            target_bottom
            + (target_height - logical_height * scale) / 2.0
            - logical_bottom * scale
        )

        def to_figure_rect(rect):
            rect_left, rect_bottom, rect_width, rect_height = rect
            return (
                (base_left + rect_left * scale) / figure_width,
                (base_bottom + rect_bottom * scale) / figure_height,
                rect_width * scale / figure_width,
                rect_height * scale / figure_height,
            )

        origin_rect = to_figure_rect(logical_rects["origin_map"])
        destination_rect = to_figure_rect(
            logical_rects["destination_map"]
        )
        colorbar_rect = to_figure_rect(logical_rects["colorbar"])
        matrix_frame_rect = logical_rects["matrix_frame"]
        matrix_frame_center_x = (
            base_left
            + (matrix_frame_rect[0] + matrix_frame_rect[2] / 2.0) * scale
        )
        matrix_frame_center_y = (
            base_bottom
            + (matrix_frame_rect[1] + matrix_frame_rect[3] / 2.0) * scale
        )
        matrix_axes_side = (
            scale * matrix_frame_size * self._matrix_axes_frame_ratio()
        )
        matrix_rect = (
            (matrix_frame_center_x - matrix_axes_side / 2.0)
            / figure_width,
            (matrix_frame_center_y - matrix_axes_side / 2.0)
            / figure_height,
            matrix_axes_side / figure_width,
            matrix_axes_side / figure_height,
        )
        self._logical_layout = logical_rects
        self._set_resolved_rects({
            "origin_map": origin_rect,
            "destination_map": destination_rect,
            "matrix": matrix_rect,
            "colorbar": colorbar_rect,
        })

    @staticmethod
    def _validate_bounded_positive(value, name, maximum):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")
        if value > maximum:
            raise ValueError(
                f"{name} must not exceed {maximum:g}"
            )
        return value

    @staticmethod
    def _validate_size_range(size_range, name, maximum):
        if len(size_range) != 2:
            raise ValueError(f"{name} must contain two values")
        size_range = tuple(float(value) for value in size_range)
        if (
            not np.all(np.isfinite(size_range))
            or size_range[0] <= 0
            or size_range[1] < size_range[0]
        ):
            raise ValueError(
                f"{name} must be finite, positive, and increasing"
            )
        if size_range[1] > maximum:
            raise ValueError(f"{name} must not exceed {maximum:g}")
        return size_range

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

        self._full_matrix = self._display_matrix.copy()
        self._full_raw_matrix = self._raw_matrix.copy()
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

        if self.allow_reorder:
            self.row_order_ = self._order_for_side(
                origin_ids, self.o_centroids_, side="row",
            )
            if self._same_entity_set:
                self.column_order_ = list(self.row_order_)
            else:
                self.column_order_ = self._order_for_side(
                    destination_ids, self.d_centroids_, side="column",
                )
        else:
            self.row_order_ = list(origin_ids)
            self.column_order_ = list(destination_ids)

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
        """Create a stable geography-only seed for boundary ordering.

        The fixed-angle boundary solver may refine this seed after the
        active figure transforms and matrix ports are known.
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
            x_tie_breaker = (
                float(x) if side == "column" else -float(x)
            )
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
        self.raw_matrix_ = self._full_raw_matrix[
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
                self.raw_matrix_[row, col] = np.nan
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

        self._resolve_automatic_rects(fig)

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

        self._draw_map_base(
            ax_origin,
            self.origin_zones,
            self.out_title,
        )
        self._draw_map_base(
            ax_destination,
            self.dest_zones,
            self.in_title,
        )
        self._im, self._transform = self._draw_matrix(ax_matrix)
        self._draw_colorbar(fig, ax_colorbar)

        # The active matrix axes rectangle is only known after equal-aspect
        # adjustment, so leader geometry is resolved after a canvas draw.
        fig.canvas.draw()
        self._routing_diagnostics = None
        self._routing_started_at = None
        if self.leader_routing == "diagonal-horizontal":
            row_order, column_order = self._solve_boundary_orders(
                fig, ax_origin, ax_destination, ax_matrix,
            )
            if (
                row_order != self.row_order_
                or column_order != self.column_order_
            ):
                self.row_order_ = row_order
                self.column_order_ = column_order
                self.o_order_ = self.row_order_
                self.d_order_ = self.column_order_
                self.o_ids_ = np.asarray(self.row_order_)
                self.d_ids_ = np.asarray(self.column_order_)
                self._apply_ordering()
                ax_matrix.clear()
                ax_colorbar.clear()
                self._im, self._transform = self._draw_matrix(ax_matrix)
                self._draw_colorbar(fig, ax_colorbar)
                fig.canvas.draw()
        self._leader_geometry = self._draw_leaders(
            fig, ax_origin, ax_destination, ax_matrix,
        )
        self._draw_map_symbols(
            ax_origin,
            self.row_order_,
            self._outflows,
            self.o_sites_,
            self.origin_line_color,
        )
        self._draw_map_symbols(
            ax_destination,
            self.column_order_,
            self._inflows,
            self.d_sites_,
            self.destination_line_color,
        )
        self._draw_matrix_ports(ax_matrix)
        fig.canvas.draw()
        if self.show_layout_frames:
            self._draw_layout_frames(
                fig, ax_origin, ax_destination, ax_matrix, ax_colorbar,
            )
            fig.canvas.draw()
        self.layout_ = self._export_layout(fig, ax_origin, ax_destination, ax_matrix)
        return fig

    def _draw_map_base(self, ax, zones, title):
        """Draw zone polygons and establish a stable map transform."""
        allocated_position = ax.get_position(original=True).frozen()
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
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
            minx, maxx = minx - pad_x, maxx + pad_x
            miny, maxy = miny - pad_y, maxy + pad_y

            data_aspect = 1.0
            if zones.crs is not None and zones.crs.is_geographic:
                mean_latitude = float((miny + maxy) / 2.0)
                cosine = abs(np.cos(np.deg2rad(mean_latitude)))
                data_aspect = 1.0 / max(cosine, 1e-6)
            figure_width = float(ax.figure.bbox.width)
            figure_height = float(ax.figure.bbox.height)
            box_aspect = (
                allocated_position.width * figure_width
                / (allocated_position.height * figure_height)
            )
            target_range_ratio = data_aspect * box_aspect
            data_width = float(maxx - minx)
            data_height = float(maxy - miny)
            if data_width / data_height < target_range_ratio:
                expanded_width = data_height * target_range_ratio
                centre = (minx + maxx) / 2.0
                minx, maxx = (
                    centre - expanded_width / 2.0,
                    centre + expanded_width / 2.0,
                )
            else:
                expanded_height = data_width / target_range_ratio
                centre = (miny + maxy) / 2.0
                miny, maxy = (
                    centre - expanded_height / 2.0,
                    centre + expanded_height / 2.0,
                )
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)

        ax.set_ylabel(
            title,
            fontsize=self.title_fontsize,
            fontweight="semibold",
            color="#24292F",
            labelpad=self.map_title_pad,
            rotation=90,
            va="center",
        )
        ax.set_position(allocated_position)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_map_symbols(
        self, ax, zone_order, flow_totals, points, accent,
    ):
        """Draw flow symbols and labels at the final leader sites."""
        active_points = {z: points[z] for z in zone_order if z in points}
        values = np.asarray(
            [flow_totals.get(z, 0.0) for z in active_points], dtype=float,
        )
        sizes = self._scale_sizes(values, self.centroid_size_range)
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
                fontsize=self.map_label_fontsize,
            )

    def _draw_matrix(self, ax):
        # The rotated matrix only occupies a diamond inside its rectangular
        # axes.  Keep the axes patch transparent so that, when users choose
        # zero or negative map-to-matrix gaps, the empty corner triangles
        # do not cover the maps.  NaN matrix cells remain explicitly white
        # through the colormap below.
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)
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
                    np.asarray(raw_sizes),
                    (
                        min(self.max_matrix_symbol_size * 0.12, 30.0),
                        self.max_matrix_symbol_size,
                    ),
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

        if self.show_labels:
            self._draw_matrix_labels(ax, transform)

        for spine in ax.spines.values():
            spine.set_visible(False)
        return im, transform

    def _draw_matrix_labels(self, ax, transform):
        """Label the two matrix edges opposite the leader-line edges."""
        rows, cols = self.matrix_.shape

        # Origin leaders terminate on the left row edge, so origin names are
        # placed on the opposite (right) row edge and extend outwards.
        for row, label in enumerate(self.row_order_):
            point = _calculate_rotated_point(
                transform, rows, row, cols - 0.5,
            )
            ax.annotate(
                str(label),
                xy=point,
                xytext=(4, -4),
                textcoords="offset points",
                fontsize=self.matrix_label_fontsize,
                color="#24292F",
                rotation=self.origin_matrix_label_rotation,
                rotation_mode="anchor",
                ha="left",
                va="top",
                clip_on=False,
                annotation_clip=False,
                zorder=8,
            )

        # Destination leaders terminate on the bottom column edge, so
        # destination names are placed on the opposite (top) column edge.
        for col, label in enumerate(self.column_order_):
            point = _calculate_rotated_point(
                transform, rows, -0.5, col,
            )
            ax.annotate(
                str(label),
                xy=point,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=self.matrix_label_fontsize,
                color="#24292F",
                rotation=self.destination_matrix_label_rotation,
                rotation_mode="anchor",
                ha="left",
                va="bottom",
                clip_on=False,
                annotation_clip=False,
                zorder=8,
            )

    def _draw_colorbar(self, fig, ax_colorbar):
        kwds = {"orientation": "vertical"}
        kwds.update(self.cbar_kwds)
        colorbar = fig.colorbar(self._im, cax=ax_colorbar, **kwds)
        ax_colorbar.set_zorder(-1)
        colorbar.ax.tick_params(labelsize=self.cbar_tick_fontsize)
        colorbar.outline.set_visible(False)
        colorbar.set_label(
            self.cbar_label,
            fontsize=self.cbar_label_fontsize,
            color="#57606A",
        )

    def _matrix_frame_figure_rect(self, fig, ax_matrix):
        """Return the rotated matrix's axis-aligned frame in figure units."""
        rows, cols = self.matrix_.shape
        corners = np.asarray([
            (0.0, 0.0),
            (float(cols), 0.0),
            (float(cols), float(rows)),
            (0.0, float(rows)),
        ])
        rotated = self._transform.transform(corners)
        display = ax_matrix.transData.transform(rotated)
        x0, y0 = display.min(axis=0)
        x1, y1 = display.max(axis=0)
        return (
            float(x0 / fig.bbox.width),
            float(y0 / fig.bbox.height),
            float((x1 - x0) / fig.bbox.width),
            float((y1 - y0) / fig.bbox.height),
        )

    def _component_frame_rects(
        self, fig, ax_origin, ax_destination, ax_matrix, ax_colorbar,
    ):
        """Return all four visible component frames in figure units."""
        return {
            "origin_map": tuple(ax_origin.get_position().bounds),
            "destination_map": tuple(ax_destination.get_position().bounds),
            "matrix": self._matrix_frame_figure_rect(fig, ax_matrix),
            "colorbar": tuple(ax_colorbar.get_position().bounds),
        }

    def _draw_layout_frames(
        self, fig, ax_origin, ax_destination, ax_matrix, ax_colorbar,
    ):
        """Draw optional pale debug frames in figure coordinates."""
        for name, rect in self._component_frame_rects(
            fig, ax_origin, ax_destination, ax_matrix, ax_colorbar,
        ).items():
            frame = Rectangle(
                rect[:2],
                rect[2],
                rect[3],
                transform=fig.transFigure,
                fill=False,
                edgecolor=self._LAYOUT_FRAME_COLOR,
                linewidth=self._LAYOUT_FRAME_LINEWIDTH,
                zorder=30,
                clip_on=False,
            )
            frame.set_gid(f"maptrix-layout-frame-{name}")
            fig.add_artist(frame)

    def _solve_boundary_orders(
        self, fig, ax_origin, ax_destination, ax_matrix,
    ):
        """Jointly assign ports and in-zone sites for fixed-angle leaders."""
        rows, cols = self.matrix_.shape
        resolved_solver = self.routing_solver
        if resolved_solver == "auto":
            resolved_solver = (
                "exact"
                if max(rows, cols) <= self.exact_routing_zone_limit
                else "heuristic"
            )
        self._routing_started_at = time.perf_counter()
        self._routing_diagnostics = {
            "requested_solver": self.routing_solver,
            "solver": resolved_solver,
            "optimal": resolved_solver == "exact",
            "iterations": 0,
            "candidate_routes": 0,
        }
        origin_ports = self._matrix_port_points(
            fig, ax_matrix, "left", rows,
        )
        destination_ports = self._matrix_port_points(
            fig, ax_matrix, "bottom", cols,
        )
        origin_group = self._build_route_options(
            fig,
            ax_origin,
            list(self.row_order_),
            self.o_centroids_,
            self.origin_zones,
            self.zone_id_col,
            origin_ports,
            "origin",
            self._outflows,
            site_grid_size=(0 if resolved_solver == "heuristic" else None),
        )
        destination_group = self._build_route_options(
            fig,
            ax_destination,
            list(self.column_order_),
            self.d_centroids_,
            self.dest_zones,
            self.dest_zone_id_col,
            destination_ports,
            "destination",
            self._inflows,
            site_grid_size=(0 if resolved_solver == "heuristic" else None),
        )
        self._routing_diagnostics["candidate_routes"] = sum(
            len(routes)
            for group in (origin_group, destination_group)
            for slots in group["options"].values()
            for routes in slots.values()
        )
        if resolved_solver == "heuristic":
            try:
                return self._solve_heuristic_boundary_orders(
                    fig,
                    ax_origin,
                    ax_destination,
                    origin_ports,
                    destination_ports,
                    origin_group,
                    destination_group,
                )
            except RuntimeError as exc:
                if "No feasible fixed-slope port assignment" not in str(exc):
                    raise
                fallback_grid_size = min(self.site_grid_size, 3)
                origin_group = self._build_route_options(
                    fig,
                    ax_origin,
                    list(self.row_order_),
                    self.o_centroids_,
                    self.origin_zones,
                    self.zone_id_col,
                    origin_ports,
                    "origin",
                    self._outflows,
                    site_grid_size=fallback_grid_size,
                )
                destination_group = self._build_route_options(
                    fig,
                    ax_destination,
                    list(self.column_order_),
                    self.d_centroids_,
                    self.dest_zones,
                    self.dest_zone_id_col,
                    destination_ports,
                    "destination",
                    self._inflows,
                    site_grid_size=fallback_grid_size,
                )
                self._routing_diagnostics["fallback_grid_size"] = (
                    fallback_grid_size
                )
                self._routing_diagnostics["candidate_routes"] += sum(
                    len(routes)
                    for group in (origin_group, destination_group)
                    for slots in group["options"].values()
                    for routes in slots.values()
                )
                return self._solve_heuristic_boundary_orders(
                    fig,
                    ax_origin,
                    ax_destination,
                    origin_ports,
                    destination_ports,
                    origin_group,
                    destination_group,
                )
        if not self.allow_reorder:
            origin_assignment = self._solve_port_site_assignment(
                list(self.row_order_),
                [origin_group],
                fixed_order=self.row_order_,
            )
            destination_assignment = self._solve_port_site_assignment(
                list(self.column_order_),
                [destination_group],
                fixed_order=self.column_order_,
            )
            self._fixed_layout_solution = {
                "origin": self._routes_from_assignment(
                    origin_assignment,
                )["origin"],
                "destination": self._routes_from_assignment(
                    destination_assignment,
                )["destination"],
            }
            return list(self.row_order_), list(self.column_order_)
        if self._same_entity_set:
            try:
                ids = list(self.row_order_)
                candidate_orders = self._solve_compatible_orders(
                    ids,
                    [origin_group, destination_group],
                )
                best_layout = None
                best_gap = -np.inf
                for shared_order in candidate_orders:
                    try:
                        origin_assignment = self._solve_port_site_assignment(
                            ids,
                            [origin_group],
                            fixed_order=shared_order,
                            allow_crossings=False,
                        )
                        destination_assignment = (
                            self._solve_port_site_assignment(
                                ids,
                                [destination_group],
                                fixed_order=shared_order,
                                allow_crossings=False,
                            )
                        )
                    except RuntimeError:
                        continue
                    achieved_gap = min(
                        self._assignment_minimum_clearance(
                            origin_assignment,
                        ),
                        self._assignment_minimum_clearance(
                            destination_assignment,
                        ),
                    )
                    if achieved_gap > best_gap:
                        best_gap = achieved_gap
                        best_layout = (
                            list(shared_order),
                            origin_assignment,
                            destination_assignment,
                        )
                    if achieved_gap >= self.min_leader_gap - 1e-6:
                        break
                if best_layout is None:
                    raise RuntimeError(
                        "No globally crossing-free shared order was found "
                        "for both map corridors."
                    )
                (
                    shared_order,
                    origin_assignment,
                    destination_assignment,
                ) = best_layout
                self._fixed_layout_solution = {
                    "origin": self._routes_from_assignment(
                        origin_assignment,
                    )["origin"],
                    "destination": self._routes_from_assignment(
                        destination_assignment,
                    )["destination"],
                }
                return shared_order, list(shared_order)
            except RuntimeError as exc:
                if self.allow_leader_crossings:
                    joint_assignment = (
                        self._solve_joint_crossing_assignment(
                            ids,
                            [origin_group, destination_group],
                        )
                    )
                    shared_order = self._order_from_assignment(
                        ids, joint_assignment,
                    )
                    self._fixed_layout_solution = (
                        self._routes_from_assignment(joint_assignment)
                    )
                    return shared_order, list(shared_order)
                warnings.warn(
                    "Shared corridor order is infeasible "
                    f"({exc}); falling back to independent "
                    "origin/destination ordering.",
                    stacklevel=2,
                )

        origin_assignment = self._solve_port_site_assignment(
            list(self.row_order_),
            [origin_group],
        )
        destination_assignment = self._solve_port_site_assignment(
            list(self.column_order_),
            [destination_group],
        )
        row_order = self._order_from_assignment(
            list(self.row_order_), origin_assignment,
        )
        column_order = self._order_from_assignment(
            list(self.column_order_), destination_assignment,
        )
        origin_assignment = self._solve_port_site_assignment(
            list(self.row_order_),
            [origin_group],
            fixed_order=row_order,
        )
        destination_assignment = self._solve_port_site_assignment(
            list(self.column_order_),
            [destination_group],
            fixed_order=column_order,
        )
        self._fixed_layout_solution = {
            "origin": self._routes_from_assignment(
                origin_assignment,
            )["origin"],
            "destination": self._routes_from_assignment(
                destination_assignment,
            )["destination"],
        }
        return row_order, column_order

    def _fixed_display_route(self, site_display, port_display):
        """Construct one fixed-slope do-leader in display coordinates."""
        site_display = np.asarray(site_display, dtype=float)
        port_display = np.asarray(port_display, dtype=float)
        delta_y = port_display[1] - site_display[1]
        if abs(delta_y) < 1e-6:
            return None
        gradient = np.copysign(self.leader_slope, delta_y)
        bend_x = site_display[0] + delta_y / gradient
        if (
            bend_x <= site_display[0] + 1.0
            or bend_x >= port_display[0] - 1.0
        ):
            return None
        bend = np.asarray((bend_x, port_display[1]), dtype=float)
        path = [site_display, bend, port_display]
        return {
            "path": path,
            "length": float(
                np.linalg.norm(site_display - bend)
                + np.linalg.norm(bend - port_display)
            ),
            "band": "up" if delta_y > 0 else "down",
            "slope": float(-gradient),
            "display_slope": float(gradient),
        }

    def _build_route_options(
        self,
        fig,
        ax,
        ids,
        initial_sites,
        zones,
        zone_id_col,
        ports,
        kind,
        flow_totals,
        *,
        site_grid_size=None,
        slots_by_zone=None,
    ):
        """Create fixed-slope candidates for every zone/port pairing."""
        if site_grid_size is None:
            site_grid_size = self.site_grid_size
        prepared = _prepare_zones(zones, zone_id_col=zone_id_col)
        geometries = dict(zip(prepared["zone_id"], prepared.geometry))
        width_map = self._leader_width_map(ids, flow_totals)
        options = {}

        for zone_id in ids:
            geometry = geometries[zone_id]
            initial = np.asarray(initial_sites[zone_id], dtype=float)
            initial_display = ax.transData.transform(initial)
            minx, miny, maxx, maxy = geometry.bounds
            x_values = np.linspace(
                minx, maxx, site_grid_size + 2,
            )[1:-1] if site_grid_size else []
            y_values = np.linspace(
                miny, maxy, site_grid_size + 2,
            )[1:-1] if site_grid_size else []
            site_candidates = [tuple(initial)]
            site_candidates.extend(
                (float(x), float(y))
                for y in y_values
                for x in x_values
                if geometry.covers(Point(float(x), float(y)))
            )

            by_slot = {}
            active_slots = (
                range(len(ports))
                if slots_by_zone is None
                else slots_by_zone[zone_id]
            )
            if isinstance(active_slots, (int, np.integer)):
                active_slots = [int(active_slots)]
            for slot in active_slots:
                port_figure = ports[slot]
                port_display = fig.transFigure.transform(port_figure)
                candidates = []
                seen = set()
                for site_data in site_candidates:
                    site_display = ax.transData.transform(site_data)
                    route = self._fixed_display_route(
                        site_display, port_display,
                    )
                    if route is None:
                        continue
                    key = tuple(
                        np.round(np.concatenate(route["path"]), 5)
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    figure_path = [
                        fig.transFigure.inverted().transform(point)
                        for point in route["path"]
                    ]
                    displacement = float(
                        np.linalg.norm(site_display - initial_display)
                    )
                    candidates.append(
                        {
                            "kind": kind,
                            "zone_id": zone_id,
                            "slot": slot,
                            "site": tuple(site_data),
                            "path": figure_path,
                            "line": LineString(route["path"]),
                            "band": route["band"],
                            "slope": route["slope"],
                            "diagonal_intercept": float(
                                site_display[1]
                                - route["display_slope"] * site_display[0]
                            ),
                            "linewidth_px": float(
                                width_map[zone_id] * fig.dpi / 72.0
                            ),
                            "score": displacement
                            + 0.03 * route["length"],
                        }
                    )
                candidates.sort(key=lambda item: item["score"])
                if candidates:
                    by_slot[slot] = candidates[:12]
            options[zone_id] = by_slot

        return {"kind": kind, "ports": ports, "options": options}

    def _solve_heuristic_boundary_orders(
        self,
        fig,
        ax_origin,
        ax_destination,
        origin_ports,
        destination_ports,
        origin_group,
        destination_group,
    ):
        """Resolve a large fixed-angle layout without a quadratic MILP."""
        iterations = 0
        if not self.allow_reorder:
            row_order = list(self.row_order_)
            column_order = list(self.column_order_)
        elif self._same_entity_set:
            row_order, order_iterations = self._heuristic_port_order(
                list(self.row_order_),
                [origin_group, destination_group],
            )
            column_order = list(row_order)
            iterations += order_iterations
        else:
            row_order, origin_iterations = self._heuristic_port_order(
                list(self.row_order_), [origin_group],
            )
            column_order, destination_iterations = (
                self._heuristic_port_order(
                    list(self.column_order_), [destination_group],
                )
            )
            iterations += origin_iterations + destination_iterations

        origin_group = self._build_route_options(
            fig,
            ax_origin,
            list(self.row_order_),
            self.o_centroids_,
            self.origin_zones,
            self.zone_id_col,
            origin_ports,
            "origin",
            self._outflows,
            slots_by_zone={
                zone_id: range(
                    max(0, slot - 2),
                    min(len(row_order), slot + 3),
                )
                for slot, zone_id in enumerate(row_order)
            },
        )
        destination_group = self._build_route_options(
            fig,
            ax_destination,
            list(self.column_order_),
            self.d_centroids_,
            self.dest_zones,
            self.dest_zone_id_col,
            destination_ports,
            "destination",
            self._inflows,
            slots_by_zone={
                zone_id: range(
                    max(0, slot - 2),
                    min(len(column_order), slot + 3),
                )
                for slot, zone_id in enumerate(column_order)
            },
        )
        self._routing_diagnostics["candidate_routes"] += sum(
            len(routes)
            for group in (origin_group, destination_group)
            for slots in group["options"].values()
            for routes in slots.values()
        )
        if self.allow_reorder and self._same_entity_set:
            row_order, refined_iterations = self._heuristic_port_order(
                list(self.row_order_),
                [origin_group, destination_group],
            )
            column_order = list(row_order)
            iterations += refined_iterations
        elif self.allow_reorder:
            row_order, refined_origin_iterations = (
                self._heuristic_port_order(
                    list(self.row_order_), [origin_group],
                )
            )
            column_order, refined_destination_iterations = (
                self._heuristic_port_order(
                    list(self.column_order_), [destination_group],
                )
            )
            iterations += (
                refined_origin_iterations
                + refined_destination_iterations
            )

        origin_assignment, origin_site_iterations = (
            self._heuristic_route_assignment(row_order, origin_group)
        )
        destination_assignment, destination_site_iterations = (
            self._heuristic_route_assignment(
                column_order, destination_group,
            )
        )
        iterations += origin_site_iterations + destination_site_iterations

        origin_crossings = self._assignment_crossing_count(
            origin_assignment,
        )
        destination_crossings = self._assignment_crossing_count(
            destination_assignment,
        )
        total_crossings = origin_crossings + destination_crossings
        if not self.allow_leader_crossings and total_crossings:
            raise RuntimeError(
                "The heuristic did not find a crossing-free fixed-slope "
                "MapTrix layout. Use routing_solver='exact', enable "
                "allow_leader_crossings, or adjust the layout."
            )

        self._fixed_layout_solution = {
            "origin": self._routes_from_assignment(
                origin_assignment,
            )["origin"],
            "destination": self._routes_from_assignment(
                destination_assignment,
            )["destination"],
        }
        self._routing_diagnostics.update({
            "iterations": iterations,
            "optimal": total_crossings == 0,
            "search_crossings": {
                "origin": origin_crossings,
                "destination": destination_crossings,
                "total": total_crossings,
            },
        })
        return row_order, column_order

    def _heuristic_port_order(self, ids, groups):
        """Build and locally improve a feasible one-to-one port order."""
        count = len(ids)
        costs = np.full((count, count), np.inf, dtype=float)
        for zone_index, zone_id in enumerate(ids):
            for slot in range(count):
                if not all(
                    slot in group["options"][zone_id]
                    for group in groups
                ):
                    continue
                costs[zone_index, slot] = sum(
                    group["options"][zone_id][slot][0]["score"]
                    for group in groups
                )
        try:
            zone_indices, slots = linear_sum_assignment(costs)
        except ValueError as exc:
            raise RuntimeError(
                "No feasible fixed-slope port assignment was found."
            ) from exc
        if (
            len(zone_indices) != count
            or not np.all(np.isfinite(costs[zone_indices, slots]))
        ):
            raise RuntimeError(
                "No feasible fixed-slope port assignment was found."
            )

        order = [None] * count
        for zone_index, slot in zip(zone_indices, slots):
            order[int(slot)] = ids[int(zone_index)]

        pair_cache = {}
        best_score = self._heuristic_order_objective(
            order, groups, pair_cache,
        )
        iterations = 0
        if count <= 12:
            swap_distances = range(1, count)
        else:
            swap_distances = sorted({
                distance
                for distance in (1, 2, 4, 8, 16, count // 2)
                if distance < count
            })
        for _ in range(self.routing_max_iterations):
            candidate_order = None
            candidate_score = best_score
            for distance in swap_distances:
                for first in range(count - distance):
                    second = first + distance
                    trial = list(order)
                    trial[first], trial[second] = (
                        trial[second], trial[first]
                    )
                    score = self._heuristic_swap_objective(
                        order,
                        groups,
                        best_score,
                        first,
                        second,
                        pair_cache,
                    )
                    if score < candidate_score:
                        candidate_order = trial
                        candidate_score = score
            if candidate_order is None:
                break
            order = candidate_order
            best_score = candidate_score
            iterations += 1
            if best_score[0] == 0 and best_score[1] <= 1e-9:
                break
        return order, iterations

    def _heuristic_swap_objective(
        self,
        order,
        groups,
        current_score,
        first_position,
        second_position,
        pair_cache,
    ):
        """Update an order objective using only swap-affected pairs."""
        first_zone = order[first_position]
        second_zone = order[second_position]
        crossings, gap_deficit, route_cost = current_score
        for group_index, group in enumerate(groups):
            first_old = group["options"][first_zone].get(first_position)
            second_old = group["options"][second_zone].get(second_position)
            first_new = group["options"][first_zone].get(second_position)
            second_new = group["options"][second_zone].get(first_position)
            if not first_new or not second_new:
                return (np.inf, np.inf, np.inf)
            route_cost += (
                first_new[0]["score"]
                + second_new[0]["score"]
                - first_old[0]["score"]
                - second_old[0]["score"]
            )
            affected = [
                position
                for position in range(len(order))
                if position not in {first_position, second_position}
            ]
            for other_position in affected:
                other_zone = order[other_position]
                old_first = self._heuristic_cached_pair_metrics(
                    group_index,
                    group,
                    first_zone,
                    first_position,
                    other_zone,
                    other_position,
                    pair_cache,
                )
                new_first = self._heuristic_cached_pair_metrics(
                    group_index,
                    group,
                    first_zone,
                    second_position,
                    other_zone,
                    other_position,
                    pair_cache,
                )
                old_second = self._heuristic_cached_pair_metrics(
                    group_index,
                    group,
                    second_zone,
                    second_position,
                    other_zone,
                    other_position,
                    pair_cache,
                )
                new_second = self._heuristic_cached_pair_metrics(
                    group_index,
                    group,
                    second_zone,
                    first_position,
                    other_zone,
                    other_position,
                    pair_cache,
                )
                crossings += (
                    new_first[0] + new_second[0]
                    - old_first[0] - old_second[0]
                )
                gap_deficit += (
                    self._clearance_deficit(new_first[1])
                    + self._clearance_deficit(new_second[1])
                    - self._clearance_deficit(old_first[1])
                    - self._clearance_deficit(old_second[1])
                )
            old_pair = self._heuristic_cached_pair_metrics(
                group_index,
                group,
                first_zone,
                first_position,
                second_zone,
                second_position,
                pair_cache,
            )
            new_pair = self._heuristic_cached_pair_metrics(
                group_index,
                group,
                first_zone,
                second_position,
                second_zone,
                first_position,
                pair_cache,
            )
            crossings += new_pair[0] - old_pair[0]
            gap_deficit += (
                self._clearance_deficit(new_pair[1])
                - self._clearance_deficit(old_pair[1])
            )
        return int(crossings), float(gap_deficit), float(route_cost)

    def _heuristic_cached_pair_metrics(
        self,
        group_index,
        group,
        first_zone,
        first_slot,
        second_zone,
        second_slot,
        pair_cache,
    ):
        key = (
            group_index,
            first_zone,
            first_slot,
            second_zone,
            second_slot,
        )
        metrics = pair_cache.get(key)
        if metrics is None:
            first = group["options"][first_zone].get(first_slot)
            second = group["options"][second_zone].get(second_slot)
            if not first or not second:
                return (np.inf, -np.inf)
            metrics = self._heuristic_route_pair_metrics(
                first[0], second[0],
            )
            pair_cache[key] = metrics
        return metrics

    def _clearance_deficit(self, clearance):
        if not np.isfinite(clearance):
            return 0.0
        return max(self.min_leader_gap - clearance, 0.0)

    def _heuristic_order_objective(self, order, groups, pair_cache):
        """Score an order using each zone/slot pair's cheapest route."""
        selected = []
        route_cost = 0.0
        for group_index, group in enumerate(groups):
            group_routes = []
            for slot, zone_id in enumerate(order):
                candidates = group["options"][zone_id].get(slot)
                if not candidates:
                    return (np.inf, np.inf, np.inf)
                route = candidates[0]
                group_routes.append((zone_id, slot, route))
                route_cost += route["score"]
            selected.append((group_index, group_routes))

        crossings = 0
        gap_deficit = 0.0
        for group_index, group_routes in selected:
            for position, (first_zone, first_slot, first_route) in enumerate(
                group_routes
            ):
                for second_zone, second_slot, second_route in group_routes[
                    position + 1:
                ]:
                    key = (
                        group_index,
                        first_zone,
                        first_slot,
                        second_zone,
                        second_slot,
                    )
                    metrics = pair_cache.get(key)
                    if metrics is None:
                        metrics = self._heuristic_route_pair_metrics(
                            first_route, second_route,
                        )
                        pair_cache[key] = metrics
                    pair_crossings, clearance = metrics
                    crossings += pair_crossings
                    gap_deficit += self._clearance_deficit(clearance)
        return crossings, gap_deficit, route_cost

    def _heuristic_route_assignment(self, order, group):
        """Choose map sites for a fixed order by coordinate descent."""
        selected = {}
        for slot, zone_id in enumerate(order):
            candidates = group["options"][zone_id].get(slot)
            if not candidates:
                raise RuntimeError(
                    f"No feasible fixed-slope route for zone {zone_id!r} "
                    f"at slot {slot}."
                )
            selected[zone_id] = candidates[0]

        iterations = 0
        max_passes = min(self.routing_max_iterations, 8)
        for _ in range(max_passes):
            changed = False
            for slot, zone_id in enumerate(order):
                candidates = group["options"][zone_id][slot]
                best_route = selected[zone_id]
                best_score = self._heuristic_site_objective(
                    zone_id, best_route, selected,
                )
                for candidate in candidates[1:]:
                    score = self._heuristic_site_objective(
                        zone_id, candidate, selected,
                    )
                    if score < best_score:
                        best_route = candidate
                        best_score = score
                if best_route is not selected[zone_id]:
                    selected[zone_id] = best_route
                    changed = True
            if not changed:
                break
            iterations += 1

        assignment = {
            zone_id: {
                "slot": slot,
                "routes": (selected[zone_id],),
                "score": selected[zone_id]["score"],
            }
            for slot, zone_id in enumerate(order)
        }
        return assignment, iterations

    def _heuristic_site_objective(
        self, zone_id, candidate, selected,
    ):
        crossings = 0
        gap_deficit = 0.0
        for other_zone, other in selected.items():
            if other_zone == zone_id:
                continue
            pair_crossings, clearance = (
                self._heuristic_route_pair_metrics(candidate, other)
            )
            crossings += pair_crossings
            if np.isfinite(clearance):
                gap_deficit += max(
                    self.min_leader_gap - clearance, 0.0,
                )
        return crossings, gap_deficit, candidate["score"]

    def _heuristic_route_pair_metrics(self, first, second):
        crossing_count = int(first["line"].intersects(second["line"]))
        if first["band"] != second["band"]:
            return crossing_count, np.inf
        center_distance = abs(
            first["diagonal_intercept"]
            - second["diagonal_intercept"]
        ) / np.sqrt(self.leader_slope ** 2 + 1.0)
        clearance = center_distance - 0.5 * (
            first["linewidth_px"] + second["linewidth_px"]
        )
        return crossing_count, clearance

    def _assignment_crossing_count(self, assignment):
        entries = list(assignment.values())
        return sum(
            self._entry_crossing_count(first, second)
            for index, first in enumerate(entries)
            for second in entries[index + 1:]
        )

    @staticmethod
    def _solve_compatible_orders(
        ids, groups, max_orders=30,
    ):
        """Find shared orders with pairwise-compatible site choices."""
        assignments = []
        costs = []
        for zone_id in ids:
            for slot in range(len(ids)):
                if not all(
                    slot in group["options"][zone_id] for group in groups
                ):
                    continue
                assignments.append((zone_id, slot))
                costs.append(
                    sum(
                        group["options"][zone_id][slot][0]["score"]
                        for group in groups
                    )
                )

        variable_count = len(assignments)
        count = len(ids)
        zone_indices = {zone_id: index for index, zone_id in enumerate(ids)}
        variable_lookup = {
            assignment: index
            for index, assignment in enumerate(assignments)
        }
        incompatible = []
        for first_zone_index, first_zone in enumerate(ids):
            for second_zone in ids[first_zone_index + 1:]:
                for first_slot in range(count):
                    first_index = variable_lookup.get(
                        (first_zone, first_slot)
                    )
                    if first_index is None:
                        continue
                    for second_slot in range(count):
                        if first_slot == second_slot:
                            continue
                        second_index = variable_lookup.get(
                            (second_zone, second_slot)
                        )
                        if second_index is None:
                            continue
                        compatible = True
                        for group in groups:
                            first_routes = group["options"][first_zone][
                                first_slot
                            ]
                            second_routes = group["options"][second_zone][
                                second_slot
                            ]
                            if not any(
                                not first["line"].intersects(
                                    second["line"]
                                )
                                for first in first_routes
                                for second in second_routes
                            ):
                                compatible = False
                                break
                        if not compatible:
                            incompatible.append(
                                (first_index, second_index)
                            )

        matrix = lil_matrix(
            (count * 2 + len(incompatible), variable_count),
            dtype=float,
        )
        for variable_index, (zone_id, slot) in enumerate(assignments):
            matrix[zone_indices[zone_id], variable_index] = 1.0
            matrix[count + slot, variable_index] = 1.0
        for row, (first, second) in enumerate(
            incompatible, start=count * 2,
        ):
            matrix[row, first] = 1.0
            matrix[row, second] = 1.0

        lower = np.concatenate(
            [np.ones(count * 2), np.full(len(incompatible), -np.inf)]
        )
        upper = np.ones(count * 2 + len(incompatible))
        costs = np.asarray(costs, dtype=float)
        costs /= max(float(costs.max()), 1.0)
        orders = []
        selected_sets = []
        for _ in range(max_orders):
            row_count = matrix.shape[0] + len(selected_sets)
            active_matrix = lil_matrix(
                (row_count, variable_count), dtype=float,
            )
            active_matrix[: matrix.shape[0]] = matrix
            active_lower = np.concatenate(
                [lower, np.full(len(selected_sets), -np.inf)]
            )
            active_upper = np.concatenate(
                [upper, np.full(len(selected_sets), count - 1.0)]
            )
            for offset, selected in enumerate(selected_sets):
                row = matrix.shape[0] + offset
                for variable_index in selected:
                    active_matrix[row, variable_index] = 1.0
            result = milp(
                c=costs,
                integrality=np.ones(variable_count),
                bounds=Bounds(
                    np.zeros(variable_count), np.ones(variable_count),
                ),
                constraints=LinearConstraint(
                    active_matrix.tocsr(), active_lower, active_upper,
                ),
                options={"time_limit": 20.0},
            )
            if not result.success or result.x is None:
                break
            selected = np.flatnonzero(result.x > 0.5).tolist()
            selected_sets.append(selected)
            order = [None] * count
            for variable_index in selected:
                zone_id, slot = assignments[variable_index]
                order[slot] = zone_id
            orders.append(order)
        if not orders:
            raise RuntimeError(
                "No shared fixed-slope port order is feasible for both "
                "map corridors."
            )
        return orders

    def _solve_joint_crossing_assignment(self, ids, groups):
        """Optimise a shared order without multiplying corridor sites.

        One binary variable chooses each zone/slot pair and separate binary
        variables choose the route site in every corridor.  This keeps the
        two maps coupled through the shared matrix order while avoiding the
        Cartesian product of their site candidates.
        """
        count = len(ids)
        zone_indices = {zone_id: index for index, zone_id in enumerate(ids)}
        order_assignments = []
        for zone_id in ids:
            for slot in range(count):
                if all(
                    slot in group["options"][zone_id]
                    for group in groups
                ):
                    order_assignments.append((zone_id, slot))
        if not order_assignments:
            raise RuntimeError("No shared fixed-slope port order is feasible.")

        order_lookup = {
            assignment: index
            for index, assignment in enumerate(order_assignments)
        }
        route_variables = []
        routes_by_assignment = {}
        for group_index, group in enumerate(groups):
            for zone_id, slot in order_assignments:
                indices = []
                for route in group["options"][zone_id][slot]:
                    indices.append(
                        len(order_assignments) + len(route_variables)
                    )
                    route_variables.append(
                        {
                            "group": group_index,
                            "zone_id": zone_id,
                            "slot": slot,
                            "route": route,
                        }
                    )
                routes_by_assignment[(group_index, zone_id, slot)] = indices

        base_variable_count = len(order_assignments) + len(route_variables)
        equality_count = count * 2 + len(groups) * len(order_assignments)
        base = lil_matrix(
            (equality_count, base_variable_count), dtype=float,
        )
        for variable_index, (zone_id, slot) in enumerate(order_assignments):
            base[zone_indices[zone_id], variable_index] = 1.0
            base[count + slot, variable_index] = 1.0
        link_row = count * 2
        for group_index in range(len(groups)):
            for zone_id, slot in order_assignments:
                order_variable = order_lookup[(zone_id, slot)]
                base[link_row, order_variable] = -1.0
                for route_variable in routes_by_assignment[
                    (group_index, zone_id, slot)
                ]:
                    base[link_row, route_variable] = 1.0
                link_row += 1

        crossing_pairs = []
        clearances = {}
        for group_index in range(len(groups)):
            group_variables = [
                len(order_assignments) + index
                for index, item in enumerate(route_variables)
                if item["group"] == group_index
            ]
            for position, first_index in enumerate(group_variables):
                first = route_variables[
                    first_index - len(order_assignments)
                ]
                first_route = first["route"]
                for second_index in group_variables[position + 1:]:
                    second = route_variables[
                        second_index - len(order_assignments)
                    ]
                    if (
                        first["zone_id"] == second["zone_id"]
                        or first["slot"] == second["slot"]
                    ):
                        continue
                    second_route = second["route"]
                    pair = (first_index, second_index)
                    if first_route["line"].intersects(second_route["line"]):
                        crossing_pairs.append(pair)
                    if first_route["band"] == second_route["band"]:
                        center_distance = abs(
                            first_route["diagonal_intercept"]
                            - second_route["diagonal_intercept"]
                        ) / np.sqrt(self.leader_slope ** 2 + 1.0)
                        clearances[pair] = center_distance - 0.5 * (
                            first_route["linewidth_px"]
                            + second_route["linewidth_px"]
                        )

        crossing_variable_offset = base_variable_count
        crossing_variable_count = len(crossing_pairs)
        route_costs = np.asarray(
            [item["route"]["score"] for item in route_variables],
            dtype=float,
        )
        route_costs /= max(float(route_costs.max()), 1.0)
        base_costs = np.zeros(base_variable_count, dtype=float)
        base_costs[len(order_assignments):] = route_costs

        def solve(conflicts, *, crossing_limit=None, objective="cost"):
            limit_count = int(crossing_limit is not None)
            constraint_count = (
                equality_count + len(conflicts)
                + crossing_variable_count + limit_count
            )
            variable_count = base_variable_count + crossing_variable_count
            matrix = lil_matrix(
                (constraint_count, variable_count), dtype=float,
            )
            matrix[:equality_count, :base_variable_count] = base
            lower = np.full(constraint_count, -np.inf)
            upper = np.full(constraint_count, np.inf)
            lower[:equality_count] = np.concatenate(
                [np.ones(count * 2), np.zeros(equality_count - count * 2)]
            )
            upper[:equality_count] = lower[:equality_count]

            conflict_start = equality_count
            for offset, (first, second) in enumerate(sorted(conflicts)):
                row = conflict_start + offset
                matrix[row, first] = 1.0
                matrix[row, second] = 1.0
                upper[row] = 1.0

            crossing_start = conflict_start + len(conflicts)
            for offset, (first, second) in enumerate(crossing_pairs):
                row = crossing_start + offset
                matrix[row, first] = 1.0
                matrix[row, second] = 1.0
                matrix[row, crossing_variable_offset + offset] = -1.0
                upper[row] = 1.0

            if crossing_limit is not None:
                row = constraint_count - 1
                for offset in range(crossing_variable_count):
                    matrix[row, crossing_variable_offset + offset] = 1.0
                upper[row] = float(crossing_limit)

            objective_costs = np.zeros(variable_count, dtype=float)
            if objective == "crossings":
                objective_costs[crossing_variable_offset:] = 1.0
            else:
                objective_costs[:base_variable_count] = base_costs
            result = milp(
                c=objective_costs,
                integrality=np.ones(variable_count),
                bounds=Bounds(
                    np.zeros(variable_count), np.ones(variable_count),
                ),
                constraints=LinearConstraint(
                    matrix.tocsr(), lower, upper,
                ),
                options={"time_limit": 20.0},
            )
            if not result.success or result.x is None:
                return None
            return np.flatnonzero(
                result.x[:base_variable_count] > 0.5
            ).tolist()

        chosen = solve(set(), objective="crossings")
        if chosen is None:
            raise RuntimeError(
                "No optimal fixed-slope MapTrix layout was found."
            )
        selected = set(chosen)
        crossing_limit = sum(
            first in selected and second in selected
            for first, second in crossing_pairs
        )
        chosen = solve(set(), crossing_limit=crossing_limit)
        if chosen is None:
            raise RuntimeError(
                "No optimal fixed-slope MapTrix layout was found."
            )

        selected = set(chosen)
        selected_route_indices = {
            index for index in selected if index >= len(order_assignments)
        }
        selected_gaps = [
            clearance
            for (first, second), clearance in clearances.items()
            if first in selected_route_indices and second in selected_route_indices
        ]
        current_gap = min(selected_gaps) if selected_gaps else np.inf
        target_gap = max(float(self.min_leader_gap), 0.0)
        if np.isfinite(current_gap) and current_gap < target_gap:
            low = current_gap
            high = target_gap
            best = chosen
            for _ in range(10):
                threshold = (low + high) / 2.0
                spacing_conflicts = {
                    pair
                    for pair, clearance in clearances.items()
                    if clearance < threshold
                }
                candidate = solve(
                    spacing_conflicts,
                    crossing_limit=crossing_limit,
                )
                if candidate is None:
                    high = threshold
                else:
                    low = threshold
                    best = candidate
            chosen = best

        selected = set(chosen)
        selected_order = {
            zone_id: slot
            for index, (zone_id, slot) in enumerate(order_assignments)
            if index in selected
        }
        selected_routes = {}
        for index in selected:
            if index < len(order_assignments):
                continue
            item = route_variables[index - len(order_assignments)]
            selected_routes[(item["group"], item["zone_id"])] = item["route"]
        return {
            zone_id: {
                "slot": selected_order[zone_id],
                "routes": tuple(
                    selected_routes[(group_index, zone_id)]
                    for group_index in range(len(groups))
                ),
                "score": sum(
                    selected_routes[(group_index, zone_id)]["score"]
                    for group_index in range(len(groups))
                ),
            }
            for zone_id in ids
        }

    def _solve_port_site_assignment(
        self, ids, groups, fixed_order=None, allow_crossings=None,
    ):
        """Solve shared port assignment and crossing-free site selection."""
        if allow_crossings is None:
            allow_crossings = self.allow_leader_crossings
        domains = {}
        for zone_id in ids:
            entries = []
            common_slots = set(range(len(ids)))
            for group in groups:
                common_slots &= set(group["options"][zone_id])
            for slot in sorted(common_slots):
                if (
                    fixed_order is not None
                    and fixed_order[slot] != zone_id
                ):
                    continue
                route_lists = [
                    group["options"][zone_id][slot] for group in groups
                ]
                combinations = [()]
                for route_list in route_lists:
                    combinations = [
                        combination + (route,)
                        for combination in combinations
                        for route in route_list
                    ]
                for routes in combinations:
                    entries.append(
                        {
                            "slot": slot,
                            "routes": routes,
                            "score": sum(
                                route["score"] for route in routes
                            ),
                        }
                    )
            entries.sort(key=lambda item: item["score"])
            domains[zone_id] = entries
            if not entries:
                raise RuntimeError(
                    f"No feasible fixed-slope route for zone {zone_id!r} "
                    f"at leader_angle={self.leader_angle:g}°. Adjust the "
                    "angle, map/matrix rectangles, or site sampling."
                )

        variables = []
        for zone_id in ids:
            variables.extend(domains[zone_id])
        variable_count = len(variables)
        count = len(ids)
        zone_indices = {zone_id: index for index, zone_id in enumerate(ids)}

        base = lil_matrix((count * 2, variable_count), dtype=float)
        for variable_index, entry in enumerate(variables):
            zone_id = entry["routes"][0]["zone_id"]
            base[zone_indices[zone_id], variable_index] = 1.0
            base[count + entry["slot"], variable_index] = 1.0

        conflict_pairs = set()
        costs = np.asarray(
            [entry["score"] for entry in variables], dtype=float,
        )
        scale = max(float(costs.max()), 1.0)
        costs = costs / scale

        if fixed_order is not None or allow_crossings:
            return self._solve_spaced_fixed_assignment(
                variables,
                base,
                costs,
                ids,
                allow_crossings=allow_crossings,
            )

        for _ in range(500):
            constraint_count = count * 2 + len(conflict_pairs)
            matrix = lil_matrix(
                (constraint_count, variable_count), dtype=float,
            )
            matrix[: count * 2] = base
            lower = np.concatenate(
                [np.ones(count * 2), np.full(len(conflict_pairs), -np.inf)]
            )
            upper = np.ones(constraint_count)
            for row, (first, second) in enumerate(
                sorted(conflict_pairs), start=count * 2,
            ):
                matrix[row, first] = 1.0
                matrix[row, second] = 1.0

            result = milp(
                c=costs,
                integrality=np.ones(variable_count),
                bounds=Bounds(
                    np.zeros(variable_count), np.ones(variable_count),
                ),
                constraints=LinearConstraint(
                    matrix.tocsr(), lower, upper,
                ),
                options={"time_limit": 20.0},
            )
            if not result.success or result.x is None:
                break

            chosen_indices = np.flatnonzero(result.x > 0.5).tolist()
            new_conflicts = set()
            for position, first_index in enumerate(chosen_indices):
                first = variables[first_index]
                for second_index in chosen_indices[position + 1:]:
                    second = variables[second_index]
                    if any(
                        first_route["line"].intersects(
                            second_route["line"]
                        )
                        for first_route, second_route in zip(
                            first["routes"], second["routes"]
                        )
                    ):
                        new_conflicts.add(
                            tuple(sorted((first_index, second_index)))
                        )
            if not new_conflicts:
                return {
                    entry["routes"][0]["zone_id"]: entry
                    for entry in (
                        variables[index] for index in chosen_indices
                    )
                }
            # A selected route that crosses another candidate can never
            # coexist with it.  Adding the whole conflict neighbourhood,
            # rather than only the currently selected pairs, makes the
            # lazy-constraint loop converge in a few MILP solves.
            for first_index in chosen_indices:
                first = variables[first_index]
                first_zone = first["routes"][0]["zone_id"]
                for second_index, second in enumerate(variables):
                    if first_index == second_index:
                        continue
                    if (
                        first_zone == second["routes"][0]["zone_id"]
                        or first["slot"] == second["slot"]
                    ):
                        continue
                    pair = tuple(sorted((first_index, second_index)))
                    if pair in conflict_pairs or pair in new_conflicts:
                        continue
                    if any(
                        first_route["line"].intersects(
                            second_route["line"]
                        )
                        for first_route, second_route in zip(
                            first["routes"], second["routes"]
                        )
                    ):
                        new_conflicts.add(pair)
            previous_count = len(conflict_pairs)
            conflict_pairs.update(new_conflicts)
            if len(conflict_pairs) == previous_count:
                break

        raise RuntimeError(
            "No crossing-free fixed-slope MapTrix layout was found. "
            "Adjust leader_angle, layout rectangles, or site_grid_size."
        )

    def _solve_spaced_fixed_assignment(
        self, variables, base, costs, ids, allow_crossings=None,
    ):
        """Lexicographically optimise crossings, spacing, then route cost."""
        if allow_crossings is None:
            allow_crossings = self.allow_leader_crossings
        count = len(ids)
        hard_conflicts = set()
        crossing_counts = {}
        clearances = {}
        for first_index, first in enumerate(variables):
            first_zone = first["routes"][0]["zone_id"]
            for second_index in range(first_index + 1, len(variables)):
                second = variables[second_index]
                if (
                    first_zone == second["routes"][0]["zone_id"]
                    or first["slot"] == second["slot"]
                ):
                    continue
                pair = (first_index, second_index)
                crossing_count = self._entry_crossing_count(first, second)
                if crossing_count:
                    crossing_counts[pair] = crossing_count
                if not allow_crossings and crossing_count:
                    hard_conflicts.add(pair)
                clearance = self._entry_diagonal_clearance(first, second)
                if np.isfinite(clearance):
                    clearances[pair] = clearance

        crossing_items = sorted(crossing_counts.items())
        active_crossing_items = (
            crossing_items if allow_crossings else []
        )
        crossing_variable_offset = len(variables)
        crossing_variable_count = len(active_crossing_items)

        def solve(
            conflicts,
            *,
            crossing_limit=None,
            objective="cost",
        ):
            linearization_count = crossing_variable_count
            limit_count = int(crossing_limit is not None)
            constraint_count = (
                count * 2 + len(conflicts)
                + linearization_count + limit_count
            )
            variable_count = len(variables) + crossing_variable_count
            matrix = lil_matrix(
                (constraint_count, variable_count), dtype=float,
            )
            matrix[: count * 2, : len(variables)] = base
            lower = np.full(constraint_count, -np.inf)
            upper = np.ones(constraint_count)
            lower[: count * 2] = 1.0
            for row, (first, second) in enumerate(
                sorted(conflicts), start=count * 2,
            ):
                matrix[row, first] = 1.0
                matrix[row, second] = 1.0

            linearization_start = count * 2 + len(conflicts)
            for offset, ((first, second), _) in enumerate(
                active_crossing_items
            ):
                row = linearization_start + offset
                crossing_variable = crossing_variable_offset + offset
                matrix[row, first] = 1.0
                matrix[row, second] = 1.0
                matrix[row, crossing_variable] = -1.0

            if crossing_limit is not None:
                row = constraint_count - 1
                for offset, (_, crossing_count) in enumerate(
                    active_crossing_items
                ):
                    matrix[
                        row, crossing_variable_offset + offset
                    ] = crossing_count
                upper[row] = float(crossing_limit)

            objective_costs = np.zeros(variable_count, dtype=float)
            if objective == "crossings":
                for offset, (_, crossing_count) in enumerate(
                    active_crossing_items
                ):
                    objective_costs[
                        crossing_variable_offset + offset
                    ] = crossing_count
            else:
                objective_costs[: len(variables)] = costs
            result = milp(
                c=objective_costs,
                integrality=np.ones(variable_count),
                bounds=Bounds(
                    np.zeros(variable_count), np.ones(variable_count),
                ),
                constraints=LinearConstraint(
                    matrix.tocsr(), lower, upper,
                ),
                options={"time_limit": 20.0},
            )
            if not result.success or result.x is None:
                return None
            return np.flatnonzero(
                result.x[: len(variables)] > 0.5
            ).tolist()

        crossing_limit = None
        if allow_crossings:
            minimum_crossing_solution = solve(
                set(), objective="crossings",
            )
            if minimum_crossing_solution is None:
                raise RuntimeError(
                    "No fixed-slope MapTrix layout was found."
                )
            crossing_limit = self._selected_crossing_count(
                minimum_crossing_solution,
                variables,
            )
            chosen = solve(set(), crossing_limit=crossing_limit)
        else:
            chosen = solve(hard_conflicts)
        if chosen is None:
            raise RuntimeError(
                "No crossing-free fixed-slope MapTrix layout was found."
            )

        current_gap = self._selected_minimum_clearance(
            chosen, variables,
        )
        target_gap = max(float(self.min_leader_gap), 0.0)
        if not np.isfinite(current_gap) or current_gap >= target_gap:
            return self._assignment_from_indices(chosen, variables)

        low = current_gap
        high = target_gap
        best = chosen
        for _ in range(10):
            threshold = (low + high) / 2.0
            spacing_conflicts = {
                pair
                for pair, clearance in clearances.items()
                if clearance < threshold
            }
            candidate = solve(
                hard_conflicts | spacing_conflicts,
                crossing_limit=crossing_limit,
            )
            if candidate is None:
                high = threshold
            else:
                low = threshold
                best = candidate

        return self._assignment_from_indices(best, variables)

    @staticmethod
    def _entry_crossing_count(first, second):
        """Count pairwise route intersections across matching corridors."""
        return sum(
            first_route["line"].intersects(second_route["line"])
            for first_route, second_route in zip(
                first["routes"], second["routes"]
            )
        )

    def _selected_crossing_count(self, chosen, variables):
        """Count crossings in one selected assignment."""
        return sum(
            self._entry_crossing_count(
                variables[first_index], variables[second_index],
            )
            for position, first_index in enumerate(chosen)
            for second_index in chosen[position + 1:]
        )

    def _entry_diagonal_clearance(self, first, second):
        """Return the minimum visible gap between parallel diagonals."""
        gaps = []
        for first_route, second_route in zip(
            first["routes"], second["routes"]
        ):
            if first_route["band"] != second_route["band"]:
                continue
            center_distance = abs(
                first_route["diagonal_intercept"]
                - second_route["diagonal_intercept"]
            ) / np.sqrt(self.leader_slope ** 2 + 1.0)
            gaps.append(
                center_distance
                - 0.5
                * (
                    first_route["linewidth_px"]
                    + second_route["linewidth_px"]
                )
            )
        return min(gaps) if gaps else np.inf

    def _selected_minimum_clearance(self, chosen, variables):
        gaps = []
        for position, first_index in enumerate(chosen):
            for second_index in chosen[position + 1:]:
                clearance = self._entry_diagonal_clearance(
                    variables[first_index], variables[second_index],
                )
                if np.isfinite(clearance):
                    gaps.append(clearance)
        return min(gaps) if gaps else np.inf

    def _assignment_minimum_clearance(self, assignment):
        entries = list(assignment.values())
        gaps = []
        for index, first in enumerate(entries):
            for second in entries[index + 1:]:
                clearance = self._entry_diagonal_clearance(first, second)
                if np.isfinite(clearance):
                    gaps.append(clearance)
        return min(gaps) if gaps else np.inf

    @staticmethod
    def _assignment_from_indices(chosen, variables):
        return {
            entry["routes"][0]["zone_id"]: entry
            for entry in (variables[index] for index in chosen)
        }

    @staticmethod
    def _order_from_assignment(ids, assignment):
        order = [None] * len(ids)
        for zone_id, entry in assignment.items():
            order[entry["slot"]] = zone_id
        return order

    @staticmethod
    def _routes_from_assignment(assignment):
        routes = {"origin": {}, "destination": {}}
        for zone_id, entry in assignment.items():
            for route in entry["routes"]:
                routes[route["kind"]][zone_id] = route
        return routes

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
                self._outflows,
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
                self._inflows,
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
            initial_sites,
            zones,
            zone_id_col,
            flow_totals,
            matrix_side,
            expected_count,
            color,
            extra_gap,
        ) in groups:
            if len(order) != expected_count:
                raise RuntimeError("Matrix dimensions and leader order are inconsistent")

            width_map = self._leader_width_map(order, flow_totals)

            map_box = map_ax.get_position()
            port_points = self._matrix_port_points(
                fig, ax_matrix, matrix_side, len(order),
            )

            if self.leader_routing == "diagonal-horizontal":
                resolved = self._fixed_layout_solution[kind]
                sites = {
                    zone_id: resolved[zone_id]["site"]
                    for zone_id in order
                }
                paths = [
                    resolved[zone_id]["path"] for zone_id in order
                ]
                bands = [
                    resolved[zone_id]["band"] for zone_id in order
                ]
                slopes = [
                    resolved[zone_id]["slope"] for zone_id in order
                ]
            else:
                sites = self._optimize_map_sites(
                    fig,
                    map_ax,
                    order,
                    initial_sites,
                    zones,
                    zone_id_col,
                )
                min_port_x = min(point[0] for point in port_points)
                bend_x = min(
                    map_box.x1 + self.corridor_gap + extra_gap,
                    min_port_x - max(self.corridor_gap * 0.35, 0.006),
                )
                bend_x = max(bend_x, map_box.x1 + 0.004)
                paths = []
                for zone_id, port in zip(order, port_points):
                    site = np.asarray(
                        _ax_to_fig(map_ax, fig, *sites[zone_id])
                    )
                    bend = np.asarray((bend_x, site[1]), dtype=float)
                    paths.append([site, bend, np.asarray(port)])
                bands = [None] * len(paths)
                slopes = [None] * len(paths)

            if kind == "origin":
                self.o_sites_ = sites
            else:
                self.d_sites_ = sites

            for index, (zone_id, path, band, slope) in enumerate(
                zip(order, paths, bands, slopes)
            ):
                site, bend, port = path
                linewidth = width_map[zone_id]
                fig.add_artist(
                    Line2D(
                        [p[0] for p in path],
                        [p[1] for p in path],
                        transform=fig.transFigure,
                        color=color,
                        linewidth=linewidth,
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
                        "value": float(flow_totals.get(zone_id, 0.0)),
                        "linewidth": float(linewidth),
                        "band": band,
                        "slope": slope,
                    }
                )
        return geometry

    def _optimize_map_sites(
        self, fig, ax, order, initial_sites, zones, zone_id_col,
    ):
        """Spread legacy-route sites using in-polygon candidates."""
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
            x_values = np.linspace(
                minx, maxx, self.site_grid_size + 2,
            )[1:-1]
            y_values = np.linspace(
                miny, maxy, self.site_grid_size + 2,
            )[1:-1]
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
                    and display[1]
                    > previous_display_y - self.min_leader_gap
                ):
                    continue
                displacement = np.sum((display - initial_display) ** 2)
                scored.append(
                    (float(displacement), candidate, display[1])
                )

            if scored:
                _, chosen, chosen_display_y = min(
                    scored, key=lambda item: item[0],
                )
            else:
                chosen = tuple(initial)
                chosen_display_y = initial_display[1]
            result[zone_id] = chosen
            previous_display_y = chosen_display_y
        return result

    def _matrix_port_points(self, fig, ax_matrix, side, count):
        rows, cols = self.matrix_.shape
        points = []
        for index in range(count):
            x, y = _calculate_matrix_anchor_point(
                self._transform, rows, cols, side, index,
            )
            points.append(_ax_to_fig(ax_matrix, fig, x, y))
        return points

    def _leader_width_map(self, order, flow_totals):
        """Map regional totals to leader widths for one leader group."""
        if self.leader_width_range is None:
            return {zone_id: float(self.leader_linewidth) for zone_id in order}
        values = np.asarray(
            [flow_totals.get(zone_id, 0.0) for zone_id in order],
            dtype=float,
        )
        low, high = self.leader_width_range
        if values.size == 0:
            return {}
        if np.all(values == 0):
            widths = np.full(values.shape, low, dtype=float)
        elif np.ptp(values) == 0:
            widths = np.full(values.shape, (low + high) / 2.0, dtype=float)
        else:
            widths = _linear_scaling(values, (low, high))
        return {
            zone_id: float(width)
            for zone_id, width in zip(order, widths)
        }

    def _draw_matrix_ports(self, ax):
        if not self.show_matrix_ports or not self._leader_geometry:
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

    @staticmethod
    def _figure_rect_to_screen(fig, rect):
        left, bottom, width, height = rect
        return {
            "x": float(left * fig.bbox.width),
            "y": float((1.0 - bottom - height) * fig.bbox.height),
            "w": float(width * fig.bbox.width),
            "h": float(height * fig.bbox.height),
        }

    @staticmethod
    def _exported_minimum_diagonal_gap(fig, leaders):
        gaps = []
        for index, first in enumerate(leaders):
            if first["band"] is None:
                continue
            first_intercept = (
                first["site"]["y"]
                - first["slope"] * first["site"]["x"]
            )
            for second in leaders[index + 1:]:
                if first["band"] != second["band"]:
                    continue
                second_intercept = (
                    second["site"]["y"]
                    - second["slope"] * second["site"]["x"]
                )
                center_distance = abs(
                    first_intercept - second_intercept
                ) / np.sqrt(first["slope"] ** 2 + 1.0)
                stroke_radius = (
                    0.5
                    * (first["linewidth"] + second["linewidth"])
                    * fig.dpi
                    / 72.0
                )
                gaps.append(center_distance - stroke_radius)
        return float(min(gaps)) if gaps else None

    @staticmethod
    def _exported_leader_crossings(leaders):
        """Count intersections between exported leader paths."""
        lines = [
            LineString(
                [(point["x"], point["y"]) for point in leader["path"]]
            )
            for leader in leaders
        ]
        return sum(
            first.intersects(second)
            for index, first in enumerate(lines)
            for second in lines[index + 1:]
        )

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
                        **{
                            k: leader[k]
                            for k in (
                                "id", "kind", "order", "value", "linewidth",
                                "band", "slope",
                            )
                        },
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
        origin_crossings = self._exported_leader_crossings(origin_leaders)
        destination_crossings = self._exported_leader_crossings(
            destination_leaders,
        )
        routing_diagnostics = None
        if self._routing_diagnostics is not None:
            routing_diagnostics = dict(self._routing_diagnostics)
            routing_diagnostics["elapsed_seconds"] = float(
                time.perf_counter() - self._routing_started_at
            )
            routing_diagnostics["crossings"] = {
                "origin": origin_crossings,
                "destination": destination_crossings,
                "total": origin_crossings + destination_crossings,
            }
            routing_diagnostics["optimal"] = bool(
                routing_diagnostics["solver"] == "exact"
                or origin_crossings + destination_crossings == 0
            )
        origin_rect = self._axes_rect_to_screen(fig, ax_origin)
        destination_rect = self._axes_rect_to_screen(
            fig, ax_destination,
        )
        matrix_axes_rect = self._axes_rect_to_screen(fig, ax_matrix)
        colorbar_axes_rect = self._axes_rect_to_screen(
            fig, self.axes_["colorbar"],
        )
        frame_rects = self._component_frame_rects(
            fig,
            ax_origin,
            ax_destination,
            ax_matrix,
            self.axes_["colorbar"],
        )
        component_frames = {
            name: self._figure_rect_to_screen(fig, rect)
            for name, rect in frame_rects.items()
        }
        matrix_frame_rect = component_frames["matrix"]
        map_right = max(
            origin_rect["x"] + origin_rect["w"],
            destination_rect["x"] + destination_rect["w"],
        )

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
                "origin": origin_rect,
                "destination": destination_rect,
            },
            "axes_rects": {
                "origin_map": origin_rect,
                "destination_map": destination_rect,
                "matrix": matrix_axes_rect,
                "colorbar": colorbar_axes_rect,
            },
            "component_frames": component_frames,
            "resolved_rects": {
                "origin_map": self.origin_map_rect,
                "destination_map": self.destination_map_rect,
                "matrix": self.matrix_rect,
                "colorbar": self.colorbar_rect,
            },
            "layout_rect": self.layout_rect,
            "layout_mode": "semantic",
            "layout_parameters": {
                "map_vertical_gap": self.map_vertical_gap,
                "map_matrix_gap": self.map_matrix_gap,
                "matrix_colorbar_gap": self.matrix_colorbar_gap,
                "colorbar_height_ratio": self.colorbar_height_ratio,
                "matrix_scale": self.matrix_scale,
                "show_layout_frames": self.show_layout_frames,
            },
            "layout_gaps": {
                "resolved_map_to_matrix": float(
                    frame_rects["matrix"][0]
                    - max(
                        self.origin_map_rect[0]
                        + self.origin_map_rect[2],
                        self.destination_map_rect[0]
                        + self.destination_map_rect[2],
                    )
                ),
                "map_vertical": float(
                    destination_rect["y"]
                    - origin_rect["y"]
                    - origin_rect["h"]
                ),
                "map_to_matrix": float(
                    matrix_frame_rect["x"] - map_right
                ),
                "matrix_to_colorbar": float(
                    colorbar_axes_rect["x"]
                    - matrix_frame_rect["x"]
                    - matrix_frame_rect["w"]
                ),
                "matrix_top_to_origin_top": float(
                    matrix_frame_rect["y"] - origin_rect["y"]
                ),
                "matrix_bottom_to_destination_bottom": float(
                    destination_rect["y"] + destination_rect["h"]
                    - matrix_frame_rect["y"] - matrix_frame_rect["h"]
                ),
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
            "allow_leader_crossings": self.allow_leader_crossings,
            "allow_reorder": self.allow_reorder,
            "leader_routing": self.leader_routing,
            "leader_angle": self.leader_angle,
            "leader_crossings": {
                "origin": origin_crossings,
                "destination": destination_crossings,
                "total": origin_crossings + destination_crossings,
            },
            "routing_diagnostics": routing_diagnostics,
            "minimum_diagonal_gap": {
                "origin": self._exported_minimum_diagonal_gap(
                    fig, origin_leaders,
                ),
                "destination": self._exported_minimum_diagonal_gap(
                    fig, destination_leaders,
                ),
            },
        }

    @staticmethod
    def _scale_sizes(values, size_range):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        minimum, maximum = size_range
        if np.ptp(values) == 0:
            return np.full(values.shape, (minimum + maximum) / 2.0)
        return _linear_scaling(values, (minimum, maximum))
