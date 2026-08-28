"""Internal utility functions for flow visualization.

Provides zone assignment, OD matrix building, coordinate transforms,
and rendering helpers shared by od_matrix and maptrix modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

import geopandas as gpd

from geoflowkit.clustering._graph_utils import _assign_zones_gdf


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------

def _prepare_zones(zones, zone_id_col=None):
    """Ensure *zones* has a ``'zone_id'`` column expected by the backend.

    Parameters
    ----------
    zones : GeoDataFrame
        Polygon geometries defining spatial zones.  Each row is a zone.
    zone_id_col : str, optional
        Column to use as the zone identifier.  When ``None`` (default)
        the GeoDataFrame index is used.

    Returns
    -------
    zones : GeoDataFrame
        A copy with a ``'zone_id'`` column set.
    """
    zones = zones.copy()
    if zone_id_col is None:
        zones['zone_id'] = zones.index
    elif zone_id_col in zones.columns:
        zones['zone_id'] = zones[zone_id_col]
    elif zone_id_col == zones.index.name:
        zones['zone_id'] = zones.index
    else:
        raise KeyError(
            f"zone_id_col={zone_id_col!r} not found in zones columns "
            f"{list(zones.columns)} or as the index name."
        )
    return zones


def _assign_zones(fdf, zones, zone_id_col=None,
                  dest_zones=None, dest_zone_id_col=None):
    """Assign each flow to an origin and destination zone.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    zones : GeoDataFrame
        Zone polygons for origin assignment.
    zone_id_col : str, optional
        Column to use as the zone identifier.  ``None`` uses the index.
    dest_zones : GeoDataFrame, optional
        Zone polygons for destination assignment.  ``None`` (default)
        reuses *zones* (symmetric case).
    dest_zone_id_col : str, optional
        Column in *dest_zones* for zone identifier.  ``None`` uses
        *zone_id_col*.

    Returns
    -------
    o_zones : np.ndarray
        Origin zone ID for each flow.
    d_zones : np.ndarray
        Destination zone ID for each flow.
    o_centroids : dict
        Mapping from origin zone ID to ``(x, y)`` centroid.
    d_centroids : dict
        Mapping from destination zone ID to ``(x, y)`` centroid.
    """
    zones_prepared = _prepare_zones(zones, zone_id_col=zone_id_col)

    o_centroids = {}
    for _, row in zones_prepared.iterrows():
        c = row['geometry'].centroid
        o_centroids[row['zone_id']] = (c.x, c.y)

    if dest_zones is not None:
        d_zone_id_col = dest_zone_id_col if dest_zone_id_col is not None else zone_id_col
        dest_prepared = _prepare_zones(dest_zones, zone_id_col=d_zone_id_col)
        o_zones, d_zones, _ = _assign_zones_gdf(fdf, zones_prepared, dest_zones=dest_prepared)
        d_centroids = {}
        for _, row in dest_prepared.iterrows():
            c = row['geometry'].centroid
            d_centroids[row['zone_id']] = (c.x, c.y)
    else:
        o_zones, d_zones, _ = _assign_zones_gdf(fdf, zones_prepared)
        d_centroids = o_centroids

    return o_zones, d_zones, o_centroids, d_centroids


def _compute_representative_points(zones, zone_id_col=None):
    """Compute representative points for each zone polygon.

    Uses :meth:`geopandas.GeoSeries.representative_point` which guarantees
    the point lies inside the polygon (unlike ``.centroid`` which may fall
    outside for concave shapes).

    Parameters
    ----------
    zones : GeoDataFrame
        Polygon geometries defining spatial zones.
    zone_id_col : str, optional
        Column to use as the zone identifier.  ``None`` uses the index.

    Returns
    -------
    rep_points : dict
        Mapping from zone ID to ``(x, y)`` representative point.
    """
    zones = _prepare_zones(zones, zone_id_col=zone_id_col)
    rep_points = {}
    for _, row in zones.iterrows():
        zid = row['zone_id']
        rp = row['geometry'].representative_point()
        rep_points[zid] = (rp.x, rp.y)
    return rep_points


# ---------------------------------------------------------------------------
# OD matrix construction
# ---------------------------------------------------------------------------

def _get_weights(fdf, weight='count'):
    """Compute per-flow weight values based on the chosen metric.

    Parameters
    ----------
    fdf : FlowDataFrame
        Input flow data.
    weight : str, default='count'
        Aggregation weight:
        - ``'count'`` — unit weight (each flow counts as 1)
        - ``'length'`` — flow length (Euclidean distance)
        - ``'divergence'`` — flow angle in radians
        - any numeric column name — use that column directly;
        - ``'volume'`` retains the historical unit-weight fallback when the
          column does not exist.

    Returns
    -------
    w_values : np.ndarray
        Per-flow weight array, same length as *fdf*.
    """
    if weight == 'length':
        return fdf.length.values
    if weight == 'divergence':
        return fdf.angle.values
    if weight in fdf.columns:
        values = np.asarray(fdf[weight], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Weight column {weight!r} must contain finite values")
        return values
    return np.ones(len(fdf))


# ---------------------------------------------------------------------------
# Value scaling
# ---------------------------------------------------------------------------

def _linear_scaling(arr, out_range=(0.5, 2.0)):
    """Linearly scale array values to a target range.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    out_range : tuple of (float, float), default=(0.5, 2.0)
        ``(low, high)`` target range.

    Returns
    -------
    scaled : np.ndarray
        Scaled array with same shape as input.
    """
    a = np.min(arr)
    b = np.max(arr)
    out_low, out_high = out_range
    if a == b:
        return np.full_like(arr, (out_low + out_high) / 2.0, dtype=float)
    scaled = out_low + (arr - a) * (out_high - out_low) / (b - a)
    return scaled


# ---------------------------------------------------------------------------
# Rotated matrix rendering (used by MapTrix)
# ---------------------------------------------------------------------------

def _rotate_matrix(ax, matrix, cmap='OrRd', vmin=None, vmax=None):
    """Draw a matrix rotated 45 degrees clockwise around its centre.

    Matplotlib uses an upward-pointing y axis, so a ``-45`` degree data
    transform is the equivalent of the clockwise screen-space rotation
    used by the static MapTrix layout.  Rotating around the matrix centre
    keeps rectangular and asymmetric matrices centred in their axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    matrix : np.ndarray
        2-D array to display (shape ``(n_rows, n_cols)``).
    cmap : str or Colormap, default='OrRd'
        Colormap for the heatmap.
    vmin, vmax : float, optional
        Colormap range.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The ``imshow`` artist.
    transform : matplotlib.transforms.Affine2D
        The affine transform used for the rotation (may be reused to map
        cell coordinates to rotated display coordinates).
    """
    rows, cols = matrix.shape
    center_x, center_y = cols / 2.0, rows / 2.0

    transform = (
        Affine2D()
        .translate(-center_x, -center_y)
        .rotate_deg(-45)
        .translate(center_x, center_y)
    )

    im = ax.imshow(
        matrix,
        cmap=cmap,
        extent=[0, cols, 0, rows],
        interpolation='nearest',
        origin='upper',
        transform=transform + ax.transData,
        vmin=vmin, vmax=vmax,
    )

    # Compute rotated bounding box and set axis limits
    theta = np.radians(-45)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    corners = np.array([
        [0, 0], [cols, 0], [cols, rows], [0, rows],
    ])
    rotated_corners = (
        (corners - [center_x, center_y]) @ rot_mat.T + [center_x, center_y]
    )

    pad = max(rows, cols) * 0.04
    ax.set_xlim(
        rotated_corners[:, 0].min() - pad,
        rotated_corners[:, 0].max() + pad,
    )
    ax.set_ylim(
        rotated_corners[:, 1].min() - pad,
        rotated_corners[:, 1].max() + pad,
    )

    # Draw grid lines through the same transform
    for j in range(cols + 1):
        start = transform.transform_point((j, 0))
        end = transform.transform_point((j, rows))
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color='k', linestyle='-', linewidth=0.5, alpha=0.8,
            zorder=2,
        )

    for i in range(rows + 1):
        start = transform.transform_point((0, i))
        end = transform.transform_point((cols, i))
        ax.plot(
            [start[0], end[0]], [start[1], end[1]],
            color='k', linestyle='-', linewidth=0.5, alpha=0.8,
            zorder=2,
        )

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_aspect('equal', adjustable='box')

    return im, transform


def _calculate_rotated_point(transform, n_row, i, j):
    """Map matrix cell ``(i, j)`` to rotated display coordinates.

    The coordinate system matches :func:`_rotate_matrix`'s ``imshow``
    convention: ``origin='upper'`` with ``extent=[0, cols, 0, rows]``.
    In this system the data y of pixel ``(i, j)`` is
    ``y = n_row - i - 0.5`` (row 0 → top, row n-1 → bottom).

    The flip ``n_row - i - 1`` is required because the extent
    ``[0, cols, 0, rows]`` places y=0 at the bottom and y=rows at the
    top, while ``origin='upper'`` puts array row 0 at the top of the
    image.

    Parameters
    ----------
    transform : matplotlib.transforms.Transform
        The affine transform produced by :func:`_rotate_matrix`.
    n_row : int
        Total number of rows in the matrix.
    i : float or int
        Row index (may be fractional for edge positions).
        ``i=-0.5`` selects the top edge, ``i=n_row-0.5`` the bottom edge.
    j : float or int
        Column index (may be fractional for edge positions).
        ``j=-0.5`` selects the left edge, ``j=cols-0.5`` the right edge.

    Returns
    -------
    x : float
    y : float
        Transformed coordinates in data space.
    """
    # With origin='upper' + extent=[0, cols, 0, rows]:
    #   data y = rows - (i + 0.5) = rows - i - 0.5
    # The flip i → n_row-i-1 maps row 0 (top) → data y ≈ n_row-0.5
    # and row n_row-1 (bottom) → data y ≈ 0.5.
    i = n_row - i - 1
    x = j + 0.5
    y = i + 0.5
    point = transform.transform_point((x, y))
    return point[0], point[1]


def _calculate_matrix_anchor_point(transform, n_rows, n_cols, side, index):
    """Calculate matrix anchor point on rotated matrix edge.

    Parameters
    ----------
    transform : matplotlib.transforms.Transform
        The affine transform returned by ``_rotate_matrix``.
    n_rows : int
        Number of matrix rows.
    n_cols : int
        Number of matrix columns.
    side : {'top', 'bottom', 'left'}
        Matrix edge used by MapTrix guide lines.

        - ``'top'``: anchors on the upper edge of the unrotated matrix
          (upper-right diamond edge with CW rotation).
        - ``'bottom'``: anchors on the lower edge of the unrotated matrix
          (lower-left diamond edge with CW rotation), used for destination.
        - ``'left'``: anchors on the left edge of the unrotated matrix
          (upper-left diamond edge with CW rotation), used for origin.

    index : int
        Column index for ``side='top'`` or row index for ``side='left'``.

    Returns
    -------
    x, y : float
        Rotated matrix anchor point in matrix axes data coordinates.

    Notes
    -----
    The unrotated matrix is drawn by ``imshow`` with::

        extent=[0, n_cols, 0, n_rows]
        origin='upper'

    Therefore:

    - top edge is ``y = n_rows``;
    - left edge is ``x = 0``;
    - column center on top edge is ``x = index + 0.5``;
    - row center on left edge is ``y = n_rows - index - 0.5``.

    These points lie exactly on the same grid lines drawn by
    ``_rotate_matrix``.
    """
    if side == "top":
        x = index + 0.5
        y = n_rows

    elif side == "bottom":
        x = index + 0.5
        y = 0.0

    elif side == "left":
        x = 0.0
        y = n_rows - index - 0.5

    else:
        raise ValueError("side must be 'top', 'bottom', or 'left'")

    point = transform.transform_point((x, y))
    return point[0], point[1]


# ---------------------------------------------------------------------------
# Cross-axes coordinate conversion
# ---------------------------------------------------------------------------

def _ax_to_fig(ax, fig, x, y):
    """Convert data coordinates on *ax* to normalised figure coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The source axes.
    fig : matplotlib.figure.Figure
        The target figure (must contain *ax*).
    x, y : float
        Data coordinates on *ax*.

    Returns
    -------
    fig_x : float
    fig_y : float
        Coordinates in figure space (0–1 normalised).
    """
    display_coords = ax.transData.transform((x, y))
    fig_coords = fig.transFigure.inverted().transform(display_coords)
    return fig_coords


# ---------------------------------------------------------------------------
# Proportional circles overlays
# ---------------------------------------------------------------------------


def _draw_size_overlay(ax, xs, ys, sizes):
    """Draw proportional size circles at given display positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xs, ys : array-like
        Coordinates for each circle.
    sizes : array-like
        Raw size values (linearly scaled to 20-800 for display).
    """
    if len(sizes) == 0:
        return
    s_arr = np.asarray(sizes)
    if np.ptp(s_arr) > 0:
        scaled = _linear_scaling(s_arr, (20, 800))
    else:
        scaled = np.full_like(s_arr, 200.0)
    ax.scatter(xs, ys, s=scaled, facecolors='none',
               edgecolors='gray', linewidths=0.5, alpha=0.7, zorder=5)


# ---------------------------------------------------------------------------
# Proportional circles on maps
# ---------------------------------------------------------------------------

def _plot_centroids(ax, centroids, sizes=None, colors='k',
                    cmap=None, vmin=None, vmax=None,
                    alpha=0.8, zorder=2):
    """Draw proportional circles at zone centroid positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    centroids : dict
        Mapping from zone ID to ``(x, y)`` centroid coordinates.
    sizes : array-like, optional
        Marker sizes (one per centroid).
    colors : array-like or str, default='k'
        Marker colour(s).
    cmap : str or Colormap, optional
        Colormap when *colors* is numeric.
    vmin, vmax : float, optional
        Colormap range.
    alpha : float, default=0.8
        Marker transparency.
    zorder : int, default=2
        Drawing z-order.

    Returns
    -------
    path : matplotlib.collections.PathCollection
        The scatter artist.
    """
    xs = np.array([c[0] for c in centroids.values()])
    ys = np.array([c[1] for c in centroids.values()])

    path = ax.scatter(
        xs, ys,
        c=colors, cmap=cmap, s=sizes,
        alpha=alpha, zorder=zorder,
        edgecolor='none',
        vmin=vmin, vmax=vmax,
    )
    return path


# ---------------------------------------------------------------------------
# Zone labels
# ---------------------------------------------------------------------------

def _plot_labels(ax, centroids, labels, fontsize=10, color='black'):
    """Draw text labels offset above zone centroid positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    centroids : dict
        Mapping from zone ID to ``(x, y)`` centroid.
    labels : dict
        Mapping from zone ID to label string.
    fontsize : int, default=10
        Label font size.
    color : str, default='black'
        Label colour.

    Returns
    -------
    texts : list of matplotlib.text.Text
        The text artists.
    """
    texts = []
    for zid, (x, y) in centroids.items():
        label = labels.get(zid, str(zid))
        t = ax.annotate(
            str(label), (x, y),
            xytext=(0, 3), textcoords='offset points',
            ha='center', va='bottom',
            fontsize=fontsize, color=color,
        )
        texts.append(t)
    return texts
