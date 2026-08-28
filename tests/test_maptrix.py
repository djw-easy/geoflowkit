import inspect

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.sparse import lil_matrix
from shapely.geometry import LineString, box

from geoflowkit import FlowDataFrame, flows_from_od
from geoflowkit.visualization import MapTrixVisualizer
from geoflowkit.visualization._utils import _calculate_rotated_point


def _flow_dataframe(origins, destinations):
    geometry = flows_from_od(
        np.asarray(origins, dtype=float),
        np.asarray(destinations, dtype=float),
        crs="EPSG:3857",
    )
    return FlowDataFrame({"geometry": geometry}, crs="EPSG:3857")


def _same_set_fixture():
    zones = gpd.GeoDataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "geometry": [
                box(0.0, 2.0, 1.0, 3.0),
                box(1.2, 2.0, 2.2, 3.0),
                box(0.0, 0.0, 1.0, 1.0),
                box(1.2, 0.0, 2.2, 1.0),
            ],
        },
        crs="EPSG:3857",
    )
    origins = [
        (0.5, 2.5),
        (1.7, 2.5), (1.7, 2.5),
        (0.5, 0.5), (0.5, 0.5), (0.5, 0.5),
        (1.7, 0.5), (1.7, 0.5), (1.7, 0.5), (1.7, 0.5),
    ]
    destinations = [
        (1.7, 2.5),
        (0.5, 2.5), (0.5, 0.5),
        (0.5, 2.5), (1.7, 2.5), (1.7, 0.5),
        (0.5, 2.5), (1.7, 2.5), (0.5, 0.5), (0.5, 2.5),
    ]
    return zones, _flow_dataframe(origins, destinations)


def _assert_group_has_no_crossings(leaders):
    lines = [
        LineString([(point["x"], point["y"]) for point in leader["path"]])
        for leader in leaders
    ]
    for index, first in enumerate(lines):
        for second in lines[index + 1:]:
            assert not first.crosses(second)
            assert not first.overlaps(second)


def test_same_entity_set_uses_shared_order_and_non_crossing_leaders():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        include_self_flows=False,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_

    assert visualizer.matrix_.shape == (4, 4)
    assert layout["same_entity_set"] is True
    assert layout["row_order"] == layout["column_order"]
    assert set(layout["row_order"]) == {"A", "B", "C", "D"}
    assert len(layout["origin_leaders"]) == 4
    assert len(layout["destination_leaders"]) == 4
    assert np.isnan(np.diag(visualizer.matrix_)).all()

    for leader in layout["origin_leaders"] + layout["destination_leaders"]:
        assert leader["path"][-1] == leader["port"]
        assert np.isclose(leader["bend"]["y"], leader["port"]["y"])
        assert leader["site"]["x"] < leader["bend"]["x"]
        assert leader["bend"]["x"] < leader["port"]["x"]
        diagonal_slope = (
            leader["bend"]["y"] - leader["site"]["y"]
        ) / (
            leader["bend"]["x"] - leader["site"]["x"]
        )
        assert np.isclose(diagonal_slope, leader["slope"])
        assert np.isclose(
            abs(diagonal_slope), np.tan(np.radians(45.0)),
        )
        assert leader["band"] in {"up", "down"}

    assert layout["leader_routing"] == "diagonal-horizontal"
    assert layout["leader_angle"] == 45.0
    assert layout["minimum_diagonal_gap"]["origin"] >= 11.9
    assert layout["minimum_diagonal_gap"]["destination"] >= 11.9
    assert visualizer.axes_["matrix"].patch.get_alpha() == 0.0

    for group in (
        layout["origin_leaders"], layout["destination_leaders"],
    ):
        for band in ("up", "down"):
            slopes = {
                round(leader["slope"], 10)
                for leader in group
                if leader["band"] == band
            }
            assert len(slopes) <= 1

    # A leader starts at the exact final site used by the proportional
    # symbol.  Site separation, when needed, moves both together.
    for group, axis_name, points in [
        (layout["origin_leaders"], "origin", visualizer.o_sites_),
        (layout["destination_leaders"], "destination", visualizer.d_sites_),
    ]:
        ax = visualizer.axes_[axis_name]
        symbol_offsets = np.asarray(ax.collections[-1].get_offsets())
        assert {
            tuple(point) for point in symbol_offsets
        } == {
            tuple(points[zone_id]) for zone_id in layout["row_order"]
        }
        for leader in group:
            display_x, display_y = ax.transData.transform(points[leader["id"]])
            assert np.isclose(leader["site"]["x"], display_x)
            assert np.isclose(
                leader["site"]["y"], fig.bbox.height - display_y,
            )

    origin_widths = {
        leader["id"]: leader["linewidth"]
        for leader in layout["origin_leaders"]
    }
    ordered_by_flow = sorted(visualizer._outflows, key=visualizer._outflows.get)
    assert [
        origin_widths[zone_id] for zone_id in ordered_by_flow
    ] == sorted(origin_widths.values())
    assert np.isclose(min(origin_widths.values()), 0.8)
    assert np.isclose(max(origin_widths.values()), 4.5)

    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)


def test_allow_reorder_false_preserves_input_zone_order():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        allow_reorder=False,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))

    assert visualizer.row_order_ == ["A", "B", "C", "D"]
    assert visualizer.column_order_ == ["A", "B", "C", "D"]
    assert visualizer.layout_["allow_reorder"] is False
    plt.close(fig)


def test_allow_reorder_requires_boolean():
    zones, _ = _same_set_fixture()
    with pytest.raises(TypeError, match="allow_reorder must be a boolean"):
        MapTrixVisualizer(zones, zone_id_col="id", allow_reorder="no")


def test_allow_leader_crossings_allows_infeasible_fixed_order():
    zones, _ = _same_set_fixture()
    base = lil_matrix((4, 2), dtype=float)
    base[0, 0] = base[1, 1] = 1.0
    base[2, 0] = base[3, 1] = 1.0
    variables = [
        {
            "slot": 0,
            "routes": ({
                "zone_id": "A",
                "line": LineString([(0, 0), (1, 1)]),
                "band": "up",
                "diagonal_intercept": 0.0,
                "linewidth_px": 1.0,
            },),
        },
        {
            "slot": 1,
            "routes": ({
                "zone_id": "B",
                "line": LineString([(0, 1), (1, 0)]),
                "band": "down",
                "diagonal_intercept": 0.0,
                "linewidth_px": 1.0,
            },),
        },
    ]

    strict = MapTrixVisualizer(zones, zone_id_col="id")
    with pytest.raises(RuntimeError, match="No crossing-free"):
        strict._solve_spaced_fixed_assignment(
            variables, base, np.zeros(2), ["A", "B"],
        )

    permissive = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        allow_leader_crossings=True,
    )
    assignment = permissive._solve_spaced_fixed_assignment(
        variables, base, np.zeros(2), ["A", "B"],
    )
    assert set(assignment) == {"A", "B"}


def test_allow_leader_crossings_minimises_crossings_before_route_cost():
    zones, _ = _same_set_fixture()
    base = lil_matrix((6, 4), dtype=float)
    base[0, 0] = base[0, 1] = 1.0
    base[1, 2] = base[2, 3] = 1.0
    base[3, 0] = base[3, 1] = 1.0
    base[4, 2] = base[5, 3] = 1.0

    long_route = LineString([(0, -2), (0, 2)])
    short_route = LineString([(0, -0.5), (0, 0.5)])
    route_specs = [
        ("A", long_route, "a"),
        ("A", short_route, "a"),
        ("B", LineString([(-1, 0), (1, 0)]), "b"),
        ("C", LineString([(-1, 1), (1, 1)]), "c"),
    ]
    variables = [
        {
            "slot": 0 if zone_id == "A" else index - 1,
            "routes": ({
                "zone_id": zone_id,
                "line": line,
                "band": band,
                "diagonal_intercept": 0.0,
                "linewidth_px": 1.0,
            },),
        }
        for index, (zone_id, line, band) in enumerate(route_specs)
    ]
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        allow_leader_crossings=True,
    )

    assignment = visualizer._solve_spaced_fixed_assignment(
        variables,
        base,
        np.asarray([0.0, 1.0, 0.0, 0.0]),
        ["A", "B", "C"],
    )

    assert assignment["A"]["routes"][0]["line"].equals(short_route)


def test_allow_leader_crossings_exports_optimised_crossing_count():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        allow_leader_crossings=True,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))

    assert visualizer.layout_["leader_crossings"] == {
        "origin": 0,
        "destination": 0,
        "total": 0,
    }
    assert visualizer.row_order_ == visualizer.column_order_
    plt.close(fig)


def test_allow_leader_crossings_requires_boolean():
    zones, _ = _same_set_fixture()
    with pytest.raises(
        TypeError, match="allow_leader_crossings must be a boolean",
    ):
        MapTrixVisualizer(
            zones,
            zone_id_col="id",
            allow_leader_crossings="yes",
        )


def test_matrix_labels_use_ordered_zone_names_on_leader_free_edges():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        matrix_label_fontsize=11,
        origin_matrix_label_rotation=-30,
        destination_matrix_label_rotation=60,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    labels = visualizer.axes_["matrix"].texts
    row_count = len(visualizer.row_order_)

    assert [label.get_text() for label in labels[:row_count]] == [
        str(zone_id) for zone_id in visualizer.row_order_
    ]
    assert [label.get_text() for label in labels[row_count:]] == [
        str(zone_id) for zone_id in visualizer.column_order_
    ]
    assert {label.get_fontsize() for label in labels} == {11.0}
    assert {label.get_rotation() for label in labels[:row_count]} == {330.0}
    assert {label.get_rotation() for label in labels[row_count:]} == {60.0}

    rows, cols = visualizer.matrix_.shape
    expected_row_points = [
        _calculate_rotated_point(
            visualizer._transform, rows, row, cols - 0.5,
        )
        for row in range(rows)
    ]
    expected_column_points = [
        _calculate_rotated_point(
            visualizer._transform, rows, -0.5, col,
        )
        for col in range(cols)
    ]
    assert np.allclose(
        [label.xy for label in labels[:row_count]], expected_row_points,
    )
    assert np.allclose(
        [label.xy for label in labels[row_count:]], expected_column_points,
    )
    plt.close(fig)


def test_matrix_labels_follow_show_labels_setting():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))

    assert len(visualizer.axes_["matrix"].texts) == 0
    plt.close(fig)


@pytest.mark.parametrize(
    "keyword",
    [
        "origin_matrix_label_rotation",
        "destination_matrix_label_rotation",
    ],
)
def test_matrix_label_rotations_must_be_finite(keyword):
    zones, _ = _same_set_fixture()
    with pytest.raises(ValueError, match=f"{keyword} must be finite"):
        MapTrixVisualizer(
            zones,
            zone_id_col="id",
            **{keyword: np.nan},
        )


def test_previous_horizontal_diagonal_routing_remains_available():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        leader_routing="horizontal-diagonal",
        leader_width_range=None,
        leader_linewidth=2.25,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_

    assert layout["leader_routing"] == "horizontal-diagonal"
    assert layout["minimum_diagonal_gap"] == {
        "origin": None,
        "destination": None,
    }
    for leader in layout["origin_leaders"] + layout["destination_leaders"]:
        assert np.isclose(leader["site"]["y"], leader["bend"]["y"])
        assert np.isclose(leader["linewidth"], 2.25)
        assert leader["band"] is None
        assert leader["slope"] is None
    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)


def test_unreachable_spacing_target_never_relaxes_non_crossing():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        min_leader_gap=1000,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_

    assert layout["minimum_diagonal_gap"]["origin"] < 1000
    assert layout["minimum_diagonal_gap"]["destination"] < 1000
    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)


def test_semantic_layout_defaults_are_public_signature_defaults():
    parameters = inspect.signature(MapTrixVisualizer).parameters

    assert parameters["map_vertical_gap"].default == 0.14
    assert parameters["map_matrix_gap"].default == 0.10
    assert parameters["matrix_colorbar_gap"].default == 0.05
    assert parameters["colorbar_height_ratio"].default == 0.88
    assert parameters["matrix_scale"].default == 1.0
    assert parameters["layout_rect"].default == (0.04, 0.08, 0.894, 0.84)
    for removed in (
        "origin_map_rect",
        "destination_map_rect",
        "matrix_rect",
        "colorbar_rect",
        "height_ratios",
        "width_ratios",
    ):
        assert removed not in parameters


def test_layout_rect_bounds_the_semantic_component_frames():
    zones, flows = _same_set_fixture()
    layout_rect = (0.10, 0.12, 0.76, 0.72)
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        layout_rect=layout_rect,
        leader_routing="horizontal-diagonal",
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    frames = visualizer.layout_["component_frames"].values()
    left, bottom, width, height = layout_rect
    target_left = left * fig.bbox.width
    target_right = (left + width) * fig.bbox.width
    target_top = (1.0 - bottom - height) * fig.bbox.height
    target_bottom = (1.0 - bottom) * fig.bbox.height

    assert min(frame["x"] for frame in frames) >= target_left - 1e-6
    assert max(
        frame["x"] + frame["w"] for frame in frames
    ) <= target_right + 1e-6
    assert min(frame["y"] for frame in frames) >= target_top - 1e-6
    assert max(
        frame["y"] + frame["h"] for frame in frames
    ) <= target_bottom + 1e-6
    assert visualizer.layout_["layout_rect"] == layout_rect
    plt.close(fig)


def test_semantic_layout_resolves_equal_maps_and_frame_based_gaps():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        map_vertical_gap=0.12,
        map_matrix_gap=-0.08,
        matrix_colorbar_gap=0.03,
        colorbar_height_ratio=0.70,
        show_layout_frames=True,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_
    frames = layout["component_frames"]
    matrix_height = frames["matrix"]["h"]

    assert layout["layout_mode"] == "semantic"
    assert np.isclose(
        frames["origin_map"]["w"], frames["destination_map"]["w"],
    )
    assert np.isclose(
        frames["origin_map"]["h"], frames["destination_map"]["h"],
    )
    assert np.isclose(
        layout["layout_gaps"]["map_vertical"],
        0.12 * matrix_height,
    )
    assert np.isclose(
        layout["layout_gaps"]["map_to_matrix"],
        -0.08 * matrix_height,
    )
    assert np.isclose(
        layout["layout_gaps"]["matrix_to_colorbar"],
        0.03 * matrix_height,
    )
    assert np.isclose(
        frames["colorbar"]["h"], 0.70 * matrix_height,
    )
    frame_artists = [
        artist for artist in fig.artists
        if (artist.get_gid() or "").startswith("maptrix-layout-frame-")
    ]
    assert len(frame_artists) == 4
    assert {artist.get_edgecolor() for artist in frame_artists} == {
        (0.7803921568627451, 0.8, 0.8196078431372549, 1.0),
    }
    plt.close(fig)


@pytest.mark.parametrize("matrix_scale", [0.65, 1.0, 1.35])
def test_matrix_scale_is_vertically_symmetric(matrix_scale):
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        matrix_scale=matrix_scale,
        colorbar_height_ratio=0.72,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_
    frames = layout["component_frames"]
    gaps = layout["layout_gaps"]
    map_stack_height = (
        frames["destination_map"]["y"]
        + frames["destination_map"]["h"]
        - frames["origin_map"]["y"]
    )

    assert np.isclose(
        frames["matrix"]["h"] / map_stack_height,
        matrix_scale,
    )
    assert np.isclose(
        gaps["matrix_top_to_origin_top"],
        gaps["matrix_bottom_to_destination_bottom"],
    )
    assert np.isclose(
        frames["colorbar"]["h"],
        0.72 * frames["matrix"]["h"],
    )
    assert layout["layout_parameters"]["matrix_scale"] == matrix_scale
    plt.close(fig)


def test_weight_transform_is_applied_after_count_aggregation():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        weight_transform=np.log1p,
        cbar_label="log(1 + count)",
        allow_reorder=False,
        show_labels=False,
    )
    visualizer.fit(flows)

    assert np.allclose(
        visualizer.matrix_,
        np.log1p(visualizer.raw_matrix_),
        equal_nan=True,
    )
    assert visualizer._outflows == {
        zone_id: float(visualizer.raw_matrix_[row].sum())
        for row, zone_id in enumerate(visualizer.row_order_)
    }

    fig = visualizer.plot(figsize=(12, 8))
    assert visualizer.axes_["colorbar"].get_ylabel() == "log(1 + count)"
    plt.close(fig)


def test_callable_weight_is_count_transform_shorthand():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        weight=lambda matrix: np.sqrt(matrix),
        allow_reorder=False,
        show_labels=False,
    ).fit(flows)

    assert visualizer.weight == "count"
    assert visualizer.cbar_label == "Transformed count"
    assert np.allclose(
        visualizer.matrix_,
        np.sqrt(visualizer.raw_matrix_),
        equal_nan=True,
    )


def test_callable_weight_cannot_be_combined_with_transform():
    zones, _ = _same_set_fixture()
    with pytest.raises(ValueError, match="cannot be combined"):
        MapTrixVisualizer(
            zones,
            zone_id_col="id",
            weight=np.log1p,
            weight_transform=np.sqrt,
        )
    with pytest.raises(TypeError, match="weight_transform must be callable"):
        MapTrixVisualizer(
            zones,
            zone_id_col="id",
            weight_transform="log1p",
        )


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda matrix: matrix[:, 0], "same shape"),
        (lambda matrix: np.full_like(matrix, np.nan), "finite values"),
        (lambda matrix: matrix + 1, "preserve zero values"),
    ],
)
def test_weight_transform_rejects_invalid_results(transform, message):
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        weight_transform=transform,
    )
    with pytest.raises(ValueError, match=message):
        visualizer.fit(flows)


def test_layout_frames_are_hidden_by_default():
    zones, flows = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones, zone_id_col="id", show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))

    assert not [
        artist for artist in fig.artists
        if (artist.get_gid() or "").startswith("maptrix-layout-frame-")
    ]
    assert not visualizer.axes_["colorbar"]._colorbar.outline.get_visible()
    assert visualizer.axes_["colorbar"].get_zorder() == -1
    plt.close(fig)


def test_semantic_layout_rejects_invalid_arguments():
    zones, _ = _same_set_fixture()
    with pytest.raises(
        ValueError, match="map_vertical_gap must be between -1 and 1",
    ):
        MapTrixVisualizer(
            zones, zone_id_col="id", map_vertical_gap=-1.0,
        )
    with pytest.raises(
        TypeError, match="show_layout_frames must be a boolean",
    ):
        MapTrixVisualizer(
            zones, zone_id_col="id", show_layout_frames="yes",
        )
    with pytest.raises(ValueError, match="matrix_scale must be between"):
        MapTrixVisualizer(
            zones, zone_id_col="id", matrix_scale=1.51,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("centroid_size_range", (30, 2000.01), "must not exceed 2000"),
        (
            "centroid_size_range",
            (100, 30),
            "must be finite, positive, and increasing",
        ),
        ("max_matrix_symbol_size", 1600.01, "must not exceed 1600"),
        (
            "leader_width_range",
            (0.8, 12.01),
            "maximum must not exceed 12 points",
        ),
        ("leader_linewidth", 12.01, "must not exceed 12"),
    ],
)
def test_visual_size_limits_reject_unreasonable_values(
    keyword, value, message,
):
    zones, _ = _same_set_fixture()
    with pytest.raises(ValueError, match=message):
        MapTrixVisualizer(
            zones,
            zone_id_col="id",
            **{keyword: value},
        )


def test_visual_size_limits_accept_documented_boundaries():
    zones, _ = _same_set_fixture()
    visualizer = MapTrixVisualizer(
        zones,
        zone_id_col="id",
        centroid_size_range=(30, 2000),
        max_matrix_symbol_size=1600,
        leader_width_range=(0.8, 12),
        leader_linewidth=12,
    )

    assert visualizer.centroid_size_range == (30.0, 2000.0)
    assert visualizer.map_cmap == "viridis"
    assert visualizer.max_matrix_symbol_size == 1600
    assert visualizer.leader_width_range == (0.8, 12.0)
    assert visualizer.leader_linewidth == 12


def test_different_entity_sets_keep_independent_rectangular_axes():
    origin_zones = gpd.GeoDataFrame(
        {
            "id": ["O1", "O2", "O3"],
            "geometry": [
                box(0, 4, 1, 5),
                box(0, 2, 1, 3),
                box(0, 0, 1, 1),
            ],
        },
        crs="EPSG:3857",
    )
    destination_zones = gpd.GeoDataFrame(
        {
            "id": ["D1", "D2"],
            "geometry": [box(3, 3, 4, 4), box(3, 0, 4, 1)],
        },
        crs="EPSG:3857",
    )
    flows = _flow_dataframe(
        [(0.5, 4.5), (0.5, 2.5), (0.5, 0.5), (0.5, 4.5)],
        [(3.5, 3.5), (3.5, 0.5), (3.5, 3.5), (3.5, 0.5)],
    )

    visualizer = MapTrixVisualizer(
        origin_zones,
        dest_zones=destination_zones,
        zone_id_col="id",
        dest_zone_id_col="id",
        leader_angle=35,
        show_labels=False,
    )
    fig = visualizer.fit_plot(flows, figsize=(12, 8))
    layout = visualizer.layout_

    assert visualizer.matrix_.shape == (3, 2)
    assert layout["same_entity_set"] is False
    assert set(layout["row_order"]) == {"O1", "O2", "O3"}
    assert set(layout["column_order"]) == {"D1", "D2"}
    assert len(layout["row_ports"]) == 3
    assert len(layout["column_ports"]) == 2
    assert layout["matrix"]["rows"] == 3
    assert layout["matrix"]["columns"] == 2
    assert layout["leader_angle"] == 35.0
    for leader in layout["origin_leaders"] + layout["destination_leaders"]:
        assert np.isclose(
            abs(leader["slope"]), np.tan(np.radians(35.0)),
        )
    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)
