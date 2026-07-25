import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, box

from geoflowkit import FlowDataFrame, flows_from_od
from geoflowkit.visualization import MapTrixVisualizer


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
    for leader in layout["origin_leaders"] + layout["destination_leaders"]:
        assert np.isclose(leader["site"]["y"], leader["bend"]["y"])
        assert np.isclose(leader["linewidth"], 2.25)
        assert leader["band"] is None
        assert leader["slope"] is None
    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)


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
