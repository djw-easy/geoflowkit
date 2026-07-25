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
        (0.5, 2.5), (0.5, 2.5), (1.7, 2.5), (1.7, 2.5),
        (0.5, 0.5), (0.5, 0.5), (1.7, 0.5), (1.7, 0.5),
    ]
    destinations = [
        (1.7, 2.5), (0.5, 0.5), (0.5, 2.5), (1.7, 0.5),
        (0.5, 2.5), (1.7, 0.5), (1.7, 2.5), (0.5, 0.5),
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
    _assert_group_has_no_crossings(layout["origin_leaders"])
    _assert_group_has_no_crossings(layout["destination_leaders"])
    plt.close(fig)
