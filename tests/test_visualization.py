"""Tests for geoflowkit.visualization — OD matrix and MapTrix modules."""

import unittest
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import box, Point

from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries
from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.visualization._utils import (
    _prepare_zones,
    _assign_zones,
    _linear_scaling,
    _rotate_matrix,
    _calculate_rotated_point,
    _ax_to_fig,
    _plot_centroids,
    _plot_labels,
    _compute_representative_points,
)
from geoflowkit.visualization.od_matrix import ODMatrixVisualizer
from geoflowkit.visualization.maptrix import MapTrixVisualizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_fdf(n_flows=6):
    """Build a small FlowDataFrame for testing."""
    np.random.seed(42)
    origins = np.array([
        [1.2, 1.1], [1.3, 1.15], [1.1, 1.2],
        [6.1, 1.1], [6.2, 1.15], [6.1, 1.2],
    ])[:n_flows]
    destinations = np.array([
        [1.1, 6.1], [1.2, 6.2], [1.15, 6.15],
        [6.1, 6.1], [6.15, 6.2], [6.2, 6.15],
    ])[:n_flows]

    flows = FlowSeries([
        Flow([origins[i], destinations[i]]) for i in range(n_flows)
    ])
    fdf = FlowDataFrame(
        {'value': np.arange(1, n_flows + 1) * 10},
        geometry=flows, crs="EPSG:4326",
    )
    return fdf


def _make_zone_gdf():
    """Build a small GeoDataFrame of 4 square zones."""
    polygons = [
        box(0, 0, 5, 5), box(5, 0, 10, 5),
        box(0, 5, 5, 10), box(5, 5, 10, 10),
    ]
    return gpd.GeoDataFrame(
        {'zone_id': [0, 1, 2, 3]},
        geometry=polygons, crs="EPSG:4326",
    )


def _make_named_zone_gdf():
    """Build a GeoDataFrame with string zone IDs."""
    polygons = [
        box(0, 0, 5, 5), box(5, 0, 10, 5),
        box(0, 5, 5, 10), box(5, 5, 10, 10),
    ]
    return gpd.GeoDataFrame(
        {'name': ['A', 'B', 'C', 'D']},
        geometry=polygons, crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Test _utils
# ---------------------------------------------------------------------------

class TestPrepareZones(unittest.TestCase):
    def test_default_column(self):
        zones = _make_zone_gdf()
        result = _prepare_zones(zones)
        self.assertIn('zone_id', result.columns)

    def test_custom_column(self):
        zones = _make_named_zone_gdf()
        result = _prepare_zones(zones, zone_id_col='name')
        self.assertIn('zone_id', result.columns)
        self.assertEqual(list(result['zone_id']), ['A', 'B', 'C', 'D'])

    def test_none_uses_index(self):
        zones = _make_named_zone_gdf()
        zones.index = [10, 20, 30, 40]
        result = _prepare_zones(zones, zone_id_col=None)
        self.assertEqual(list(result['zone_id']), [10, 20, 30, 40])

    def test_missing_column_raises(self):
        zones = _make_named_zone_gdf()
        with self.assertRaises(KeyError):
            _prepare_zones(zones, zone_id_col='bogus')


class TestLinearScaling(unittest.TestCase):
    def test_basic_scale(self):
        arr = np.array([0, 5, 10])
        scaled = _linear_scaling(arr, out_range=(0, 1))
        np.testing.assert_almost_equal(scaled[0], 0.0)
        np.testing.assert_almost_equal(scaled[1], 0.5)
        np.testing.assert_almost_equal(scaled[2], 1.0)

    def test_constant_input(self):
        arr = np.array([3.0, 3.0, 3.0])
        scaled = _linear_scaling(arr, out_range=(0, 1))
        np.testing.assert_almost_equal(scaled, 0.5)

    def test_custom_range(self):
        arr = np.array([0, 2])
        scaled = _linear_scaling(arr, out_range=(0.5, 2.0))
        self.assertAlmostEqual(scaled[0], 0.5)
        self.assertAlmostEqual(scaled[1], 2.0)


class TestAssignZones(unittest.TestCase):
    def setUp(self):
        self.fdf = _make_synthetic_fdf(6)
        self.zones = _make_zone_gdf()

    def test_gdf_method(self):
        o_z, d_z, o_cent, d_cent = _assign_zones(self.fdf, self.zones)
        self.assertEqual(len(o_z), 6)
        self.assertEqual(len(d_z), 6)
        self.assertIsInstance(o_cent, dict)
        self.assertEqual(len(o_cent), 4)
        self.assertEqual(len(d_cent), 4)

    def test_custom_zone_id_col(self):
        zones = _make_named_zone_gdf()
        o_z, d_z, o_cent, d_cent = _assign_zones(
            self.fdf, zones, zone_id_col='name',
        )
        self.assertEqual(len(o_z), 6)
        self.assertEqual(len(o_cent), 4)
        self.assertEqual(len(d_cent), 4)


class TestRotateMatrix(unittest.TestCase):
    def test_returns_im_and_transform(self):
        fig, ax = plt.subplots()
        mat = np.array([[1, 2], [3, 4]])
        im, transform = _rotate_matrix(ax, mat)
        self.assertIsNotNone(im)
        self.assertIsNotNone(transform)
        plt.close(fig)

    def test_calculate_rotated_point(self):
        fig, ax = plt.subplots()
        mat = np.array([[1, 2], [3, 4]])
        _, transform = _rotate_matrix(ax, mat)
        x, y = _calculate_rotated_point(transform, 2, 0, 0)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        plt.close(fig)


class TestAxToFig(unittest.TestCase):
    def test_conversion(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        fig_x, fig_y = _ax_to_fig(ax, fig, 5, 5)
        self.assertGreaterEqual(fig_x, 0)
        self.assertLessEqual(fig_x, 1)
        self.assertGreaterEqual(fig_y, 0)
        self.assertLessEqual(fig_y, 1)
        plt.close(fig)


class TestPlotCentroidsAndLabels(unittest.TestCase):
    def test_plot_centroids(self):
        fig, ax = plt.subplots()
        centroids = {0: (0, 0), 1: (5, 5)}
        sizes = np.array([100, 200])
        path = _plot_centroids(ax, centroids, sizes=sizes)
        self.assertIsNotNone(path)
        plt.close(fig)

    def test_plot_labels(self):
        fig, ax = plt.subplots()
        centroids = {0: (0, 0), 1: (5, 5)}
        labels = {0: 'A', 1: 'B'}
        texts = _plot_labels(ax, centroids, labels)
        self.assertEqual(len(texts), 2)
        plt.close(fig)


class TestRepresentativePoints(unittest.TestCase):
    def test_returns_dict(self):
        zones = _make_zone_gdf()
        rep = _compute_representative_points(zones)
        self.assertIsInstance(rep, dict)
        self.assertEqual(len(rep), 4)
        for zid, (x, y) in rep.items():
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_custom_zone_id_col(self):
        zones = _make_named_zone_gdf()
        rep = _compute_representative_points(zones, zone_id_col='name')
        self.assertEqual(set(rep.keys()), {'A', 'B', 'C', 'D'})

    def test_none_uses_index(self):
        zones = _make_named_zone_gdf()
        zones.index = [100, 200, 300, 400]
        rep = _compute_representative_points(zones, zone_id_col=None)
        self.assertEqual(set(rep.keys()), {100, 200, 300, 400})

    def test_point_inside_polygon(self):
        """representative_point should always be inside its polygon."""
        from shapely.geometry import Polygon
        # A concave L-shaped polygon
        l_shape = Polygon([
            (0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10),
        ])
        zones = gpd.GeoDataFrame(
            {'zone_id': [0]}, geometry=[l_shape], crs="EPSG:4326",
        )
        rep = _compute_representative_points(zones)
        x, y = rep[0]
        self.assertTrue(l_shape.contains(Point(x, y)))


# ---------------------------------------------------------------------------
# Test ODMatrixVisualizer
# ---------------------------------------------------------------------------

class TestODMatrixVisualizer(unittest.TestCase):
    def setUp(self):
        self.fdf = _make_synthetic_fdf(6)
        self.zones = _make_zone_gdf()

    def test_fit_sets_attributes(self):
        vis = ODMatrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        self.assertIsNotNone(vis.matrix_)
        self.assertIsNotNone(vis.o_ids_)
        self.assertIsNotNone(vis.d_ids_)
        self.assertIsNotNone(vis.o_centroids_)
        self.assertIsNotNone(vis.d_centroids_)
        self.assertIsNotNone(vis.o_zones_)
        self.assertIsNotNone(vis.d_zones_)
        self.assertGreater(len(vis.o_ids_), 0)
        self.assertGreater(len(vis.d_ids_), 0)
        self.assertEqual(vis.matrix_.shape[0], len(vis.o_ids_))
        self.assertEqual(vis.matrix_.shape[1], len(vis.d_ids_))

    def test_plot_returns_ax(self):
        vis = ODMatrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        ax = vis.plot()
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_plot_before_fit_raises(self):
        vis = ODMatrixVisualizer(origin_zones=self.zones)
        with self.assertRaises(RuntimeError):
            vis.plot()

    def test_fit_plot_returns_ax(self):
        ax = ODMatrixVisualizer(origin_zones=self.zones).fit_plot(self.fdf)
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_with_figsize(self):
        ax = ODMatrixVisualizer(origin_zones=self.zones).fit_plot(self.fdf, figsize=(8, 6))
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_no_colorbar(self):
        vis = ODMatrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        ax = vis.plot(colorbar=False)
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_external_ax(self):
        fig, ax = plt.subplots()
        result = ODMatrixVisualizer(origin_zones=self.zones).fit_plot(self.fdf, ax=ax)
        self.assertIs(result, ax)
        plt.close(fig)

    def test_custom_zone_id_col(self):
        zones = _make_named_zone_gdf()
        ax = ODMatrixVisualizer(origin_zones=zones, zone_id_col='name').fit_plot(self.fdf)
        self.assertIsNotNone(ax)
        plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Test MapTrixVisualizer
# ---------------------------------------------------------------------------

class TestMapTrixVisualizer(unittest.TestCase):
    def setUp(self):
        self.fdf = _make_synthetic_fdf(6)
        self.zones = _make_zone_gdf()

    def test_fit_sets_attributes(self):
        vis = MapTrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        self.assertIsNotNone(vis.matrix_)
        self.assertIsNotNone(vis.zone_ids_)
        self.assertIsNotNone(vis.o_centroids_)
        self.assertIsNotNone(vis.d_centroids_)
        self.assertIsNotNone(vis._outflows)
        self.assertIsNotNone(vis._inflows)
        self.assertIsNotNone(vis.o_order_)
        self.assertIsNotNone(vis.d_order_)
        # o_order_ / d_order_ only include zones with nonzero flows
        self.assertGreater(len(vis.o_order_), 0)
        self.assertGreater(len(vis.d_order_), 0)
        self.assertTrue(set(vis.o_order_).issubset(vis.zone_ids_))
        self.assertTrue(set(vis.d_order_).issubset(vis.zone_ids_))
        for z in vis.o_order_:
            self.assertGreater(vis._outflows[z], 0)
        for z in vis.d_order_:
            self.assertGreater(vis._inflows[z], 0)

    def test_outflows_inflows_sum(self):
        vis = MapTrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        total_out = sum(vis._outflows.values())
        total_in = sum(vis._inflows.values())
        self.assertEqual(total_out, total_in)

    def test_plot_returns_fig(self):
        vis = MapTrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        fig = vis.plot(figsize=(10, 6))
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_before_fit_raises(self):
        vis = MapTrixVisualizer(origin_zones=self.zones)
        with self.assertRaises(RuntimeError):
            vis.plot()

    def test_fit_plot_returns_fig(self):
        fig = MapTrixVisualizer(origin_zones=self.zones).fit_plot(self.fdf, figsize=(10, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_titles(self):
        fig = MapTrixVisualizer(
            origin_zones=self.zones, out_title='SOURCE', in_title='SINK',
        ).fit_plot(self.fdf, figsize=(10, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_labels(self):
        fig = MapTrixVisualizer(
            origin_zones=self.zones, show_labels=False,
        ).fit_plot(self.fdf, figsize=(10, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_subplot_structure(self):
        vis = MapTrixVisualizer(origin_zones=self.zones)
        vis.fit(self.fdf)
        fig = vis.plot(figsize=(10, 6))
        self.assertGreaterEqual(len(fig.axes), 3)
        plt.close(fig)

    def test_external_figure(self):
        fig = plt.figure(figsize=(12, 6))
        result = MapTrixVisualizer(origin_zones=self.zones).fit_plot(self.fdf, fig=fig)
        self.assertIs(result, fig)
        plt.close(fig)

    def test_custom_zone_id_col(self):
        zones = _make_named_zone_gdf()
        fig = MapTrixVisualizer(origin_zones=zones, zone_id_col='name').fit_plot(
            self.fdf, figsize=(10, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_maptrix_with_figsize(self):
        fig = MapTrixVisualizer(origin_zones=self.zones).fit_plot(
            self.fdf, figsize=(10, 6),
        )
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.zones = _make_zone_gdf()

    def test_empty_flowdataframe_od(self):
        fdf = FlowDataFrame(
            {'a': []}, geometry=FlowSeries([]), crs="EPSG:4326",
        )
        vis = ODMatrixVisualizer(origin_zones=self.zones)
        vis.fit(fdf)
        self.assertEqual(vis.matrix_.shape, (4, 4))  # all zones, even with no flows
        ax = vis.plot()
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_empty_flowdataframe_maptrix(self):
        fdf = FlowDataFrame(
            {'a': []}, geometry=FlowSeries([]), crs="EPSG:4326",
        )
        vis = MapTrixVisualizer(origin_zones=self.zones)
        vis.fit(fdf)
        self.assertEqual(vis.matrix_.shape, (4, 4))  # all zones, even with no flows

    def test_single_flow_od(self):
        flows = FlowSeries([Flow([[1, 1], [6, 6]])])
        fdf = FlowDataFrame({'v': [1]}, geometry=flows, crs="EPSG:4326")
        ax = ODMatrixVisualizer(origin_zones=self.zones).fit_plot(fdf)
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_single_flow_maptrix(self):
        flows = FlowSeries([Flow([[1, 1], [6, 6]])])
        fdf = FlowDataFrame({'v': [1]}, geometry=flows, crs="EPSG:4326")
        fig = MapTrixVisualizer(origin_zones=self.zones).fit_plot(fdf, figsize=(10, 6))
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Asymmetric zones
# ---------------------------------------------------------------------------

class TestAsymmetricZones(unittest.TestCase):
    def setUp(self):
        self.fdf = _make_synthetic_fdf(6)
        self.origin_zones = _make_zone_gdf()           # 4 zones: [0,1,2,3]

        # dest_zones cover the synthetic flow destinations:
        #   flows 0-2 → d ≈ (1.2, 6.2)   (zone 'a')
        #   flows 3-5 → d ≈ (6.2, 6.2)   (zone 'b')
        #   zone 'c' is extra (no inflow)
        dest_polys = [
            box(0, 5, 5, 10),    # 'a'
            box(5, 5, 10, 10),   # 'b'
            box(0, 0, 2, 2),     # 'c' — no flows
        ]
        self.dest_zones = gpd.GeoDataFrame(
            {'zone_id': ['a', 'b', 'c']},
            geometry=dest_polys, crs="EPSG:4326",
        )

    # --- ODMatrixVisualizer ---

    def test_od_asymmetric_fit(self):
        vis = ODMatrixVisualizer(
            origin_zones=self.origin_zones, dest_zones=self.dest_zones,
            dest_zone_id_col='zone_id',
        )
        vis.fit(self.fdf)
        self.assertTrue(vis._asymmetric)
        # o_ids=[0,1] (nonzero outflow), d_ids=['a','b'] (nonzero inflow)
        self.assertEqual(len(vis.o_ids_), 2)
        self.assertEqual(len(vis.d_ids_), 2)
        self.assertIn(0, vis.o_ids_)
        self.assertIn('a', vis.d_ids_)

    def test_od_asymmetric_fit_plot(self):
        vis = ODMatrixVisualizer(
            origin_zones=self.origin_zones, dest_zones=self.dest_zones,
            dest_zone_id_col='zone_id',
        )
        vis.fit(self.fdf)
        ax = vis.plot(figsize=(8, 6))
        self.assertIsNotNone(ax)
        plt.close(ax.figure)

    def test_od_asymmetric_origin_zones_only(self):
        vis = ODMatrixVisualizer(origin_zones=self.origin_zones)
        vis.fit(self.fdf)
        self.assertFalse(vis._asymmetric)
        self.assertEqual(len(vis.o_ids_), 2)   # zones 0,1
        self.assertEqual(len(vis.d_ids_), 2)   # zones 2,3

    def test_od_asymmetric_init_raises_without_args(self):
        with self.assertRaises(TypeError):
            ODMatrixVisualizer()

    # --- MapTrixVisualizer ---

    def test_maptrix_asymmetric_fit(self):
        vis = MapTrixVisualizer(
            origin_zones=self.origin_zones, dest_zones=self.dest_zones,
            dest_zone_id_col='zone_id',
        )
        vis.fit(self.fdf)
        self.assertTrue(vis._asymmetric)
        self.assertIsNotNone(vis.matrix_)
        self.assertIsNotNone(vis.o_order_)
        self.assertIsNotNone(vis.d_order_)
        self.assertEqual(len(vis.o_order_), vis.matrix_.shape[1])
        self.assertEqual(len(vis.d_order_), vis.matrix_.shape[0])

    def test_maptrix_asymmetric_plot(self):
        vis = MapTrixVisualizer(
            origin_zones=self.origin_zones, dest_zones=self.dest_zones,
            dest_zone_id_col='zone_id',
        )
        vis.fit(self.fdf)
        fig = vis.plot(figsize=(12, 8))
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_maptrix_asymmetric_origin_zones_only(self):
        vis = MapTrixVisualizer(origin_zones=self.origin_zones)
        vis.fit(self.fdf)
        self.assertFalse(vis._asymmetric)
        self.assertEqual(len(vis.o_order_), 2)
        self.assertEqual(len(vis.d_order_), 2)


if __name__ == '__main__':
    unittest.main()
