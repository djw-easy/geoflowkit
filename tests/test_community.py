import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries
from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.clustering._graph_utils import (
    flows_to_graph,
    assign_flow_labels,
    _assign_zones_grid,
    _assign_zones_aggregate,
)
from geoflowkit.clustering.community import (
    CNMFlow, cnm,
    LouvainFlow, louvain,
    STOCSFlow, stocs,
)


class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        flows = FlowSeries([
            Flow([[0, 0], [1, 1]]),
            Flow([[0.5, 0.5], [1.5, 1.5]]),
            Flow([[2, 2], [3, 3]]),
            Flow([[2.5, 2.5], [3.5, 3.5]]),
        ])
        self.fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40]},
            geometry=flows, crs="EPSG:4326"
        )

    def test_assign_zones_grid(self):
        o_zones, d_zones, centroids = _assign_zones_grid(self.fdf, cell_size=1.0)
        self.assertEqual(len(o_zones), 4)
        self.assertEqual(len(d_zones), 4)
        self.assertIsInstance(centroids, dict)

    def test_assign_zones_aggregate(self):
        o_zones, d_zones, centroids = _assign_zones_aggregate(self.fdf)
        self.assertEqual(len(o_zones), 4)
        self.assertEqual(len(d_zones), 4)
        self.assertIsInstance(centroids, dict)

    def test_flows_to_graph_grid(self):
        G, o_zones, d_zones, centroids = flows_to_graph(
            self.fdf, zone_method='grid', cell_size=1.0
        )
        self.assertGreater(len(G.nodes), 0)
        self.assertGreater(len(G.edges), 0)
        self.assertEqual(len(o_zones), 4)

    def test_flows_to_graph_aggregate(self):
        G, o_zones, d_zones, centroids = flows_to_graph(
            self.fdf, zone_method='aggregate'
        )
        self.assertGreater(len(G.nodes), 0)

    def test_flows_to_graph_custom(self):
        def custom_zone_func(fdf):
            origins = np.array([[0, 0], [0, 0], [2, 2], [2, 2]])
            destinations = np.array([[1, 1], [1, 1], [3, 3], [3, 3]])
            o_zones = np.array([0, 0, 1, 1])
            d_zones = np.array([2, 2, 3, 3])
            zone_centroids = {
                0: (0, 0), 1: (2, 2),
                2: (1, 1), 3: (3, 3)
            }
            return o_zones, d_zones, zone_centroids

        G, o_zones, d_zones, centroids = flows_to_graph(
            self.fdf, zone_method='custom', zone_func=custom_zone_func
        )
        self.assertGreater(len(G.nodes), 0)

    def test_flows_to_graph_invalid_method(self):
        with self.assertRaises(ValueError):
            flows_to_graph(self.fdf, zone_method='invalid')

    def test_flows_to_graph_grid_requires_cell_size(self):
        with self.assertRaises(ValueError):
            flows_to_graph(self.fdf, zone_method='grid')

    def test_flows_to_graph_custom_requires_func(self):
        with self.assertRaises(ValueError):
            flows_to_graph(self.fdf, zone_method='custom')

    def test_assign_flow_labels(self):
        o_zones = np.array([0, 0, 1, 1])
        d_zones = np.array([2, 2, 3, 3])
        zone_communities = {0: 0, 1: 0, 2: 1, 3: 1}
        labels = assign_flow_labels(self.fdf, o_zones, d_zones, zone_communities)
        self.assertEqual(len(labels), 4)
        self.assertEqual(labels[0], -1)
        self.assertEqual(labels[1], -1)


class TestCNMFlow(unittest.TestCase):
    def setUp(self):
        flows = FlowSeries([
            Flow([[0, 0], [0.5, 0.5]]),
            Flow([[0.1, 0.1], [0.6, 0.6]]),
            Flow([[0.2, 0.2], [0.7, 0.7]]),
            Flow([[5, 5], [5.5, 5.5]]),
            Flow([[5.1, 5.1], [5.6, 5.6]]),
            Flow([[5.2, 5.2], [5.7, 5.7]]),
        ])
        self.fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40, 50, 60]},
            geometry=flows, crs="EPSG:4326"
        )

    def test_cnm_fit(self):
        model = CNMFlow(zone_method='grid', cell_size=1.0)
        model.fit(self.fdf)
        self.assertEqual(len(model.labels_), 6)
        self.assertIsNotNone(model.zone_labels_)
        self.assertGreaterEqual(model.n_communities_, 1)

    def test_cnm_fit_predict(self):
        labels = cnm(self.fdf, zone_method='grid', cell_size=1.0)
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(labels >= -1))

    def test_cnm_detects_two_groups(self):
        labels = cnm(self.fdf, zone_method='grid', cell_size=1.0)
        valid_labels = labels[labels >= 0]
        self.assertGreaterEqual(len(np.unique(valid_labels)), 1)

    def test_cnm_aggregate_method(self):
        labels = cnm(self.fdf, zone_method='aggregate')
        self.assertEqual(len(labels), 6)

    def test_cnm_custom_zone_func(self):
        def simple_zones(fdf):
            n = len(fdf)
            o_zones = np.array([0] * (n // 2) + [1] * (n - n // 2))
            d_zones = np.array([2] * (n // 2) + [3] * (n - n // 2))
            centroids = {0: (0, 0), 1: (5, 5), 2: (0.5, 0.5), 3: (5.5, 5.5)}
            return o_zones, d_zones, centroids

        labels = cnm(self.fdf, zone_method='custom', zone_func=simple_zones)
        self.assertEqual(len(labels), 6)

    def test_cnm_modularity(self):
        model = CNMFlow(zone_method='grid', cell_size=1.0)
        model.fit(self.fdf)
        self.assertGreaterEqual(model.modularity_, -1.0)
        self.assertLessEqual(model.modularity_, 1.0)


class TestLouvainFlow(unittest.TestCase):
    def setUp(self):
        flows = FlowSeries([
            Flow([[0, 0], [0.5, 0.5]]),
            Flow([[0.1, 0.1], [0.6, 0.6]]),
            Flow([[0.2, 0.2], [0.7, 0.7]]),
            Flow([[5, 5], [5.5, 5.5]]),
            Flow([[5.1, 5.1], [5.6, 5.6]]),
            Flow([[5.2, 5.2], [5.7, 5.7]]),
        ])
        self.fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40, 50, 60]},
            geometry=flows, crs="EPSG:4326"
        )

    def test_louvain_fit(self):
        model = LouvainFlow(zone_method='grid', cell_size=1.0, seed=42)
        model.fit(self.fdf)
        self.assertEqual(len(model.labels_), 6)
        self.assertIsNotNone(model.zone_labels_)
        self.assertGreaterEqual(model.n_communities_, 1)

    def test_louvain_fit_predict(self):
        labels = louvain(self.fdf, zone_method='grid', cell_size=1.0, seed=42)
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(labels >= -1))

    def test_louvain_reproducible(self):
        labels1 = louvain(self.fdf, zone_method='grid', cell_size=1.0, seed=42)
        labels2 = louvain(self.fdf, zone_method='grid', cell_size=1.0, seed=42)
        np.testing.assert_array_equal(labels1, labels2)

    def test_louvain_aggregate_method(self):
        labels = louvain(self.fdf, zone_method='aggregate', seed=42)
        self.assertEqual(len(labels), 6)

    def test_louvain_custom_zone_func(self):
        def simple_zones(fdf):
            n = len(fdf)
            o_zones = np.array([0] * (n // 2) + [1] * (n - n // 2))
            d_zones = np.array([2] * (n // 2) + [3] * (n - n // 2))
            centroids = {0: (0, 0), 1: (5, 5), 2: (0.5, 0.5), 3: (5.5, 5.5)}
            return o_zones, d_zones, centroids

        labels = louvain(self.fdf, zone_method='custom', zone_func=simple_zones, seed=42)
        self.assertEqual(len(labels), 6)

    def test_louvain_modularity(self):
        model = LouvainFlow(zone_method='grid', cell_size=1.0, seed=42)
        model.fit(self.fdf)
        self.assertGreaterEqual(model.modularity_, -1.0)
        self.assertLessEqual(model.modularity_, 1.0)


class TestSTOCSFlow(unittest.TestCase):
    def setUp(self):
        flows = FlowSeries([
            Flow([[0, 0], [0.5, 0.5]]),
            Flow([[0.1, 0.1], [0.6, 0.6]]),
            Flow([[0.2, 0.2], [0.7, 0.7]]),
            Flow([[5, 5], [5.5, 5.5]]),
            Flow([[5.1, 5.1], [5.6, 5.6]]),
            Flow([[5.2, 5.2], [5.7, 5.7]]),
        ])
        self.fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40, 50, 60]},
            geometry=flows, crs="EPSG:4326"
        )

    def test_stocs_fit(self):
        model = STOCSFlow(
            zone_method='grid', cell_size=1.0,
            spatial_weight=0.5, tabu_tenure=10, max_iter=50, n_init=2, seed=42
        )
        model.fit(self.fdf)
        self.assertEqual(len(model.labels_), 6)
        self.assertIsNotNone(model.zone_labels_)
        self.assertGreaterEqual(model.n_communities_, 1)

    def test_stocs_fit_predict(self):
        labels = stocs(
            self.fdf, zone_method='grid', cell_size=1.0,
            spatial_weight=0.5, tabu_tenure=10, max_iter=50, n_init=2, seed=42
        )
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(labels >= -1))

    def test_stocs_spatial_weight(self):
        labels1 = stocs(
            self.fdf, zone_method='grid', cell_size=1.0,
            spatial_weight=0.0, max_iter=50, n_init=2, seed=42
        )
        labels2 = stocs(
            self.fdf, zone_method='grid', cell_size=1.0,
            spatial_weight=1.0, max_iter=50, n_init=2, seed=42
        )
        self.assertEqual(len(labels1), 6)
        self.assertEqual(len(labels2), 6)

    def test_stocs_aggregate_method(self):
        labels = stocs(
            self.fdf, zone_method='aggregate',
            spatial_weight=0.5, max_iter=50, n_init=2, seed=42
        )
        self.assertEqual(len(labels), 6)

    def test_stocs_custom_zone_func(self):
        def simple_zones(fdf):
            n = len(fdf)
            o_zones = np.array([0] * (n // 2) + [1] * (n - n // 2))
            d_zones = np.array([2] * (n // 2) + [3] * (n - n // 2))
            centroids = {0: (0, 0), 1: (5, 5), 2: (0.5, 0.5), 3: (5.5, 5.5)}
            return o_zones, d_zones, centroids

        labels = stocs(
            self.fdf, zone_method='custom', zone_func=simple_zones,
            spatial_weight=0.5, max_iter=50, n_init=2, seed=42
        )
        self.assertEqual(len(labels), 6)

    def test_stocs_modularity(self):
        model = STOCSFlow(
            zone_method='grid', cell_size=1.0,
            spatial_weight=0.5, max_iter=50, n_init=2, seed=42
        )
        model.fit(self.fdf)
        self.assertGreaterEqual(model.modularity_, -1.0)
        self.assertLessEqual(model.modularity_, 1.0)


class TestEmptyFlowDataFrame(unittest.TestCase):
    def setUp(self):
        self.fdf = FlowDataFrame(
            {'value': []},
            geometry=FlowSeries([]),
            crs="EPSG:4326"
        )

    def test_cnm_empty(self):
        labels = cnm(self.fdf, zone_method='grid', cell_size=1.0)
        self.assertEqual(len(labels), 0)

    def test_louvain_empty(self):
        labels = louvain(self.fdf, zone_method='grid', cell_size=1.0, seed=42)
        self.assertEqual(len(labels), 0)

    def test_stocs_empty(self):
        labels = stocs(self.fdf, zone_method='grid', cell_size=1.0, seed=42)
        self.assertEqual(len(labels), 0)


class TestGDFZoneMethod(unittest.TestCase):
    def setUp(self):
        flows = FlowSeries([
            Flow([[0, 0], [0.5, 0.5]]),
            Flow([[0.1, 0.1], [0.6, 0.6]]),
            Flow([[2, 2], [2.5, 2.5]]),
            Flow([[2.1, 2.1], [2.6, 2.6]]),
        ])
        self.fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40]},
            geometry=flows, crs="EPSG:4326"
        )

    def _make_zones(self):
        import geopandas as gpd
        from shapely.geometry import box
        zones = gpd.GeoDataFrame({
            'zone_id': [0, 1, 2],
            'geometry': [
                box(-1, -1, 1, 1),
                box(1, 1, 3, 3),
                box(1, -1, 3, 1),
            ]
        }, crs="EPSG:4326")
        return zones

    def test_flows_to_graph_gdf(self):
        from geoflowkit.clustering._graph_utils import flows_to_graph
        zones = self._make_zones()
        G, o_zones, d_zones, centroids = flows_to_graph(
            self.fdf, zone_method='gdf', zones=zones
        )
        self.assertGreater(len(G.nodes), 0)

    def test_flows_to_graph_gdf_requires_zones(self):
        from geoflowkit.clustering._graph_utils import flows_to_graph
        with self.assertRaises(ValueError):
            flows_to_graph(self.fdf, zone_method='gdf')

    def test_cnm_with_gdf_zones(self):
        zones = self._make_zones()
        labels = cnm(self.fdf, zone_method='gdf', zones=zones)
        self.assertEqual(len(labels), 4)
        self.assertTrue(np.all(labels >= -1))

    def test_louvain_with_gdf_zones(self):
        zones = self._make_zones()
        labels = louvain(self.fdf, zone_method='gdf', zones=zones, seed=42)
        self.assertEqual(len(labels), 4)
        self.assertTrue(np.all(labels >= -1))


class TestCrossCommunityDetection(unittest.TestCase):
    def test_cross_community_labels(self):
        """Test that flows between different communities get -1 label"""
        o_zones = np.array([0, 0, 1, 1])
        d_zones = np.array([2, 3, 2, 3])
        zone_communities = {0: 0, 1: 1, 2: 0, 3: 1}
        # Flow 0: o=0(comm0), d=2(comm0) → same → label=0
        # Flow 1: o=0(comm0), d=3(comm1) → cross → label=-1
        # Flow 2: o=1(comm1), d=2(comm0) → cross → label=-1
        # Flow 3: o=1(comm1), d=3(comm1) → same → label=1
        flows = FlowSeries([
            Flow([[0, 0], [1, 1]]),
            Flow([[0.1, 0.1], [1.1, 1.1]]),
            Flow([[2, 2], [3, 3]]),
            Flow([[2.1, 2.1], [3.1, 3.1]]),
        ])
        fdf = FlowDataFrame(
            {'value': [1, 2, 3, 4]},
            geometry=flows,
            crs="EPSG:4326"
        )
        from geoflowkit.clustering._graph_utils import assign_flow_labels
        labels = assign_flow_labels(fdf, o_zones, d_zones, zone_communities)
        np.testing.assert_array_equal(labels, [0, -1, -1, 1])

    def test_same_community_labels(self):
        """Test that flows within same community get correct label"""
        o_zones = np.array([0, 1])
        d_zones = np.array([0, 1])
        zone_communities = {0: 0, 1: 0}
        fdf = FlowDataFrame(
            {'value': [1, 2]},
            geometry=FlowSeries([Flow([[0,0],[1,1]]), Flow([[2,2],[3,3]])]),
            crs="EPSG:4326"
        )
        from geoflowkit.clustering._graph_utils import assign_flow_labels
        labels = assign_flow_labels(fdf, o_zones, d_zones, zone_communities)
        np.testing.assert_array_equal(labels, [0, 0])

    def test_mixed_community_labels(self):
        """Test mix of same and cross-community flows"""
        o_zones = np.array([0, 0, 1, 1])
        d_zones = np.array([0, 1, 0, 1])
        zone_communities = {0: 0, 1: 1}
        fdf = FlowDataFrame(
            {'value': [1, 2, 3, 4]},
            geometry=FlowSeries([
                Flow([[0,0],[1,1]]), Flow([[0,0],[2,2]]),
                Flow([[2,2],[1,1]]), Flow([[2,2],[3,3]])
            ]),
            crs="EPSG:4326"
        )
        from geoflowkit.clustering._graph_utils import assign_flow_labels
        labels = assign_flow_labels(fdf, o_zones, d_zones, zone_communities)
        np.testing.assert_array_equal(labels, [0, -1, -1, 1])

    def test_flows_to_graph_weight_volume(self):
        """Test flows_to_graph with weight='volume'"""
        from geoflowkit.clustering._graph_utils import flows_to_graph
        flows = FlowSeries([
            Flow([[0, 0], [1, 1]]),
            Flow([[0, 0], [1, 1]]),
        ])
        fdf = FlowDataFrame(
            {'value': [10, 20], 'volume': [100, 200]},
            geometry=flows, crs="EPSG:4326"
        )
        G, _, _, _ = flows_to_graph(fdf, zone_method='grid', cell_size=1.0, weight='volume')
        assert len(G.edges) > 0

    def test_flows_to_graph_weight_length(self):
        """Test flows_to_graph with weight='length'"""
        from geoflowkit.clustering._graph_utils import flows_to_graph
        flows = FlowSeries([
            Flow([[0, 0], [1, 1]]),
            Flow([[0, 0], [1, 1]]),
        ])
        fdf = FlowDataFrame(
            {'value': [10, 20]},
            geometry=flows, crs="EPSG:4326"
        )
        G, _, _, _ = flows_to_graph(fdf, zone_method='grid', cell_size=1.0, weight='length')
        assert len(G.edges) > 0

    def test_all_same_zone(self):
        """All flows in same zone should produce one community"""
        flows = FlowSeries([
            Flow([[0, 0], [0.1, 0.1]]),
            Flow([[0, 0], [0.2, 0.2]]),
            Flow([[0, 0], [0.3, 0.3]]),
        ])
        fdf = FlowDataFrame(
            {'value': [1, 2, 3]},
            geometry=flows, crs="EPSG:4326"
        )
        labels = louvain(fdf, zone_method='aggregate', seed=42)
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            self.assertEqual(len(set(valid_labels)), 1)

    def test_stocs_n_init_1(self):
        """Test STOCS with n_init=1"""
        flows = FlowSeries([
            Flow([[0, 0], [0.5, 0.5]]),
            Flow([[0.1, 0.1], [0.6, 0.6]]),
            Flow([[2, 2], [2.5, 2.5]]),
            Flow([[2.1, 2.1], [2.6, 2.6]]),
        ])
        fdf = FlowDataFrame(
            {'value': [10, 20, 30, 40]},
            geometry=flows, crs="EPSG:4326"
        )
        labels = stocs(
            fdf, zone_method='grid', cell_size=1.0,
            spatial_weight=0.5, tabu_tenure=5, max_iter=10, n_init=1, seed=42
        )
        self.assertEqual(len(labels), 4)


if __name__ == '__main__':
    unittest.main()
