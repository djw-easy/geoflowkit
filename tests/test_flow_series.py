import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries


class TestFlowSeries(unittest.TestCase):
    def setUp(self):
        # Create sample Flow objects
        self.flow1 = Flow([[0, 0], [1, 1]])
        self.flow2 = Flow([[1, 1], [2, 2]])
        self.flow3 = Flow([[2, 2], [3, 3]])

        self.fs1 = FlowSeries([self.flow1, self.flow2], crs="EPSG:4326")
        self.fs2 = FlowSeries([self.flow2, self.flow3], crs="EPSG:4326")

    def test_creation(self):
        """Test basic FlowSeries creation"""
        self.assertEqual(len(self.fs1), 2)
        self.assertIsInstance(self.fs1, FlowSeries)
        self.assertTrue(all(isinstance(f, Flow) for f in self.fs1))

    def test_invalid_data(self):
        """Test validation of input data"""
        with self.assertRaises(TypeError):
            FlowSeries([1, 2, 3])  # Non-Flow objects
        with self.assertRaises(TypeError):
            FlowSeries([self.flow1, "invalid"])  # Mixed types

    def test_item_assignment(self):
        """Test item assignment validation"""
        with self.assertRaises(TypeError):
            self.fs1[0] = "invalid"  # Non-Flow object
        self.fs1[0] = self.flow3  # Valid Flow object
        self.assertEqual(self.fs1[0], self.flow3)

    def test_concat(self):
        """Test concatenation of FlowSeries"""
        combined = pd.concat([self.fs1, self.fs2])
        self.assertIsInstance(combined, FlowSeries)
        self.assertEqual(len(combined), 4)
        self.assertEqual(combined.crs, self.fs1.crs)

        # Test invalid concatenation
        with self.assertRaises(TypeError):
            pd.concat([self.fs1, pd.Series([1, 2, 3])])

    def test_astype(self):
        """Test type conversion prevention"""
        with self.assertRaises(TypeError):
            self.fs1.astype(float)
        # Valid conversion to same type
        converted = self.fs1.astype(object)
        self.assertIsInstance(converted, FlowSeries)

    def test_crs_handling(self):
        """Test CRS handling"""
        fs = FlowSeries([self.flow1, self.flow2], crs="EPSG:4326")
        self.assertEqual(fs.crs, "EPSG:4326")

        # Replacing CRS without allow_override raises ValueError
        with self.assertRaises(ValueError):
            fs.set_crs("EPSG:3857")

        # to_crs transforms geometries correctly
        self.assertIsInstance(fs.to_crs("EPSG:3857"), FlowSeries)
        # set_crs with allow_override works
        self.assertIsInstance(fs.set_crs("EPSG:3857", allow_override=True), FlowSeries)

    def test_od(self):
        """Test origin and destination properties"""
        self.assertIsInstance(self.fs1.o, gpd.GeoSeries)
        self.assertIsInstance(self.fs1.d, gpd.GeoSeries)

    def test_length(self):
        """Test length (distance) property"""
        lengths = self.fs1.length
        self.assertEqual(len(lengths), 2)
        self.assertTrue(all(lengths > 0))

    def test_angle(self):
        """Test angle property"""
        import warnings
        fs = FlowSeries([Flow([[0, 0], [1, 0]]), Flow([[0, 0], [0, 1]])], crs="EPSG:32618")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angles = fs.angle
        self.assertEqual(len(angles), 2)

    def test_within(self):
        """Test within method"""
        from shapely.geometry import box
        mask = box(-1, -1, 3, 3)
        result = self.fs1.within(mask)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.all())

        # Partial within
        partial_fs = FlowSeries(
            [Flow([[0, 0], [10, 10]]), Flow([[0, 0], [1, 1]])],
            crs="EPSG:4326"
        )
        result = partial_fs.within(mask)
        self.assertFalse(result.iloc[0])
        self.assertTrue(result.iloc[1])

    def test_clip(self):
        """Test clip method"""
        from shapely.geometry import box
        mask = box(-1, -1, 3, 3)
        clipped = self.fs1.clip(mask)
        self.assertEqual(len(clipped), 2)
        self.assertIsInstance(clipped, FlowSeries)

    def test_distance(self):
        """Test distance method between two FlowSeries"""
        result = self.fs1.distance(self.fs2, distance="max")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result >= 0))

        result_sum = self.fs1.distance(self.fs2, distance="sum")
        self.assertEqual(len(result_sum), 2)

        result_min = self.fs1.distance(self.fs2, distance="min")
        self.assertEqual(len(result_min), 2)

        result_mean = self.fs1.distance(self.fs2, distance="mean")
        self.assertEqual(len(result_mean), 2)

        result_weighted = self.fs1.distance(self.fs2, distance="weighted", w1=1, w2=2)
        self.assertEqual(len(result_weighted), 2)

        # Single Flow
        result_single = self.fs1.distance(self.flow1, distance="max")
        self.assertEqual(len(result_single), 2)

        # Invalid distance method
        with self.assertRaises(ValueError):
            self.fs1.distance(self.fs2, distance="invalid")

    def test_to_crs(self):
        """Test CRS transformation"""
        fs = FlowSeries([Flow([[0, 0], [1, 1]]), Flow([[1, 2], [3, 4]])], crs="EPSG:4326")
        reprojected = fs.to_crs("EPSG:32618")
        self.assertIsInstance(reprojected, FlowSeries)
        self.assertNotEqual(reprojected.crs, fs.crs)

    def test_set_crs(self):
        """Test set_crs method"""
        fs = FlowSeries([Flow([[0, 0], [1, 1]]), Flow([[1, 2], [3, 4]])], crs="EPSG:4326")
        fs2 = fs.set_crs("EPSG:32618", allow_override=True)
        self.assertEqual(fs2.crs, "EPSG:32618")

    def test_plot(self):
        """Test plot method runs without error"""
        import matplotlib
        matplotlib.use("Agg")
        fs = FlowSeries([Flow([[0, 0], [1, 1]]), Flow([[1, 2], [3, 4]])], crs="EPSG:4326")
        ax = fs.plot()
        self.assertIsNotNone(ax)

    def test_geometry_access(self):
        """Test geometry access"""
        self.assertIsInstance(self.fs1.geometry, FlowSeries)


if __name__ == "__main__":
    unittest.main()
