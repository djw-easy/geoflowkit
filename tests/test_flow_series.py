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

        # Test CRS mismatch
        with self.assertRaises(ValueError):
            FlowSeries([self.flow1, self.flow2], crs="EPSG:4326").set_crs("EPSG:3857")

        self.assertIsInstance(fs.to_crs("EPSG:3857"), FlowSeries)
        self.assertIsInstance(fs.set_crs("EPSG:3857", allow_override=True), FlowSeries)
        self.assertNotEqual(fs.crs, None)
        self.assertNotEqual(fs.to_crs("EPSG:3857").crs, None)
        self.assertNotEqual(fs.set_crs("EPSG:3857", allow_override=True).crs, None)

    def test_od(self):
        """Test origin and destination properties"""
        self.assertIsInstance(self.fs1.o, gpd.GeoSeries)
        self.assertIsInstance(self.fs1.d, gpd.GeoSeries)

    def test_geometry_access(self):
        """Test geometry access"""
        self.assertIsInstance(self.fs1.geometry, FlowSeries)


if __name__ == "__main__":
    unittest.main()
