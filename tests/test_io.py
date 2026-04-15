import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries
from geoflowkit.flowdataframe import FlowDataFrame
from geoflowkit.io import flows_from_od, flows_from_geometry, read_csv, read_file


class TestFlowsFromOD(unittest.TestCase):
    def test_basic(self):
        """Test creation from origin-destination arrays"""
        o = np.array([[0, 0], [1, 2], [3, 4]])
        d = np.array([[1, 1], [2, 3], [5, 6]])
        fs = flows_from_od(o, d)
        self.assertIsInstance(fs, FlowSeries)
        self.assertEqual(len(fs), 3)

    def test_list_input(self):
        """Test with list input"""
        o = [[0, 0], [1, 2]]
        d = [[1, 1], [3, 4]]
        fs = flows_from_od(o, d, crs="EPSG:4326")
        self.assertEqual(fs.crs, "EPSG:4326")
        self.assertEqual(len(fs), 2)

    def test_invalid_dim(self):
        """Test error on invalid dimensions"""
        with self.assertRaises(ValueError):
            flows_from_od([1, 2, 3], [[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            flows_from_od([[0, 0]], [[1, 1], [2, 2]])  # length mismatch


class TestFlowsFromGeometry(unittest.TestCase):
    def test_flow_input(self):
        """Test with Flow objects"""
        flows = [Flow([[0, 0], [1, 1]]), Flow([[2, 2], [3, 3]])]
        fs = flows_from_geometry(flows)
        self.assertEqual(len(fs), 2)
        self.assertIsInstance(fs[0], Flow)

    def test_linestring(self):
        """Test conversion from LineString"""
        from shapely.geometry import LineString
        lines = [LineString([[0, 0], [1, 1]]), LineString([[2, 2], [3, 3]])]
        fs = flows_from_geometry(lines)
        self.assertEqual(len(fs), 2)

    def test_multipoint(self):
        """Test conversion from MultiPoint"""
        from shapely.geometry import MultiPoint
        mps = [MultiPoint([[0, 0], [1, 1]]), MultiPoint([[2, 2], [3, 3]])]
        fs = flows_from_geometry(mps)
        self.assertEqual(len(fs), 2)

    def test_mixed(self):
        """Test with mixed geometry types"""
        from shapely.geometry import LineString
        flows = [Flow([[0, 0], [1, 1]]), LineString([[2, 2], [3, 3]])]
        fs = flows_from_geometry(flows)
        self.assertEqual(len(fs), 2)

    def test_invalid_type(self):
        """Test error on invalid geometry type"""
        from shapely.geometry import Point
        with self.assertRaises(TypeError):
            flows_from_geometry([Point(0, 0), Point(1, 1)])


class TestReadCSV(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.temp_dir, "test_flows.csv")

    def tearDown(self):
        import gc
        gc.collect()  # release file handles
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def test_basic(self):
        """Test reading from CSV"""
        df = pd.DataFrame({
            "o_x": [0.0, 1.0],
            "o_y": [0.0, 2.0],
            "d_x": [1.0, 3.0],
            "d_y": [1.0, 4.0],
        })
        df.to_csv(self.csv_path, index=False)
        result = read_csv(self.csv_path, use_cols=["o_x", "o_y", "d_x", "d_y"])
        self.assertIsInstance(result, FlowDataFrame)
        self.assertEqual(len(result), 2)

    def test_int_columns(self):
        """Test reading with integer column indices"""
        df = pd.DataFrame({
            "o_x": [0.0, 1.0],
            "o_y": [0.0, 2.0],
            "d_x": [1.0, 3.0],
            "d_y": [1.0, 4.0],
        })
        df.to_csv(self.csv_path, index=False)
        result = read_csv(self.csv_path, use_cols=[0, 1, 2, 3])
        self.assertEqual(len(result), 2)

    def test_invalid_columns(self):
        """Test error on invalid column count"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(self.csv_path, index=False)
        with self.assertRaises(ValueError):
            read_csv(self.csv_path, use_cols=["a", "b"])


if __name__ == "__main__":
    unittest.main()
