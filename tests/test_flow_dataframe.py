import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from geoflowkit.flow import Flow
from geoflowkit.flowseries import FlowSeries
from geoflowkit.flowdataframe import FlowDataFrame


class TestFlowDataFrame(unittest.TestCase):
    def setUp(self):
        # Create sample Flow objects
        self.flow1 = Flow([[0, 0], [1, 1]])
        self.flow2 = Flow([[1, 1], [2, 2]])
        self.flow3 = Flow([[2, 2], [3, 3]])

        # Create sample data
        self.data = {
            'id': [1, 2, 3],
            'value': [10, 20, 30],
            'geometry': FlowSeries([self.flow1, self.flow2, self.flow3])
        }

    def test_init_with_flow_series(self):
        """Test initialization with FlowSeries geometry"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.crs, "EPSG:4326")

    def test_init_with_flow_list(self):
        """Test initialization with list of Flow objects"""
        data = self.data.copy()
        data['geometry'] = [self.flow1, self.flow2, self.flow3]
        df = FlowDataFrame(data, crs="EPSG:4326")
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.crs, "EPSG:4326")

    def test_init_with_invalid_geometry(self):
        """Test initialization with invalid geometry"""
        data = self.data.copy()
        data['geometry'] = [[0, 0], [1, 1], [2, 2]]  # Not Flow objects
        with self.assertRaises(TypeError):
            FlowDataFrame(data)

    def test_set_geometry(self):
        """Test setting geometry column"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")
        new_flows = FlowSeries([self.flow3, self.flow2, self.flow1], crs="EPSG:4326")
        df.set_geometry(new_flows, inplace=True)
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.crs, "EPSG:4326")

    def test_set_geometry_with_column(self):
        """Test setting geometry from existing column"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")
        df['new_geom'] = FlowSeries([self.flow3, self.flow2, self.flow1], crs="EPSG:4326")
        df.set_geometry('new_geom', inplace=True)
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(df._geometry_column_name, 'new_geom')
        self.assertEqual(df.crs, "EPSG:4326")

    def test_set_invalid_geometry(self):
        """Test setting invalid geometry"""
        df = FlowDataFrame(self.data)
        with self.assertRaises(TypeError):
            df.set_geometry([[0, 0], [1, 1], [2, 2]], inplace=True)

    def test_copy(self):
        """Test copying FlowDataFrame"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")
        df_copy = df.copy()
        self.assertIsInstance(df_copy, FlowDataFrame)
        self.assertIsInstance(df_copy.geometry, FlowSeries)
        self.assertEqual(len(df_copy), 3)
        self.assertEqual(df_copy.crs, "EPSG:4326")

    def test_setitem_geometry(self):
        """Test setting geometry column directly"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")
        new_flows = FlowSeries([self.flow3, self.flow2, self.flow1], crs="EPSG:4526")
        df['geometry'] = new_flows
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(df.crs, "EPSG:4526")
        df['geometry'] = FlowSeries([self.flow3, self.flow2, self.flow1], crs="EPSG:4526")
        self.assertIsInstance(df.geometry, FlowSeries)
        self.assertEqual(df.crs, "EPSG:4526")

    def test_setitem_invalid_geometry(self):
        """Test setting invalid geometry directly"""
        df = FlowDataFrame(self.data)
        with self.assertRaises(TypeError):
            df.geometry = [[0, 0], [1, 1], [2, 2]]
        with self.assertRaises(TypeError):
            df['geometry'] = [[0, 0], [1, 1], [2, 2]]

    def test_basic_operations(self):
        """Test basic GeoDataFrame operations"""
        df = FlowDataFrame(self.data, crs="EPSG:4326")

        # Test slicing
        subset = df[:2]
        self.assertIsInstance(subset, FlowDataFrame)
        self.assertIsInstance(subset.geometry, FlowSeries)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset.crs, "EPSG:4326")

        # Test column operations
        df['new_col'] = [1, 2, 3]
        self.assertEqual(len(df.columns), 4)  # id, value, geometry, new_col
        self.assertEqual(df.crs, "EPSG:4326")

    def test_concat_operations(self):
        """Test concatenation operations"""
        df1 = FlowDataFrame(self.data, crs="EPSG:4326")
        df2 = FlowDataFrame(self.data, crs="EPSG:4326")

        # Test concatenation
        result = pd.concat([df1, df2])
        self.assertIsInstance(result, FlowDataFrame)
        self.assertIsInstance(result.geometry, FlowSeries)
        self.assertEqual(len(result), 6)
        self.assertEqual(result.crs, "EPSG:4326")

    def test_plot_arrow(self):
        """Test plot with kind='arrow'"""
        import matplotlib
        matplotlib.use("Agg")
        import tempfile
        import os

        df = FlowDataFrame(self.data, crs="EPSG:4326")
        ax = df.plot(kind='arrow')
        self.assertIsNotNone(ax)

        # Save to file to verify image is generated
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig = ax.get_figure()
            fig.savefig(f.name)
            temp_path = f.name
        self.assertTrue(os.path.exists(temp_path))
        os.unlink(temp_path)

    def test_plot_arrow_with_numeric_column(self):
        """Test plot with numeric column"""
        import matplotlib
        matplotlib.use("Agg")
        import tempfile
        import os

        df = FlowDataFrame(self.data, crs="EPSG:4326")
        ax = df.plot(kind='arrow', column='value')
        self.assertIsNotNone(ax)

        # Save to file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig = ax.get_figure()
            fig.savefig(f.name)
            temp_path = f.name
        self.assertTrue(os.path.exists(temp_path))
        os.unlink(temp_path)

    def test_plot_arrow_with_categorical_column(self):
        """Test plot with categorical column"""
        import matplotlib
        matplotlib.use("Agg")
        import tempfile
        import os

        data = self.data.copy()
        data['category'] = ['A', 'B', 'A']
        df = FlowDataFrame(data, crs="EPSG:4326")
        ax = df.plot(kind='arrow', column='category')
        self.assertIsNotNone(ax)

        # Check that legend was added for categorical data
        legend = ax.get_legend()
        self.assertIsNotNone(legend)

        # Save to file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            fig = ax.get_figure()
            fig.savefig(f.name)
            temp_path = f.name
        self.assertTrue(os.path.exists(temp_path))
        os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
