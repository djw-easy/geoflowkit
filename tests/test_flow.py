import unittest
import numpy as np
import shapely
from shapely.geometry import Point, MultiPoint
from geoflowkit.flow import Flow


class TestFlowCreation(unittest.TestCase):
    def test_from_coordinate_pairs(self):
        flow = Flow([[0, 0], [1, 1]])
        self.assertIsInstance(flow, Flow)
        self.assertEqual(len(flow.geoms), 2)

    def test_from_tuple_pairs(self):
        flow = Flow([(0, 0), (1, 1)])
        self.assertIsInstance(flow, Flow)

    def test_from_point_objects(self):
        """Test creating Flow from a list of Point objects"""
        flow = Flow([Point(0, 0), Point(1, 1)])
        self.assertIsInstance(flow, Flow)
        self.assertEqual(flow.o.x, 0.0)
        self.assertEqual(flow.d.x, 1.0)

    def test_from_numpy_array(self):
        arr = np.array([[0.0, 0.0], [1.0, 2.0]])
        flow = Flow(arr)
        self.assertIsInstance(flow, Flow)

    def test_from_another_flow(self):
        flow1 = Flow([[0, 0], [1, 1]])
        flow2 = Flow(flow1)
        self.assertIs(flow1, flow2)

    def test_empty_flow(self):
        flow = Flow(None)
        self.assertTrue(flow.is_empty)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            Flow([[0, 0]])

    def test_too_many_points_raises(self):
        with self.assertRaises(ValueError):
            Flow([[0, 0], [1, 1], [2, 2]])

    def test_from_line_string(self):
        from shapely.geometry import LineString
        flow = Flow(LineString([[0, 0], [1, 1]]))
        self.assertIsInstance(flow, Flow)

    def test_from_3d_coords(self):
        flow = Flow([[0, 0, 10], [1, 1, 20]])
        self.assertEqual(flow.o.z, 10)
        self.assertEqual(flow.d.z, 20)


class TestFlowProperties(unittest.TestCase):
    def setUp(self):
        self.flow = Flow([[1.0, 2.0], [3.0, 4.0]])

    def test_origin(self):
        self.assertIsInstance(self.flow.o, Point)
        self.assertEqual(self.flow.o.x, 1.0)
        self.assertEqual(self.flow.o.y, 2.0)

    def test_destination(self):
        self.assertIsInstance(self.flow.d, Point)
        self.assertEqual(self.flow.d.x, 3.0)
        self.assertEqual(self.flow.d.y, 4.0)

    def test_geoms(self):
        self.assertEqual(len(self.flow.geoms), 2)
        self.assertEqual(self.flow.geoms[0].x, 1.0)
        self.assertEqual(self.flow.geoms[1].x, 3.0)


class TestFlowRepresentation(unittest.TestCase):
    def setUp(self):
        self.flow = Flow([[0.0, 0.0], [1.0, 2.0]])

    def test_wkt(self):
        wkt = self.flow.wkt
        self.assertIn("FLOW", wkt)
        self.assertIn("0", wkt)
        self.assertIn("2", wkt)

    def test_str(self):
        s = str(self.flow)
        self.assertIn("FLOW", s)

    def test_repr(self):
        r = repr(self.flow)
        self.assertIn("FLOW", r)

    def test_svg(self):
        svg = self.flow.svg()
        self.assertIn("<g>", svg)
        self.assertIn("</g>", svg)

    def test_svg_empty(self):
        flow = Flow(None)
        svg = flow.svg()
        self.assertEqual(svg, "<g />")

    def test_geo_interface(self):
        gi = self.flow.__geo_interface__
        self.assertEqual(gi["type"], "Flow")
        self.assertEqual(len(gi["coordinates"]), 2)


class TestFlowOperations(unittest.TestCase):
    def test_bounds(self):
        flow = Flow([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(flow.bounds, (1.0, 2.0, 3.0, 4.0))

    def test_bounds_reversed(self):
        flow = Flow([[3.0, 4.0], [1.0, 2.0]])
        self.assertEqual(flow.bounds, (1.0, 2.0, 3.0, 4.0))

    def test_is_valid(self):
        flow = Flow([[0, 0], [1, 1]])
        self.assertTrue(flow.is_valid)

    def test_is_empty(self):
        flow = Flow(None)
        self.assertTrue(flow.is_empty)

    def test_distance_between_flows(self):
        f1 = Flow([[0, 0], [1, 1]])
        f2 = Flow([[0, 0], [2, 2]])
        d = f1.distance(f2)
        self.assertIsInstance(d, float)

    def test_contains_point(self):
        from shapely.geometry import box
        flow = Flow([[1, 1], [3, 3]])
        mask = box(0, 0, 4, 4)
        self.assertTrue(flow.within(mask))


class TestFlowShapelyCompat(unittest.TestCase):
    def test_centroid(self):
        flow = Flow([[0, 0], [2, 2]])
        centroid = flow.centroid
        self.assertAlmostEqual(centroid.x, 1.0)
        self.assertAlmostEqual(centroid.y, 1.0)

    def test_area(self):
        flow = Flow([[0, 0], [1, 1]])
        self.assertEqual(flow.area, 0.0)

    def test_is_simple(self):
        flow = Flow([[0, 0], [1, 1]])
        self.assertTrue(flow.is_simple)

    def test_geoms_iterator(self):
        flow = Flow([[0, 0], [1, 1]])
        points = list(flow.geoms)
        self.assertEqual(len(points), 2)

    def test_hash(self):
        flow1 = Flow([[0, 0], [1, 1]])
        flow2 = Flow([[0, 0], [1, 1]])
        self.assertEqual(hash(flow1), hash(flow2))

    def test_equals(self):
        flow1 = Flow([[0, 0], [1, 1]])
        flow2 = Flow([[0, 0], [1, 1]])
        self.assertTrue(flow1.equals(flow2))


if __name__ == "__main__":
    unittest.main()
