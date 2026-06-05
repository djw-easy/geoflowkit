import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import box
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.spatial.kl_function import k_func, l_func, local_l_func
from geoflowkit.spatial.centrality import i_index
from geoflowkit.spatial.utils import second_order_density


def _make_fdf(n=10, spread=10.0, crs="EPSG:3857"):
    rng = np.random.RandomState(42)
    o = rng.rand(n, 2) * spread
    d = o + rng.randn(n, 2) * 1.0
    fs = flows_from_od(o, d, crs=crs)
    return FlowDataFrame({"value": np.arange(n)}, geometry=fs, crs=crs)


class TestSecondOrderDensity:
    def test_basic(self):
        fdf = _make_fdf(8)
        result = second_order_density(fdf, distance='max')
        assert isinstance(result, float)
        assert result > 0

    def test_with_precomputed_matrix(self):
        from geoflowkit.flowmetrics import pairwise_distances
        fdf = _make_fdf(8)
        dis = pairwise_distances(fdf, distance='max')
        result = second_order_density(dis_matrix=dis, distance='max')
        assert result > 0

    def test_non_max_distance_raises(self):
        fdf = _make_fdf(8)
        with pytest.raises(NotImplementedError):
            second_order_density(fdf, distance='sum')


class TestNthLargest:
    def test_1d(self):
        from geoflowkit.spatial.utils import nth_largest
        arr = np.array([1, 3, 2, 5, 4])
        result = nth_largest(arr, 1, axis=0)
        assert result == 5

    def test_2d(self):
        from geoflowkit.spatial.utils import nth_largest
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = nth_largest(arr, 1, axis=1)
        np.testing.assert_array_equal(result, np.array([3, 6]))

    def test_2d_axis0(self):
        from geoflowkit.spatial.utils import nth_largest
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = nth_largest(arr, 1, axis=0)
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))

    def test_n_too_large(self):
        from geoflowkit.spatial.utils import nth_largest
        arr = np.array([[1, 2], [3, 4]])
        with np.testing.assert_raises(ValueError):
            nth_largest(arr, 5, axis=1)


class TestKFunc:
    def test_basic(self):
        fdf = _make_fdf(10)
        r_list, kr_list = k_func(fdf, dr=1.0, k=1)
        assert len(r_list) > 0
        assert len(r_list) == len(kr_list)
        assert all(kr >= 0 for kr in kr_list)

    def test_with_mask(self):
        fdf = _make_fdf(10)
        mask = box(-1, -1, 5, 5)
        r_list, kr_list = k_func(fdf, dr=1.0, mask=mask)
        assert len(r_list) > 0


class TestLFunc:
    def test_basic(self):
        fdf = _make_fdf(10)
        r_list, lr_list = l_func(fdf, dr=1.0, k=1)
        assert len(r_list) > 0
        assert len(r_list) == len(lr_list)


class TestLocalLFunc:
    def test_basic(self):
        fdf = _make_fdf(10)
        result = local_l_func(fdf, r=2.0)
        assert result.shape == (10,)
        assert all(isinstance(v, float) for v in result)

    def test_with_precomputed_matrix(self):
        from geoflowkit.flowmetrics import pairwise_distances
        fdf = _make_fdf(10)
        dis = pairwise_distances(fdf, distance='max')
        result = local_l_func(fdf, r=2.0, dis_matrix=dis)
        assert result.shape == (10,)


class TestIIndex:
    def setup_method(self):
        self.fdf = _make_fdf(15, spread=10.0)

    def _make_zones(self):
        zones = gpd.GeoDataFrame({
            'zone_id': [0, 1, 2, 3],
            'geometry': [
                box(0, 0, 5, 5),
                box(5, 0, 10, 5),
                box(0, 5, 5, 10),
                box(5, 5, 10, 10),
            ]
        }, crs="EPSG:3857")
        return zones

    def test_basic(self):
        zones = self._make_zones()
        result = i_index(self.fdf, zones)
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'I_index' in result.columns
        assert 'flow_count' in result.columns
        assert 'total_length' in result.columns
        assert len(result) == 4

    def test_with_alpha(self):
        zones = self._make_zones()
        result = i_index(self.fdf, zones, alpha=500.0)
        assert 'alpha' in result.columns
        assert all(result['alpha'] == 500.0)

    def test_origin_type(self):
        zones = self._make_zones()
        result = i_index(self.fdf, zones, od_type='o')
        assert len(result) == 4

    def test_missing_zone_id_raises(self):
        zones = gpd.GeoDataFrame({
            'geometry': [box(0, 0, 10, 10)]
        }, crs="EPSG:3857")
        with pytest.raises(ValueError):
            i_index(self.fdf, zones)

    def test_invalid_od_type_raises(self):
        zones = self._make_zones()
        with pytest.raises(ValueError):
            i_index(self.fdf, zones, od_type='x')

    def test_zone_with_no_flows(self):
        zones = gpd.GeoDataFrame({
            'zone_id': [0, 1],
            'geometry': [
                box(-100, -100, -90, -90),
                box(100, 100, 110, 110),
            ]
        }, crs="EPSG:3857")
        result = i_index(self.fdf, zones)
        assert all(result['I_index'] == 0)
        assert all(result['flow_count'] == 0)
