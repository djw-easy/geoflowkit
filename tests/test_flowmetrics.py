import numpy as np
import pytest
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.flowmetrics import (
    pairwise_distances,
    k_neighbor_distances,
    snn_distance,
    flow_entropy,
    flow_divergence
)


@pytest.fixture
def sample_fdf():
    """Create a sample FlowDataFrame for testing."""
    o_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    d_points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    fs = flows_from_od(o_points, d_points, crs="EPSG:3857")
    data = {'geometry': fs, 'value': [1, 2, 3, 4, 5]}
    return FlowDataFrame(data, crs="EPSG:3857")


class TestPairwiseDistances:
    def test_basic(self, sample_fdf):
        result = pairwise_distances(sample_fdf)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.all(result == result.T)  # symmetric

    def test_distance_types(self, sample_fdf):
        for dist_type in ['max', 'min', 'sum', 'mean']:
            result = pairwise_distances(sample_fdf, distance=dist_type)
            assert result.shape == (5, 5)


class TestKNeighborDistances:
    def test_k1(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=1)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_k2(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=2)
        assert result.shape == (5,)
        assert np.all(result >= 0)

    def test_k_greater_than_n(self, sample_fdf):
        with pytest.raises(IndexError):
            k_neighbor_distances(sample_fdf, k=10)

    def test_k_must_be_positive(self, sample_fdf):
        with pytest.raises(ValueError):
            k_neighbor_distances(sample_fdf, k=0)


class TestSNNDistance:
    def test_basic(self, sample_fdf):
        result = snn_distance(sample_fdf, k=2)
        assert result.shape == (5, 5)
        assert np.all((result >= 0) & (result <= 1))
        assert np.all(result == result.T)  # symmetric
        assert np.all(result.diagonal() == 0)  # diagonal is 0

    def test_same_flows_same_snn(self, sample_fdf):
        # Same flow should have SNN distance 0 with itself
        result = snn_distance(sample_fdf, k=4)
        assert result[0, 0] == 0

    def test_snn_symmetric(self, sample_fdf):
        result = snn_distance(sample_fdf, k=2)
        for i in range(len(sample_fdf)):
            for j in range(i + 1, len(sample_fdf)):
                assert result[i, j] == result[j, i]


class TestFlowEntropy:
    def test_basic(self, sample_fdf):
        result = flow_entropy(sample_fdf)
        assert isinstance(result, float)
        assert result >= 0

    def test_uniform_distribution(self, sample_fdf):
        # For n flows, maximum entropy is log2(n)
        result = flow_entropy(sample_fdf)
        n = len(sample_fdf)
        max_entropy = np.log2(n)
        assert result <= max_entropy


class TestFlowDivergence:
    def test_basic(self, sample_fdf):
        result = flow_divergence(sample_fdf, n_directions=6)
        assert isinstance(result, float)
        assert result >= -1e-9  # Allow small negative due to floating point

    def test_n_directions(self, sample_fdf):
        for n in [4, 6, 8]:
            result = flow_divergence(sample_fdf, n_directions=n)
            max_entropy = np.log2(n)
            assert result <= max_entropy

    def test_invalid_n_directions(self, sample_fdf):
        with pytest.raises(ValueError):
            flow_divergence(sample_fdf, n_directions=1)

    def test_all_same_direction(self):
        # All flows going in same direction should give 0 divergence
        o_points = np.array([[0, 0], [0, 0], [0, 0]])
        d_points = np.array([[1, 0], [1, 0], [1, 0]])  # All going east
        fs = flows_from_od(o_points, d_points, crs="EPSG:3857")
        fdf = FlowDataFrame({'geometry': fs}, crs="EPSG:3857")
        result = flow_divergence(fdf, n_directions=4)
        assert abs(result) < 1e-9  # Should be ~0
