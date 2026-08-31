import numpy as np
import pytest
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.flowmetrics import (
    _haversine_pdist,
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

    def test_empty_dataframe(self):
        fdf = FlowDataFrame(geometry=FlowSeries([]), crs="EPSG:3857")
        assert pairwise_distances(fdf).shape == (0, 0)

    def test_haversine_known_distance_and_order(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        distances = _haversine_pdist(coords)
        assert distances.shape == (3,)
        assert distances[0] == pytest.approx(111.195, rel=1e-4)
        assert distances[1] == pytest.approx(111.195, rel=1e-4)
        assert distances[2] == pytest.approx(157.249, rel=1e-4)


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
        with pytest.raises(ValueError, match="smaller than the number of flows"):
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

    def test_k_must_be_smaller_than_sample(self, sample_fdf):
        with pytest.raises(ValueError, match="smaller than the number of flows"):
            snn_distance(sample_fdf, k=len(sample_fdf))


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

    def test_empty_dataframe(self):
        fdf = FlowDataFrame(geometry=FlowSeries([]), crs="EPSG:3857")
        assert flow_entropy(fdf) == 0.0

    def test_single_flow_is_exactly_zero(self):
        fs = flows_from_od(np.array([[0, 0]]), np.array([[1, 0]]), crs="EPSG:3857")
        fdf = FlowDataFrame(geometry=fs, crs="EPSG:3857")
        assert flow_entropy(fdf) == 0.0


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


class TestPairwiseDistancesExtra:
    def test_invalid_geographic_mode_raises(self, sample_fdf):
        with pytest.raises(ValueError, match="handle_geographic"):
            pairwise_distances(sample_fdf, handle_geographic='warning')

    def test_weighted_distance_no_length(self, sample_fdf):
        result = pairwise_distances(sample_fdf, distance='weighted', length=False)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.allclose(np.diag(result), 0.0)
        assert np.allclose(result, result.T)

    def test_weighted_distance_with_length(self, sample_fdf):
        result = pairwise_distances(sample_fdf, distance='weighted', length=True)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)
        assert np.allclose(np.diag(result), 0.0)
        assert np.allclose(result, result.T)

    def test_weighted_length_matches_flowbase_definition(self):
        origins = np.array([[0, 0], [4, 0]])
        destinations = np.array([[3, 0], [4, 4]])
        fs = flows_from_od(origins, destinations, crs="EPSG:3857")
        fdf = FlowDataFrame(geometry=fs, crs="EPSG:3857")

        result = pairwise_distances(
            fdf, distance='weighted', w1=2, w2=3, length=True
        )
        expected = np.sqrt((2 * 4 ** 2 + 3 * (1 ** 2 + 4 ** 2)) / (3 * 4))
        assert result[0, 1] == pytest.approx(expected)

    def test_weighted_length_handles_zero_length_flows(self):
        origins = np.array([[0, 0], [1, 1]])
        destinations = np.array([[0, 0], [2, 1]])
        fs = flows_from_od(origins, destinations, crs="EPSG:3857")
        fdf = FlowDataFrame(geometry=fs, crs="EPSG:3857")

        result = pairwise_distances(fdf, distance='weighted', length=True)
        assert result[0, 0] == 0.0
        assert np.isinf(result[0, 1])

    def test_weighted_with_weights(self, sample_fdf):
        result = pairwise_distances(sample_fdf, distance='weighted', w1=1, w2=2, length=False)
        assert result.shape == (5, 5)
        assert np.all(result >= 0)

    def test_mean_distance(self, sample_fdf):
        result = pairwise_distances(sample_fdf, distance='mean')
        assert result.shape == (5, 5)
        assert np.all(result >= 0)

    def test_single_flow(self):
        o = np.array([[0, 0]])
        d = np.array([[1, 1]])
        fs = flows_from_od(o, d, crs="EPSG:3857")
        fdf = FlowDataFrame({'geometry': fs}, crs="EPSG:3857")
        result = pairwise_distances(fdf)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0

    def test_diagonal_zero(self, sample_fdf):
        result = pairwise_distances(sample_fdf)
        assert np.allclose(np.diag(result), 0.0)

    def test_symmetric(self, sample_fdf):
        result = pairwise_distances(sample_fdf)
        assert np.allclose(result, result.T)


class TestKNeighborDistancesExtra:
    def test_k1_non_negative(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=1)
        assert np.all(result >= 0)

    def test_k_equals_n_minus_1(self, sample_fdf):
        result = k_neighbor_distances(sample_fdf, k=4)
        assert result.shape == (5,)
        assert np.all(result >= 0)


class TestSNNDistanceExtra:
    def test_snn_diagonal_zero(self, sample_fdf):
        result = snn_distance(sample_fdf, k=4)
        assert np.allclose(np.diag(result), 0.0)

    def test_snn_range(self, sample_fdf):
        result = snn_distance(sample_fdf, k=2)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
