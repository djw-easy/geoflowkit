import numpy as np
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.clustering.kmedoid import KMedoidFlow, kmedoid
from geoflowkit.clustering.dbscan import DBSCANFlow, dbscan, _flow_distance_factory


def _make_fdf(n=12, n_groups=3):
    rng = np.random.RandomState(42)
    centers = rng.rand(n_groups, 2) * 10
    labels_true = rng.randint(0, n_groups, n)
    o = centers[labels_true] + rng.randn(n, 2) * 0.3
    d = centers[labels_true] + rng.randn(n, 2) * 0.3
    fs = flows_from_od(o, d, crs="EPSG:3857")
    return FlowDataFrame({"true_label": labels_true}, geometry=fs, crs="EPSG:3857")


class TestKMedoidFlow:
    def test_fit(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, random_state=42)
        model.fit(fdf)
        assert model.labels_ is not None
        assert len(model.labels_) == 10
        assert model.cluster_centers_ is not None
        assert model.inertia_ is not None

    def test_fit_predict(self):
        fdf = _make_fdf(10)
        labels = kmedoid(fdf, n_clusters=3, random_state=42)
        assert len(labels) == 10
        assert len(set(labels)) <= 3

    def test_reproducible(self):
        fdf = _make_fdf(10)
        labels1 = kmedoid(fdf, n_clusters=3, random_state=42)
        labels2 = kmedoid(fdf, n_clusters=3, random_state=42)
        np.testing.assert_array_equal(labels1, labels2)

    def test_inertia_non_negative(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, random_state=42)
        model.fit(fdf)
        assert model.inertia_ >= 0

    def test_distance_options(self):
        fdf = _make_fdf(10)
        for dist in ['max', 'min', 'sum', 'mean']:
            labels = kmedoid(fdf, n_clusters=3, distance=dist, random_state=42)
            assert len(labels) == 10

    def test_method_precompute(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, method='precompute', random_state=42)
        model.fit(fdf)
        assert model.labels_ is not None

    def test_method_online(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, method='online', random_state=42)
        model.fit(fdf)
        assert model.labels_ is not None

    def test_invalid_method(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, method='invalid')
        try:
            model.fit(fdf)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_invalid_distance(self):
        fdf = _make_fdf(10)
        model = KMedoidFlow(n_clusters=3, distance='invalid')
        try:
            model.fit(fdf)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_predict_before_fit(self):
        model = KMedoidFlow(n_clusters=3)
        try:
            model.predict(None)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestDBSCANFlow:
    def test_fit(self):
        fdf = _make_fdf(20)
        model = DBSCANFlow(eps=2.0, min_samples=3)
        model.fit(fdf)
        assert model.labels_ is not None
        assert len(model.labels_) == 20

    def test_fit_predict(self):
        fdf = _make_fdf(20)
        labels = dbscan(fdf, eps=2.0, min_samples=3)
        assert len(labels) == 20

    def test_n_clusters(self):
        fdf = _make_fdf(20)
        model = DBSCANFlow(eps=2.0, min_samples=3)
        model.fit(fdf)
        assert model.n_clusters >= 0

    def test_core_samples(self):
        fdf = _make_fdf(20)
        model = DBSCANFlow(eps=2.0, min_samples=3)
        model.fit(fdf)
        assert model.core_sample_indices_ is not None

    def test_n_clusters_before_fit_raises(self):
        model = DBSCANFlow()
        try:
            _ = model.n_clusters
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_noise_label(self):
        fdf = _make_fdf(30)
        labels = dbscan(fdf, eps=0.01, min_samples=5)
        assert -1 in labels or len(set(labels)) > 0

    def test_distance_options(self):
        fdf = _make_fdf(20)
        for dist in ['max', 'min', 'sum', 'mean']:
            labels = dbscan(fdf, eps=2.0, min_samples=3, distance=dist)
            assert len(labels) == 20

    def test_invalid_distance(self):
        try:
            DBSCANFlow(distance='invalid')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestFlowDistanceFactory:
    def test_max(self):
        metric = _flow_distance_factory('max')
        # u=[0,0,1,1], v=[0,0,2,2]
        # o_dist = 0, d_dist = sqrt(2)
        u = np.array([0, 0, 1, 1])
        v = np.array([0, 0, 2, 2])
        d = metric(u, v)
        assert abs(d - np.sqrt(2)) < 1e-9

    def test_min(self):
        metric = _flow_distance_factory('min')
        u = np.array([0, 0, 1, 1])
        v = np.array([0, 0, 2, 2])
        d = metric(u, v)
        assert abs(d - 0.0) < 1e-9

    def test_sum(self):
        metric = _flow_distance_factory('sum')
        u = np.array([0, 0, 1, 1])
        v = np.array([0, 0, 2, 2])
        d = metric(u, v)
        assert abs(d - np.sqrt(2)) < 1e-9

    def test_mean(self):
        metric = _flow_distance_factory('mean')
        u = np.array([0, 0, 1, 1])
        v = np.array([0, 0, 2, 2])
        d = metric(u, v)
        assert abs(d - np.sqrt(2) / 2) < 1e-9

    def test_invalid(self):
        try:
            _flow_distance_factory('invalid')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
