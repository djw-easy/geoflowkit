import numpy as np
import pytest
from geoflowkit import FlowDataFrame, FlowSeries, flows_from_od
from geoflowkit.manifold.ftsne.ftsne import FTSNE


def _make_fdf(n=20, crs="EPSG:3857"):
    rng = np.random.RandomState(42)
    o = rng.rand(n, 2) * 10
    d = o + rng.randn(n, 2) * 1.0
    fs = flows_from_od(o, d, crs=crs)
    return FlowDataFrame({"value": np.arange(n)}, geometry=fs, crs=crs)


class TestFTSNE:
    def test_init(self):
        model = FTSNE(perplexity=5, learning_rate=0.1, random_state=42,
                       early_exaggeration_iter=10, max_iter=50)
        assert model.perplexity == 5
        assert model.learning_rate == 0.1

    def test_init_with_auto_early_exaggeration(self):
        model = FTSNE(early_exaggeration_iter='auto', max_iter=100)
        assert model.early_exaggeration_iter == 25

    def test_fit_transform_local(self):
        fdf = _make_fdf(20)
        model = FTSNE(perplexity=5, learning_rate=0.1, random_state=42,
                       early_exaggeration_iter=10, max_iter=50)
        result = model.fit_transform(fdf, union={('o', 'd'): (0, 1)})
        assert result.shape == (20, 2)

    def test_fit_transform_global(self):
        fdf = _make_fdf(20)
        model = FTSNE(perplexity=5, learning_rate=0.1, random_state=42,
                       early_exaggeration_iter=10, max_iter=50)
        result = model.fit_transform(fdf, identity={'o': 0, 'd': 1})
        assert result.shape == (20, 2)

    def test_get_values_o(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, 'o')
        assert values.shape == (5, 2)

    def test_get_values_d(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, 'd')
        assert values.shape == (5, 2)

    def test_get_values_length(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, 'length')
        assert values.shape == (5, 1)

    def test_get_values_angle(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, 'angle')
        assert values.shape == (5, 1)

    def test_get_values_column(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, 'value')
        assert values.shape == (5, 1)

    def test_get_values_tuple(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        values = model._get_values(fdf, ('o', 'd'))
        assert values.shape == (5, 4)

    def test_get_values_invalid(self):
        fdf = _make_fdf(5)
        model = FTSNE()
        with pytest.raises(ValueError):
            model._get_values(fdf, 'nonexistent')

    def test_invalid_perplexity(self):
        with pytest.raises(Exception):
            FTSNE(perplexity=-1)

    def test_invalid_method(self):
        with pytest.raises(Exception):
            FTSNE(method='invalid')

    def test_invalid_loss_func(self):
        with pytest.raises(Exception):
            FTSNE(loss_func='invalid')

    def test_invalid_init(self):
        with pytest.raises(Exception):
            FTSNE(init='invalid')
