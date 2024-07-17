import pytest
from unittest.mock import patch, MagicMock

from ...settings import settings
from ...core.numpy_backend import NumpyBackend

# Mocking JAX and NumPy to demonstrate backend switching
mock_jax = MagicMock()
mock_np = MagicMock()


class TestNumpyBackend:
    def test_backend_property(self):
        settings.core["numpy_backend"] = mock_np
        np_backend = NumpyBackend()
        assert np_backend.backend is mock_np

        settings.core["numpy_backend"] = mock_jax
        assert np_backend.backend is mock_jax

    def test_getattr_numpy(self):
        settings.core["numpy_backend"] = mock_np
        np_backend = NumpyBackend()
        mock_np.sum = MagicMock(return_value="numpy_sum")
        assert np_backend.sum([1, 2, 3]) == "numpy_sum"

    def test_getattr_jax(self):
        settings.core["numpy_backend"] = mock_jax
        np_backend = NumpyBackend()
        mock_jax.sum = MagicMock(return_value="jax_sum")
        assert np_backend.sum([1, 2, 3]) == "jax_sum"

    @pytest.mark.parametrize("backend", [mock_np, mock_jax])
    def test_backend_functionality(self, backend):
        settings.core["numpy_backend"] = backend
        np_backend = NumpyBackend()

        backend.sum = MagicMock(return_value="sum_result")
        result = np_backend.sum([1, 2, 3])
        assert result == "sum_result"
        backend.sum.assert_called_with([1, 2, 3])


@pytest.fixture(scope='module')
def mock_settings():
    with patch.dict(settings.core, {"numpy_backend": mock_np}):
        yield settings
