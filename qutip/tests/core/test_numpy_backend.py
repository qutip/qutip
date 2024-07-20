import pytest
import numpy
from unittest.mock import Mock

from qutip.core.numpy_backend import np, NumpyBackend
from qutip import CoreOptions

# Mocking JAX to demonstrate backend switching
mock_jax = Mock()
mock_np = numpy


class TestNumpyBackend:
    def test_backend_property(self):
        with CoreOptions(numpy_backend=mock_np):
            np_backend = NumpyBackend()
            assert np_backend.backend is mock_np

        with CoreOptions(numpy_backend=mock_jax):
            np_backend = NumpyBackend()
            assert np_backend.backend is mock_jax

    def test_getattr_numpy(self):
        with CoreOptions(numpy_backend=mock_np):
            np_backend = NumpyBackend()
            assert np_backend.sum([1, 2, 3]) == numpy.sum([1, 2, 3])

    def test_getattr_jax(self):
        with CoreOptions(numpy_backend=mock_jax):
            np_backend = NumpyBackend()
            mock_jax.sum = Mock(return_value="jax_sum")
            assert np_backend.sum([1, 2, 3]) == "jax_sum"
