import pytest
import numpy
from unittest.mock import Mock

from qutip.core.numpy_backend import np
from qutip import CoreOptions, settings

# Mocking JAX to demonstrate backend switching
mock_jax = Mock
mock_jax.sum = Mock(return_value="jax_sum")
mock_np = numpy


class TestNumpyBackend:
    def test_getattr_numpy(self):
        with CoreOptions(numpy_backend=mock_np):
            assert np.sum([1, 2, 3]) == numpy.sum([1, 2, 3])
            assert np.sum is numpy.sum

    def test_coreoptions_setattr(self):
        with CoreOptions():
            settings.core["numpy_backend"] = mock_jax
            assert np.sum([1, 2, 3]) == "jax_sum"

    def test_getattr_jax(self):
        with CoreOptions(numpy_backend=mock_jax):
            assert np.sum([1, 2, 3]) == "jax_sum"
