import pytest
import numpy
from unittest.mock import Mock

import qutip
from qutip.core.numpy_backend import np
from qutip import CoreOptions, settings

# Mocking JAX to demonstrate backend switching
mock_jax = Mock
mock_jax.sum = Mock(return_value="jax_sum")
mock_np = numpy


class TestDefaultDType:
    def test_QobjCreation(self):
        with CoreOptions(default_dtype="CSR", default_dtype_range="creation"):
            from_list = qutip.Qobj([[1, 0], [1,0]])
            from_array = qutip.Qobj(np.array([[1, 0], [1,0]]))
            assert from_list.dtype is qutip.core.data.CSR
            assert from_list.dtype is qutip.core.data.Dense

        with CoreOptions(default_dtype="CSR", default_dtype_range="full"):
            from_list = qutip.Qobj([[1, 0], [1,0]])
            from_array = qutip.Qobj(np.array([[1, 0], [1,0]]))
            assert from_list.dtype is qutip.core.data.CSR
            assert from_list.dtype is qutip.core.data.CSR

    def test_dtype_dispatch(self):
        with CoreOptions(default_dtype="CSR", default_dtype_range="creation"):
            sparse = qutip.qeye(2, dtype="CSR")
            dense = qutip.Qobj(2, dtype="Dense")
            assert (sparse + dense).dtype is qutip.core.data.Dense

        with CoreOptions(default_dtype="CSR", default_dtype_range="missing"):
            # CSR + Dense not implemented
            sparse = qutip.qeye(2, dtype="CSR")
            dense = qutip.Qobj(2, dtype="Dense")
            assert (sparse + dense).dtype is qutip.core.data.CSR

        with CoreOptions(default_dtype="CSR", default_dtype_range="full"):
            dense = qutip.Qobj(2, dtype="Dense")
            assert (dense + dense).dtype is qutip.core.data.CSR


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
