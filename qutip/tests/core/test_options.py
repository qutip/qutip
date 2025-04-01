import pytest
import numpy

import qutip
from qutip.core.numpy_backend import np
from qutip import CoreOptions, settings


# Mocking JAX to demonstrate backend switching
class Mock_JAX:
    def sum(*a, **kw):
        return "jax_sum"

    def __getattr__(self, _):
        return "other"


mock_jax = Mock_JAX()
mock_np = numpy


class TestDefaultDType:
    def test_QobjCreation(self):
        with CoreOptions(default_dtype="CSR", default_dtype_scope="creation"):
            from_list = qutip.Qobj([[1, 0], [1,0]])
            from_array = qutip.Qobj(np.array([[1, 0], [1,0]]))
            assert from_list.dtype is qutip.core.data.CSR
            assert from_array.dtype is qutip.core.data.Dense

        with CoreOptions(default_dtype="CSR", default_dtype_scope="full"):
            from_list = qutip.Qobj([[1, 0], [1,0]])
            from_array = qutip.Qobj(np.array([[1, 0], [1,0]]))
            assert from_list.dtype is qutip.core.data.CSR
            assert from_array.dtype is qutip.core.data.CSR

    def test_dtype_dispatch(self):
        # CSR + Dense not implemented
        sparse = qutip.qeye(2, dtype="CSR")
        dense = qutip.qeye(2, dtype="Dense")
        with CoreOptions(default_dtype="CSR", default_dtype_scope="creation"):
            # sparse to dense is cheaper than the reverse.
            assert (sparse + dense).dtype is qutip.core.data.Dense
            assert (dense + dense).dtype is qutip.core.data.Dense

        with CoreOptions(default_dtype="CSR", default_dtype_scope="missing"):
            assert (sparse + dense).dtype is qutip.core.data.CSR
            assert (dense + dense).dtype is qutip.core.data.Dense

        with CoreOptions(default_dtype="CSR", default_dtype_scope="full"):
            assert (sparse + dense).dtype is qutip.core.data.CSR
            assert (dense + dense).dtype is qutip.core.data.CSR

    def test_dtype_group(self):
        qutip.core.data.to.register_group(
            ["test"],
            dense=qutip.core.data.Dia,
            sparse=qutip.core.data.Dia,
            diagonal=qutip.core.data.CSR
        )
        with CoreOptions(default_dtype="test"):
            # sparse to dense is cheaper than the reverse.
            assert qutip.qeye(3).dtype is qutip.core.data.CSR
            assert qutip.basis(2).dtype is qutip.core.data.Dia

        with CoreOptions(default_dtype="test", default_dtype_scope="full"):
            assert (
                qutip.qeye(3).dtype
                in (qutip.core.data.CSR, qutip.core.data.Dia)
            )
            assert (
                (-qutip.qeye(3, dtype="dense")).dtype in
                (qutip.core.data.CSR, qutip.core.data.Dia)
            )


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
