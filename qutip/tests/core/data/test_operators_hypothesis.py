import numpy

from hypothesis import given, strategies as st

from qutip.core import data as _data
from qutip.tests import strategies as qst


@given(qst.qobj_np(), qst.qobj_dtypes())
def test_data_create(np_array, dtype):
    if len(np_array.shape) == 1:
        np_array = numpy.atleast_2d(np_array).transpose()
    else:
        np_array = numpy.atleast_2d(np_array)
    data = _data.to(dtype, _data.create(np_array))
    numpy.testing.assert_allclose(
        data.to_array(), numpy.atleast_2d(np_array), rtol=1e-15
    )


@given(qst.qobj_datas())
def test_data_neg_operator(data):
    neg = -data
    numpy.testing.assert_allclose(
        neg.to_array(), -data.to_array(), rtol=1e-15
    )


same_shape = st.shared(qst.qobj_shapes())

@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_add_operator(a, b):
    result = a + b
    numpy.testing.assert_allclose(
        result.to_array(), a.to_array() + b.to_array(), rtol=1e-15
    )


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_minus_operator(a, b):
    result = a - b
    numpy.testing.assert_allclose(
        result.to_array(), a.to_array() - b.to_array(), rtol=1e-15
    )


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_matmul_operator(a, b):
    result = a @ b
    numpy.testing.assert_allclose(
        result.to_array(), a.to_array() @ b.to_array(), rtol=1e-15
    )


@given(st.complex_numbers(), qst.qobj_datas(shape=same_shape))
def test_data_scalar_multiplication_left_operator(x, a):
    result = x * a
    numpy.testing.assert_allclose(
        result.to_array(), x * a.to_array(), rtol=1e-15
    )


@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
def test_data_scalar_multiplication_right_operator(a, x):
    result = a * x
    numpy.testing.assert_allclose(
        result.to_array(), a.to_array() * x, rtol=1e-15
    )

@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
def test_data_scalar_division_operator(a, x):
    result = a / x
    numpy.testing.assert_allclose(
        result.to_array(), a.to_array() / x, rtol=1e-15
    )


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_equality_operator(a, b):
    result = (a == b)
    assert result == numpy.allclose(
        a.to_array(), b.to_array(), rtol=1e-15
    )
