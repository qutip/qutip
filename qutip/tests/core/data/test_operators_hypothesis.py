import numpy

from hypothesis import given, strategies as st

from qutip.core import data as _data
from qutip.tests import strategies as qst


@given(qst.qobj_np(), qst.qobj_dtypes())
def test_data_create(np_array, dtype):
    data = _data.to(dtype, _data.create(np_array))
    qst.note(data=data, np_array=np_array)
    qst.assert_allclose(data.to_array(), np_array)


@given(qst.qobj_datas())
def test_data_neg_operator(data):
    neg = -data
    qst.note(neg=neg, data=data)
    qst.assert_allclose(neg.to_array(), -data.to_array())


same_shape = st.shared(qst.qobj_shapes())

@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_add_operator(a, b):
    result = a + b
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() + b.to_array()
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_minus_operator(a, b):
    result = a - b
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() - b.to_array()
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_matmul_operator(a, b):
    result = a @ b
    qst.note(result=result, a=a, b=b)
    qst.assert_allclose(result.to_array(), a.to_array() @ b.to_array())


@given(st.complex_numbers(), qst.qobj_datas(shape=same_shape))
def test_data_scalar_multiplication_left_operator(x, a):
    result = x * a
    qst.note(result=result, x=x, a=a)
    qst.assert_allclose(result.to_array(), x * a.to_array())


@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
def test_data_scalar_multiplication_right_operator(a, x):
    result = a * x
    qst.note(result=result, a=a, x=x)
    qst.assert_allclose(result.to_array(), a.to_array() * x)


@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
def test_data_scalar_division_operator(a, x):
    result = a / x
    qst.note(result=result, a=a, x=x)
    qst.assert_allclose(result.to_array(), a.to_array() / x)


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_equality_operator(a, b):
    result = (a == b)
    qst.note(result=result, a=a, b=b)
    assert result == numpy.allclose(
        a.to_array(), b.to_array(), rtol=1e-15
    )
