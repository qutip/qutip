import numpy

from hypothesis import given, strategies as st

from qutip.core import data as _data
from qutip.tests import strategies as qst

same_shape = st.shared(qst.qobj_shapes())
[matmul_shape_a, matmul_shape_b] = qst.qobj_shared_shapes([
    ("n", "k"),
    ("k", "m"),
])


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


@given(qst.qobj_datas())
def test_data_iszero(data):
    result = _data.iszero(data)
    qst.note(result=result, data=data)
    with qst.ignore_arithmetic_warnings():
        np_array = data.to_array()
        expected = numpy.allclose(
            np_array, numpy.zeros_like(np_array), rtol=1e-15
        )
    assert result is expected


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


@given(st.complex_numbers(), qst.qobj_datas(shape=same_shape))
def test_data_scalar_multiplication_left_operator(x, a):
    result = x * a
    qst.note(result=result, x=x, a=a)
    with qst.ignore_arithmetic_warnings():
        expected = x * a.to_array()
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
def test_data_scalar_multiplication_right_operator(a, x):
    result = a * x
    qst.note(result=result, a=a, x=x)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() * x
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(
    qst.qobj_datas(shape=same_shape),
    st.complex_numbers(min_magnitude=1e-12),
)
def test_data_scalar_division_operator(a, x):
    result = a / x
    qst.note(result=result, a=a, x=x)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() / x
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_equality_operator_same_shapes(a, b):
    result = (a == b)
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = numpy.allclose(
            a.to_array(), b.to_array(), rtol=1e-15
        )
    assert result == expected


@given(qst.qobj_datas(), qst.qobj_datas())
def test_data_equality_operator_different_shapes(a, b):
    result = (a == b)
    qst.note(result=result, a=a, b=b)
    if a.shape != b.shape:
        expected = False
    else:
        with qst.ignore_arithmetic_warnings():
            expected = numpy.allclose(
                a.to_array(), b.to_array(), rtol=1e-15
            )
    assert result == expected


@given(
    qst.qobj_datas(shape=matmul_shape_a),
    qst.qobj_datas(shape=matmul_shape_b),
)
def test_data_matmul_operator(a, b):
    result = a @ b
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() @ b.to_array()
    qst.note(expected=expected)
    qst.assert_allclose(result.to_array(), expected)
