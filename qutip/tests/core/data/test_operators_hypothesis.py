import numpy

from hypothesis import given, strategies as st

import qutip
from qutip.core import data as _data
from qutip.tests import strategies as qst

same_shape = st.shared(qst.qobj_shapes())
matmul_shape_a, matmul_shape_b = qst.qobj_shared_shapes([
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
    result = _data.iszero(data, tol=1e-15)
    qst.note(result=result, data=data)
    with qst.ignore_arithmetic_warnings():
        np_array = data.to_array()
        expected = numpy.allclose(
            np_array, numpy.zeros_like(np_array), atol=1e-15, rtol=0,
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
@qst.raises_when(
    ValueError,
    "Multiplying CSR matrix by non-finite value {x!r}"
    " would generate a dense matrix of NaNs.",
    when=lambda a, x: not numpy.isfinite(x) and isinstance(a, _data.CSR),
)
def test_data_scalar_multiplication_left_operator(x, a):
    result = x * a
    qst.note(result=result, x=x, a=a)
    with qst.ignore_arithmetic_warnings():
        expected = x * a.to_array()
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), st.complex_numbers())
@qst.raises_when(
    ValueError,
    "Multiplying CSR matrix by non-finite value {x!r}"
    " would generate a dense matrix of NaNs.",
    when=lambda a, x: not numpy.isfinite(x) and isinstance(a, _data.CSR),
)
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
@qst.raises_when(
    ValueError,
    "Multiplying CSR matrix by non-finite value {inv_x!r}"
    " would generate a dense matrix of NaNs.",
    when=lambda a, x: not numpy.isfinite(1 / x) and isinstance(a, _data.CSR),
    format_args=lambda a, x: {"inv_x": 1 / x},
)
def test_data_scalar_division_operator(a, x):
    result = a / x
    qst.note(result=result, a=a, x=x)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() * (1 / x)
    qst.assert_allclose(result.to_array(), expected, treat_inf_as_nan=True)


@given(qst.qobj_datas(shape=same_shape), qst.qobj_datas(shape=same_shape))
def test_data_equality_operator_same_shapes(a, b):
    with qutip.CoreOptions(atol=1e-15):
        result = (a == b)
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = numpy.allclose(
            numpy.zeros(a.shape),
            a.to_array() - b.to_array(),
            atol=1e-15, rtol=0,
        )
    assert result == expected


@given(qst.qobj_datas(), qst.qobj_datas())
def test_data_equality_operator_different_shapes(a, b):
    with qutip.CoreOptions(atol=1e-15):
        result = (a == b)
    qst.note(result=result, a=a, b=b)
    if a.shape != b.shape:
        assert result is False
    else:
        with qst.ignore_arithmetic_warnings():
            expected = numpy.allclose(
                numpy.zeros(a.shape),
                a.to_array() - b.to_array(),
                atol=1e-15
            )
        assert result == expected


@given(
    qst.qobj_datas_matmul(shape=matmul_shape_a),
    qst.qobj_datas_matmul(shape=matmul_shape_b),
)
def test_data_matmul_operator(a, b):
    result = a @ b
    qst.note(result=result, a=a, b=b)
    with qst.ignore_arithmetic_warnings():
        expected = a.to_array() @ b.to_array()
    qst.assert_allclose(
        result.to_array(), expected,
        atol=1e-10, rtol=1e-10, treat_inf_as_nan=True,
    )


@given(qst.qobj_datas())
@qst.raises_when(
    ValueError,
    "matrix shape {shape} is not square.",
    when=lambda data: data.shape[0] != data.shape[1],
    format_args=lambda data: {"shape": data.shape},
)
def test_trace(data):
    result = data.trace()
    with qst.ignore_arithmetic_warnings():
        expected = data.to_array().trace()
    qst.note(result=result, data=data)
    qst.assert_allclose(result, expected)


@given(qst.qobj_datas())
def test_adjoint(data):
    result = data.adjoint()
    qst.note(result=result, data=data)
    qst.assert_allclose(result.to_array(), data.to_array().T.conj())


@given(qst.qobj_datas())
def test_conj(data):
    result = data.conj()
    qst.note(result=result, data=data)
    qst.assert_allclose(result.to_array(), data.to_array().conj())


@given(qst.qobj_datas())
def test_transpose(data):
    result = data.transpose()
    qst.note(result=result, data=data)
    qst.assert_allclose(result.to_array(), data.to_array().T)


@given(qst.qobj_datas())
def test_copy(data):
    result = data.copy()
    qst.note(result=result, data=data)
    qst.assert_allclose(result.to_array(), data.to_array())
