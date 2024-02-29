import pytest
import numpy
import qutip
from qutip.core import data as _data
from qutip import settings


@pytest.mark.parametrize('type_left', _data.to.dtypes)
@pytest.mark.parametrize('type_right', _data.to.dtypes)
@pytest.mark.parametrize(['operator', 'dispatch'], [
    pytest.param(lambda left, right: left + right, _data.add, id="add"),
    pytest.param(lambda left, right: left - right, _data.sub, id="sub"),
    pytest.param(lambda left, right: left @ right, _data.matmul, id="matmul"),
])
def test_data_binary_operator(type_left, type_right, operator, dispatch):
    left = qutip.rand_dm(5, dtype=type_left).data
    right = qutip.rand_dm(5, dtype=type_right).data
    numpy.testing.assert_allclose(
        operator(left, right).to_array(),
        dispatch(left, right).to_array(),
        rtol=1e-15
    )


@pytest.mark.parametrize('type_', _data.to.dtypes)
@pytest.mark.parametrize(['operator', 'dispatch'], [
    pytest.param(lambda data, number: data * number, _data.mul, id="mul"),
    pytest.param(lambda data, number: number * data, _data.mul, id="rmul"),
    pytest.param(lambda data, number: data / number,
                 lambda data, number: _data.mul(data, 1/number), id="div"),
])
def test_data_scalar_operator(type_, operator, dispatch):
    data = qutip.qeye(2, dtype=type_).data
    number = 3
    numpy.testing.assert_allclose(
        operator(data, number).to_array(),
        dispatch(data, number).to_array(),
        rtol=1e-15
    )


@pytest.mark.parametrize('type_', _data.to.dtypes)
def test_data_neg_operator(type_):
    data = qutip.qeye(2, dtype=type_).data
    numpy.testing.assert_allclose(
        (-data).to_array(), -data.to_array(), rtol=1e-15
    )


@pytest.mark.parametrize('type_left', _data.to.dtypes)
@pytest.mark.parametrize('type_right', _data.to.dtypes)
def test_data_eq_operator(type_left, type_right):
    mat = qutip.rand_dm(5)
    noise = qutip.rand_dm(5) * settings.core["atol"] / 10

    left = mat.to(type_left).data
    right = mat.to(type_right).data
    assert left == right
    right = (mat + noise).to(type_right).data
    assert left == right

    right = (mat + noise * 100).to(type_right).data
    assert left != right

    right = qutip.operator_to_vector(mat).to(type_right).data
    assert left != right

    assert left != mat
    assert numpy.all(left != mat.full())
