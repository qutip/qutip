import pytest
import numpy
import qutip
from qutip.core import data as _data


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
def test_data_unitary_operator(type_, operator, dispatch):
    data = qutip.qeye(2, dtype=type_).data
    number = 3
    numpy.testing.assert_allclose(
        operator(data, number).to_array(),
        dispatch(data, number).to_array(),
        rtol=1e-15
    )


def imul(data, number):
    data *= number


def idiv(data, number):
    data /= number


@pytest.mark.parametrize('type_', _data.to.dtypes)
@pytest.mark.parametrize(['operator', 'dispatch'], [
    pytest.param(imul, _data.mul, id="imul"),
    pytest.param(idiv, lambda data, num: _data.mul(data, 1/num), id="idiv"),
])
def test_data_inplace_operator(type_, operator, dispatch):
    data = qutip.qeye(2, dtype=type_).data
    data_copy = data.copy()
    number = 3
    operator(data, number)
    numpy.testing.assert_allclose(
        data.to_array(),
        dispatch(data_copy, number).to_array(),
        rtol=1e-15
    )


@pytest.mark.parametrize('type_', _data.to.dtypes)
def test_data_neg_operator(type_):
    data = qutip.qeye(2, dtype=type_).data
    numpy.testing.assert_allclose(
        (-data).to_array(), -data.to_array(), rtol=1e-15
    )
