import numpy as np
import pytest
from scipy import sparse
from qutip import data
from .test_mathematics import UnaryOpMixin


def test_init_empty_data():
    shape = (3, 3)
    base_data = data.Data(shape)
    assert base_data.shape[0] == shape[0]
    assert base_data.shape[1] == shape[1]


@pytest.mark.parametrize(['base', 'dtype'], [
    pytest.param(data.dense.zeros(2, 2), data.Dense, id='data.Dense'),
    pytest.param(data.csr.zeros(2, 2), data.CSR, id='data.CSR'),
    pytest.param(data.dia.zeros(2, 2), data.Dia, id='data.Dia'),
    pytest.param(np.zeros((10, 10), dtype=np.complex128), data.Dense,
                 id='array'),
    pytest.param(sparse.eye(10, dtype=np.complex128, format='csr'), data.CSR,
                 id='sparse'),
    pytest.param(sparse.eye(10, dtype=np.complex128, format='dia'), data.Dia,
                 id='diag'),
    pytest.param(np.zeros((10, 10), dtype=np.int32), data.Dense, id='array'),
    pytest.param(sparse.eye(10, dtype=float, format='dia'), data.Dia,
                 id='diag'),
    pytest.param(sparse.eye(10, dtype=float, format='csr'), data.CSR,
                 id='sparse'),
])
def test_create(base, dtype):
    # The test of exactitude is done in test_csr, test_dense.
    created = data.create(base)
    assert isinstance(created, dtype)


@pytest.mark.parametrize(['from_', 'base'], [
    pytest.param('dense', data.dense.zeros(2, 2), id='from Dense str'),
    pytest.param('Dense', data.dense.zeros(2, 2), id='from Dense STR'),
    pytest.param(data.Dense, data.dense.zeros(2, 2), id='from Dense type'),
    pytest.param('csr', data.csr.zeros(2, 2), id='from CSR str'),
    pytest.param('CSR', data.csr.zeros(2, 2), id='from CSR STR'),
    pytest.param(data.CSR, data.csr.zeros(2, 2), id='from CSR type'),
    pytest.param('Dia', data.dia.zeros(2, 2), id='from Dia STR'),
    pytest.param('dia', data.dia.zeros(2, 2), id='from Dia str'),
    pytest.param(data.Dia, data.dia.zeros(2, 2), id='from Dia type'),
])
@pytest.mark.parametrize(['to_', 'dtype'], [
    pytest.param('dense', data.Dense, id='to Dense str'),
    pytest.param('Dense', data.Dense, id='to Dense STR'),
    pytest.param(data.Dense, data.Dense, id='to Dense type'),
    pytest.param('csr', data.CSR, id='to CSR str'),
    pytest.param('CSR', data.CSR, id='to CSR STR'),
    pytest.param(data.CSR, data.CSR, id='to CSR type'),
    pytest.param('Dia', data.Dia, id='to Dia STR'),
    pytest.param('dia', data.Dia, id='to Dia str'),
    pytest.param(data.Dia, data.Dia, id='to Dia type'),
])
def test_converters(from_, base, to_, dtype):
    converter = data.to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = data.to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(data.to(to_, base), dtype)


dtype_names = list(data.to._str2type.keys()) + list(data.to.dtypes)
dtype_types = list(data.to._str2type.values()) + list(data.to.dtypes)
dtype_combinations = list(zip(dtype_names, dtype_types))
@pytest.mark.parametrize(['input', 'type_'], dtype_combinations,
                         ids=[str(dtype) for dtype in dtype_names])
def test_parse(input, type_):
    assert data.to.parse(input) is type_


@pytest.mark.parametrize(['input', 'error', 'msg'], [
    pytest.param(5, TypeError,
                  "Invalid dtype is neither a type nor a type name: 5",
                 id="wrong type"),
    pytest.param("__this_is_not_a_known_type_name__", ValueError,
                  "Type name is not known to the data-layer: "
                  "'__this_is_not_a_known_type_name__'",
                 id="not alias"),
    pytest.param(object, ValueError,
                  "Type is not a data-layer type: <class 'object'>",
                 id="not Data"),
])
def test_parse_error(input, error, msg):
    with pytest.raises(error) as exc:
        data.to.parse(input)
    assert str(exc.value) == msg


class TestConvert(UnaryOpMixin):
    def op_numpy(self, mat):
        return mat

    specialisations = [
        pytest.param(data.dense.from_csr, data.CSR, data.Dense),
        pytest.param(data.dense.from_dia, data.Dia, data.Dense),
        pytest.param(data.csr.from_dense, data.Dense, data.CSR),
        pytest.param(data.csr.from_dia, data.Dia, data.CSR),
        pytest.param(data.dia.from_dense, data.Dense, data.Dia),
        pytest.param(data.dia.from_csr, data.CSR, data.Dia),
    ]
