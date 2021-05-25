import numpy as np
import pytest
from scipy import sparse
from qutip.core.data import to, create, CSR, Dense, dense, csr


@pytest.mark.parametrize(['base', 'dtype'], [
    pytest.param(dense.zeros(2, 2), Dense, id='data.Dense'),
    pytest.param(csr.zeros(2, 2), CSR, id='data.CSR'),
    pytest.param(np.zeros((10, 10), dtype=np.complex128), Dense, id='array'),
    pytest.param(sparse.eye(10, dtype=np.complex128, format='csr'), CSR,
                 id='sparse'),
    pytest.param(np.zeros((10, 10), dtype=np.int32), Dense, id='array'),
    pytest.param(sparse.eye(10, dtype=np.float, format='csr'), CSR,
                 id='sparse'),
])
def test_create(base, dtype):
    # The test of exactitude is done in test_csr, test_dense.
    created = create(base)
    assert isinstance(created, dtype)


@pytest.mark.parametrize(['from_', 'base'], [
    pytest.param('dense', dense.zeros(2, 2), id='from Dense str'),
    pytest.param('Dense', dense.zeros(2, 2), id='from Dense STR'),
    pytest.param(Dense, dense.zeros(2, 2), id='from Dense type'),
    pytest.param('csr', csr.zeros(2, 2), id='from CSR str'),
    pytest.param('CSR', csr.zeros(2, 2), id='from CSR STR'),
    pytest.param(CSR, csr.zeros(2, 2), id='from CSR type'),
])
@pytest.mark.parametrize(['to_', 'dtype'], [
    pytest.param('dense', Dense, id='to Dense str'),
    pytest.param('Dense', Dense, id='to Dense STR'),
    pytest.param(Dense, Dense, id='to Dense type'),
    pytest.param('csr', CSR, id='to CSR str'),
    pytest.param('CSR', CSR, id='to CSR STR'),
    pytest.param(CSR, CSR, id='to CSR type'),
])
def test_converters(from_, base, to_, dtype):
    converter = to[to_, from_]
    assert isinstance(converter(base), dtype)
    converter = to[to_]
    assert isinstance(converter(base), dtype)
    assert isinstance(to(to_, base), dtype)


def test_parse():
    assert to.parse("dense") is Dense
    assert to.parse("Dense") is Dense
    assert to.parse("DeNsE") is Dense

    assert to.parse("csr") is CSR
    assert to.parse("CSR") is CSR
    assert to.parse("cSr") is CSR

    with pytest.raises(TypeError) as exc:
        to.parse(5)
    assert str(exc.value) == (
        "Invalid dtype is neither a type nor a type name: 5"
    )

    with pytest.raises(ValueError) as exc:
        to.parse("__this_is_not_a_known_type_name__")
    assert str(exc.value) == (
        "Type name is not known to the data-layer: "
        "'__this_is_not_a_known_type_name__'"
    )

    with pytest.raises(ValueError) as exc:
        to.parse(object)
    assert str(exc.value) == "Type is not a data-layer type: <class 'object'>"
