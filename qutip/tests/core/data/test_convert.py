import numpy as np
import pytest

from qutip.core.data import to
from qutip import rand_herm, Qobj

# Set up some fixtures for automatic parametrisation.

@pytest.mark.parametrize(["shape"], [
    pytest.param((1, 5), id='ket'),
    pytest.param((5, 1), id='bra'),
    pytest.param((5, 5), id='square'),
    pytest.param((2, 4), id='wide'),
    pytest.param((4, 2), id='tall'),
])
@pytest.mark.parametrize(['from_type'],
    [pytest.param(dtype, id=dtype.__name__) for dtype in to.dtypes])
@pytest.mark.parametrize(['to_type'],
    [pytest.param(dtype, id=dtype.__name__) for dtype in to.dtypes])
def test_convertion(shape, from_type, to_type):
    # There is no creation function for arbitrary data types, so 2 conversions
    # will be done. But this test will ensure all convertions are tested.
    obj = Qobj(np.random.rand(shape[0]*shape[1]).reshape(shape)).to(from_type)
    obj_changed = to(to_type, obj.data)
    assert isinstance(obj_changed, to_type)
    np.testing.assert_allclose(obj.full(), obj_changed.to_array())
