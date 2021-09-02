import numpy as np
from qutip import (
    rand_ket, rand_dm, rand_herm, rand_unitary, rand_ket_haar, rand_dm_hs,
    rand_super, rand_unitary_haar, rand_dm_ginibre, rand_super_bcsz, qeye,
    rand_stochastic,
)
import pytest


@pytest.mark.repeat(5)
@pytest.mark.parametrize('func', [rand_unitary, rand_unitary_haar])
def test_rand_unitary(func):
    """
    Random Qobjs: Tests that unitaries are actually unitary.
    """
    random_qobj = func(5)
    I = qeye(5)
    assert random_qobj * random_qobj.dag() == I


@pytest.mark.repeat(5)
@pytest.mark.parametrize('density', [0.2, 0.8], ids=["sparse", "dense"])
@pytest.mark.parametrize('pos_def', [True, False])
def test_rand_herm(density, pos_def):
    """
    Random Qobjs: Hermitian matrix
    """
    random_qobj = rand_herm(5, density=density, pos_def=pos_def)
    if pos_def:
        assert all(random_qobj.eigenenergies() > -1e14)
    assert random_qobj.isherm


@pytest.mark.repeat(5)
def test_rand_herm_Eigs():
    """
    Random Qobjs: Hermitian matrix - Eigs given
    """
    eigs = np.random.random(5)
    eigs /= np.sum(eigs)
    eigs.sort()
    random_qobj = rand_herm(eigs)
    np.testing.assert_allclose(random_qobj.eigenenergies(), eigs)
    # verify hermitian
    assert random_qobj.isherm


@pytest.mark.repeat(5)
@pytest.mark.parametrize('func', [rand_dm, rand_dm_hs])
def test_rand_dm(func):
    """
    Random Qobjs: Density matrix
    """
    random_qobj = func(5)
    assert abs(random_qobj.tr() - 1.0) < 1e-14
    # verify all eigvals are >=0
    assert all(random_qobj.eigenenergies() >= -1e-14)
    # verify hermitian
    assert random_qobj.isherm


@pytest.mark.repeat(5)
def test_rand_dm_Eigs():
    """
    Random Qobjs: Density matrix - Eigs given
    """
    eigs = np.random.random(5)
    eigs /= np.sum(eigs)
    eigs.sort()
    random_qobj = rand_dm(eigs)
    assert abs(random_qobj.tr() - 1.0) < 1e-14
    np.testing.assert_allclose(random_qobj.eigenenergies(), eigs)
    # verify hermitian
    assert random_qobj.isherm


@pytest.mark.repeat(5)
def test_rand_dm_ginibre_rank():
    """
    Random Qobjs: Ginibre-random density ops have correct rank.
    """
    random_qobj = rand_dm_ginibre(5, rank=3)
    rank = sum([abs(E) >= 1e-10 for E in random_qobj.eigenenergies()])
    assert rank == 3


@pytest.mark.repeat(5)
@pytest.mark.parametrize('kind', ["left", "right"])
def test_rand_stochastic(kind):
    """
    Random Qobjs: Test random stochastic
    """
    random_qobj = rand_stochastic(5, kind=kind)
    axis = {"left":0, "right":1}[kind]
    np.testing.assert_allclose(np.sum(random_qobj.full(), axis=axis), 1,
                               atol=1e-14)


@pytest.mark.repeat(5)
@pytest.mark.parametrize('func', [rand_ket, rand_ket_haar])
def test_rand_ket(func):
    """
    Random Qobjs: Test random ket type and norm.
    """
    random_qobj = func(5)
    assert random_qobj.type == 'ket'
    assert abs(random_qobj.norm() - 1) < 1e-14


@pytest.mark.repeat(5)
def test_rand_super():
    """
    Random Qobjs: Super operator.
    """
    random_qobj = rand_super(5)
    assert random_qobj.issuper


@pytest.mark.repeat(5)
def test_rand_super_bcsz_cptp():
    """
    Random Qobjs: Tests that BCSZ-random superoperators are CPTP.
    """
    random_qobj = rand_super_bcsz(5)
    assert random_qobj.issuper
    assert random_qobj.iscptp


@pytest.mark.parametrize('func', [
    rand_unitary, rand_unitary_haar, rand_herm,
    rand_dm, rand_dm_hs, rand_dm_ginibre,
    rand_ket, rand_ket_haar,
    rand_super, rand_super_bcsz
])
def test_random_seeds(func):
    """
    Random Qobjs: Random number generator seed
    """
    seed = 12345
    U0 = func(5, seed=seed)
    U1 = func(5, seed=None)
    U2 = func(5, seed=seed)
    assert U0 != U1
    assert U0 == U2


@pytest.mark.parametrize('func', [rand_ket, rand_ket_haar])
@pytest.mark.parametrize(('args', 'kwargs', 'dims'), [
    pytest.param((6,), {}, [[6], [1]], id="N"),
    pytest.param((), {'dims': [[2, 3], [1, 1]]}, [[2, 3], [1, 1]], id="dims"),
    pytest.param((6,), {'dims': [[2, 3], [1, 1]]}, [[2, 3], [1, 1]],
                 id="both"),
])
def test_rand_vector_dims(func, args, kwargs, dims):
    shape = np.prod(dims[0]), np.prod(dims[1])
    output = func(*args, **kwargs)
    assert output.shape == shape
    assert output.dims == dims


@pytest.mark.parametrize('func', [rand_ket, rand_ket_haar])
def test_rand_ket_raises_if_no_args(func):
    with pytest.raises(ValueError):
        func()


@pytest.mark.parametrize('func', [
    rand_unitary, rand_herm, rand_dm, rand_unitary_haar, rand_dm_ginibre,
    rand_dm_hs, rand_stochastic,
])
@pytest.mark.parametrize(('args', 'kwargs', 'dims'), [
    pytest.param((6,), {}, [[6], [6]], id="N"),
    pytest.param((6,), {'dims': [[2, 3], [2, 3]]}, [[2, 3], [2, 3]],
                 id="both"),
])
def test_rand_oper_dims(func, args, kwargs, dims):
    shape = np.prod(dims[0]), np.prod(dims[1])
    output = func(*args, **kwargs)
    assert output.shape == shape
    assert output.dims == dims


_super_dims = [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]


@pytest.mark.parametrize('func', [rand_super, rand_super_bcsz])
@pytest.mark.parametrize(('args', 'kwargs', 'dims'), [
    pytest.param((6,), {}, [[[6]]*2]*2, id="N"),
    pytest.param((6,), {'dims': _super_dims}, _super_dims,
                 id="both"),
])
def test_rand_super_dims(func, args, kwargs, dims):
    shape = np.prod(dims[0]), np.prod(dims[1])
    output = func(*args, **kwargs)
    assert output.shape == shape
    assert output.dims == dims
