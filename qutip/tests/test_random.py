import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import pytest

from qutip import qeye
from qutip import data as _data
from qutip.random_objects import (
    rand_jacobi_rotation,
    rand_herm,
    rand_unitary,
    rand_dm,
    rand_ket,
    rand_stochastic,
    rand_super,
)


@pytest.fixture(params=[
    24,
    [24],
    [2, 3, 4],
    [[8], [3]]
])
def dimensions(self, request):
    return request.param


@pytest.fixture(params=[
    _data.to.dtypes
])
def dtype(self, request):
    return request.param


def _assert_density(qobj, density):
    N = np.sum(qobj.full() != 0)
    density = N / np.prod(qobj.shape)
    assert density == pytest.approx(density, abs=0.2)


def _assert_metadata(qobj, dims, dtype=None):
    if isinstance(dims, int):
        dims = [dims]
    assert random_qobj.dims == [dims, dims]
    N = np.prod(dims)
    assert random_qobj.shape == (N, N)
    if dtype:
        assert isinstance(random_qobj.data, dtype)


rand_jacobi_rotation


@pytest.mark.repeat(3)
@pytest.mark.parametrize('density', [0.2, 0.8], ids=["sparse", "dense"])
@pytest.mark.parametrize('pos_def', [True, False])
def test_rand_herm(dimensions, density, pos_def, dtype):
    """
    Random Qobjs: Hermitian matrix
    """
    random_qobj = rand_herm(
        dimensions,
        density=density,
        pos_def=pos_def,
        dtype=dtype
    )
    if pos_def:
        assert all(random_qobj.eigenenergies() > -1e14)
    assert random_qobj.isherm
    assert _data.ishem(random_qobj.data)
    _assert_metadata(random_qobj, dimensions, dtype)
    _assert_density(random_qobj, density)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('density', [0.2, 0.8], ids=["sparse", "dense"])
def test_rand_herm_Eigs(dimensions, density):
    """
    Random Qobjs: Hermitian matrix - Eigs given
    """
    N = np.prod(dimensions)
    eigs = np.random.random(N)
    eigs /= np.sum(eigs)
    eigs.sort()
    random_qobj = rand_herm(dimensions, density=density, eigenvalues=eigs)
    np.testing.assert_allclose(random_qobj.eigenenergies(), eigs)
    # verify hermitian
    assert random_qobj.isherm
    assert _data.isherm(random_qobj.data)
    _assert_metadata(random_qobj, dimensions)
    _assert_density(random_qobj, density)


@pytest.mark.repeat(5)
@pytest.mark.parametrize('distribution', ["haar", "exp"])
def test_rand_unitary(dimensions, distribution, dtype):
    """
    Random Qobjs: Tests that unitaries are actually unitary.
    """
    random_qobj = rand_unitary(dimensions, distribution, dtype=dtype)
    I = qeye(5)
    assert random_qobj * random_qobj.dag() == I
    _assert_metadata(random_qobj, dimensions, dtype)


@pytest.mark.repeat(3)
@pytest.mark.parametrize(["distribution", "kw"], [
    pytest.param("ginibre", {"rank": 3}),
    pytest.param("ginibre", {"rank": 1}),
    pytest.param("hs", {"rank": 0}),
    pytest.param("pure", {"rank": 1, "density": 0.5}),
    pytest.param("eigen", {"eigenvalues": True}),
    pytest.param("uniform", {"density": 0.7}),
    pytest.param("uniform", {"density": 0.3}),
])
def test_rand_dm(dimensions, kw, dtype, distribution):
    """
    Random Qobjs: Density matrix
    """
    if "eigenvalues" in kw:
        N = np.prod(dimensions)
        eigs = np.random.random(N)
        eigs /= np.sum(eigs)
        eigs.sort()
        kw["eigenvalues"] = eigs

    random_qobj = rand_dm(
        dimensions,
        dtype=dtype,
        **kw
    )
    assert abs(random_qobj.tr() - 1.0) < 1e-14
    # verify all eigvals are >=0
    assert all(random_qobj.eigenenergies() >= -1e-14)
    # verify hermitian
    assert random_qobj.isherm
    assert _data.isherm(random_qobj.data)

    _assert_metadata(random_qobj, dimensions, dtype)

    eigenvalues = random_qobj.eigenenergies()

    if "rank" in kw:
        kw["rank"] == kw["rank"] or N  # 'hs' is full rank.
        rank = sum([abs(E) >= 1e-10 for E in eigenvalues])
        assert rank == kw["rank"]

    if "eigenvalues" in kw:
        np.testing.assert_allclose(eigenvalues, kw["eigenvalues"])

    if "density" in kw:
        _assert_density(random_qobj, kw["density"])


@pytest.mark.repeat(5)
@pytest.mark.parametrize('kind', ["left", "right"])
def test_rand_stochastic(dimensions, kind, dtype):
    """
    Random Qobjs: Test random stochastic
    """
    random_qobj = rand_stochastic(dimensions, kind=kind, dtype=dtype)
    axis = {"left":0, "right":1}[kind]
    np.testing.assert_allclose(
        np.sum(random_qobj.full(), axis=axis),
        1, atol=1e-14
    )
    _assert_metadata(random_qobj, dimensions, dtype)


@pytest.mark.repeat(5)
@pytest.mark.parametrize('distribution', ["haar", "fill"])
def test_rand_ket(dimensions, func, dtype):
    """
    Random Qobjs: Test random ket type and norm.
    """
    random_qobj = func(dimensions, distribution=distribution, dtype=dtype)

    assert random_qobj.type == 'ket'
    assert abs(random_qobj.norm() - 1) < 1e-14

    if isinstance(dimensions, int):
        dims = [dimensions]
    N = np.prod(dimensions)
    assert random_qobj.dims == [dimensions, [1]]
    assert random_qobj.shape == (N, N)
    assert isinstance(random_qobj.data, dtype)


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
