import numpy as np
from numpy.random import SeedSequence, default_rng
import scipy.sparse as sp
import scipy.linalg as la
import pytest

from qutip import qeye, num, to_kraus, kraus_to_choi, CoreOptions, Qobj
from qutip import data as _data
from qutip.core.dimensions import Space
from qutip.random_objects import (
    rand_herm,
    rand_unitary,
    rand_dm,
    rand_ket,
    rand_stochastic,
    rand_super,
    rand_super_bcsz,
    rand_kraus_map,
)


@pytest.fixture(params=[
    12,
    [8],
    [2, 2, 3],
    [[2], [2]],
    Space(3),
], ids=["int", "list", "tensor", "super", "Space"])
def dimensions(request):
    return request.param


@pytest.fixture(
    params=list(_data.to.dtypes),
    ids=lambda dtype: str(dtype)[:-2].split(".")[-1]
)
def dtype(request):
    return request.param


def _assert_density(qobj, density):
    N = np.sum(qobj.full() != 0)
    density = N / np.prod(qobj.shape)
    assert density == pytest.approx(density, abs=0.2)


def _assert_metadata(random_qobj, dims, dtype=None, super=False, ket=False):
    if isinstance(dims, int):
        dims = [dims]
    elif isinstance(dims, Space):
        dims = dims.as_list()
    N = np.prod(dims)
    if super and not isinstance(dims[0], list):
        target_dims_0 = [dims, dims]
        shape0 = N**2
    else:
        target_dims_0 = dims
        shape0 = N

    if ket:
        target_dims_1 = [1]
        shape1 = 1
    else:
        target_dims_1 = target_dims_0
        shape1 = shape0

    assert random_qobj.dims[0] == target_dims_0
    assert random_qobj.dims[1] == target_dims_1
    assert random_qobj.shape == (shape0, shape1)

    if dtype:
        assert isinstance(random_qobj.data, dtype)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('density', [0.2, 0.8], ids=["sparse", "dense"])
@pytest.mark.parametrize('distribution', ["fill", "pos_def"])
def test_rand_herm(dimensions, density, distribution, dtype):
    """
    Random Qobjs: Hermitian matrix
    """
    random_qobj = rand_herm(
        dimensions,
        density=density,
        distribution=distribution,
        dtype=dtype
    )
    if distribution == "pos_def":
        assert all(random_qobj.eigenenergies() > -1e14)
    assert random_qobj.isherm
    assert _data.isherm(random_qobj.data)
    _assert_metadata(random_qobj, dimensions, dtype)
    _assert_density(random_qobj, density)


@pytest.mark.repeat(3)
@pytest.mark.parametrize('density', [0.2, 0.8], ids=["sparse", "dense"])
def test_rand_herm_Eigs(dimensions, density):
    """
    Random Qobjs: Hermitian matrix - Eigs given
    """
    if isinstance(dimensions, Space):
        N = dimensions.size
    else:
        N = np.prod(dimensions)
    eigs = np.random.random(N)
    eigs /= np.sum(eigs)
    eigs.sort()
    random_qobj = rand_herm(dimensions, density, "eigen", eigenvalues=eigs)
    np.testing.assert_allclose(random_qobj.eigenenergies(), eigs)
    # verify hermitian
    assert random_qobj.isherm
    assert _data.isherm(random_qobj.data)
    _assert_metadata(random_qobj, dimensions)
    _assert_density(random_qobj, density)


@pytest.mark.repeat(5)
@pytest.mark.parametrize('distribution', ["haar", "exp"])
@pytest.mark.parametrize('density', [0.2, 0.8])
def test_rand_unitary(dimensions, distribution, density, dtype):
    """
    Random Qobjs: Tests that unitaries are actually unitary.
    """
    N = np.prod(dimensions)
    random_qobj = rand_unitary(
        dimensions, distribution=distribution,
        density=density, dtype=dtype
    )
    I = qeye(dimensions)
    assert random_qobj * random_qobj.dag() == I
    _assert_metadata(random_qobj, dimensions, dtype)
    if distribution == "exp":
        _assert_density(random_qobj, density)


@pytest.mark.repeat(3)
@pytest.mark.parametrize(["distribution", "kw"], [
    pytest.param("ginibre", {"rank": 3}),
    pytest.param("ginibre", {"rank": 1}),
    pytest.param("hs", {"rank": 0}),
    pytest.param("pure", {"rank": 1, "density": 0.5}),
    pytest.param("eigen", {"eigenvalues": True}),
    pytest.param("herm", {"density": 0.7}),
    pytest.param("herm", {"density": 0.3}),
])
def test_rand_dm(dimensions, kw, dtype, distribution):
    """
    Random Qobjs: Density matrix
    """
    if isinstance(dimensions, Space):
        N = dimensions.size
    else:
        N = np.prod(dimensions)

    if "eigenvalues" in kw:
        eigs = np.random.random(N)
        eigs /= np.sum(eigs)
        eigs.sort()
        kw["eigenvalues"] = eigs

    random_qobj = rand_dm(
        dimensions,
        distribution=distribution,
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
        print(kw["rank"], N, kw["rank"] or N)
        desired_rank = kw["rank"] or N  # 'hs' is full rank.
        rank = sum([abs(E) >= 1e-10 for E in eigenvalues])
        assert rank == desired_rank

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
def test_rand_ket(dimensions, distribution, dtype):
    """
    Random Qobjs: Test random ket type and norm.
    """
    random_qobj = rand_ket(dimensions, distribution=distribution, dtype=dtype)

    target_type = "ket"
    if isinstance(dimensions, list) and isinstance(dimensions[0], list):
        target_type = "operator-ket"
    assert random_qobj.type == target_type
    assert abs(random_qobj.norm() - 1) < 1e-14

    if isinstance(dimensions, int):
        dims = [dimensions]
    N = np.prod(dimensions)
    _assert_metadata(random_qobj, dimensions, dtype, ket=True)


@pytest.mark.repeat(2)
@pytest.mark.parametrize('superrep', ["choi", "super"])
def test_rand_super(dimensions, dtype, superrep):
    """
    Random Qobjs: Super operator.
    """
    random_qobj = rand_super(dimensions, dtype=dtype, superrep=superrep)
    assert random_qobj.issuper
    with CoreOptions(atol=2e-9):
        assert random_qobj.iscptp
    assert random_qobj.superrep == superrep
    _assert_metadata(random_qobj, dimensions, dtype, super=True)


@pytest.mark.repeat(2)
@pytest.mark.parametrize('rank', [None, 4])
@pytest.mark.parametrize('superrep', ["choi", "super"])
def test_rand_super_bcsz(dimensions, dtype, rank, superrep):
    """
    Random Qobjs: Tests that BCSZ-random superoperators are CPTP.
    """

    random_qobj = rand_super_bcsz(dimensions, rank=rank,
                                  dtype=dtype, superrep=superrep)
    if isinstance(dimensions, Space):
        dimensions = dimensions.as_list()
    assert random_qobj.issuper
    with CoreOptions(atol=1e-9):
        assert random_qobj.iscptp
    assert random_qobj.superrep == superrep
    _assert_metadata(random_qobj, dimensions, dtype, super=True)
    if (
        not rank
        and isinstance(dimensions, list)
        and isinstance(dimensions[0], list)
    ):
        # dimensions = [[a], [a]], qobj.dims = [[[a], [a]], [[a], [a]]]
        rank = np.prod(dimensions)
    elif not rank:
        # dimensions = [a], qobj.dims = [[[a], [a]], [[a], [a]]]
        rank = np.prod(dimensions)**2
    rank = rank or N
    obtained_rank = len(to_kraus(random_qobj, tol=1e-13))
    assert obtained_rank == rank


@pytest.mark.parametrize("function", [
    pytest.param(rand_herm, id="rand_herm"),
    pytest.param(rand_unitary, id="rand_unitary"),
    pytest.param(rand_dm, id="rand_dm"),
    pytest.param(rand_ket, id="rand_ket"),
    pytest.param(rand_stochastic, id="rand_stochastic"),
    pytest.param(rand_super, id="rand_super"),
    pytest.param(rand_super_bcsz, id="rand_super_bcsz"),
])
@pytest.mark.parametrize("seed", [
    pytest.param(lambda : 123, id="int"),
    pytest.param(lambda : SeedSequence(123), id="SeedSequence"),
    pytest.param(lambda : default_rng(123), id="Generator")
])
def test_random_seeds(function, seed):
    """
    Random Qobjs: Random number generator seed
    """
    U0 = function(5, seed=seed())
    U1 = function(5, seed=None)
    U2 = function(5, seed=seed())
    assert U0 != U1
    assert U0 == U2


def test_kraus_map(dimensions, dtype):
    if isinstance(dimensions, list) and isinstance(dimensions[0], list):
        # Each element of a kraus map cannot be a super operators
        with pytest.raises(TypeError) as err:
            kmap = rand_kraus_map(dimensions, dtype=dtype)
        assert "super operator" in str(err.value)
    else:
        kmap = rand_kraus_map(dimensions, dtype=dtype)
        _assert_metadata(kmap[0], dimensions, dtype)
        with CoreOptions(atol=1e-9):
            assert kraus_to_choi(kmap).iscptp


dtype_names = list(_data.to._str2type.keys()) + list(_data.to.dtypes)
dtype_types = list(_data.to._str2type.values()) + list(_data.to.dtypes)
dtype_combinations = list(zip(dtype_names, dtype_types))
@pytest.mark.parametrize(['alias', 'dtype'], dtype_combinations,
                         ids=[str(dtype) for dtype in dtype_names])
@pytest.mark.parametrize('func', [
    rand_herm,
    rand_unitary,
    rand_dm,
    rand_ket,
    rand_stochastic,
    rand_super,
    rand_super_bcsz,
    rand_kraus_map,
])
def test_random_dtype(func, alias, dtype):
    with CoreOptions(default_dtype=alias):
        object = func(2)
        if isinstance(object, Qobj):
            assert isinstance(object.data, dtype)
        else:
            for obj in object:
                assert isinstance(obj.data, dtype)
