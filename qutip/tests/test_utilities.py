import numpy as np
from qutip import convert_unit, clebsch, n_thermal
import qutip.utilities as utils
from qutip.utilities import available_cpu_count
from functools import partial
import pytest


@pytest.mark.parametrize(['w', 'w_th', 'expected'], [
    pytest.param(np.log(2), 1, 1, id='log(2)'),
    pytest.param(np.log(2)*5, 5, 1, id='5*log(2)'),
    pytest.param(0, 1, 0, id="0_energy"),
    pytest.param(1, -1, 0, id="neg_temp"),
    pytest.param(np.array([np.log(2), np.log(3), np.log(4)]), 1,
                 np.array([1, 1/2, 1/3]), id="array"),
])
def test_n_thermal(w, w_th, expected):
    np.testing.assert_allclose(n_thermal(w, w_th), expected)


def _get_converter(orig, target):
    """get funtion 'convert_{}_to_{}' when available for coverage """
    try:
        func = getattr(utils, f'convert_{orig}_to_{target}')
    except AttributeError:
        func = partial(convert_unit, orig=orig, to=target)
    return func


@pytest.mark.parametrize('orig', ["J", "eV", "meV", "GHz", "mK"])
@pytest.mark.parametrize('target', ["J", "eV", "meV", "GHz", "mK"])
def test_unit_conversions(orig, target):
    T = np.random.rand() * 100.0
    T_converted = convert_unit(T, orig=orig, to=target)
    T_back = convert_unit(T_converted, orig=target, to=orig)

    assert T == pytest.approx(T_back)

    T_converted = _get_converter(orig=orig, target=target)(T)
    T_back = _get_converter(orig=target, target=orig)(T_converted)

    assert T == pytest.approx(T_back)


@pytest.mark.parametrize('orig', ["J", "eV", "meV", "GHz", "mK"])
@pytest.mark.parametrize('middle', ["J", "eV", "meV", "GHz", "mK"])
@pytest.mark.parametrize('target', ["J", "eV", "meV", "GHz", "mK"])
def test_unit_conversions_loop(orig, middle, target):
    T = np.random.rand() * 100.0
    T_middle = convert_unit(T, orig=orig, to=middle)
    T_converted = convert_unit(T_middle, orig=middle, to=target)
    T_back = convert_unit(T_converted, orig=target, to=orig)

    assert T == pytest.approx(T_back)


def test_unit_conversions_bad_unit():
    with pytest.raises(TypeError):
        convert_unit(10, orig="bad", to="J")
    with pytest.raises(TypeError):
        convert_unit(10, orig="J", to="bad")


@pytest.mark.parametrize('j1', [0.5, 1.0, 1.5, 2.0, 5, 7.5, 10, 12.5])
@pytest.mark.parametrize('j2', [0.5, 1.0, 1.5, 2.0, 5, 7.5, 10, 12.5])
def test_unit_clebsch_delta_j(j1, j2):
    """sum_m1 sum_m2 C(j1,j2,j3,m1,m2,m3) * C(j1,j2,j3',m1,m2,m3') =
    delta j3,j3' delta m3,m3'"""
    for _ in range(10):
        j3 = np.random.choice(np.arange(abs(j1-j2), j1+j2+1))
        j3p = np.random.choice(np.arange(abs(j1-j2), j1+j2+1))
        m3 = np.random.choice(np.arange(-j3, j3+1))
        m3p = np.random.choice(np.arange(-j3p, j3p+1))

        sum_match = 0
        sum_differ = 0
        for m1 in np.arange(-j1, j1+1):
            for m2 in np.arange(-j2, j2+1):
                c1 = clebsch(j1, j2, j3, m1, m2, m3)
                c2 = clebsch(j1, j2, j3p, m1, m2, m3p)
                sum_match += c1**2
                sum_differ += c1*c2
        assert sum_match == pytest.approx(1)
        assert sum_differ == pytest.approx(int(j3 == j3p and m3 == m3p))


@pytest.mark.parametrize('j1', [0.5, 1.0, 1.5, 2.0, 5, 7.5, 10, 12.5])
@pytest.mark.parametrize('j2', [0.5, 1.0, 1.5, 2.0, 5, 7.5, 10, 12.5])
def test_unit_clebsch_delta_m(j1, j2):
    """sum_j3 sum_m3 C(j1,j2,j3,m1,m2,m3)*C(j1,j2,j3,m1',m2',m3) =
    delta m1,m1' delta m2,m2'"""
    for _ in range(10):
        m1 = np.random.choice(np.arange(-j1, j1+1))
        m1p = np.random.choice(np.arange(-j1, j1+1))
        m2 = np.random.choice(np.arange(-j2, j2+1))
        m2p = np.random.choice(np.arange(-j2, j2+1))

        sum_match = 0
        sum_differ = 0
        for j3 in np.arange(abs(j1-j2), j1+j2+1):
            for m3 in np.arange(-j3, j3+1):
                c1 = clebsch(j1, j2, j3, m1, m2, m3)
                c2 = clebsch(j1, j2, j3, m1p, m2p, m3)
                sum_match += c1**2
                sum_differ += c1*c2
        assert sum_match == pytest.approx(1)
        assert sum_differ == pytest.approx(int(m1 == m1p and m2 == m2p))


def test_cpu_count(monkeypatch):
    ncpus = available_cpu_count()
    assert isinstance(ncpus, int)
    assert ncpus >= 1

    monkeypatch.setenv("QUTIP_NUM_PROCESSES", str(ncpus + 2))
    new_ncpus = available_cpu_count()
    assert new_ncpus == ncpus + 2

    monkeypatch.setenv("QUTIP_NUM_PROCESSES", str(0))
    new_ncpus = available_cpu_count()
    assert new_ncpus >= 1
