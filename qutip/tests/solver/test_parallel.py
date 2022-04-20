import numpy as np
import time
import pytest
import threading

from qutip.solver.parallel import parallel_map, serial_map, loky_pmap


def _func1(x):
    return x**2


def _func2(x, a, b, c, d=0, e=0, f=0):
    assert d > 0
    assert e > 0
    assert f > 0
    time.sleep(np.random.rand() * 0.1)  # random delay
    return x**2

@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map(map, num_cpus):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': threading.TIMEOUT_MAX,
        'timeout': threading.TIMEOUT_MAX,
        'num_cpus': num_cpus,
    }

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = map(_func2, x, args, kwargs, map_kw=map_kw)
    assert ((np.array(y1) == np.array(y2)).all())


@pytest.mark.parametrize('map', [
    pytest.param(parallel_map, id='parallel_map'),
    pytest.param(loky_pmap, id='loky_pmap'),
    pytest.param(serial_map, id='serial_map'),
])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map_accumulator(map, num_cpus):
    if map is loky_pmap:
        loky = pytest.importorskip("loky")
    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': threading.TIMEOUT_MAX,
        'timeout': threading.TIMEOUT_MAX,
        'num_cpus': num_cpus,
    }
    y2 = []

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    map(_func2, x, args, kwargs, reduce_func=y2.append, map_kw=map_kw)
    assert ((np.array(y1) == np.array(y2)).all())
