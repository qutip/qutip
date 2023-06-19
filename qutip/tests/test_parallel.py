import numpy as np
import time
from numpy.testing import assert_

from qutip.parallel import parfor, parallel_map, serial_map


def _func1(x):
    time.sleep(np.random.rand() * 0.25)  # random delay
    return x**2


def _func2(x, a, b, c, d=0, e=0, f=0):
    time.sleep(np.random.rand() * 0.25)  # random delay
    return x**2


def test_parfor1():
    "parfor"

    x = np.arange(10)
    y1 = list(map(_func1, x))
    y2 = parfor(_func1, x)

    assert_((np.array(y1) == np.array(y2)).all())


def test_parallel_map():
    "parallel_map"

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}

    x = np.arange(10)
    y1 = list(map(_func1, x))
    y1 = [_func2(xx, *args, **kwargs)for xx in x]

    y2 = parallel_map(_func2, x, args, kwargs, num_cpus=1)
    assert_((np.array(y1) == np.array(y2)).all())

    y2 = parallel_map(_func2, x, args, kwargs, num_cpus=2)
    assert_((np.array(y1) == np.array(y2)).all())


def test_serial_map():
    "serial_map"

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}

    x = np.arange(10)
    y1 = list(map(_func1, x))
    y1 = [_func2(xx, *args, **kwargs)for xx in x]

    y2 = serial_map(_func2, x, args, kwargs, num_cpus=1)
    assert_((np.array(y1) == np.array(y2)).all())

    y2 = serial_map(_func2, x, args, kwargs, num_cpus=2)
    assert_((np.array(y1) == np.array(y2)).all())
