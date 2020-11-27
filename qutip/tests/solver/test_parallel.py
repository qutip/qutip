# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
import time
import pytest

from qutip.solver.parallel import parallel_map, serial_map, loky_pmap
try:
    import loky
except ModuleNotFoundError:
    loky = False

def _func1(x):
    return x**2


def _func2(x, a, b, c, d=0, e=0, f=0):
    assert d > 0
    assert e > 0
    assert f > 0
    time.sleep(np.random.rand() * 0.1)  # random delay
    return x**2

@pytest.mark.parametrize('map',
                         [parallel_map, loky_pmap, serial_map],
                         ids=['parallel_map', 'loky_pmap', 'serial_map'])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map(map, num_cpus):
    if map is loky_pmap and not loky:
        pytest.skip(reason="module loky not available")

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': 1e8,
        'timeout': 1e8,
        'num_cpus': num_cpus,
    }

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    y2 = parallel_map(_func2, x, args, kwargs, map_kw=map_kw)
    assert ((np.array(y1) == np.array(y2)).all())


@pytest.mark.parametrize('map',
                         [parallel_map, loky_pmap, serial_map],
                         ids=['parallel_map', 'loky_pmap', 'serial_map'])
@pytest.mark.parametrize('num_cpus',
                         [1, 2],
                         ids=['1', '2'])
def test_map_accumulator(map, num_cpus):
    if map is loky_pmap and not loky:
        pytest.skip(reason="module loky not available")

    args = (1, 2, 3)
    kwargs = {'d': 4, 'e': 5, 'f': 6}
    map_kw = {
        'job_timeout': 1e8,
        'timeout': 1e8,
        'num_cpus': num_cpus,
    }
    y2 = []

    x = np.arange(10)
    y1 = [_func1(xx) for xx in x]

    parallel_map(_func2, x, args, kwargs, reduce_func=y2.append, map_kw=map_kw)
    assert ((np.array(y1) == np.array(y2)).all())
