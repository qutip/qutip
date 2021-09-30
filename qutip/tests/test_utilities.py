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
from qutip import convert_unit, clebsch, n_thermal
import qutip.utilities as utils
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
