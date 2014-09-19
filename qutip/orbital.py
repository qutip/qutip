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

__all__ = ['orbital']

import numpy as np
from scipy.misc import factorial


def orbital(theta, phi, *args):
    """Calculates an angular wave function on a sphere.
    ``psi = orbital(theta,phi,ket1,ket2,...)`` calculates
    the angular wave function on a sphere at the mesh of points
    defined by theta and phi which is
    :math:`\sum_{lm} c_{lm} Y_{lm}(theta,phi)` where :math:`C_{lm}` are the
    coefficients specified by the list of kets. Each ket has 2l+1 components
    for some integer l.

    Parameters
    ----------
    theta : list/array
        Polar angles

    phi : list/array
        Azimuthal angles

    args : list/array
        ``list`` of ket vectors.

    Returns
    -------
    ``array`` for angular wave function

    """
    psi = 0.0
    if isinstance(args[0], list):
        # use the list in args[0]
        args = args[0]

    for k in range(len(args)):
        ket = args[k]
        if not ket.type == 'ket':
            raise TypeError('Invalid input ket in orbital')
        sk = ket.shape
        nchk = (sk[0] - 1) / 2.0
        if nchk != np.floor(nchk):
            raise ValueError(
                'Kets must have odd number of components in orbital')
        l = int((sk[0] - 1) / 2)
        if l == 0:
            SPlm = np.sqrt(2) * np.ones((np.size(theta), 1), dtype=complex)
        else:
            SPlm = _sch_lpmv(l, np.cos(theta))
        fac = np.sqrt((2.0 * l + 1) / (8 * np.pi))
        kf = ket.full()
        psi += np.sqrt(2) * fac * kf[l, 0] * np.ones((np.size(phi),
                                                      np.size(theta)),
                                                     dtype=complex) * SPlm[0]
        for m in range(1, l + 1):
            psi += ((-1.0) ** m * fac * kf[l - m, 0]) * \
                np.array([np.exp(1.0j * 1 * phi)]).T * \
                np.ones((np.size(phi), np.size(theta)),
                        dtype=complex) * SPlm[1]
        for m in range(-l, 0):
            psi = psi + (fac * kf[l - m, 0]) * \
                np.array([np.exp(1.0j * 1 * phi)]).T * \
                np.ones((np.size(phi), np.size(theta)), dtype=complex) * \
                SPlm[abs(m)]
    return psi


# Schmidt Semi-normalized Associated Legendre Functions
def _sch_lpmv(n, x):
    '''
    Outputs array of Schmidt Seminormalized Associated Legendre Functions
    S_{n}^{m} for m<=n.

    Parameters
    ----------
    n : int
        Degree of polynomial.

    x : float
        Point at which to evaluate

    Returns
    -------
    array of values for Legendre functions.

    '''
    from scipy.special import lpmv
    n = int(n)
    sch = np.array([1.0])
    sch2 = np.array([(-1.0) ** m * np.sqrt(
        (2.0 * factorial(n - m)) / factorial(n + m)) for m in range(1, n + 1)])
    sch = np.append(sch, sch2)
    if isinstance(x, float) or len(x) == 1:
        leg = lpmv(np.arange(0, n + 1), n, x)
        return np.array([sch * leg]).T
    else:
        for j in range(0, len(x)):
            leg = lpmv(range(0, n + 1), n, x[j])
            if j == 0:
                out = np.array([sch * leg]).T
            else:
                out = np.append(out, np.array([sch * leg]).T, axis=1)
    return out
