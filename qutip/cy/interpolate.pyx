#!python
#cython: language_level=3
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project.
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
cimport numpy as cnp
from libc.math cimport (fabs, fmin)
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double phi(double t):

    cdef double abs_t = fabs(t)
    if abs_t <= 1:
        return 4 - 6 * abs_t**2 + 3 * abs_t**3
    elif abs_t <= 2:
        return (2-abs_t)**3
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double interp(double x, double a, double b, double[::1] c):
    cdef int n = c.shape[0] - 3
    cdef double h = (b-a) / n
    cdef int l = <int>((x-a)/h) + 1
    cdef int m = <int>(fmin(l+3, n+3))
    cdef size_t ii
    cdef double s = 0, _tmp
    cdef double pos = (x-a)/h + 2

    for ii in range(l, m+1):
        _tmp = phi(pos - ii)
        if _tmp:
            s += c[ii-1] * _tmp
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef complex zinterp(double x, double a, double b, complex[::1] c):
    cdef int n = c.shape[0] - 3
    cdef double h = (b-a) / n
    cdef int l = <int>((x-a)/h) + 1
    cdef int m = <int>(fmin(l+3, n+3))
    cdef size_t ii
    cdef complex s = 0
    cdef double _tmp, pos = (x-a)/h + 2

    for ii in range(l, m+1):
        _tmp = phi(pos - ii)
        if _tmp:
            s += c[ii-1] * _tmp
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arr_interp(double[::1] x, double a, double b, double[::1] c):
    cdef int lenx = x.shape[0]
    cdef int lenc = c.shape[0]
    cdef int n = lenc - 3
    cdef double h = (b-a) / n
    cdef size_t ii, jj
    cdef int l, m
    cdef double pos, _tmp
    cdef cnp.ndarray[double, ndim=1, mode="c"] out = np.zeros(lenx, dtype=float)

    for jj in range(lenx):
        l = <int>((x[jj]-a)/h) + 1
        m = <int>(fmin(l+3, n+3))
        pos = (x[jj]-a)/h + 2

        for ii in range(l, m+1):
            _tmp = phi(pos - ii)
            if _tmp:
                out[jj] += c[ii-1] * _tmp

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arr_zinterp(double[::1] x, double a, double b, complex[::1] c):
    cdef int lenx = x.shape[0]
    cdef int lenc = c.shape[0]
    cdef int n = lenc - 3
    cdef double h = (b-a) / n
    cdef size_t ii, jj
    cdef int l, m
    cdef double pos, _tmp
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out = np.zeros(lenx, dtype=complex)

    for jj in range(lenx):
        l = <int>((x[jj]-a)/h) + 1
        m = <int>(fmin(l+3, n+3))
        pos = (x[jj]-a)/h + 2

        for ii in range(l, m+1):
            _tmp = phi(pos - ii)
            if _tmp:
                out[jj] = out[jj] + c[ii-1] * _tmp

    return out
