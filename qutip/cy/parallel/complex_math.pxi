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

cdef extern from "complex.h" nogil:
    double         cabs(double complex x)
    double complex cacos(double complex x)
    double complex cacosh(double complex x)
    double         carg(double complex x)
    double complex casin(double complex x)
    double complex casinh(double complex x)
    double complex catan(double complex x)
    double complex catanh(double complex x)
    double complex ccos(double complex x)
    double complex ccosh(double complex x)
    double complex cexp(double complex x)
    double         cimag(double complex x)
    double complex clog(double complex x)
    double complex conj(double complex x)
    double complex cpow(double complex x, double complex y)
    double complex cproj(double complex x)
    double         creal(double complex x)
    double complex csin(double complex x)
    double complex csinh(double complex x)
    double complex csqrt(double complex x)
    double complex ctan(double complex x)
    double complex ctanh(double complex x)


cdef inline double abs(double complex x):
    return cabs(x)
cdef inline double complex acos(double complex x):
    return cacos(x)
cdef inline double complex acosh(double complex x):
    return cacosh(x)
cdef inline double arg(double complex x):
    return carg(x)
cdef inline double complex asin(double complex x):
    return casin(x)
cdef inline double complex asinh(double complex x):
    return casinh(x)
cdef inline double complex atan(double complex x):
    return catan(x)
cdef inline double complex atanh(double complex x):
    return catanh(x)
cdef inline double complex cos(double complex x):
    return ccos(x)
cdef inline double complex cosh(double complex x):
    return ccosh(x)
cdef inline double complex exp(double complex x):
    return cexp(x)
cdef inline double imag(double complex x):
    return cimag(x)
cdef inline double complex log(double complex x):
    return clog(x)
cdef inline double complex pow(double complex x, double complex y):
    return cpow(x,y)
cdef inline double complex proj(double complex x):
    return cproj(x)
cdef inline double real(double complex x):
    return creal(x)
cdef inline double complex sin(double complex x):
    return csin(x)
cdef inline double complex sinh(double complex x):
    return csinh(x)
cdef inline double complex sqrt(double complex x):
    return csqrt(x)
cdef inline double complex tan(double complex x):
    return ctan(x)
cdef inline double complex tanh(double complex x):
    return ctanh(x)