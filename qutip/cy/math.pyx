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
cimport cython
from libc.math cimport fabs, exp, copysign

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double erf(double x):
    """
    A Cython version of the erf function from the cdflib in SciPy.
    """
    cdef double c = 0.564189583547756
    
    cdef double a[5] 
    a[:] = [0.771058495001320e-4, -0.133733772997339e-2, 0.323076579225834e-1,
                     0.479137145607681e-1, 0.128379167095513]
    
    cdef double b[3] 
    b[:] = [0.301048631703895e-2, 0.538971687740286e-1, .375795757275549]
    
    cdef double p[8] 
    p[:] = [-1.36864857382717e-7, 5.64195517478974e-1, 7.21175825088309,
                     4.31622272220567e1, 1.52989285046940e2, 3.39320816734344e2,
                     4.51918953711873e2, 3.00459261020162e2]
    
    cdef double q[8] 
    q[:]= [1.0, 1.27827273196294e1, 7.70001529352295e1, 2.77585444743988e2,
                     6.38980264465631e2, 9.31354094850610e2, 7.90950925327898e2,
                     3.00459260956983e2]
    
    cdef double r[5] 
    r[:] = [2.10144126479064, 2.62370141675169e1, 2.13688200555087e1, 
                      4.65807828718470, 2.82094791773523e-1]
    
    cdef double s[4] 
    s[:] = [9.41537750555460e1, 1.87114811799590e2, 9.90191814623914e1,
                     1.80124575948747e1]
    
    cdef double ax = fabs(x)
    cdef double t, x2, top, bot, erf
    
    if ax <= 0.5:
        t = x*x
        top = ((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4]) + 1.0
        bot = ((b[0]*t+b[1])*t+b[2])*t + 1.0
        erf = x * (top/bot)
        return erf
    elif ax <= 4.0:
        x2 = x*x
        top = ((((((p[0]*ax+p[1])*ax+p[2])*ax+p[3])*ax+p[4])*ax+p[5])*ax+p[6])*ax + p[7]
        bot = ((((((q[0]*ax+q[1])*ax+q[2])*ax+q[3])*ax+q[4])*ax+q[5])*ax+q[6])*ax + q[7]
        erf = 0.5 + (0.5-exp(-x2)*top/bot)
        if x < 0.0:
            erf = -erf
        return erf
    elif ax < 5.8:
        x2 = x*x
        t = 1.0/x2
        top = (((r[0]*t+r[1])*t+r[2])*t+r[3])*t + r[4]
        bot = (((s[0]*t+s[1])*t+s[2])*t+s[3])*t + 1.0
        erf = (c-top/(x2*bot))/ax
        erf = 0.5 + (0.5-exp(-x2)*erf)
        if x < 0.0:
            erf = -erf
        return erf
    else:
        erf = copysign(1.0, x)
        return erf