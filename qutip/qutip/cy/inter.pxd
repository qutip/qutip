#cython: language_level=3
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

cdef complex _spline_complex_t_second(double x, double[::1] t,
                                      complex[::1] y, complex[::1] M,
                                      int N)

cdef complex _spline_complex_cte_second(double x, double[::1] t,
                                        complex[::1] y, complex[::1] M,
                                        int N, double dt)

cdef double _spline_float_t_second(double x, double[::1] t,
                                   double[::1] y, double[::1] M,
                                   int N)

cdef double _spline_float_cte_second(double x, double[::1] t,
                                     double[::1] y, double[::1] M,
                                     int N, double dt)

cdef double _step_float_cte(double x, double[::1] t, double[::1] y, int n_t)

cdef complex _step_complex_cte(double x, double[::1] t, complex[::1] y, int n_t)

cdef double _step_float_t(double x, double[::1] t, double[::1] y, int n_t)

cdef complex _step_complex_t(double x, double[::1] t, complex[::1] y, int n_t)
