// This file is part of QuTiP: Quantum Toolbox in Python.
//
//    Copyright (c) 2011 and later, QuSTaR.
//   All rights reserved.
//
//    Redistribution and use in source and binary forms, with or without 
//    modification, are permitted provided that the following conditions are 
//    met:
//
//   1. Redistributions of source code must retain the above copyright notice, 
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//   3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
//       of its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
//    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
//    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
//    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
//    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
//    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
//    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
//    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
//    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
//    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//#############################################################################
#include <complex>

#ifdef __GNUC__
void zspmvpy_openmp(const std::complex<double> * __restrict__ data, const int * __restrict__ ind, 
            const int *__restrict__ ptr,
            const std::complex<double> * __restrict__ vec, const std::complex<double> a, 
            std::complex<double> * __restrict__ out,
            const unsigned int nrows, const unsigned int nthr);
#elif defined(_MSC_VER)
void zspmvpy_openmp(const std::complex<double> * __restrict data, const int * __restrict ind, 
            const int *__restrict ptr,
            const std::complex<double> * __restrict vec, const std::complex<double> a, 
            std::complex<double> * __restrict out,
            const int nrows, const unsigned int nthr);
#else
void zspmvpy_openmp(const std::complex<double> * data, const int * ind, 
            const int * ptr,
            const std::complex<double> * vec, const std::complex<double> a, 
            std::complex<double> * out,
            const unsigned int nrows, const unsigned int nthr);           
#endif