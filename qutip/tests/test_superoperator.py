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

from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite
import scipy

from qutip import *


class TestMatrixVector:
    """
    A test class for the QuTiP function for matrix/vector conversion.
    """

    def testOperatorVector(self):
        """
        Superoperator: Test unitary transformation with operators and superoperators.
        """
        N = 3
        rho = rand_dm(N)
        U = rand_unitary(N)

        rho1 = U * rho * U.dag()
        rho2_vec = spre(U) * spost(U.dag()) * operator_to_vector(rho)
        rho2 = vector_to_operator(rho2_vec)

        assert_((rho1 - rho2).norm() < 1e-8)

    def testMatrixVectorMatrix(self):
        """
        Superoperator: Conversion matrix to vector to matrix
        """
        M = scipy.rand(10, 10)
        V = mat2vec(M)
        M2 = vec2mat(V)
        assert_(norm(M - M2) == 0.0)

    def testVectorMatrixVector(self):
        """
        Superoperator: Conversion vector to matrix to vector
        """
        V = scipy.rand(100)     # a row vector
        M = vec2mat(V)
        V2 = mat2vec(M).T  # mat2vec returns a column vector
        assert_(norm(V - V2) == 0.0)

    def testVectorMatrixIndexConversion(self):
        """
        Superoperator: Conversion between matrix and vector indices
        """
        N = 10
        for I in range(N * N):
            i, j = vec2mat_index(N, I)
            I2 = mat2vec_index(N, i, j)
            assert_(I == I2)

    def testVectorMatrixIndexCompability(self):
        """
        Superoperator: Test compability between matrix/vector conversion and
        the corresponding index conversion.
        """
        N = 10
        M = scipy.rand(N, N)
        V = mat2vec(M)
        for I in range(N * N):
            i, j = vec2mat_index(N, I)
            assert_(V[I][0] == M[i, j])

    def testLiouvillianImplementations(self):
        """
        Superoperator: Randomized comparison of standard and optimized
        liouvillian.
        """
        N1 = 3
        N2 = 4
        N3 = 5

        a1 = tensor(rand_dm(N1, density=0.75), identity(N2), identity(N3))
        a2 = tensor(identity(N1), rand_dm(N2, density=0.75), identity(N3))
        a3 = tensor(identity(N1), identity(N2), rand_dm(N3, density=0.75))
        H = a1.dag() * a1 + a2.dag() * a2 + a3.dag() * a3

        c_ops = [sqrt(0.01) * a1, sqrt(0.025) * a2, sqrt(0.05) * a3]

        L1 = liouvillian(H, c_ops)
        L2 = liouvillian_fast(H, c_ops)

        assert_((L1 - L2).norm() < 1e-8)


if __name__ == "__main__":
    run_module_suite()
