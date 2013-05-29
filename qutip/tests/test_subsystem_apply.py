# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from numpy.linalg import norm
from numpy.testing import assert_, run_module_suite
from scipy import rand



class TestSubsystemApply:
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    """


    def test_SimpleApply(self):
        """
        Superoperator: Efficient numerics and reference return same result,
        acting on non-composite system  
        """
        h_5 = rand(5, 5)
        h_5 = h_5 * h_5.H
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
