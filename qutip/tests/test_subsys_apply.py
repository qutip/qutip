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

from qutip.random_objects import rand_dm, rand_unitary, rand_kraus_map
from qutip.subsystem_apply import subsystem_apply
from qutip.superop_reps import kraus_to_super
from qutip.superoperator import mat2vec, vec2mat
from qutip.tensor import tensor
from qutip.qobj import Qobj


class TestSubsysApply(object):
    """
    A test class for the QuTiP function for applying superoperators to
    subsystems.
    The four tests below determine whether efficient numerics, naive numerics
    and semi-analytic results are identical.
    """

    def test_SimpleSingleApply(self):
        """
        Non-composite system, operator on Hilbert space.
        """
        tol = 1e-12
        rho_3 = rand_dm(3)
        single_op = rand_unitary(3)
        analytic_result = single_op * rho_3 * single_op.dag()
        naive_result = subsystem_apply(rho_3, single_op, [True],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).data.todense()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="SimpleSingle: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))
                                       
        efficient_result = subsystem_apply(rho_3, single_op, [True])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        efficient_diff_norm = norm(efficient_diff)
        assert_(efficient_diff_norm < tol,
                msg="SimpleSingle: efficient_diff_norm {} "
                    "is beyond tolerance {}".format(
                        efficient_diff_norm, tol))

    def test_SimpleSuperApply(self):
        """
        Non-composite system, operator on Liouville space.
        """
        tol = 1e-12
        rho_3 = rand_dm(3)
        superop = kraus_to_super(rand_kraus_map(3))
        analytic_result = vec2mat(superop.data.todense() *
                                  mat2vec(rho_3.data.todense()))

        naive_result = subsystem_apply(rho_3, superop, [True],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).data.todense()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="SimpleSuper: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_3, superop, [True])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        efficient_diff_norm = norm(efficient_diff)
        assert_(efficient_diff_norm < tol,
                msg="SimpleSuper: efficient_diff_norm {} "
                    "is beyond tolerance {}".format(
                        efficient_diff_norm, tol))

    def test_ComplexSingleApply(self):
        """
        Composite system, operator on Hilbert space.
        """
        tol = 1e-12
        rho_list = list(map(rand_dm, [2, 3, 2, 3, 2]))
        rho_input = tensor(rho_list)
        single_op = rand_unitary(3)

        analytic_result = rho_list
        analytic_result[1] = single_op * analytic_result[1] * single_op.dag()
        analytic_result[3] = single_op * analytic_result[3] * single_op.dag()
        analytic_result = tensor(analytic_result)

        naive_result = subsystem_apply(rho_input, single_op,
                                       [False, True, False, True, False],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).data.todense()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="ComplexSingle: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_input, single_op,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        efficient_diff_norm = norm(efficient_diff)
        assert_(efficient_diff_norm < tol,
                msg="ComplexSingle: efficient_diff_norm {} "
                    "is beyond tolerance {}".format(
                        efficient_diff_norm, tol))

    def test_ComplexSuperApply(self):
        """
        Superoperator: Efficient numerics and reference return same result,
        acting on non-composite system
        """
        tol = 1e-10
        rho_list = list(map(rand_dm, [2, 3, 2, 3, 2]))
        rho_input = tensor(rho_list)
        superop = kraus_to_super(rand_kraus_map(3))
        
        analytic_result = rho_list
        analytic_result[1] = Qobj(vec2mat(superop.data.todense() *
                                  mat2vec(analytic_result[1].data.todense())))
        analytic_result[3] = Qobj(vec2mat(superop.data.todense() *
                                  mat2vec(analytic_result[3].data.todense())))
        analytic_result = tensor(analytic_result)

        naive_result = subsystem_apply(rho_input, superop,
                                       [False, True, False, True, False],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).data.todense()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="ComplexSuper: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_input, superop,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        efficient_diff_norm = norm(efficient_diff)
        assert_(efficient_diff_norm < tol,
                msg="ComplexSuper: efficient_diff_norm {} "
                    "is beyond tolerance {}".format(
                        efficient_diff_norm, tol))


if __name__ == "__main__":
    run_module_suite()
