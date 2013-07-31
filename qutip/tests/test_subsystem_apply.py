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

from qutip.random_objects import rand_dm, rand_unitary, rand_kraus_map
from qutip.subsystem_apply import subsystem_apply
from qutip.superop_reps import kraus_to_super
from qutip.superoperator import mat2vec, vec2mat
from qutip.tensor import tensor
from qutip.qobj import Qobj


class TestSubsystemApply(object):
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
        rho_3 = rand_dm(3)
        single_op = rand_unitary(3)
        analytic_result = single_op * rho_3 * single_op.dag()
        naive_result = subsystem_apply(rho_3, single_op, [True],
                                       reference=True)
        efficient_result = subsystem_apply(rho_3, single_op, [True])
        naive_diff = (analytic_result - naive_result).data.todense()
        efficient_diff = (efficient_result - analytic_result).data.todense()
        assert_(norm(naive_diff) < 1e-12 and norm(efficient_diff) < 1e-12)

    def test_SimpleSuperApply(self):
        """
        Non-composite system, operator on Liouville space.
        """
        rho_3 = rand_dm(3)
        superop = kraus_to_super(rand_kraus_map(3))
        analytic_result = vec2mat(superop.data.todense() *
                                  mat2vec(rho_3.data.todense()))

        naive_result = subsystem_apply(rho_3, superop, [True],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).data.todense()
        assert_(norm(naive_diff) < 1e-12)

        efficient_result = subsystem_apply(rho_3, superop, [True])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        assert_(norm(efficient_diff) < 1e-12)

    def test_ComplexSingleApply(self):
        """
        Composite system, operator on Hilbert space.
        """
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
        assert_(norm(naive_diff) < 1e-12)

        efficient_result = subsystem_apply(rho_input, single_op,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        assert_(norm(efficient_diff) < 1e-12)

    def test_ComplexSuperApply(self):
        """
        Superoperator: Efficient numerics and reference return same result,
        acting on non-composite system
        """
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
        assert_(norm(naive_diff) < 1e-12)

        efficient_result = subsystem_apply(rho_input, superop,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).data.todense()
        assert_(norm(efficient_diff) < 1e-12)


if __name__ == "__main__":
    run_module_suite()
