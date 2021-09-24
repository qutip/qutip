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
        naive_diff = (analytic_result - naive_result).full()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="SimpleSingle: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_3, single_op, [True])
        efficient_diff = (efficient_result - analytic_result).full()
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
        analytic_result = vec2mat(superop.full() @ mat2vec(rho_3.full()))
        naive_result = subsystem_apply(rho_3, superop, [True],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).full()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="SimpleSuper: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_3, superop, [True])
        efficient_diff = (efficient_result - analytic_result).full()
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
        naive_diff = (analytic_result - naive_result).full()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="ComplexSingle: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_input, single_op,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).full()
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
        analytic_result[1] = Qobj(
            vec2mat(superop.full() @ mat2vec(analytic_result[1].full())))
        analytic_result[3] = Qobj(
            vec2mat(superop.full() @ mat2vec(analytic_result[3].full())))
        analytic_result = tensor(analytic_result)

        naive_result = subsystem_apply(rho_input, superop,
                                       [False, True, False, True, False],
                                       reference=True)
        naive_diff = (analytic_result - naive_result).full()
        naive_diff_norm = norm(naive_diff)
        assert_(naive_diff_norm < tol,
                msg="ComplexSuper: naive_diff_norm {} "
                    "is beyond tolerance {}".format(
                        naive_diff_norm, tol))

        efficient_result = subsystem_apply(rho_input, superop,
                                           [False, True, False, True, False])
        efficient_diff = (efficient_result - analytic_result).full()
        efficient_diff_norm = norm(efficient_diff)
        assert_(efficient_diff_norm < tol,
                msg="ComplexSuper: efficient_diff_norm {} "
                    "is beyond tolerance {}".format(
                        efficient_diff_norm, tol))


if __name__ == "__main__":
    run_module_suite()
