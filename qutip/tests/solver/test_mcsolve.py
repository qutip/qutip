import pytest
import numpy as np
import qutip
from qutip.solver.mcsolve import mcsolve, McSolver
from qutip.solver.solver_base import Solver
from qutip.solver.options import SolverOptions

def _return_constant(t, args):
    return args['constant']


def _return_decay(t, args):
    return args['constant'] * np.exp(-args['rate'] * t)


class callable_qobj:
    def __init__(self, oper, coeff=None):
        self.oper = oper
        self.coeff = coeff

    def __call__(self, t, args):
        if self.coeff is not None:
            return self.oper * self.coeff(t, args)
        return self.oper


@pytest.mark.usefixtures("in_temporary_directory")
class StatesAndExpectOutputCase:
    """
    Mixin class to test the states and expectation values from ``mcsolve``.
    """
    size = 10
    h = qutip.num(size)
    state = qutip.basis(size, size-1)
    times = np.linspace(0, 1, 101)
    e_ops = [qutip.num(size)]
    ntraj = 2000

    def _assert_states(self, result, expected, tol):
        assert hasattr(result, 'states')
        assert len(result.states) == len(self.times)
        for test_operator, expected_part in zip(self.e_ops, expected):
            test = qutip.expect(test_operator, result.states)
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def _assert_expect(self, result, expected, tol):
        assert hasattr(result, 'expect')
        assert len(result.expect) == len(self.e_ops)
        for test, expected_part in zip(result.expect, expected):
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def test_states_and_expect(self, hamiltonian, args, c_ops, expected, tol):
        options = SolverOptions(store_states=True)
        result = mcsolve(hamiltonian, self.state, self.times, args=args,
                               c_ops=c_ops, e_ops=self.e_ops, ntraj=self.ntraj,
                               options=options, target_tol=0.01)
        self._assert_expect(result, expected, tol)
        self._assert_states(result, expected, tol)


class TestNoCollapse(StatesAndExpectOutputCase):
    """
    Test that `mcsolve` correctly solves the system when there is a constant
    Hamiltonian and no collapses.
    """
    def pytest_generate_tests(self, metafunc):
        tol = 1e-8
        expect = (qutip.expect(self.e_ops[0], self.state)
                  * np.ones_like(self.times))
        hamiltonian_types = [
            (self.h, "Qobj"),
            ([self.h], "list"),
            (qutip.QobjEvo([self.h, [self.h, '0']]), "QobjEvo"),
            (callable_qobj(self.h), "callable"),
        ]
        cases = [pytest.param(hamiltonian, {}, [], [expect], tol, id=id)
                 for hamiltonian, id in hamiltonian_types]
        metafunc.parametrize(
            ['hamiltonian', 'args', 'c_ops', 'expected', 'tol'],
            cases)

    # Previously the "states_only" and "expect_only" tests were mixed in to
    # every other test case.  We move them out into the simplest set so that
    # their behaviour remains tested, but isn't repeated as often to keep test
    # runtimes shorter.  The known-good cases are still tested in the other
    # test cases, this is just testing the single-output behaviour.

    def test_states_only(self, hamiltonian, args, c_ops, expected, tol):
        options = SolverOptions(store_states=None)
        result = mcsolve(hamiltonian, self.state, self.times, args=args,
                               c_ops=c_ops, e_ops=[], ntraj=self.ntraj,
                               options=options)
        self._assert_states(result, expected, tol)

    def test_expect_only(self, hamiltonian, args, c_ops, expected, tol):
        result = mcsolve(hamiltonian, self.state, self.times, args=args,
                               c_ops=c_ops, e_ops=self.e_ops, ntraj=self.ntraj)
        self._assert_expect(result, expected, tol)


class TestConstantCollapse(StatesAndExpectOutputCase):
    """
    Test that `mcsolve` correctly solves the system when there is a constant
    collapse operator.
    """
    def pytest_generate_tests(self, metafunc):
        tol = 0.05
        coupling = 0.2
        expect = (qutip.expect(self.e_ops[0], self.state)
                  * np.exp(-coupling * self.times))
        collapse_op = qutip.destroy(self.size)
        c_op_types = [
            (np.sqrt(coupling)*collapse_op, {}, "constant"),
            ([collapse_op, 'sqrt({})'.format(coupling)], {}, "string"),
            (callable_qobj(collapse_op, _return_constant),
             {'constant': np.sqrt(coupling)}, "function"),
        ]
        cases = [pytest.param(self.h, args, [c_op], [expect], tol, id=id)
                 for c_op, args, id in c_op_types]
        metafunc.parametrize(
            ['hamiltonian', 'args', 'c_ops', 'expected', 'tol'],
            cases)


class TestTimeDependentCollapse(StatesAndExpectOutputCase):
    """
    Test that `mcsolve` correctly solves the system when the collapse operators
    are time-dependent.
    """
    def pytest_generate_tests(self, metafunc):
        tol = 0.05
        coupling = 0.2
        expect = (qutip.expect(self.e_ops[0], self.state)
                  * np.exp(-coupling * (1 - np.exp(-self.times))))
        collapse_op = qutip.destroy(self.size)
        collapse_args = {'constant': np.sqrt(coupling), 'rate': 0.5}
        collapse_string = 'sqrt({} * exp(-t))'.format(coupling)
        c_op_types = [
            ([collapse_op, _return_decay], collapse_args, "function"),
            ([collapse_op, collapse_string], {}, "string"),
        ]
        cases = [pytest.param(self.h, args, [c_op], [expect], tol, id=id)
                 for c_op, args, id in c_op_types]
        metafunc.parametrize(
            ['hamiltonian', 'args', 'c_ops', 'expected', 'tol'],
            cases)


def test_stored_collapse_operators_and_times():
    """
    Test that the output contains information on which collapses happened and
    at what times, and make sure that this information makes sense.
    """
    size = 10
    a = qutip.destroy(size)
    H = qutip.num(size)
    state = qutip.basis(size, size-1)
    times = np.linspace(0, 10, 100)
    c_ops = [a, a]
    result = mcsolve(H, state, times, c_ops, ntraj=3)
    assert len(result.col_times[0]) > 0
    assert len(result.col_which) == len(result.col_times)
    assert all(col in [0, 1] for col in result.col_which[0])


def test_expectation_dtype(options):
    # We're just testing the output value, so it's important whether certain
    # things are complex or real, but not what the magnitudes of constants are.
    focks = 5
    a = qutip.tensor(qutip.destroy(focks), qutip.qeye(2))
    sm = qutip.tensor(qutip.qeye(focks), qutip.sigmam())
    H = 1j*a.dag()*sm + a
    H = H + H.dag()
    state = qutip.basis([focks, 2], [0, 1])
    times = np.linspace(0, 10, 5)
    c_ops = [a, sm]
    e_ops = [a.dag()*a, sm.dag()*sm, a]
    data = mcsolve(H, state, times, c_ops, e_ops, ntraj=5)
    assert isinstance(data.expect[0][1], float)
    assert isinstance(data.expect[1][1], float)
    assert isinstance(data.expect[2][1], complex)


class TestSeeds:
    sizes = [6, 6, 6]
    dampings = [0.1, 0.4, 0.1]
    ntraj = 25  # Big enough to ensure there are differences without being slow
    a = [qutip.destroy(size) for size in sizes]
    H = 1j * (qutip.tensor(a[0], a[1].dag(), a[2].dag())
              - qutip.tensor(a[0].dag(), a[1], a[2]))
    state = qutip.tensor(qutip.coherent(sizes[0], np.sqrt(2)),
                         qutip.basis(sizes[1:], [0, 0]))
    times = np.linspace(0, 10, 2)
    c_ops = [
        np.sqrt(2*dampings[0]) * qutip.tensor(a[0], qutip.qeye(sizes[1:])),
        (np.sqrt(2*dampings[1])
         * qutip.tensor(qutip.qeye(sizes[0]), a[1], qutip.qeye(sizes[2]))),
        np.sqrt(2*dampings[2]) * qutip.tensor(qutip.qeye(sizes[:2]), a[2]),
    ]

    @pytest.mark.xfail(reason="current limitation of SolverOptions")
    def test_seeds_can_be_reused(self):
        args = (self.H, self.state, self.times)
        kwargs = {'c_ops': self.c_ops, 'ntraj': self.ntraj}
        first = mcsolve(*args, **kwargs)
        options = SolverOptions(seeds=first.seeds)
        second = mcsolve(*args, options=options, **kwargs)
        for first_t, second_t in zip(first.col_times, second.col_times):
            np.testing.assert_equal(first_t, second_t)
        for first_w, second_w in zip(first.col_which, second.col_which):
            np.testing.assert_equal(first_w, second_w)

    def test_seeds_are_not_reused_by_default(self):
        args = (self.H, self.state, self.times)
        kwargs = {'c_ops': self.c_ops, 'ntraj': self.ntraj}
        first = mcsolve(*args, **kwargs)
        second = mcsolve(*args, **kwargs)
        assert not all(np.array_equal(first_t, second_t)
                       for first_t, second_t in zip(first.col_times,
                                                    second.col_times))
        assert not all(np.array_equal(first_w, second_w)
                       for first_w, second_w in zip(first.col_which,
                                                    second.col_which))


def test_list_ntraj():
    """Test that `ntraj` can be a list."""
    size = 5
    a = qutip.destroy(size)
    H = qutip.num(size)
    state = qutip.basis(size, 1)
    times = np.linspace(0, 0.8, 100)
    # Arbitrary coupling and bath temperature.
    coupling = 1 / 0.129
    n_th = 0.063
    c_ops = [np.sqrt(coupling * (n_th + 1)) * a,
             np.sqrt(coupling * n_th) * a.dag()]
    e_ops = [qutip.num(size)]
    ntraj = [1, 5, 15, 100]
    mc = mcsolve(H, state, times, c_ops, e_ops, ntraj=ntraj)
    assert len(ntraj) == len(mc.expect)


# Defined in module-scope so it's pickleable.
def _dynamic(t, args):
    return 0 if args["collapse"] else 1

@pytest.mark.xfail(reason="current limitation of SolverOptions")
def test_dynamic_arguments():
    """Test dynamically updated arguments are usable."""
    size = 5
    a = qutip.destroy(size)
    H = qutip.num(size)
    times = np.linspace(0, 1, 11)
    state = qutip.basis(size, 2)

    c_ops = [[a, _dynamic], [a.dag(), _dynamic]]
    mc = mcsolve(H, state, times, c_ops, ntraj=25, args={"collapse": []})
    assert all(len(collapses) <= 1 for collapses in mc.col_which)
