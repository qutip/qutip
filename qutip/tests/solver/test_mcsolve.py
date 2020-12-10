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

import pytest
import numpy as np
import qutip
from qutip.solver import mcsolve, mesolve, SolverOptions, McSolver, MeMcSolver
from qutip.solver.evolver import all_ode_method


def _return_constant(t, args):
    return args['constant']


def _return_decay(t, args):
    return args['constant'] * np.exp(-args['rate'] * t)


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
    ntraj = 250

    def _assert_states(self, result, expected, tol):
        assert hasattr(result, 'average_states')
        assert len(result.average_states) == len(self.times)
        for test_operator, expected_part in zip(self.e_ops, expected):
            test = qutip.expect(test_operator, result.average_states)
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def _assert_expect(self, result, expected, tol):
        assert hasattr(result, 'average_expect')
        assert len(result.average_expect) == len(self.e_ops)
        for test, expected_part in zip(result.average_expect, expected):
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def test_states_and_expect(self, hamiltonian, args, c_ops, expected, tol):
        options = SolverOptions(store_states=True)
        result = mcsolve(hamiltonian, self.state, self.times, args=args,
                         c_ops=c_ops, e_ops=self.e_ops, ntraj=self.ntraj,
                         options=options)
        self._assert_expect(result, expected, tol)
        self._assert_states(result, expected, tol)

    def _assert_photocurrent(self, result, expected, tol):
        assert len(result.photocurrent) == 1
        # Too noisy to do a simple meaningful test that would always pass


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
            (self.h, {}, "Qobj", False),
            ([self.h], {}, "list", False),
            ([self.h, [self.h, '0']], {}, "string", True),
            ([self.h, [self.h, _return_constant]], {'constant': 0},
             "function", False),
        ]
        cases = []
        for hamiltonian, args, id, slow in hamiltonian_types:
            if slow and 'only' in metafunc.function.__name__:
                # Skip the single-output test if it's a slow case.
                continue
            marks = [pytest.mark.slow] if slow else []
            cases.append(pytest.param(hamiltonian, args, [self.h*0], [expect], tol,
                                      id=id, marks=marks))
        metafunc.parametrize(['hamiltonian', 'args', 'c_ops', 'expected',
                              'tol'],
                             cases)

    # Previously the "states_only" and "expect_only" tests were mixed in to
    # every other test case.  We move them out into the simplest set so that
    # their behaviour remains tested, but isn't repeated as often to keep test
    # runtimes shorter.  The known-good cases are still tested in the other
    # test cases, this is just testing the single-output behaviour.

    def test_states_only(self, hamiltonian, args, c_ops, expected, tol):
        options = SolverOptions(keep_runs_results=False, store_states=True)
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
            ([collapse_op, _return_constant], {'constant': np.sqrt(coupling)},
             "function"),
        ]
        cases = []
        for c_op, args, id in c_op_types:
            for method in all_ode_method:
                for ahs in [False, True]:
                    fullid = id + "_" + method + ("_AHS" if ahs else "")
                    options = SolverOptions(store_states=True,
                                            method=method, ahs=ahs)
                    cases.append(pytest.param(self.h, args, [c_op],
                                              [expect], tol, options,
                                              id=fullid,
                                              marks=[pytest.mark.slow]))
        metafunc.parametrize(['hamiltonian', 'args', 'c_ops', 'expected',
                              'tol', 'options'],
                             cases)

    def test_states_and_expect(self, hamiltonian, args, c_ops,
                               expected, tol, options):
        result = mcsolve(hamiltonian, self.state, self.times, args=args,
                         c_ops=c_ops, e_ops=self.e_ops, ntraj=self.ntraj,
                         options=options)
        self._assert_expect(result, expected, tol)
        self._assert_states(result, expected, tol)
        self._assert_photocurrent(result, expected[0], 0.3)


class TestTimeDependentCollapse(StatesAndExpectOutputCase):
    """
    Test that `mcsolve` correctly solves the system when the collapse operators
    are time-dependent.
    """
    def pytest_generate_tests(self, metafunc):
        tol = 0.1
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
        cases = []
        for c_op, args, id in c_op_types:
            cases.append(pytest.param(self.h, args, [c_op], [expect], tol,
                                      id=id, marks=[pytest.mark.slow]))
        metafunc.parametrize(['hamiltonian', 'args', 'c_ops', 'expected',
                              'tol'],
                             cases)


class TestDiagonalized():
    """
    Test that `mcsolve` correctly solves the system by diagonalizing the
    effective hamiltonian.
    """
    size = 10
    h = qutip.num(size)
    state = qutip.basis(size, size-1)
    times = np.linspace(0, 1, 101)
    e_ops = [qutip.num(size)]
    ntraj = 100

    def test_diag(self):
        # The hamiltonian must not already be diagonal.
        tol = 0.05
        a = qutip.destroy(self.size)
        h = self.h + a + a.dag()
        expected = mesolve(h, self.state, self.times, c_ops=[a],
                           e_ops=self.e_ops).expect[0]
        opt = SolverOptions(store_states=True, method="diag")
        result = mcsolve(h, self.state, self.times, c_ops=[a],
                         e_ops=self.e_ops, ntraj=self.ntraj,
                         options=opt)
        self._assert_expect(result, [expected], tol)
        self._assert_states(result, [expected], tol)
        self._assert_photocurrent(result, expected, 0.5)

    def _assert_photocurrent(self, result, expected, tol):
        assert len(result.photocurrent) == 1
        expected = np.array(expected)
        expected = (expected[1:] + expected[:-1]) / 2
        ratio = np.mean(result.photocurrent[0] / expected)
        np.testing.assert_allclose(ratio, 1, rtol=tol)

    def _assert_states(self, result, expected, tol):
        assert hasattr(result, 'average_states')
        assert len(result.average_states) == len(self.times)
        for test_operator, expected_part in zip(self.e_ops, expected):
            test = qutip.expect(test_operator, result.average_states)
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def _assert_expect(self, result, expected, tol):
        assert hasattr(result, 'average_expect')
        assert len(result.average_expect) == len(self.e_ops)
        for test, expected_part in zip(result.average_expect, expected):
            np.testing.assert_allclose(test, expected_part, rtol=tol)

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
    result = mcsolve(H, state, times, c_ops, ntraj=1)
    assert len(result.col_times[0]) > 0
    assert len(result.col_which) == len(result.col_times)
    assert all(col in [0, 1] for col in result.col_which[0])


def test_expectation_dtype():
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
    assert isinstance(data.average_expect[0][1], float)
    assert isinstance(data.average_expect[1][1], float)
    assert isinstance(data.average_expect[2][1], complex)


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

    #@pytest.mark.xfail(reason="current limitation of SolverOptions")
    def test_seeds_can_be_reused(self):
        args = (self.H, self.state, self.times)
        kwargs = {'c_ops': self.c_ops, 'ntraj': self.ntraj}
        first = mcsolve(*args, **kwargs)
        kwargs['seeds'] = first.seeds
        second = mcsolve(*args, **kwargs)
        for first_t, second_t in zip(first.col_times, second.col_times):
            assert np.array_equal(first_t, second_t)
        for first_w, second_w in zip(first.col_which, second.col_which):
            assert np.array_equal(first_w, second_w)

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

# Defined in module-scope so it's pickleable.
def _dynamic(t, args):
    return 0 if args["collapse"] else 1


def test_dynamic_arguments():
    """Test dynamically updated arguments are usable."""
    size = 5
    a = qutip.destroy(size)
    H = qutip.num(size)
    times = np.linspace(0, 1, 11)
    state = qutip.basis(size, 2)

    c_ops = [[a, _dynamic], [a.dag(), _dynamic]]
    mc = mcsolve(H, state, times, c_ops, ntraj=25,
                 args={"collapse": []}, feedback_args={"collapse":"collapse"})
    assert all(len(collapses) <= 1 for collapses in mc.col_which)


def _regression_490_f1(t, args):
    return t-1


def _regression_490_f2(t, args):
    return -t


def test_regression_490():
    """Test for regression of gh-490."""
    # Both call sesolve
    h = [qutip.sigmax(),
         [qutip.sigmay(), _regression_490_f1],
         [qutip.sigmaz(), _regression_490_f2]]
    state = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 3, 10)
    result_me = mesolve(h, state, times)
    result_mc = mcsolve(h, state, times, ntraj=1)
    for state_me, state_mc in zip(result_me.states, result_mc.states):
        np.testing.assert_allclose(state_me.full(), state_mc.full(), atol=1e-8)


def _photo_func(t, args):
    return args["a"] * t


def test_McSolver():
    N = 10
    solver = McSolver(qutip.qeye(N),
                      c_ops=[qutip.destroy(N)],
                      e_ops=[qutip.num(N)])
    res = solver.run(qutip.basis(N, N-1), np.linspace(0,1,11), ntraj=10)
    assert res.num_traj == 10
    res = solver.add_traj(5)
    assert res.num_traj == 15
    res = solver.run(qutip.basis(N, N-1), np.linspace(0,1,11), ntraj=5)
    assert res.num_traj == 5


def test_MeMcSolver():
    H = qutip.tensor([qutip.qeye(4), qutip.num(2)])
    a = qutip.tensor([qutip.qeye(4), qutip.destroy(2)])
    b = qutip.tensor([qutip.destroy(4), qutip.qeye(2)])
    e = qutip.tensor([qutip.num(4), qutip.qeye(2)])
    psi0 = qutip.tensor([qutip.basis(4,3), qutip.basis(2,1)])
    solver = MeMcSolver(H, c_ops=[a], sc_ops=[b], e_ops=[e],
                        options=SolverOptions(map="serial_map"))
    res = solver.run(psi0, np.linspace(0,1,11), ntraj=20).expect[0]
    me_res = mesolve(H, psi0, np.linspace(0,1,11),
                     c_ops=[a,b], e_ops=[e]).expect[0]
    np.testing.assert_allclose(res, me_res, 5e-1) # High tol for faster tests
