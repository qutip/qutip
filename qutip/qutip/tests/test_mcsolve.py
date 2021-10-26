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
    ntraj = 750

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
        options = qutip.Options(average_states=True, store_states=True)
        result = qutip.mcsolve(hamiltonian, self.state, self.times, args=args,
                               c_ops=c_ops, e_ops=self.e_ops, ntraj=self.ntraj,
                               options=options)
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
            cases.append(pytest.param(hamiltonian, args, [], [expect], tol,
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
        options = qutip.Options(average_states=True, store_states=True)
        result = qutip.mcsolve(hamiltonian, self.state, self.times, args=args,
                               c_ops=c_ops, e_ops=[], ntraj=self.ntraj,
                               options=options)
        self._assert_states(result, expected, tol)

    def test_expect_only(self, hamiltonian, args, c_ops, expected, tol):
        result = qutip.mcsolve(hamiltonian, self.state, self.times, args=args,
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
            cases.append(pytest.param(self.h, args, [c_op], [expect], tol,
                                      id=id, marks=[pytest.mark.slow]))
        metafunc.parametrize(['hamiltonian', 'args', 'c_ops', 'expected',
                              'tol'],
                             cases)


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
    result = qutip.mcsolve(H, state, times, c_ops, ntraj=1)
    assert len(result.col_times[0]) > 0
    assert len(result.col_which) == len(result.col_times)
    assert all(col in [0, 1] for col in result.col_which[0])


@pytest.mark.parametrize('options', [
    pytest.param(qutip.Options(average_expect=True), id="average_expect=True"),
    pytest.param(qutip.Options(average_states=False),
                 id="average_states=False"),
])
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
    data = qutip.mcsolve(H, state, times, c_ops, e_ops, ntraj=5,
                         options=options)
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

    def test_seeds_can_be_reused(self):
        args = (self.H, self.state, self.times)
        kwargs = {'c_ops': self.c_ops, 'ntraj': self.ntraj}
        first = qutip.mcsolve(*args, **kwargs)
        options = qutip.Options(seeds=first.seeds)
        second = qutip.mcsolve(*args, options=options, **kwargs)
        for first_t, second_t in zip(first.col_times, second.col_times):
            np.testing.assert_equal(first_t, second_t)
        for first_w, second_w in zip(first.col_which, second.col_which):
            np.testing.assert_equal(first_w, second_w)

    def test_seeds_are_not_reused_by_default(self):
        args = (self.H, self.state, self.times)
        kwargs = {'c_ops': self.c_ops, 'ntraj': self.ntraj}
        first = qutip.mcsolve(*args, **kwargs)
        second = qutip.mcsolve(*args, **kwargs)
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
    mc = qutip.mcsolve(H, state, times, c_ops, e_ops, ntraj=ntraj)
    assert len(ntraj) == len(mc.expect)


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
    mc = qutip.mcsolve(H, state, times, c_ops, ntraj=25, args={"collapse": []})
    assert all(len(collapses) <= 1 for collapses in mc.col_which)


def _regression_490_f1(t, args):
    return t-1


def _regression_490_f2(t, args):
    return -t


def test_regression_490():
    """Test for regression of gh-490."""
    h = [qutip.sigmax(),
         [qutip.sigmay(), _regression_490_f1],
         [qutip.sigmaz(), _regression_490_f2]]
    state = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    times = np.linspace(0, 3, 10)
    result_me = qutip.mesolve(h, state, times)
    result_mc = qutip.mcsolve(h, state, times, ntraj=1)
    for state_me, state_mc in zip(result_me.states, result_mc.states):
        np.testing.assert_allclose(state_me.full(), state_mc.full(), atol=1e-8)
