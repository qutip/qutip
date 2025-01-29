from copy import copy

import numpy as np
import pytest

import qutip
from qutip.solver.nm_mcsolve import nm_mcsolve, NonMarkovianMCSolver


@pytest.mark.slow
@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("mixed_initial_state", [True, False])
def test_agreement_with_mesolve_for_negative_rates(
        improved_sampling, mixed_initial_state):
    """
    A rough test that nm_mcsolve agress with mesolve in the
    presence of negative rates.
    """
    times = np.linspace(0, 0.25, 51)
    if mixed_initial_state:
        state0 = qutip.maximally_mixed_dm(2)
    else:
        state0 = qutip.basis(2, 1)
    a0 = qutip.destroy(2)
    H = a0.dag() * a0
    e_ops = [
        a0.dag() * a0,
        a0 * a0.dag(),
    ]

    # Rate functions
    kappa = 1.0 / 0.129
    nth = 0.063
    args = {
        "kappa": kappa,
        "nth": nth,
    }
    gamma1 = "kappa * nth"
    gamma2 = "kappa * (nth+1) + 12 * exp(-2*t**3) * (-sin(15*t)**2)"

    # nm_mcsolve integration
    ops_and_rates = [
        [a0.dag(), gamma1],
        [a0, gamma2],
    ]
    mc_result = nm_mcsolve(
        H, state0, times, ops_and_rates,
        args=args, e_ops=e_ops, ntraj=1000 if improved_sampling else 2000,
        options={"rtol": 1e-8, "improved_sampling": improved_sampling},
        seeds=0,
    )

    # mesolve integration for comparison
    d_ops = [
        [qutip.lindblad_dissipator(a0.dag(), a0.dag()), gamma1],
        [qutip.lindblad_dissipator(a0, a0), gamma2],
    ]
    me_result = qutip.mesolve(
        H, state0, times, d_ops,
        args=args, e_ops=e_ops,
    )

    np.testing.assert_allclose(mc_result.trace, [1.] * len(times), rtol=0.25)
    np.testing.assert_allclose(
        me_result.expect[0], mc_result.expect[0], rtol=0.25,
    )
    np.testing.assert_allclose(
        me_result.expect[1], mc_result.expect[1], rtol=0.25,
    )


def test_completeness_relation():
    """
    NonMarkovianMCSolver guarantees that the operators in solver.ops
    satisfy the completeness relation ``sum(Li.dag() * Li) = a*I`` where a is a
    constant and I the identity.
    """
    # some arbitrary H
    H = qutip.sigmaz()
    ground_state = qutip.basis(2, 1)
    # test using all combinations of the following operators
    from itertools import combinations
    all_ops_and_rates = [
        (qutip.sigmap(), 1),
        (qutip.sigmam(), 1),
        (qutip.sigmaz(), 1),
        (1j * qutip.qeye(2), 1),
    ]
    # empty ops_and_rates not allowed
    for n in range(1, len(all_ops_and_rates) + 1):
        for ops_and_rates in combinations(all_ops_and_rates, n):
            solver = NonMarkovianMCSolver(H, ops_and_rates)
            op = sum((L.dag() * L) for L in solver.ops)
            a_candidate = qutip.expect(op, ground_state)
            assert op == a_candidate * qutip.qeye(op.dims[0])


def test_solver_pickleable():
    """
    NonMarkovianMCSolver objects must be pickleable for multiprocessing.
    """
    import pickle
    # arbitrary Hamiltonian and Lindblad operator
    H = qutip.sigmaz()
    L = qutip.sigmam()
    # try various types of coefficient functions
    rates = [
        0,
        _return_constant,
        "sin(t)",
    ]
    args = [
        {},
        {'constant': 1},
        {},
    ]
    for rate, arg in zip(rates, args):
        op_and_rate = (L, qutip.coefficient(rate, args=arg))
        solver = NonMarkovianMCSolver(H, [op_and_rate])
        jar = pickle.dumps(solver)

        loaded_solver = pickle.loads(jar)
        assert len(solver.ops) == len(loaded_solver.ops)
        for i in range(len(solver.ops)):
            assert solver.ops[i] == loaded_solver.ops[i]
            _assert_functions_equal(lambda t: solver.rate(t, i),
                                    lambda t: loaded_solver.rate(t, i))
        _assert_functions_equal(solver.rate_shift, loaded_solver.rate_shift)


def _assert_functions_equal(f1, f2):
    times = np.linspace(0, 1)
    values1 = [f1(t) for t in times]
    values2 = [f2(t) for t in times]
    np.testing.assert_allclose(values1, values2)


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
    Mixin class to test the states and expectation values from nm_mcsolve.
    """
    size = 10
    h = qutip.num(size)
    pure_state = qutip.basis(size, size-1)
    mixed_state = qutip.maximally_mixed_dm(size)
    times = np.linspace(0, 1, 101)
    e_ops = [qutip.num(size)]
    ntraj = 500

    def _assert_states(self, result, expected, tol):
        assert hasattr(result, 'states')
        assert len(result.states) == len(self.times)
        assert len(self.e_ops) == len(expected)
        for test_operator, expected_part in zip(self.e_ops, expected):
            test = qutip.expect(test_operator, result.states)
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    def _assert_expect(self, result, expected, tol):
        assert hasattr(result, 'expect')
        assert len(result.expect) == len(self.e_ops)
        assert len(self.e_ops) == len(expected)
        for test, expected_part in zip(result.expect, expected):
            np.testing.assert_allclose(test, expected_part, rtol=tol)

    @pytest.mark.parametrize("improved_sampling", [True, False])
    def test_states_and_expect(
        self, hamiltonian, state, args, ops_and_rates,
        expected, tol, improved_sampling
    ):
        options = {"store_states": True, "map": "serial",
                   "improved_sampling": improved_sampling}
        result = nm_mcsolve(
            hamiltonian, state, self.times, args=args,
            ops_and_rates=ops_and_rates,
            e_ops=self.e_ops, ntraj=self.ntraj, options=options,
            # target_tol not supported for mixed initial state
            target_tol=(0.05 if state.isket else None)
        )
        self._assert_expect(result, expected, tol)
        self._assert_states(result, expected, tol)


class TestNoCollapse(StatesAndExpectOutputCase):
    """
    Test that nm_mcsolve correctly solves the system when there is a constant
    Hamiltonian and no collapses.
    """

    def pytest_generate_tests(self, metafunc):
        tol = 1e-8
        hamiltonian_types = [
            (self.h, "Qobj"),
            ([self.h], "list"),
            (qutip.QobjEvo(
                [self.h, [self.h, _return_constant]],
                args={'constant': 0}), "QobjEvo"),
            (callable_qobj(self.h), "callable"),
        ]
        cases = [
            pytest.param(hamiltonian, {}, [], tol, id=id)
            for hamiltonian, id in hamiltonian_types
        ]
        metafunc.parametrize([
            'hamiltonian', 'args', 'ops_and_rates', 'tol',
        ], cases)

        initial_state_types = [
            (self.pure_state, "pure"),
            (self.mixed_state, "mixed"),
        ]
        expect = [qutip.expect(self.e_ops[0], state) * np.ones_like(self.times)
                  for state, _ in initial_state_types]
        cases = [pytest.param(state, [exp], id=id)
                 for (state, id), exp in zip(initial_state_types, expect)]
        metafunc.parametrize(['state', 'expected'], cases)

    # Previously the "states_only" and "expect_only" tests were mixed in to
    # every other test case.  We move them out into the simplest set so that
    # their behaviour remains tested, but isn't repeated as often to keep test
    # runtimes shorter.  The known-good cases are still tested in the other
    # test cases, this is just testing the single-output behaviour.

    @pytest.mark.parametrize("improved_sampling", [True, False])
    def test_states_only(self, hamiltonian, state, args, ops_and_rates,
                         expected, tol, improved_sampling):
        options = {"store_states": True, "map": "serial",
                   "improved_sampling": improved_sampling}
        result = nm_mcsolve(
            hamiltonian, state, self.times, args=args,
            ops_and_rates=ops_and_rates,
            e_ops=[], ntraj=self.ntraj, options=options,
        )
        self._assert_states(result, expected, tol)

    @pytest.mark.parametrize("improved_sampling", [True, False])
    def test_expect_only(self, hamiltonian, state, args, ops_and_rates,
                         expected, tol, improved_sampling):
        options = {'map': 'serial', "improved_sampling": improved_sampling}
        result = nm_mcsolve(
            hamiltonian, state, self.times, args=args,
            ops_and_rates=ops_and_rates,
            e_ops=self.e_ops, ntraj=self.ntraj, options=options,
        )
        self._assert_expect(result, expected, tol)


class TestConstantCollapse(StatesAndExpectOutputCase):
    """
    Test that nm_mcsolve correctly solves the system when the
    collapse rates are constant.
    """

    def pytest_generate_tests(self, metafunc):
        tol = 0.25
        rate = 0.2
        op = qutip.destroy(self.size)
        op_and_rate_types = [
            ([op, rate], {}, "constant"),
            ([op, '1 * {}'.format(rate)], {}, "string"),
            ([op, lambda t: rate], {}, "function"),
            ([op, lambda t, w: rate], {"w": 1.0}, "function_with_args"),
        ]
        cases = [
            pytest.param(self.h, args, [op_and_rate], tol, id=id)
            for op_and_rate, args, id in op_and_rate_types
        ]
        metafunc.parametrize([
            'hamiltonian', 'args', 'ops_and_rates', 'tol',
        ], cases)

        initial_state_types = [
            (self.pure_state, "pure"),
            (self.mixed_state, "mixed"),
        ]
        expect = [(qutip.expect(self.e_ops[0], state)
                   * np.exp(-rate * self.times))
                  for state, _ in initial_state_types]
        cases = [pytest.param(state, [exp], id=id)
                 for (state, id), exp in zip(initial_state_types, expect)]
        metafunc.parametrize(['state', 'expected'], cases)


class TestTimeDependentCollapse(StatesAndExpectOutputCase):
    """
    Test that nm_mcsolve correctly solves the system when the
    collapse rates are time-dependent.
    """

    def pytest_generate_tests(self, metafunc):
        tol = 0.25
        coupling = 0.2
        op = qutip.destroy(self.size)
        rate_args = {'constant': coupling, 'rate': 0.5}
        rate_string = 'sqrt({} * exp(-t))'.format(coupling)
        op_and_rate_types = [
            ([op, rate_string], {}, "string"),
            ([op, _return_decay], rate_args, "function"),
        ]
        cases = [
            pytest.param(self.h, args, [op_and_rate], tol, id=id)
            for op_and_rate, args, id in op_and_rate_types
        ]
        metafunc.parametrize([
            'hamiltonian', 'args', 'ops_and_rates', 'tol',
        ], cases)

        initial_state_types = [
            (self.pure_state, "pure"),
            (self.mixed_state, "mixed"),
        ]
        expect = [(qutip.expect(self.e_ops[0], state)
                   * np.exp(-coupling * (1 - np.exp(-self.times))))
                  for state, _ in initial_state_types]
        cases = [pytest.param(state, [exp], id=id)
                 for (state, id), exp in zip(initial_state_types, expect)]
        metafunc.parametrize(['state', 'expected'], cases)


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
    ops_and_rates = [
        (a, 1.0),
        (a, 1.0),
    ]
    result = nm_mcsolve(
        H, state, times, ops_and_rates, ntraj=3,
        options={"map": "serial"},
    )
    assert len(result.col_times[0]) > 0
    assert len(result.col_which) == len(result.col_times)
    assert all(col in [0, 1] for col in result.col_which[0])


@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("keep_runs_results", [True, False])
@pytest.mark.parametrize("mixed_initial_state", [True, False])
def test_states_outputs(keep_runs_results, improved_sampling,
                        mixed_initial_state):
    # We're just testing the output value, so it's important whether certain
    # things are complex or real, but not what the magnitudes of constants are.
    focks = 5
    ntraj = 13
    a = qutip.destroy(focks) & qutip.qeye(2)
    sm = qutip.qeye(focks) & qutip.sigmam()
    H = 1j*a.dag()*sm + a
    H = H + H.dag()
    if mixed_initial_state:
        state = qutip.maximally_mixed_dm(focks) & qutip.fock_dm(2, 1)
    else:
        state = qutip.basis([focks, 2], [0, 1])
    times = np.linspace(0, 10, 21)
    ops_and_rates = [
        (a, 1.0),
        (sm, 1.0),
    ]
    # nm_mcsolve adds one more operator to complete the operator set
    # which results in the len(ops_and_rates) + 1 below:
    total_ops = len(ops_and_rates) + 1
    data = nm_mcsolve(
        H, state, times, ops_and_rates, ntraj=ntraj,
        options={
            "keep_runs_results": keep_runs_results,
            "map": "serial",
            "improved_sampling": improved_sampling})

    assert len(data.average_states) == len(times)
    assert isinstance(data.average_states[0], qutip.Qobj)
    assert data.average_states[0].norm() == pytest.approx(1.)
    assert data.average_states[0].isoper
    if state.isket:
        assert data.average_states[0] == qutip.ket2dm(state)
    else:
        assert data.average_states[0] == state

    assert isinstance(data.average_final_state, qutip.Qobj)
    assert data.average_final_state.norm() == pytest.approx(1.)
    assert data.average_final_state.isoper

    if keep_runs_results:
        assert len(data.runs_states) == ntraj
        assert len(data.runs_states[0]) == len(times)
        assert isinstance(data.runs_states[0][0], qutip.Qobj)
        assert data.runs_states[0][0].norm() == pytest.approx(1.)
        assert data.runs_states[0][0].isket

        assert len(data.runs_final_states) == ntraj
        assert isinstance(data.runs_final_states[0], qutip.Qobj)
        assert data.runs_final_states[0].norm() == pytest.approx(1.)
        assert data.runs_final_states[0].isket

    steady_state = data.steady_state()
    assert isinstance(steady_state, qutip.Qobj)
    assert steady_state.norm() == pytest.approx(1.)
    assert steady_state.isoper

    np.testing.assert_allclose(times, data.times)
    assert data.num_trajectories == ntraj
    assert len(data.e_ops) == 0
    assert data.stats["num_collapse"] == total_ops
    assert len(data.col_times) == ntraj
    assert np.max(np.hstack(data.col_which)) <= total_ops
    assert data.stats['end_condition'] == "ntraj reached"


@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("keep_runs_results", [True, False])
@pytest.mark.parametrize("mixed_initial_state", [True, False])
def test_expectation_outputs(keep_runs_results, improved_sampling,
                             mixed_initial_state):
    # We're just testing the output value, so it's important whether certain
    # things are complex or real, but not what the magnitudes of constants are.
    focks = 5
    ntraj = 13
    a = qutip.destroy(focks) & qutip.qeye(2)
    sm = qutip.qeye(focks) & qutip.sigmam()
    H = 1j*a.dag()*sm + a
    H = H + H.dag()
    if mixed_initial_state:
        state = qutip.maximally_mixed_dm(focks) & qutip.fock_dm(2, 1)
    else:
        state = qutip.basis([focks, 2], [0, 1])
    times = np.linspace(0, 10, 5)
    ops_and_rates = [
        (a, 1.0),
        (sm, 1.0),
    ]
    # nm_mcsolve adds one more operator to complete the operator set
    # which results in the len(ops_and_rates) + 1 below:
    total_ops = len(ops_and_rates) + 1
    e_ops = [a.dag()*a, sm.dag()*sm, a]
    data = nm_mcsolve(
        H, state, times, ops_and_rates, e_ops=e_ops, ntraj=ntraj,
        options={
            "keep_runs_results": keep_runs_results,
            "map": "serial",
            "improved_sampling": improved_sampling})
    assert isinstance(data.average_expect[0][1], float)
    assert isinstance(data.average_expect[1][1], float)
    assert isinstance(data.average_expect[2][1], complex)
    assert isinstance(data.std_expect[0][1], float)
    assert isinstance(data.std_expect[1][1], float)
    assert isinstance(data.std_expect[2][1], float)
    if keep_runs_results:
        assert len(data.runs_expect) == len(e_ops)
        assert len(data.runs_expect[0]) == ntraj
        assert isinstance(data.runs_expect[0][0][1], float)
        assert isinstance(data.runs_expect[1][0][1], float)
        assert isinstance(data.runs_expect[2][0][1], complex)
    np.testing.assert_allclose(times, data.times)
    assert data.num_trajectories == ntraj
    assert len(data.e_ops) == len(e_ops)
    assert data.stats["num_collapse"] == total_ops
    assert len(data.col_times) == ntraj
    assert np.max(np.hstack(data.col_which)) <= total_ops
    assert data.stats['end_condition'] == "ntraj reached"


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
    ops_and_rates = [
        (qutip.tensor(a[0], qutip.qeye(sizes[1:])), 2 * dampings[0]),
        (
            qutip.tensor(qutip.qeye(sizes[0]), a[1], qutip.qeye(sizes[2])),
            2 * dampings[1],
        ),
        (qutip.tensor(qutip.qeye(sizes[:2]), a[2]), 2 * dampings[2]),
    ]

    def test_seeds_can_be_reused(self):
        args = (self.H, self.state, self.times)
        kwargs = {'ops_and_rates': self.ops_and_rates, 'ntraj': self.ntraj}
        first = nm_mcsolve(*args, **kwargs)
        second = nm_mcsolve(*args, seeds=first.seeds, **kwargs)
        for first_t, second_t in zip(first.col_times, second.col_times):
            np.testing.assert_equal(first_t, second_t)
        for first_w, second_w in zip(first.col_which, second.col_which):
            np.testing.assert_equal(first_w, second_w)

    def test_seeds_are_not_reused_by_default(self):
        args = (self.H, self.state, self.times)
        kwargs = {'ops_and_rates': self.ops_and_rates, 'ntraj': self.ntraj}
        first = nm_mcsolve(*args, **kwargs)
        second = nm_mcsolve(*args, **kwargs)
        assert not all(np.array_equal(first_t, second_t)
                       for first_t, second_t in zip(first.col_times,
                                                    second.col_times))
        assert not all(np.array_equal(first_w, second_w)
                       for first_w, second_w in zip(first.col_which,
                                                    second.col_which))

    @pytest.mark.parametrize('seed', [1, np.random.SeedSequence(2)])
    def test_seed_type(self, seed):
        args = (self.H, self.state, self.times)
        kwargs = {'ops_and_rates': self.ops_and_rates, 'ntraj': self.ntraj}
        first = nm_mcsolve(*args, seeds=copy(seed), **kwargs)
        second = nm_mcsolve(*args, seeds=copy(seed), **kwargs)
        for f_seed, s_seed in zip(first.seeds, second.seeds):
            assert f_seed.state == s_seed.state

    def test_bad_seed(self):
        args = (self.H, self.state, self.times)
        kwargs = {'ops_and_rates': self.ops_and_rates, 'ntraj': self.ntraj}
        with pytest.raises(ValueError):
            nm_mcsolve(*args, seeds=[1], **kwargs)

    def test_generator(self):
        args = (self.H, self.state, self.times)
        kwargs = {'ops_and_rates': self.ops_and_rates, 'ntraj': self.ntraj}
        first = nm_mcsolve(
            *args, seeds=1, options={'bitgenerator': 'MT19937'},
            **kwargs,
        )
        second = nm_mcsolve(*args, seeds=1, **kwargs)
        for f_seed, s_seed in zip(first.seeds, second.seeds):
            assert f_seed.state == s_seed.state
        assert not all(np.array_equal(first_t, second_t)
                       for first_t, second_t in zip(first.col_times,
                                                    second.col_times))
        assert not all(np.array_equal(first_w, second_w)
                       for first_w, second_w in zip(first.col_which,
                                                    second.col_which))

    def test_stepping(self):
        size = 10
        a = qutip.destroy(size)
        H = qutip.num(size)
        ops_and_rates = [(a, qutip.coefficient('alpha', args={'alpha': 0}))]
        mcsolver = NonMarkovianMCSolver(
            H, ops_and_rates, options={'map': 'serial'},
        )
        mcsolver.start(qutip.basis(size, size-1), 0, seed=5)
        state_1 = mcsolver.step(1, args={'alpha': 1})

        mcsolver.start(qutip.basis(size, size-1), 0, seed=5)
        state_2 = mcsolver.step(1, args={'alpha': 1})
        assert state_1 == state_2


@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("mixed_initial_state", [True, False])
def test_timeout(improved_sampling, mixed_initial_state):
    size = 10
    ntraj = 1000
    a = qutip.destroy(size)
    H = qutip.num(size)
    if mixed_initial_state:
        state = qutip.maximally_mixed_dm(size)
    else:
        state = qutip.basis(size, size-1)
    times = np.linspace(0, 1.0, 100)
    coupling = 0.5
    n_th = 0.05
    ops_and_rates = [
        (a, np.sqrt(coupling * (n_th + 1)))
    ]
    e_ops = [qutip.num(size)]
    res = nm_mcsolve(
        H, state, times, ops_and_rates, e_ops=e_ops, ntraj=ntraj,
        options={'map': 'serial', "improved_sampling": improved_sampling},
        timeout=1e-6,
    )
    assert res.stats['end_condition'] == 'timeout'


@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("mixed_initial_state", [True, False])
def test_super_H(improved_sampling, mixed_initial_state):
    size = 10
    ntraj = 250
    a = qutip.destroy(size)
    H = qutip.num(size)
    if mixed_initial_state:
        state = qutip.maximally_mixed_dm(size)
    else:
        state = qutip.basis(size, size-1)
    times = np.linspace(0, 1.0, 100)
    # Arbitrary coupling and bath temperature.
    coupling = 0.5
    n_th = 0.05
    ops_and_rates = [
        (a, np.sqrt(coupling * (n_th + 1)))
    ]
    e_ops = [qutip.num(size)]
    mc_expected = nm_mcsolve(
        H, state, times, ops_and_rates, e_ops=e_ops, ntraj=ntraj,
        target_tol=(0.1 if state.isket else None),
        options={'map': 'serial', "improved_sampling": improved_sampling},
    )
    mc = nm_mcsolve(
        qutip.liouvillian(H), state, times, ops_and_rates, e_ops=e_ops,
        ntraj=ntraj, target_tol=(0.1 if state.isket else None),
        options={'map': 'serial', "improved_sampling": improved_sampling})
    np.testing.assert_allclose(mc_expected.expect[0], mc.expect[0], atol=0.65)


def test_NonMarkovianMCSolver_run():
    size = 10
    args = {'coupling': 0}
    ops_and_rates = [
        (qutip.destroy(size), qutip.coefficient('coupling', args=args))
    ]
    H = qutip.num(size)
    solver = NonMarkovianMCSolver(H, ops_and_rates)
    solver.options = {'store_final_state': True}
    res = solver.run(qutip.basis(size, size-1), np.linspace(0, 5.0, 11),
                     e_ops=[qutip.qeye(size)], args={'coupling': 1})
    assert res.final_state is not None
    assert len(res.collapse[0]) != 0
    assert res.num_trajectories == 1
    np.testing.assert_allclose(res.expect[0], np.ones(11))
    res += solver.run(
        qutip.basis(size, size-1), np.linspace(0, 5.0, 11),
        e_ops=[qutip.qeye(size)], args={'coupling': 1},
        ntraj=1000, target_tol=0.1
    )
    assert 1 < res.num_trajectories < 1001


def test_NonMarkovianMCSolver_stepping():
    size = 10
    args = {'coupling': 0}
    ops_and_rates = [
        (qutip.destroy(size), qutip.coefficient('coupling', args=args))
    ]
    H = qutip.num(size)
    solver = NonMarkovianMCSolver(H, ops_and_rates)
    solver.start(qutip.basis(size, size-1), 0, seed=0)
    state = solver.step(1)
    assert qutip.expect(qutip.qeye(size), state) == pytest.approx(1)
    assert qutip.expect(qutip.num(size), state) == pytest.approx(size - 1)
    assert state.isoper
    assert solver.rate_shift(1) == 0
    assert solver.rate(1, 0) == 0
    assert solver.sqrt_shifted_rate(1, 0) == 0
    state = solver.step(5, args={'coupling': 5})
    assert qutip.expect(qutip.qeye(size), state) == pytest.approx(1)
    assert qutip.expect(qutip.num(size), state) <= size - 1
    assert state.isoper
    assert solver.rate_shift(5) == 0
    assert solver.rate(5, 0) == 5
    assert solver.sqrt_shifted_rate(5, 0) == np.sqrt(5)


# Defined in module-scope so it's pickleable.
def _dynamic(t, args):
    return 0 if args["collapse"] else 1


@pytest.mark.xfail(reason="current limitation of NonMarkovianMCSolver")
def test_dynamic_arguments():
    """Test dynamically updated arguments are usable."""
    size = 5
    a = qutip.destroy(size)
    H = qutip.num(size)
    times = np.linspace(0, 1, 11)
    state = qutip.basis(size, 2)

    ops_and_rates = [[a, _dynamic], [a.dag(), _dynamic]]
    mc = nm_mcsolve(
        H, state, times, ops_and_rates, ntraj=25, args={"collapse": []},
    )
    assert all(len(collapses) <= 1 for collapses in mc.col_which)


@pytest.mark.parametrize(["initial_state", "ntraj"], [
    pytest.param(qutip.maximally_mixed_dm(2), 5, id="dm"),
    pytest.param([(qutip.basis(2, 0), 0.3), (qutip.basis(2, 1), 0.7)],
                 5, id="statelist"),
    pytest.param([(qutip.basis(2, 0), 0.3), (qutip.basis(2, 1), 0.7)],
                 [4, 2], id="ntraj-spec"),
    pytest.param([(qutip.basis(2, 0), 0.3),
                  ((qutip.basis(2, 0) + qutip.basis(2, 1)).unit(), 0.7)],
                 [4, 2], id="non-orthogonals"),
])
@pytest.mark.parametrize("improved_sampling", [True, False])
def test_mixed_averaging(improved_sampling, initial_state, ntraj):
    # we will only check that the initial state of the result equals the
    # intended initial state exactly
    H = qutip.sigmax()
    tlist = [0, 1]
    L = qutip.sigmam()
    rate = -1

    solver = qutip.NonMarkovianMCSolver(
        H, [(L, rate)], options={'improved_sampling': improved_sampling})
    result = solver.run(initial_state, tlist, ntraj)

    if isinstance(initial_state, qutip.Qobj):
        reference = initial_state
    else:
        reference = sum(p * psi.proj() for psi, p in initial_state)

    assert result.states[0] == reference
    assert result.num_trajectories == np.sum(ntraj)

    assert hasattr(result, 'initial_states')
    assert isinstance(result.initial_states, list)
    assert all(isinstance(st, qutip.Qobj) for st in result.initial_states)
    assert hasattr(result, 'ntraj_per_initial_state')
    assert isinstance(result.ntraj_per_initial_state, list)
    assert len(result.ntraj_per_initial_state) == len(result.initial_states)
    if isinstance(ntraj, list):
        assert result.ntraj_per_initial_state == ntraj
    else:
        assert sum(result.ntraj_per_initial_state) == ntraj
    assert (
        sum(result.deterministic_weights + result.runs_weights)
        == pytest.approx(1.)
    )


@pytest.mark.parametrize("improved_sampling", [True, False])
@pytest.mark.parametrize("p", [0, 0.25, 0.5])
def test_mixed_equals_merged(improved_sampling, p):
    # Running mcsolve with mixed ICs should be the same as running mcsolve
    # multiple times and merging the results afterwards
    initial_state1 = qutip.basis(2, 1)
    initial_state2 = (qutip.basis(2, 1) + qutip.basis(2, 0)).unit()
    H = qutip.sigmax()
    L = qutip.sigmam()
    def rate_function(t):
        return -1 + t
    tlist = np.linspace(0, 2, 20)
    ntraj = [3, 9]

    solver = qutip.NonMarkovianMCSolver(
        H, [(L, qutip.coefficient(rate_function))],
        options={'improved_sampling': improved_sampling})
    mixed_result = solver.run(
        [(initial_state1, p), (initial_state2, 1 - p)], tlist, ntraj)

    # Reuse seeds, then results should be identical
    seeds = mixed_result.seeds
    seeds1 = seeds[:ntraj[0]]
    seeds2 = seeds[ntraj[0]:]

    pure_result1 = solver.run(initial_state1, tlist, ntraj[0], seeds=seeds1)
    pure_result2 = solver.run(initial_state2, tlist, ntraj[1], seeds=seeds2)
    merged_result = pure_result1.merge(pure_result2, p)

    assert mixed_result.num_trajectories == sum(ntraj)
    assert merged_result.num_trajectories == sum(ntraj)
    for state1, state2 in zip(mixed_result.states, merged_result.states):
        assert state1 == state2

    assert hasattr(mixed_result, 'initial_states')
    assert isinstance(mixed_result.initial_states, list)
    assert mixed_result.initial_states == [initial_state1, initial_state2]
    assert hasattr(mixed_result, 'ntraj_per_initial_state')
    assert isinstance(mixed_result.ntraj_per_initial_state, list)
    assert mixed_result.ntraj_per_initial_state == ntraj
    assert (
        sum(mixed_result.runs_weights + mixed_result.deterministic_weights)
        == pytest.approx(1.)
    )
    assert (
        sum(merged_result.runs_weights + merged_result.deterministic_weights)
        == pytest.approx(1.)
    )
