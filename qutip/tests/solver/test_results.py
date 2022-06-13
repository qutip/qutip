import numpy as np
import pytest

import qutip
from qutip.solver import SolverResultsOptions
from qutip.solver.result import Result, MultiTrajResult, McResult


def e_op_state_by_time(t, state):
    """ An e_ops function that returns the state multiplied by the time. """
    return t * state


class TestResult:
    @pytest.mark.parametrize(["N", "e_ops", "options"], [
        pytest.param(10, (), {}, id="no-e-ops"),
        pytest.param(
            10, qutip.create(10), {"store_states": True}, id="store-states",
        )
    ])
    def test_states(self, N, e_ops, options):
        res = Result(e_ops, SolverResultsOptions(**options))
        for i in range(N):
            res.add(i, qutip.basis(N, i))
        np.testing.assert_allclose(np.array(res.times), np.arange(N))
        assert res.states == [qutip.basis(N, i) for i in range(N)]
        assert res.final_state == qutip.basis(N, N-1)

    @pytest.mark.parametrize(["N", "e_ops", "options"], [
        pytest.param(10, (), {
            "store_states": False, "store_final_state": True
        }, id="no-e-ops"),
        pytest.param(10, qutip.create(10), {
            "store_final_state": True,
        }, id="with-eops"),
    ])
    def test_final_state_only(self, N, e_ops, options):
        res = Result(e_ops, SolverResultsOptions(**options))
        for i in range(N):
            res.add(i, qutip.basis(N, i))
        np.testing.assert_allclose(np.array(res.times), np.arange(N))
        assert res.states == []
        assert res.final_state == qutip.basis(N, N-1)

    @pytest.mark.parametrize(["N", "e_ops", "results"], [
        pytest.param(
            10, qutip.num(10), {0: np.arange(10)}, id="single-e-op",
        ),
        pytest.param(
            10,
            [qutip.num(10), qutip.qeye(10)],
            {0: np.arange(10), 1: np.ones(10)},
            id="list-e-ops",
        ),
        pytest.param(
            10,
            {"a": qutip.num(10), "b": qutip.qeye(10)},
            {"a": np.arange(10), "b": np.ones(10)},
            id="dict-e-ops",
        ),
        pytest.param(
            5, qutip.QobjEvo(qutip.num(5)), {0: np.arange(5)}, id="qobjevo",
        ),
        pytest.param(
            5, e_op_state_by_time,
            {0: [i * qutip.basis(5, i) for i in range(5)]},
            id="function",
        )
    ])
    def test_expect_and_e_ops(self, N, e_ops, results):
        res = Result(
            e_ops,
            SolverResultsOptions(
                store_final_state=False,
                store_states=False),
        )
        if isinstance(e_ops, dict):
            raw_ops = e_ops
        elif isinstance(e_ops, (list, tuple)):
            raw_ops = dict(enumerate(e_ops))
        else:
            raw_ops = {0: e_ops}
        for i in range(N):
            res.add(i, qutip.basis(N, i))
        np.testing.assert_allclose(np.array(res.times), np.arange(N))
        assert res.final_state is None
        assert not res.states
        for i, k in enumerate(results):
            assert res.e_ops[k].op is raw_ops[k]
            e_op_call_values = [
                res.e_ops[k](i, qutip.basis(N, i)) for i in range(N)
            ]
            if isinstance(res.expect[i][0], qutip.Qobj):
                assert res.expect[i] == results[k]
                assert res.e_data[k] == results[k]
                assert e_op_call_values == results[k]
            else:
                np.testing.assert_allclose(res.expect[i], results[k])
                np.testing.assert_allclose(res.e_data[k], results[k])
                np.testing.assert_allclose(e_op_call_values, results[k])

    def test_add_processor(self):
        res = Result([], SolverResultsOptions(store_states=False))
        a = []
        b = []
        states = [{"t": 0}, {"t": 1}]

        res.add_processor(lambda t, state: a.append((t, state)))
        res.add(0, states[0])
        res.add_processor(
            lambda t, state: b.append((t, state)),
            requires_copy=True,
        )
        res.add(1, states[1])

        assert a == [(0, {"t": 0}), (1, {"t": 1})]
        assert a[0][1] is states[0]  # no copy made
        assert a[1][1] is not states[1]  # copy made (for b)
        assert b == [(1, {"t": 1})]
        assert b[0][1] is not states[1]  # copy made

    def test_repr_minimal(self):
        res = Result(
            [],
            SolverResultsOptions(store_final_state=False, store_states=False),
        )
        assert repr(res) == "\n".join([
            "<Result",
            "  Solver: None",
            "  Number of e_ops: 0",
            "  State not saved.",
            ">",
        ])

    def test_repr_full(self):
        res = Result(
            [qutip.num(5), qutip.qeye(5)],
            SolverResultsOptions(store_states=True),
            solver="test-solver",
            stats={"stat-a": 1, "stat-b": 2},
        )
        for i in range(5):
            res.add(i, qutip.basis(5, i))
        assert repr(res) == "\n".join([
            "<Result",
            "  Solver: test-solver",
            "  Solver stats:",
            "    stat-a: 1",
            "    stat-b: 2",
            "  Time interval: [0, 4] (5 steps)",
            "  Number of e_ops: 2",
            "  States saved.",
            ">",
        ])


class TestMultiTrajResult:
    def test_averages_and_states(self):
        N = 10
        e_ops = [qutip.num(N), qutip.qeye(N)]
        m_res = MultiTrajResult(3)
        opt = SolverResultsOptions(store_states=True)
        for _ in range(5):
            res = Result(e_ops, opt)
            res.collapse = []
            for i in range(N):
                res.add(i, (qutip.basis(N, i) / 2).unit())
                res.collapse.append((i+0.5, i % 2))
            m_res.add(res)

        np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
        np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
        np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
        for i in range(N):
            assert m_res.average_states[i] == qutip.fock_dm(N, i)
        assert m_res.average_final_state == qutip.fock_dm(N, N - 1)
        assert len(m_res.runs_states) == 5
        assert len(m_res.runs_states[0]) == N
        for i in range(5):
            assert m_res.runs_final_states[i] == qutip.basis(N, N - 1)
        assert np.all(np.array(m_res.col_which) < 2)


class TestMultiTrajResultAveraged:
    def test_averages_and_states(self):
        N = 10
        e_ops = [qutip.num(N), qutip.qeye(N)]
        m_res = MultiTrajResultAveraged(3)
        opt = SolverResultsOptions(
            store_final_state=True,
        )
        for _ in range(5):
            res = Result(e_ops, opt)
            res.collapse = []
            for i in range(N):
                res.add(i, (qutip.basis(N, i) / 2).unit())
                res.collapse.append((i + 0.5, i % 2))
            m_res.add(res)

        np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
        np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
        np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
        assert m_res.average_final_state == qutip.fock_dm(N, N-1)
        assert np.all(np.array(m_res.col_which) < 2)

# =============================================================================  ====================================================
@pytest.mark.parametrize('keep_runs_results', [True, False])
@pytest.mark.parametrize('format', ['dm', 'ket'])
def test_McResult(format, keep_runs_results):
    N = 10
    ntraj = 5
    e_ops = [qutip.num(N), qutip.qeye(N)]
    opt = qutip.solver.SolverResultsOptions(
        keep_runs_results=keep_runs_results,
        store_states=True,
    )
    state0 = qutip.basis(N, 0)
    if format == 'dm':
        state0 = qutip.ket2dm(state0)

    m_res = McResult(e_ops, opt, np.arange(N), state0, 2)
    m_res.set_expect_tol(None, ntraj)
    for _ in range(ntraj):
        res = m_res.spawn()
        res.collapse = []
        for i in range(1, N):
            state = qutip.basis(N, i) / 2
            if format == 'dm':
                state = qutip.ket2dm(state)
            res.add(i, state)
            res.collapse.append((i+0.5, i%2))
        m_res.add(res)

    np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
    np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
    np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
    for i in range(N):
        assert m_res.average_states[i] == qutip.fock_dm(N, i)
    assert m_res.average_final_state == qutip.fock_dm(N, N-1)
    if keep_runs_results:
        assert len(m_res.runs_states) == 5
        assert len(m_res.runs_states[0]) == N
        for i in range(ntraj):
            target = qutip.basis(N, N-1)
            if format == 'dm':
                target = target.proj()
            assert m_res.runs_final_states[i] == target
    else:
        assert m_res.runs_states is None
        assert m_res.runs_final_states is None
    assert np.all(np.array(m_res.col_which) < 2)
    np.testing.assert_allclose(np.array(m_res.times), np.arange(N))
    assert m_res.end_condition == "ntraj reached"
    assert isinstance(m_res.collapse, list)
    assert len(m_res.col_which[0]) == len(m_res.col_times[0])


@pytest.mark.parametrize('keep_runs_results', [True, False])
@pytest.mark.parametrize('e_ops',
    ['list', 'qobj', 'qobjevo', 'callable', 'dict', 'list_func'])
def test_multitraj_expect(keep_runs_results, e_ops):
    N = 10
    ntraj = 5
    opt = qutip.solver.SolverResultsOptions(
        keep_runs_results=keep_runs_results,
        store_states=True,
    )
    m_res = MultiTrajResult(
        _make_e_ops(N, e_ops),
        opt,
        np.arange(N),
        qutip.basis(N, 0)
    )
    for _ in range(ntraj):
        res = m_res.spawn()
        for i in range(1, N):
            res.add(i, qutip.basis(N, i) / 2)
        m_res.add(res)

    average_expect = _check_and_extract_expect(m_res.average_expect, e_ops)
    np.testing.assert_allclose(average_expect, np.arange(N))
    std_expect = _check_and_extract_expect(m_res.std_expect, e_ops)
    np.testing.assert_allclose(std_expect, np.zeros(N))
    if keep_runs_results:
        for i in range(ntraj):
            runs_expect = _check_and_extract_expect(m_res.runs_expect, e_ops)
            np.testing.assert_allclose(runs_expect[i], np.arange(N))
    else:
        assert m_res.runs_expect is None
    assert m_res.end_condition == "unknown"


@pytest.mark.parametrize('keep_runs_results', [True, False])
@pytest.mark.parametrize('ttol', [
    pytest.param(0.1, id='atol'),
    pytest.param([0.001, 0.1], id='rtol'),
    pytest.param([[0.001, 0.1], [0.1, 0]], id='tol_per_e_op'),
])
def test_multitraj_targettol(keep_runs_results, ttol):
    N = 10
    ntraj = 1000
    opt = qutip.solver.SolverResultsOptions(
        keep_runs_results=keep_runs_results,
        store_states=True,
    )
    m_res = MultiTrajResult(
        [qutip.num(N), qutip.qeye(N)],
        opt,
        np.arange(N),
        qutip.basis(N, 0)
    )
    m_res.set_expect_tol(ttol, ntraj)
    for _ in range(ntraj):
        res = m_res.spawn()
        for i in range(N):
            res.add(i, qutip.basis(N, i) * (1 - 0.5*np.random.rand()))
        if m_res.add(res) <= 0:
            break

    assert m_res.end_condition == "target tolerance reached"
    assert m_res.num_expect == 2
    assert m_res.num_traj <= 1000
