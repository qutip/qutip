import numpy as np
import pytest

import qutip
from qutip.solver import SolverResultsOptions
from qutip.solver.result import (
    Result, MultiTrajResult, MultiTrajResultAveraged,
)


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
