import numpy as np
import pytest

import qutip
from qutip.solver.result import Result, MultiTrajResult, McResult


def fill_options(**kwargs):
    return {
        "store_states": None,
        "store_final_state": False,
        "keep_runs_results": False,
        **kwargs
    }


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
        res = Result(e_ops, fill_options(**options))
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
        res = Result(e_ops, fill_options(**options))
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
            fill_options(
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
                assert (res.expect[i] == results[k]).all()
                assert res.e_data[k] == results[k]
                assert e_op_call_values == results[k]
            else:
                np.testing.assert_allclose(res.expect[i], results[k])
                np.testing.assert_allclose(res.e_data[k], results[k])
                np.testing.assert_allclose(e_op_call_values, results[k])

    def test_add_processor(self):
        res = Result([], fill_options(store_states=False))
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
            fill_options(store_final_state=False, store_states=False),
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
            fill_options(store_states=True),
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


def e_op_num(t, state):
    """ An e_ops function that returns the ground state occupation. """
    return state.dag() @ qutip.num(5) @ state


class TestMultiTrajResult:
    def _fill_trajectories(self, multiresult, N, ntraj,
                           collapse=False, noise=0, dm=False):
        # Fix the seed to avoid failing due to bad luck
        np.random.seed(1)
        for _ in range(ntraj):
            result = Result(multiresult._raw_ops, multiresult.options)
            result.collapse = []
            for t in range(N):
                delta = 1 + noise * np.random.randn()
                state = qutip.basis(N, t) * delta
                if dm:
                    state = state.proj()
                result.add(t, state)
                if collapse:
                    result.collapse.append((t+0.1, 0))
                    result.collapse.append((t+0.2, 1))
                    result.collapse.append((t+0.3, 1))
            if multiresult.add((0, result)) <= 0:
                break

    def _expect_check_types(self, multiresult):
        assert isinstance(multiresult.std_expect, list)
        assert isinstance(multiresult.average_e_data, dict)
        assert isinstance(multiresult.std_expect, list)
        assert isinstance(multiresult.average_e_data, dict)

        if multiresult.trajectories:
            assert isinstance(multiresult.runs_expect, list)
            assert isinstance(multiresult.runs_e_data, dict)
        else:
            assert multiresult.runs_expect == []
            assert multiresult.runs_e_data == {}

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    @pytest.mark.parametrize('dm', [True, False])
    def test_McResult(self, dm, keep_runs_results):
        N = 10
        ntraj = 5
        e_ops = [qutip.num(N), qutip.qeye(N)]
        opt = fill_options(keep_runs_results=keep_runs_results)

        m_res = McResult(e_ops, opt, stats={"num_collapse": 2})
        m_res.add_end_condition(ntraj, None)
        self._fill_trajectories(m_res, N, ntraj, collapse=True, dm=dm)

        np.testing.assert_allclose(np.array(m_res.times), np.arange(N))
        assert m_res.stats['end_condition'] == "ntraj reached"
        self._expect_check_types(m_res)

        assert np.all(np.array(m_res.col_which) < 2)
        assert isinstance(m_res.collapse, list)
        assert len(m_res.col_which[0]) == len(m_res.col_times[0])
        np.testing.assert_allclose(m_res.photocurrent[0], np.ones(N-1))
        np.testing.assert_allclose(m_res.photocurrent[1], 2 * np.ones(N-1))

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    @pytest.mark.parametrize(["e_ops", "results"], [
        pytest.param(qutip.num(5), [np.arange(5)], id="single-e-op"),
        pytest.param(
            {"a": qutip.num(5), "b": qutip.qeye(5)},
            [np.arange(5), np.ones(5)],
            id="dict-e-ops",
        ),
        pytest.param(qutip.QobjEvo(qutip.num(5)), [np.arange(5)], id="qobjevo"),
        pytest.param(e_op_num, [np.arange(5)], id="function"),
        pytest.param(
            [qutip.num(5), e_op_num],
            [np.arange(5), np.arange(5)],
            id="list-e-ops",
        ),
    ])
    def test_multitraj_expect(self, keep_runs_results, e_ops, results):
        N = 5
        ntraj = 25
        opt = fill_options(keep_runs_results=keep_runs_results)
        m_res = MultiTrajResult(e_ops, opt, stats={})
        self._fill_trajectories(m_res, N, ntraj, noise=0.01)

        for expect, expected in zip(m_res.average_expect, results):
            np.testing.assert_allclose(expect, expected,
                                       atol=1e-14, rtol=0.02)

        for variance, expected in zip(m_res.std_expect, results):
            np.testing.assert_allclose(variance, 0.02 * expected,
                                       atol=1e-14, rtol=0.9)

        if keep_runs_results:
            for runs_expect, expected in zip(m_res.runs_expect, results):
                for expect in runs_expect:
                    np.testing.assert_allclose(expect, expected,
                                               atol=1e-14, rtol=0.1)

        self._expect_check_types(m_res)

        assert m_res.stats['end_condition'] == "unknown"

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    @pytest.mark.parametrize('dm', [True, False])
    def test_multitraj_state(self, keep_runs_results, dm):
        N = 5
        ntraj = 25
        opt = fill_options(keep_runs_results=keep_runs_results)
        m_res = MultiTrajResult([], opt)
        self._fill_trajectories(m_res, N, ntraj, dm=dm)

        np.testing.assert_allclose(np.array(m_res.times), np.arange(N))

        for i in range(N):
            assert m_res.average_states[i] == qutip.fock_dm(N, i)
        assert m_res.average_final_state == qutip.fock_dm(N, N-1)

        if keep_runs_results:
            assert len(m_res.runs_states) == 25
            assert len(m_res.runs_states[0]) == N
            for i in range(ntraj):
                expected = qutip.basis(N, N-1)
                if dm:
                    expected = expected.proj()
                assert m_res.runs_final_states[i] == expected

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    @pytest.mark.parametrize('targettol', [
        pytest.param(0.1, id='atol'),
        pytest.param([0.001, 0.1], id='rtol'),
        pytest.param([[0.001, 0.1], [0.1, 0]], id='tol_per_e_op'),
    ])
    def test_multitraj_targettol(self, keep_runs_results, targettol):
        N = 10
        ntraj = 1000
        opt = fill_options(
            keep_runs_results=keep_runs_results, store_states=True
        )
        m_res = MultiTrajResult([qutip.num(N), qutip.qeye(N)], opt, stats={})
        m_res.add_end_condition(ntraj, targettol)
        self._fill_trajectories(m_res, N, ntraj, noise=0.1)

        assert m_res.stats['end_condition'] == "target tolerance reached"
        assert m_res.num_trajectories <= 1000

    def test_multitraj_steadystate(self):
        N = 5
        ntraj = 100
        opt = fill_options()
        m_res = MultiTrajResult([], opt, stats={})
        m_res.add_end_condition(1000)
        self._fill_trajectories(m_res, N, ntraj)
        assert m_res.stats['end_condition'] == "timeout"
        assert m_res.steady_state() == qutip.qeye(5) / 5

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    def test_repr(self, keep_runs_results):
        N = 10
        ntraj = 10
        opt = fill_options(keep_runs_results=keep_runs_results)
        m_res = MultiTrajResult([], opt)
        self._fill_trajectories(m_res, N, ntraj)
        repr = m_res.__repr__()
        assert "Number of trajectories: 10" in repr
        if keep_runs_results:
            assert "Trajectories saved." in repr

    @pytest.mark.parametrize('keep_runs_results', [True, False])
    def test_merge_result(self, keep_runs_results):
        N = 10
        opt = fill_options(
            keep_runs_results=keep_runs_results, store_states=True
        )
        m_res1 = MultiTrajResult([qutip.num(10)], opt, stats={"run time": 1})
        self._fill_trajectories(m_res1, N, 10, noise=0.1)

        m_res2 = MultiTrajResult([qutip.num(10)], opt, stats={"run time": 2})
        self._fill_trajectories(m_res2, N, 30, noise=0.1)

        merged_res = m_res1 + m_res2
        assert merged_res.num_trajectories == 40
        np.testing.assert_allclose(merged_res.average_expect[0],
                                   np.arange(10), rtol=0.1)
        np.testing.assert_allclose(
            np.diag(sum(merged_res.average_states).full()),
            np.ones(N),
            rtol=0.1
        )
        assert bool(merged_res.trajectories) == keep_runs_results
        assert merged_res.stats["run time"] == 3
