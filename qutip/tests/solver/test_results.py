from qutip.solver.result import *
import qutip
import numpy as np
import pytest

class _e_ops_callable:
    def __init__(self, N):
        self.N = N

    def __call__(self, t, state):
        return qutip.expect(qutip.num(self.N), state)


def _make_e_ops(N, type_):
    if type_ == 'list':
        e_op = [qutip.num(N)]
    elif type_ == 'qobj':
        e_op = qutip.num(N)
    elif type_ == 'qobjevo':
        e_op = qutip.QobjEvo(qutip.num(N))
    elif type_ == 'callable':
        e_op = _e_ops_callable(N)
    elif type_ == 'dict':
        e_op = {'0': qutip.num(N)}
    elif type_ == 'list_func':
        e_op = [qutip.num(N), _e_ops_callable(N)]
    return e_op


def _check_and_extract_expect(expect, type_):
    if type_ in ['qobj', 'qobjevo', 'callable']:
        assert isinstance(expect, np.ndarray)
    elif type_ in ['list', 'list_func']:
        assert isinstance(expect, list)
        expect = expect[0]
    elif type_ in ['dict']:
        assert isinstance(expect, dict)
        expect = expect['0']
    return expect


def test_result_states():
    N = 10
    opt = qutip.solver.SolverResultsOptions()
    res = Result([], opt, 0, qutip.basis(N, 0))
    for i in range(1, N):
        res.add(i, qutip.basis(N,i))
    for i in range(N):
        assert res.states[i] == qutip.basis(N, i)
    assert res.final_state == qutip.basis(N, N-1)
    np.testing.assert_allclose(np.array(res.times), np.arange(N))
    assert res.num_expect == 0


def test_result_expect():
    N = 10
    opt = qutip.solver.SolverResultsOptions(store_final_state=False,
                                            store_states=False)
    res = Result([qutip.num(N), qutip.qeye(N)], opt, 0, qutip.basis(N, 0))
    for i in range(1, N):
        res.add(i, qutip.basis(N,i))
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state is None
    assert not res.states
    assert res.num_expect == 2


@pytest.mark.parametrize('e_ops',
    ['list', 'qobj', 'qobjevo', 'callable', 'dict', 'list_func'])
def test_result_expect_types(e_ops):
    N = 10
    opt = qutip.solver.SolverResultsOptions(store_final_state=False,
                                            store_states=False)
    res = Result(_make_e_ops(N, e_ops), opt)
    for i in range(N):
        res.add(i, qutip.basis(N,i))
    expect = _check_and_extract_expect(res.expect, e_ops)
    np.testing.assert_allclose(expect, np.arange(N))


def test_result_final_state():
    N = 10
    opt = qutip.solver.SolverResultsOptions(store_final_state=True,
                                            store_states=False)
    res = Result([], opt, [0, N-1], qutip.basis(N, 0))
    for i in range(N):
        res.add(i, qutip.basis(N,i))
    assert res.final_state == qutip.basis(N, N-1)
    assert not res.states
    assert res.num_expect == 0


def test_result_normalize():
    N = 10
    opt = qutip.solver.SolverResultsOptions(store_states=True,
                                            normalize_output=True)
    res = Result([qutip.num(N), qutip.qeye(N)], opt)
    for i in range(N):
        res.add(i, qutip.basis(N,i)/2)
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state == qutip.basis(N, N-1)
    for i in range(N):
        assert res.states[i] == qutip.basis(N, i)
    assert res.num_collapse == 0
    assert res.num_expect == 2


@pytest.mark.parametrize('norm, format, proj', [
    pytest.param(True, 'ket', lambda x: x, id='ket'),
    pytest.param(True, 'all', lambda x: x, id='all'),
    pytest.param(False, 'dm', lambda x: x, id='not dm'),
    pytest.param(True, 'dm', qutip.ket2dm, id='dm'),
    pytest.param(False, 'ket', qutip.ket2dm, id='not ket'),
])
def test_result_normalize_opt(norm, format, proj):
    N = 10
    opt = qutip.solver.SolverResultsOptions(store_final_state=True,
                                            normalize_output=format)
    res = Result(qutip.qeye(N), opt, 0, proj(qutip.basis(N, 0)))
    for i in range(1, N):
        res.add(i, proj(qutip.basis(N, i) / 2))
    fact = 1 if norm else 0.25
    np.testing.assert_allclose(res.expect[1:], np.ones(N-1) * fact)
    assert res.final_state == proj(qutip.basis(N, N-1) * (1 if norm else 0.5))


@pytest.mark.parametrize('keep_runs_results', [True, False])
@pytest.mark.parametrize('format', ['dm', 'ket'])
def test_McResult(format, keep_runs_results):
    N = 10
    ntraj = 5
    e_ops = [qutip.num(N), qutip.qeye(N)]
    opt = qutip.solver.SolverResultsOptions(
        keep_runs_results=keep_runs_results,
        store_states=True, normalize_output=True
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
        normalize_output=True
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
        store_states=True, normalize_output=False
    )
    m_res = MultiTrajResult(
        [qutip.num(N), qutip.qeye(N)],
        opt,
        np.arange(N),
        qutip.basis(N, 0) * (1 - 0.5*np.random.rand())
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
