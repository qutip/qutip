from qutip.solver.result import *
import qutip as qt
import numpy as np

def test_result_states():
    N = 10
    res = Result([], qt.solver.SolverResultsOptions(), qt.basis(N, 0))
    for i in range(N):
        res.add(i, qt.basis(N,i))
    for i in range(N):
        assert res.states[i] == qt.basis(N,i)
    assert res.final_state == qt.basis(N,N-1)
    np.testing.assert_allclose(np.array(res.times), np.arange(N))

def test_result_expect():
    N = 10
    res = Result([qt.num(N), qt.qeye(N)],
                 qt.solver.SolverResultsOptions(store_final_state=False,
                                                store_states=False),
                 qt.basis(N, 0))
    for i in range(N):
        res.add(i, qt.basis(N,i))
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state is None
    assert not res.states

def test_result_normalize():
    N = 10
    res = Result([qt.num(N), qt.qeye(N)],
                 qt.solver.SolverResultsOptions(store_states=True,
                                                normalize_output=True),
                 qt.basis(N, 0))
    for i in range(N):
        res.add(i, qt.basis(N,i)/2)
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state == qt.basis(N,N-1)
    for i in range(N):
        assert res.states[i] == qt.basis(N,i)

def test_multitraj_results():
    N = 10
    e_ops = [qt.num(N), qt.qeye(N)]
    m_res = MultiTrajResult(3)
    opt = qt.solver.SolverResultsOptions(store_states=True,
                                         normalize_output=True)
    for _ in range(5):
        res = Result(e_ops, opt, qt.basis(N, 0))
        res.collapse = []
        for i in range(N):
            res.add(i, qt.basis(N,i)/2)
            res.collapse.append((i+0.5, i%2))
        m_res.add(res)

    np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
    np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
    np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
    for i in range(N):
        assert m_res.average_states[i] == qt.basis(N,i) * qt.basis(N,i).dag()
    assert m_res.average_final_state == qt.basis(N,N-1) * qt.basis(N,N-1).dag()
    assert len(m_res.runs_states) == 5
    assert len(m_res.runs_states[0]) == N
    for i in range(5):
        assert m_res.runs_final_states[i] == qt.basis(N,N-1)
    assert np.all(np.array(m_res.col_which) < 2)

def test_multitrajavg_results():
    N = 10
    e_ops = [qt.num(N), qt.qeye(N)]
    m_res = MultiTrajResultAveraged(3)
    opt = qt.solver.SolverResultsOptions(store_final_state=True,
                                         normalize_output=True)
    for _ in range(5):
        res = Result(e_ops, opt, qt.basis(N, 0))
        res.collapse = []
        for i in range(N):
            res.add(i, qt.basis(N,i)/2)
            res.collapse.append((i+0.5, i%2))
        m_res.add(res)

    np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
    np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
    np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
    assert m_res.average_final_state == qt.basis(N,N-1) * qt.basis(N,N-1).dag()
    assert np.all(np.array(m_res.col_which) < 2)
