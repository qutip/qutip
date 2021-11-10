from qutip.solver.result import *
import qutip
import numpy as np

def test_result_states():
    N = 10
    res = Result([], qutip.solver.SolverResultsOptions(),
                 _super=False, oper_state=False)
    for i in range(N):
        res.add(i, qutip.basis(N,i))
    for i in range(N):
        assert res.states[i] == qutip.basis(N, i)
    assert res.final_state == qutip.basis(N, N-1)
    np.testing.assert_allclose(np.array(res.times), np.arange(N))

def test_result_expect():
    N = 10
    res = Result(
        [qutip.num(N), qutip.qeye(N)],
        qutip.solver.SolverResultsOptions(store_final_state=False,
                                          store_states=False),
        _super=False, oper_state=False
    )
    for i in range(N):
        res.add(i, qutip.basis(N,i))
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state is None
    assert not res.states

def test_result_normalize():
    N = 10
    res = Result([qutip.num(N), qutip.qeye(N)],
                 qutip.solver.SolverResultsOptions(store_states=True,
                                                   normalize_output=True),
                 _super=False, oper_state=False)
    for i in range(N):
        res.add(i, qutip.basis(N,i)/2)
    np.testing.assert_allclose(res.expect[0], np.arange(N))
    np.testing.assert_allclose(res.expect[1], np.ones(N))
    assert res.final_state == qutip.basis(N, N-1)
    for i in range(N):
        assert res.states[i] == qutip.basis(N, i)

def test_multitraj_results():
    N = 10
    e_ops = [qutip.num(N), qutip.qeye(N)]
    m_res = MultiTrajResult(3)
    opt = qutip.solver.SolverResultsOptions(store_states=True,
                                            normalize_output=True)
    for _ in range(5):
        res = Result(e_ops, opt, _super=False, oper_state=False)
        res.collapse = []
        for i in range(N):
            res.add(i, qutip.basis(N, i) / 2)
            res.collapse.append((i+0.5, i%2))
        m_res.add(res)

    np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
    np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
    np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
    for i in range(N):
        assert m_res.average_states[i] == qutip.fock_dm(N, i)
    assert m_res.average_final_state == qutip.fock_dm(N, N-1)
    assert len(m_res.runs_states) == 5
    assert len(m_res.runs_states[0]) == N
    for i in range(5):
        assert m_res.runs_final_states[i] == qutip.basis(N, N-1)
    assert np.all(np.array(m_res.col_which) < 2)

def test_multitrajavg_results():
    N = 10
    e_ops = [qutip.num(N), qutip.qeye(N)]
    m_res = MultiTrajResultAveraged(3)
    opt = qutip.solver.SolverResultsOptions(store_final_state=True,
                                            normalize_output=True)
    for _ in range(5):
        res = Result(e_ops, opt, _super=False, oper_state=False)
        res.collapse = []
        for i in range(N):
            res.add(i, qutip.basis(N, i) / 2)
            res.collapse.append((i+0.5, i%2))
        m_res.add(res)

    np.testing.assert_allclose(m_res.average_expect[0], np.arange(N))
    np.testing.assert_allclose(m_res.average_expect[1], np.ones(N))
    np.testing.assert_allclose(m_res.std_expect[1], np.zeros(N))
    assert m_res.average_final_state == qutip.fock_dm(N, N-1)
    assert np.all(np.array(m_res.col_which) < 2)
