import numpy as np
from numpy.testing import assert_, assert_equal
import qutip as qt
from qutip.solver import config

"""
def test_rhs_reuse():
    "" "
    rhs_reuse : pyx filenames match for rhs_reus= True
    "" "
    N = 10
    a = qt.destroy(N)
    H = [a.dag()*a, [a+a.dag(), 'sin(t)']]
    psi0 = qt.fock(N,3)
    tlist = np.linspace(0,10,10)
    e_ops = [a.dag()*a]
    c_ops = [0.25*a]

    # Test sesolve
    out1 = qt.mesolve(H, psi0,tlist, e_ops=e_ops)

    _temp_config_name = config.tdname

    out2 = qt.mesolve(H, psi0,tlist, e_ops=e_ops)

    assert_(config.tdname != _temp_config_name)
    _temp_config_name = config.tdname

    out3 = qt.mesolve(H, psi0,tlist, e_ops=e_ops,
                        options=qt.Options(rhs_reuse=True))

    assert_(config.tdname == _temp_config_name)

    # Test mesolve

    out1 = qt.mesolve(H, psi0,tlist, c_ops=c_ops, e_ops=e_ops)

    _temp_config_name = config.tdname

    out2 = qt.mesolve(H, psi0,tlist, c_ops=c_ops, e_ops=e_ops)

    assert_(config.tdname != _temp_config_name)
    _temp_config_name = config.tdname

    out3 = qt.mesolve(H, psi0,tlist, e_ops=e_ops, c_ops=c_ops,
                        options=qt.Options(rhs_reuse=True))

    assert_(config.tdname == _temp_config_name)

"""
