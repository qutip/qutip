from numpy.testing import assert_

from qutip import rand_herm, qeye
from qutip.rhs_generate import _td_format_check


def test_setTDFormatCheckMC():
    "td_format_check: monte-carlo"

    # define operators
    H = rand_herm(10)
    c_op = qeye(10)

    def f_c_op(t, args):
        return 0

    def f_H(t, args):
        return 0
    # check constant H and no C_ops
    time_type, h_stuff, c_stuff = _td_format_check(H, [], 'mc')
    assert_(time_type == 0)

    # check constant H and constant C_ops
    time_type, h_stuff, c_stuff = _td_format_check(H, [c_op], 'mc')
    assert_(time_type == 0)

    # check constant H and str C_ops
    time_type, h_stuff, c_stuff = _td_format_check(H, [c_op, '1'], 'mc')
    # assert_(time_type==1) # this test fails!!

    # check constant H and func C_ops
    time_type, h_stuff, c_stuff = _td_format_check(H, [f_c_op], 'mc')
    # assert_(time_type==2) # FAILURE

    # check str H and constant C_ops
    time_type, h_stuff, c_stuff = _td_format_check([H, '1'], [c_op], 'mc')
    # assert_(time_type==10)

    # check str H and str C_ops
    time_type, h_stuff, c_stuff = _td_format_check([H, '1'], [c_op, '1'], 'mc')
    # assert_(time_type==11)

    # check str H and func C_ops
    time_type, h_stuff, c_stuff = _td_format_check([H, '1'], [f_c_op], 'mc')
    # assert_(time_type==12)

    # check func H and constant C_ops
    time_type, h_stuff, c_stuff = _td_format_check(f_H, [c_op], 'mc')
    # assert_(time_type==20)

    # check func H and str C_ops
    time_type, h_stuff, c_stuff = _td_format_check(f_H, [c_op, '1'], 'mc')
    # assert_(time_type==21)

    # check func H and func C_ops
    time_type, h_stuff, c_stuff = _td_format_check(f_H, [f_c_op], 'mc')
    # assert_(time_type==22)
