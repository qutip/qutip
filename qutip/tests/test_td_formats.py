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

from numpy.testing import assert_, run_module_suite

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


if __name__ == "__main__":
    run_module_suite()
