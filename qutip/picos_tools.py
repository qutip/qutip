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
"""Tools for manipulating PICOS problems, including adding constraints and
objectives based on QIP.
"""

import numpy as np
try:
    import picos as pic
except ImportError:
    pic = None

try:
    import cvxopt as cvx
except ImportError:
    cvx = None

__all__ = [
    'add_ptrace_eq_constraint'
] if pic is not None else []

def add_ptrace_eq_constraint(problem, A, B, keep_left, traceout, keep_right):
    """
    Generates a list of constraints such that Tr_{traceout}(A) == B
    and adds it to a PICOS problem instance.
    """
    # TODO: figure out how to get PICOS to properly format this
    #       list of constraints.

    return problem.add_list_of_constraints([
         # We call pic.sum to ensure that the summation index is formatted
         # when the problem is printed out.
        pic.sum(
            [
                A[
                    idx_i_left * keep_right * traceout + idx_tr * keep_right + idx_i_right,
                    idx_j_left * keep_right * traceout + idx_tr * keep_right + idx_j_right,
                ]
                for idx_tr in range(traceout)
            ], 'j', 'range({})'.format(traceout)
        )
        == B[idx_i_left * keep_right + idx_i_right, idx_j_left * keep_right + idx_j_right]
        for idx_i_left  in range(keep_left)
        for idx_i_right in range(keep_right)
        for idx_j_left  in range(keep_left)
        for idx_j_right in range(keep_right)
    ])

def to_picos_param(name, qobj):
    # We *must* call cvx.matrix here, as there's a bug in
    # pic.new_param that causes it to sometimes take the
    # transpose of np.array instances. Calling cvx.matrix
    # ourselves, we can be sure to avoid that bug.
    return pic.new_param(name, cvx.matrix(qobj.data.todense()))