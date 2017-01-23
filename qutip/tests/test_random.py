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

import scipy.sparse as sp
import scipy.linalg as la
import numpy as np
from numpy.testing import assert_equal, assert_, run_module_suite

from qutip.random_objects import (rand_ket, rand_dm, rand_herm, rand_unitary,
                                  rand_ket_haar, rand_dm_hs,
                                  rand_super, rand_unitary_haar, rand_dm_ginibre,
                                  rand_super_bcsz)
from qutip.operators import qeye

def test_rand_unitary_haar_unitarity():
    """
    Random Qobjs: Tests that unitaries are actually unitary.
    """
    U = rand_unitary_haar(5)
    I = qeye(5)

    assert_(U * U.dag() == I)

def test_rand_dm_ginibre_rank():
    """
    Random Qobjs: Ginibre-random density ops have correct rank.
    """
    rho = rand_dm_ginibre(5, rank=3)

    rank = sum([abs(E) >= 1e-10 for E in rho.eigenenergies()])
    assert_(rank == 3)

def test_rand_super_bcsz_cptp():
    """
    Random Qobjs: Tests that BCSZ-random superoperators are CPTP.
    """
    S = rand_super_bcsz(5)
    assert_(S.iscptp)

def check_func_dims(func, args, kwargs, dims):
    # TODO: promote this out of test_random, as it's generically useful
    #       in writing tests.
    resdims = func(*args, **kwargs).dims
    assert_(resdims == dims, "Checking {}; epected dimensions of {}, got {}.".format(func.__name__, dims, resdims))

def test_rand_vector_dims():
    FUNCS = [rand_ket, rand_ket_haar]
    for func in FUNCS:
        check_func_dims( func, (7, ), {}, [[7], [1]])
        check_func_dims( func, (6, ), {'dims': [[2,3], [1,1]]}, [[2,3], [1,1]])

def test_rand_oper_dims():
    FUNCS = [rand_unitary, rand_herm, rand_dm, rand_unitary_haar, rand_dm_ginibre, rand_dm_hs]
    for func in FUNCS:
        check_func_dims( func, (7, ), {}, [[7], [7]])
        check_func_dims( func, (6, ), {'dims': [[2, 3], [2, 3]]}, [[2, 3], [2, 3]])

def test_rand_super_dims():
    FUNCS = [rand_super, rand_super_bcsz]
    for func in FUNCS:
        check_func_dims(func, (7, ), {}, [[[7], [7]]] * 2)
        check_func_dims(func, (6, ), {'dims': [[[2, 3], [2, 3]], [[2, 3], [2, 3]]]}, [[[2, 3], [2, 3]], [[2, 3], [2, 3]]])

if __name__ == "__main__":
    run_module_suite()
