# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, The QuTiP Project
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
from numpy.testing import assert_
from qutip import *
from qutip.legacy.ptrace import _ptrace as _pt

def test_ptrace_rand():
    'ptrace : randomized tests'
    for k in range(10):
        A = tensor(rand_ket(5), rand_ket(2), rand_ket(3))
        B = A.ptrace([1,2])
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,2])
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,1])
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)


    for k in range(10):
        A = tensor(rand_dm(2), thermal_dm(10,1), rand_unitary(3))
        B = A.ptrace([1,2])
        bdat,bd,bs = _pt(A, [1,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,2])
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,1])
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        
    for k in range(10):
        A = tensor(rand_ket(2),rand_ket(2),rand_ket(2),
                    rand_ket(2),rand_ket(2),rand_ket(2))
        B = A.ptrace([3,2])
        bdat,bd,bs = _pt(A, [3,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,2])
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,1])
        bdat,bd,bs = _pt(A, [0,1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        


    for k in range(10):
        A = rand_dm(64,0.5,dims=[[4,4,4],[4,4,4]])
        B = A.ptrace([0])
        bdat,bd,bs = _pt(A, [0])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([1])
        bdat,bd,bs = _pt(A, [1])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)
        
        B = A.ptrace([0,2])
        bdat,bd,bs = _pt(A, [0,2])
        C = Qobj(bdat,dims=bd)
        assert_(B==C)