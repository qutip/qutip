# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2015 and later, Xiang Gao.
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

import numpy as np
from numpy.testing import assert_, run_module_suite
from qutip.perturbation import Perturbation
from qutip.qobj import Qobj

def test_Perturbation():
    ntests = 10
    tol = 0.000000001
    converge_tol = 100*tol
    minhsz = 2
    maxhsz = 10
    minorder = 1
    maxorder = 5
    maxlambda = 0.1
    minlambda = 0.01
    output = False

    for i in range(ntests):
        # parameters for each individual test
        lamda = np.random.uniform(minlambda,maxlambda)
        # size of hamiltonian
        n = np.random.randint(minhsz,maxhsz)
        # number of terms
        m = np.random.randint(minorder,maxorder) 
        
        # generate a random symmetry matrix
        def random_hamiltonian():
            h = np.random.random(size=(n,n))
            h = np.tril(h) + np.tril(h, -1).T
            return h

        # hamiltionians of each order of perturbation
        hs = [random_hamiltonian() for j in range(m+1)]
        # total hamiltonian
        h = sum([lamda**j * x for x,j in zip(hs,range(m+1))])
        # convert to Qobj
        h = Qobj(h)
        hs = [Qobj(x) for x in hs]
        
        # solve for eigenvalues directly
        heigenval, heigenvec = h.eigenstates()
        
        # solve for eigenvalues using perturbation
        calculator = Perturbation(hs[0])
        for j in range(1,m+1):
            calculator.next_order(hs[j])
        calculator.goto_converge(lamda,tol)
        heigenval_p, heigenvec_p = calculator.result(lamda)
        
        # calculate the difference
        eigvaldiff = heigenval_p - heigenval
        absdiff = [ abs(x) for x in eigvaldiff ]
        diffsum = sum(absdiff)
        assert_(diffsum < converge_tol)
        
        # outputs
        if output:
            print("Test No.", i+1)
            print("lambda:",lamda)
            print("size: {}*{}".format(n,n))
            print("perturbation in Hamiltonian up to {}th order".format(m))
            print("converge until the {}th order".format(calculator.order))
            print("sum of difference:",diffsum)
            print()

if __name__ == "__main__":
    run_module_suite()