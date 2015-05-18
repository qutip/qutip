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

#Author: Neill Lambert
#Contact: nwlambert@gmail.com

"""
We take a quantum system coupled to a bosonic environment and map to a model
in which a collective mode of the environment, known as the reaction coordinate 
(RC), is incorporated within an effective system Hamiltonian. We then treat the 
residual environment within a full second-order Born-Markov master equation
formalism. Thus, all important system-bath, and indeed intrabath, correlations 
are incorporated into the system-RC Hamiltonian in the regimes we study.

Furthur information can be found at link.aps.org/doi/10.1103/PhysRevA.90.032114 
"""

import time
import numpy as np
import scipy.sparse as sp
from numpy import matrix
from numpy import linalg
from qutip import spre, spost, sprepost, thermal_dm, mesolve, Odeoptions
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis, qeye

class rcsolve(object):
    """
    Class to solve for an open quantum system using the reaction coordinate (RC)
    model. 
    """
    
    def __init__(self, Del=1.0, wc=0.05, wq=0.5, alpha=2.5/np.pi, N = 20, 
                 Temperature = 1/0.95, tlist=None, output=None):
        """
        Parameters
        ----------
        Del: Float
            The number of qubits in the system.
        wc: Float
            Cutoff frequency.
        wq: Float
            Energy of the 2-level system.
        alpha: Float
            Coupling strength.
        N: Integer
            Number of cavity fock states.
        Temperature: Float
            Temperature. 
        tlist: List.
            Time over which system evolves.
        output: List
            System evolution.
        """
        self.Del = Del
        self.wc = wc
        self.wq = wq
        self.alpha = alpha
        self.N = N
        self.Temperature = Temperature
        self.tlist = np.linspace(0, 40, 600)
        if tlist is not None:
            self.tlist = tlist
        self.output = None
             
    
    def solve(self):
        """
        Set up the high temperature master equation and solve it using mesolve.
        """
        start_time = time.time()    
        
        #Set up the master equation
        
        Hdot = 0.5 * self.wq* sigmaz() + 0.5 * self.Del* sigmax()
        dot_energy, dot_state = Hdot.eigenstates()
        deltaE = dot_energy[1] - dot_energy[0]
        gamma = deltaE / (2 * np.pi * self.wc)
        wa = 2 * np.pi * gamma *self.wc #reaction coordinate frequency
        g = np.sqrt(np.pi * wa * self.alpha / 2.0) #reaction coordinate coupling
        nb = (1 / (np.exp(wa/self.Temperature) -1))
        
        #Reaction coordinate hamiltonian/operators
        Nmax = self.N * 2        #hilbert space 
        a  = tensor(destroy(self.N), qeye(2))
        sm = tensor(qeye(self.N), destroy(2))
        unit = tensor(qeye(self.N), qeye(2))

        nq = sm.dag() * sm  # atom
        nL = sm.dag() * sm  # atom
        nR = sm * sm.dag()   # atom

        na = a.dag() * a    # cavity
        xa = a.dag() + a
        
        # decoupled Hamiltonian
        H0 = wa * a.dag() * a+ self.wq*0.5*(nL-nR) + self.Del*0.5*(sm+sm.dag())
        #interaction
        H1 = (g * (a.dag() + a) * (nL-nR))
        H = H0 + H1
        
        L=0
        PsipreEta=0
        PsipreX=0

        all_energy, all_state = H.eigenstates()
        Apre = spre((a + a.dag()))
        Apost = spost(a + a.dag())
        for j in range(Nmax):
            for k in range(Nmax):
                A = xa.matrix_element(all_state[j].dag(), all_state[k])
                delE = (all_energy[j] - all_energy[k])
                if np.absolute(A) > 0.0:
                    if abs(delE) > 0.0:
                        X = (0.5 * np.pi * gamma*(all_energy[j] - all_energy[k])
                             * (np.cosh((all_energy[j] - all_energy[k]) /
                                (2 * self.Temperature))
                             / (np.sinh((all_energy[j] - all_energy[k]) /
                                (2 * self.Temperature)))) * A)
                        eta = (0.5 * np.pi * gamma *
                               (all_energy[j] - all_energy[k]) * A)
                        PsipreX = PsipreX + X * all_state[j]*all_state[k].dag()
                        PsipreEta = PsipreEta + (eta * all_state[j]
                                                 * all_state[k].dag())
                    else:
                        X =0.5 * np.pi * gamma * A * 2 * self.Temperature
                        PsipreX=PsipreX+X*all_state[j]*all_state[k].dag()
        
        L = ((-spre((a+a.dag()) * PsipreX)) + (sprepost(a+a.dag(), PsipreX))
             +(sprepost(PsipreX, a+a.dag())) + (-spost(PsipreX*(a+a.dag())))
             +(spre((a+a.dag()) * PsipreEta)) + (sprepost(a+a.dag(), PsipreEta))
             +(-sprepost(PsipreEta,a+a.dag())) + (-spost(PsipreEta*(a+a.dag()))))           
        
        #Setup the operators and the Hamiltonian and the master equation 
        #and solve for time steps in tlist
        psi0 = (tensor(thermal_dm(self.N,nb), basis(2,1)*basis(2,1).dag()))
        self.output = mesolve(H, psi0, self.tlist, [L],
                              [a.dag()*a,nL,nR,sm,sm.dag()],
                              options=Odeoptions(nsteps=15000),
                              progress_bar=None)
        end_time = time.time()
        print("Integration required %g seconds" % (end_time - start_time))
        
    
    def plot_tls_so(self):
        """
        Plot the state occupation in the two level system for the entire time
        duration.
        """ 
        import matplotlib.pyplot as plt   
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(8,4))
        axes.plot(self.tlist, np.real(self.output.expect[3]), 'b', linewidth=2,
                  label="P12")
        fig, axes2 = plt.subplots(1, 1, sharex=True, figsize=(8,4))
        axes2.plot(self.tlist, self.output.expect[2], 'g', linewidth=2,
                   label="P1")
        axes.legend(loc=0)
        axes2.legend(loc=0)
                
