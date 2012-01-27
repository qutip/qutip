#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

from types import *
from scipy.integrate import *
from qutip.tidyup import tidyup
from qutip.Qobj import *
from qutip.superoperator import *
from qutip.expect import *
from qutip.states import *
from qutip.Odeoptions import Odeoptions
from qutip.cyQ.ode_rhs import cyq_ode_rhs
from qutip.cyQ.codegen import Codegen
from qutip.rhs_generate import rhs_generate
from qutip.Odedata import Odedata
import os,numpy,odeconfig
import scipy.sparse as sp
# ------------------------------------------------------------------------------
# 
# 
# 
#def me_ode_solve(H, rho0, tlist, c_op_list, expt_op_list, H_args, opt):
def brmesolve(R, rho0, tlist, e_ops, opt=None):
    """
    Solve the dynamics for the system using the Bloch-Redfeild master equation.
    
    ..note:: Experimental
    
    """

    if opt == None:
        opt = Odeoptions()
        opt.nsteps = 2500  # 

    if opt.tidy:
        R = tidyup(R, opt.atol)

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    # 
    n_e_ops  = len(e_ops)
    n_tsteps = len(tlist)
    dt       = tlist[1]-tlist[0]

    if n_e_ops == 0:
        result_list = []
    else:
        result_list = []
        for op in e_ops:
            if op.isherm and rho0.isherm:
                result_list.append(zeros(n_tsteps))
            else:
                result_list.append(zeros(n_tsteps,dtype=complex))

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cyq_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol, #nsteps=opt.nsteps,
                              #first_step=opt.first_step, min_step=opt.min_step,
                              max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    rho = Qobj(rho0)

    t_idx = 0
    for t in tlist:
        if not r.successful():
            break;

        rho.data = vec2mat(r.y)
        
        # calculate all the expectation values, or output rho if no operators
        if n_e_ops == 0:
            result_list.append(Qobj(rho)) # copy rho
        else:
            for m in range(0, n_e_ops):
                result_list[m][t_idx] = expect(e_ops[m], rho)

        r.integrate(r.t + dt)
        t_idx += 1
          
    return result_list

# ------------------------------------------------------------------------------
# Functions for calculting the Bloch-Redfield tensor for a time-independent 
# system.
# 
def __matrix_element(bra, op, ket):
    return (bra.data.T * op.data * ket.data)[0,0]

def bloch_redfield_tensor(H, c_ops, spectra_cb):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.
    
    ..note:: Experimental
    
    """
    
    # TODO: optimize by using data directly instead of Qobjs
        
    # TODO: check that H is a Qobj

    # use the eigenbasis         
    ekets, evals = H.eigenstates()

    N = len(evals)  
    K = len(c_ops)
    A = zeros((K, N, N), dtype=complex)
    W = zeros((N,N),dtype=complex)
    
    # pre-calculate matrix elements
    for n in xrange(N):
        for m in xrange(N):
            W[n,m] = evals[n] - evals[m]        
            for k in range(K):
                A[k,n,m] = __matrix_element(ekets[n], c_ops[k], ekets[m])
    
    
    print "A.re =\n", real(A)
    print "A.im =\n", imag(A)
                
    # unitary part
    R = -1.0j*(spre(H) - spost(H))
    R.data=R.data.tolil()
    for I in xrange(N*N):
        a,b = vec2mat_index(N, I)
        for J in xrange(N*N):
            c,d = vec2mat_index(N, J)
   
            # unitary part: use spre and spost above, same as this:
            # R[I,J] = -1j * (H[a,c] * (b == d) - H[d,b] * (a == c))   
   
            # dissipative part:
            for k in xrange(K):
                # for each operator coupling the system to the environment

                R.data[I,J] += A[k,a,c] * A[k,d,b] * 2 * pi * (spectra_cb[k](W[c,a]) + spectra_cb[k](W[d,b]))
                       
                s1 = s2 = 0
                for n in xrange(N):                         
                    s1 += A[k,a,n] * A[k,n,c] * spectra_cb[k](W[n,c])
                    s2 += A[k,d,n] * A[k,n,b] * spectra_cb[k](W[n,d])
       
                R.data[I,J] += - pi * (b == d) * s1 - pi * (a == c) * s2       
    R.data=R.data.tocsr()
    return R
    
    
