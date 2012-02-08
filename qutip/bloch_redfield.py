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
def brmesolve(R, ekets, rho0, tlist, e_ops, opt=None):
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
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    if ekets != None:
        rho0 = rho0.transform(ekets)
        for n in arange(len(e_ops)):
            e_ops[n] = e_ops[n].transform(ekets)

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
            result_list.append(Qobj(rho))
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
def bloch_redfield_tensor_cb(H, c_ops, spectra_cb, use_secular=True):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment. Use the original computational basis: not working
    
    ..note:: Experimental
    
    """
        
    # TODO: check that H is a Qobj

    # use the eigenbasis         
    evals = H.eigenenergies()

    N = len(evals)  
    K = len(c_ops)
    W = zeros((N,N))
    
    # pre-calculate matrix elements
    for n in xrange(N):
        for m in xrange(N):
            W[n,m] = real(evals[n] - evals[m])
    
    dw_min = abs(W[W.nonzero()]).min()
                    
    # unitary part
    R = -1.0j*(spre(H) - spost(H))
    R.data=R.data.tolil()
    for I in xrange(N*N):
        a,b = vec2mat_index(N, I)
        for J in xrange(N*N):
            c,d = vec2mat_index(N, J)
   
            # unitary part: use spre and spost above, same as this:
            # R[I,J] = -1j * (H[a,c] * (b == d) - H[d,b] * (a == c))   
            # in eb
            # R[I,J] = -1j * W[a,b] * (a == c) * (b == d)
 
            if use_secular == False or abs(W[a,b]-W[c,d]) < dw_min:
  
                # dissipative part:
                for k in xrange(K):
                    # for each operator coupling the system to the environment

                    R.data[I,J] +=  c_ops[k].data[a,c] * c_ops[k].data[d,b] * (spectra_cb[k](W[a,c]) + spectra_cb[k](W[b,d])) / 4
                      
                    s1 = s2 = 0
                    for n in xrange(N):                         
                        s1 += c_ops[k].data[a,n] * c_ops[k].data[n,c] * spectra_cb[k](W[n,c])
                        s2 += c_ops[k].data[d,n] * c_ops[k].data[n,b] * spectra_cb[k](W[n,b])
        
                    R.data[I,J] += - (b == d) * s1 / 4 - (a == c) * s2 / 4
                
    R.data=R.data.tocsr()
    return R, None
    
# ------------------------------------------------------------------------------
# Functions for calculting the Bloch-Redfield tensor for a time-independent 
# system.
# 
def bloch_redfield_tensor(H, c_ops, spectra_cb, use_secular=True):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment. Use the eigenbasis for calculating R.
    
    ..note:: Experimental
    
    """
        
    # TODO: check that H is a Qobj

    # use the eigenbasis         
    ekets, evals = H.eigenstates()

    N = len(evals)  
    K = len(c_ops)
    A = zeros((K,N,N), dtype=complex) # TODO: use sparse here
    W = zeros((N,N))
    
    # pre-calculate matrix elements
    for n in xrange(N):
        for m in xrange(N):
            W[n,m] = real(evals[m] - evals[n])

    for k in range(K):
        #A[k,n,m] = c_ops[k].matrix_element(ekets[n], ekets[m])
        A[k,:,:] = c_ops[k].transform(ekets).full()

    dw_min = abs(W[W.nonzero()]).min()
                  
    # unitary part
    Heb = H.transform(ekets)
    R = -1.0j*(spre(Heb) - spost(Heb))
    R.data=R.data.tolil()
    for I in xrange(N*N):
        a,b = vec2mat_index(N, I)
        for J in xrange(N*N):
            c,d = vec2mat_index(N, J)
   
            # unitary part: use spre and spost above, same as this:
            # R[I,J] = -1j * (H[a,c] * (b == d) - H[d,b] * (a == c))   
            # in eb
            # R[I,J] = -1j * W[a,b] * (a == c) * (b == d)
 
            if use_secular == False or abs(W[a,b]-W[c,d]) < dw_min/10.0:
  
                # dissipative part:
                for k in xrange(K):
                    # for each operator coupling the system to the environment

                    R.data[I,J] += (A[k,a,c] * A[k,d,b] / 4) * (spectra_cb[k](W[a,c]) + spectra_cb[k](W[b,d]))                      
                    
                    s1 = s2 = 0
                    for n in xrange(N):                         
                        s1 += A[k,a,n] * A[k,n,c] * spectra_cb[k](W[n,c])
                        s2 += A[k,d,n] * A[k,n,b] * spectra_cb[k](W[n,d])
        
                    R.data[I,J] += - (b == d) * s1 / 4 - (a == c) * s2 / 4
                
    R.data=R.data.tocsr()
    return R, ekets    
    
    
