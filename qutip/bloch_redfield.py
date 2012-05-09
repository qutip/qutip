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


#-------------------------------------------------------------------------------
# Solve the Bloch-Redfield master quation
# 
# 
def brmesolve(H, psi0, tlist, c_ops, e_ops=[], spectra_cb=[], args={}, options=Odeoptions()):
    """
    Solve the dynamics for the system using the Bloch-Redfeild master equation.

    .. note:: 
    
        This solver does not currently support time-dependent Hamiltonian or
        collapse operators.
   
    Parameters
    ----------
    
    H : :class:`qutip.Qobj`
        system Hamiltonian.
        
    rho0 / psi0: :class:`qutip.Qobj`
        initial density matrix or state vector (ket).
     
    tlist : *list* / *array*    
        list of times for :math:`t`.
        
    c_ops : list of :class:`qutip.Qobj`
        list of collapse operators.
    
    expt_ops : list of :class:`qutip.Qobj` / callback function
        list of operators for which to evaluate expectation values.
     
    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and collapse operators.
     
    options : :class:`qutip.Qdeoptions`
        with options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.Odedata`

        An instance of the class :class:`qutip.Odedata`, which contains either
        an *array* of expectation values for the times specified by `tlist`.
    """

    if len(spectra_cb) == 0:
        for n in range(len(c_ops)):
            spectra_cb.append(lambda w: 1.0) # add white noise callbacks if absent

    R, ekets = bloch_redfield_tensor(H, c_ops, spectra_cb)
        
    output = Odedata()
    output.times = tlist

    output.expect = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops, options)

    return output

#-------------------------------------------------------------------------------
# Evolution of the Bloch-Redfield master equation given the Bloch-Redfield
# tensor.
# 
def bloch_redfield_solve(R, ekets, rho0, tlist, e_ops=[], opt=None):
    """
    Evolve the ODEs defined by Bloch-Redfeild master equation.
    """

    if opt == None:
        opt = Odeoptions()
        opt.nsteps = 2500  # 

    if opt.tidy:
        R.tidyup()

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()
       
    #
    # prepare output array
    # m
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
            e_ops[n] = e_ops[n].transform(ekets, False)

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
def bloch_redfield_tensor(H, c_ops, spectra_cb, use_secular=True):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.   
   
    Parameters
    ----------
    
    H : :class:`qutip.Qobj`
        system Hamiltonian.
                
    c_ops : list of :class:`qutip.Qobj`
        list of collapse operators.
    
    spectra_cb : list of callback functions
        list of callback functions that evaluate the noise power spectrum
        at a given frequency.
        
    use_secular : bool
        flag (True of False) that indicates if the secular approximation should
        be used.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """
        
    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise "H must be a quantum object"

    # use the eigenbasis         
    evals, ekets = H.eigenstates()

    N = len(evals)  
    K = len(c_ops)
    A = zeros((K,N,N), dtype=complex) # TODO: use sparse here
    W = zeros((N,N))
    
    # pre-calculate matrix elements
    for n in xrange(N):
        for m in xrange(N):
            W[m,n] = real(evals[m] - evals[n])

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
            # R[I,J] = -1j * W[a,b] * (a == c) * (b == d)
 
            if use_secular == False or abs(W[a,b]-W[c,d]) < dw_min/10.0:
  
                # dissipative part:
                for k in xrange(K):
                    # for each operator coupling the system to the environment

                    R.data[I,J] += (A[k,a,c] * A[k,d,b] / 2) * (spectra_cb[k](W[c,a]) + spectra_cb[k](W[d,b]))                      
                    
                    s1 = s2 = 0
                    for n in xrange(N): 
                        s1 += A[k,a,n] * A[k,n,c] * spectra_cb[k](W[c,n])
                        s2 += A[k,d,n] * A[k,n,b] * spectra_cb[k](W[d,n])
        
                    R.data[I,J] += - (b == d) * s1 / 2 - (a == c) * s2 / 2
                
    R.data=R.data.tocsr()
    return R, ekets    
    
    
