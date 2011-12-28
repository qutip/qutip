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

from scipy import *
from qutip.Qobj import *

from qutip.superoperator import *

from qutip.odesolve import *
from qutip.essolve import *
from qutip.states import basis

def propagator(H, t, c_op_list, H_args=None):
    """
    Calculate the propagator U(t) for the density matrix or wave function such that
    :math:`\psi(t) = U(t)\psi(0)` or :math:`\\rho_{\mathrm vec}(t) = U(t) \\rho_{\mathrm vec}(0)`
    where :math:`\\rho_{\mathrm vec}` is the vector representation of the density matrix.
    
    Arguments:
    
        `H` (:class:`qutip.Qobj`) Hamiltonian.
        
        `t` (*float*) time.
        
        `c_op_list` (list of :class:`qutip.Qobj`) collapse operators.
        
        `H_args` (*list/array*) [optional] parameters to callback functions for time-dependent Hamiltonians.
    
    Returns a :class:`qutip.Qobj` instance representing the propagator :math:`U(t)`.
    """

    if len(c_op_list) == 0:
        # calculate propagator for the wave function

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        else:
            N = H.shape[0]

        u = zeros([N, N], dtype=complex)
        
        for n in range(0, N):

            psi0 = basis(N, n)
            psi_t = odesolve(H, psi0, [0, t], c_op_list, [], H_args)

            u[:,n] = psi_t[1].full().T

    else:
        # calculate the propagator for the vector representation of the 
        # density matrix

        if isinstance(H, FunctionType):
            H0 = H(0.0, H_args)
            N = H0.shape[0]
        else:
            N = H.shape[0]

        u = zeros([N*N, N*N], dtype=complex)
        
        for n in xrange(0, N*N):
            psi0  = basis(N*N, n)
            rho0  = Qobj(vec2mat(psi0.full()))
            rho_t = odesolve(H, rho0, [0, t], c_op_list, [], H_args)
            u[:,n] = mat2vec(rho_t[1].full()).T


    return Qobj(u)

def get_min_and_index(lst): 
    """
    Private function for obtaining min and max indicies.
    """
    minval,minidx = lst[0],0 
    for i,v in enumerate(lst[1:]): 
        if v < minval: 
            minval,minidx = v,i+1 
    return minval,minidx 

def propagator_steadystate(U):
    """
    Find the steady state for successive applications of the propagator :math:`U`.
    
    Arguments:
    
        `U` (:class:`qutip.Qobj`) Operator representing the propagator.
    
    Returns a :class:`qutip.Qobj` instance representing the steady-state vector.
    """

    evals,evecs = la.eig(U.full())

    ev_min, ev_idx = get_min_and_index(abs(evals-1.0))

    N = int(sqrt(len(evals)))

    evecs = evecs.T
    rho = Qobj(vec2mat(evecs[ev_idx]))

    return rho * (1 / real(rho.tr()))


def floquet_modes(H, T, H_args=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.
    
    Returns a list of :class:`qutip.Qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]
         
    .. note:: Experimental
    """

    # get the unitary propagator
    U = propagator(H, T, [], H_args)

    # find the eigenstates for the propagator
    evals,evecs = la.eig(U.full())

    eargs = angle(evals)
    
    # make sure that the phase is in the interval [-pi, pi], so that the
    # quasi energy is in the interval [-pi/T, pi/T] where T is the period of the
    # driving.
    #eargs  += (eargs <= -2*pi) * (2*pi) + (eargs > 0) * (-2*pi)
    eargs  += (eargs <= -pi) * (2*pi) + (eargs > pi) * (-2*pi)
    e_quasi = -eargs/T

    # sort by the quasi energy
    order = argsort(e_quasi)

    # prepare a list of kets for the floquet states
    new_dims  = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(matrix(evecs[:,o]).T, dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]

def floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args=None):
    """
    Calculate the Floquet modes at times tlist Phi_alpha(tlist) propagting the
    initial Floquet modes Phi_alpha(0)
    
    .. note:: Experimental    
    """

    # find t in [0,T] such that t_orig = t + n * T for integer n
    t = t - int(t/T) * T
    
    f_modes_t = []
        
    # get the unitary propagator from 0 to t
    if t > 0.0:
        U = propagator(H, t, [], H_args)

        for n in arange(len(f_modes_0)):
            f_modes_t.append(U * f_modes_0[n])

    else:

        f_modes_t = f_modes_0
        #for n in arange(len(f_states_0)):
        #    f_modes_t.append(f_states_0[n])


    return f_modes_t
    
def floquet_states_t(f_modes_0, f_energies, t, H, T, H_args=None):
    """
    Evaluate the floquet states at time t.
    
    Returns a list of the wavefunctions.
        
    .. note:: Experimental    
    """
    
    f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)
    return [(f_modes_t[i] * exp(-1j * f_energies[i]*t)) for i in arange(len(f_energies))]    
    
    
def floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, H_args=None):
    """
    Evaluate the wavefunction for a time t using the Floquet states decompositon.
    
    Returns the wavefunction.
        
    .. note:: Experimental    
    """
    
    f_states_t = floquet_states_t(f_modes_0, f_energies, t, H, T, H_args)
    return sum([f_states_t[i] * f_coeff[i] for i in arange(len(f_energies))])

def floquet_state_decomposition(f_modes_0, f_energies, psi0):
    """
    Decompose the wavefunction psi in the Floquet states, return the coefficients
    in the decomposition as an array of complex amplitudes.
    """
    return [(f_modes_0[i].dag() * psi0).data[0,0] for i in arange(len(f_energies))]
    
    
    
    
def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T, H_args, kmax=5):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.
    """
    
    N = len(f_energies)
    M = 2*kmax + 1
    
    omega = (2*pi)/T
    
    Delta = zeros((N, N, M))
    Xlist = zeros((N, N, M), dtype=complex)
    
    nT = 100
    tlist = linspace(0, T, nT)
    dT = T/nT

    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax,kmax+1, 1):
                Delta[a,b,k_idx] = f_energies[a] - f_energies[b] + k * omega
                k_idx += 1

    for t in tlist:
        # TODO: repeated invocations of floquet_modes_t is inefficient...
        # make a and b outer loops and use the odesolve instead of the propagator.
        f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)   
        for a in range(N):
            for b in range(N):
                k_idx = 0
                for k in range(-kmax,kmax+1, 1):
                    Xlist[a,b,k_idx] += (dT/T) * exp(-1j * k * omega * t) * (f_modes_t[a].dag() * c_op * f_modes_t[b]).full()[0,0]
                    k_idx += 1
    
    return Delta, Xlist
        
    
    
    
    
    
    
    

