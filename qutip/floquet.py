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
from qutip.mesolve import *
from qutip.essolve import *
from qutip.steady import steadystate
from qutip.states import basis
from qutip.states import projection
from qutip.Odeoptions import Odeoptions
from qutip.propagator import propagator


def floquet_modes(H, T, H_args=None, sort=False):
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
    if sort == True:
        order = argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

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
            f_modes_t.append(U * f_modes_0[n] * exp(1j * f_energies[n]*t))

    else:
        f_modes_t = f_modes_0

    return f_modes_t
    
def floquet_modes_table(f_modes_0, f_energies, tlist, H, T, H_args=None):
    """
    Pre-calculate the Floquet modes for a range of times spanning the floquet
    period. Can later be used as a table to look up the floquet modes for
    any time.
    
    .. note:: Experimental    
    """

    # truncate tlist to the driving period
    tlist_period = tlist[where(tlist <= T)]

    f_modes_table_t = [[] for t in tlist_period]

    opt = Odeoptions()
    opt.rhs_reuse = True

    for n, f_mode in enumerate(f_modes_0):
        output = mesolve(H, f_mode, tlist_period, [], [], H_args, opt)
        for t_idx, f_state_t in enumerate(output.states):
            f_modes_table_t[t_idx].append(f_state_t * exp(1j * f_energies[n]*tlist_period[t_idx]))
        
    return f_modes_table_t    
    
def floquet_modes_t_lookup(f_modes_table_t, t, T):
    """
    Lookup the floquet mode at time t in the pre-calculated table of floquet
    modes in the first period of the time-dependence.
    """

    # find t_wrap in [0,T] such that t = t_wrap + n * T for integer n
    t_wrap = t - int(t/T) * T

    # find the index in the table that corresponds to t_wrap (= tlist[t_idx])
    t_idx = int(t_wrap/T * len(f_modes_table_t))
    
    # XXX: might want to give a warning if the cast of t_idx to int discard
    # a significant fraction in t_idx, which would happen if the list of time
    # values isn't perfect matching the driving period
    #if debug: print "t = %f -> t_wrap = %f @ %d of %d" % (t, t_wrap, t_idx, N)

    return f_modes_table_t[t_idx]

def floquet_states(f_modes, f_energies, t):
    """
    Evaluate the floquet states at time t.
    
    Returns a list of the wavefunctions.
        
    """
    
    return [(f_modes[i] * exp(-1j * f_energies[i]*t)) for i in arange(len(f_energies))]   
        
def floquet_states_t(f_modes_0, f_energies, t, H, T, H_args=None):
    """
    Evaluate the floquet states at time t.
    
    Returns a list of the wavefunctions.
        
    .. note:: Experimental    
    """
    
    f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)
    return [(f_modes_t[i] * exp(-1j * f_energies[i]*t)) for i in arange(len(f_energies))]    
    
def floquet_wavefunction(f_modes, f_energies, f_coeff, t):
    """
    Evaluate the wavefunction for a time t using the Floquet states decompositon.
    
    Returns the wavefunction.
        
    """
    return sum([f_modes[i] * exp(-1j * f_energies[i]*t) * f_coeff[i] for i in arange(len(f_energies))])
    
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

# should be moved to a utility library?    
def n_thermal(w, w_th):
    if (w > 0): 
        return 1.0/(exp(w/w_th) - 1.0)
    else: 
        return 0.0
    
def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T, H_args, J_cb, w_th, kmax=5,f_modes_table_t=None):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.
    """
    
    N = len(f_energies)
    M = 2*kmax + 1
    
    omega = (2*pi)/T
    
    Delta = zeros((N, N, M))
    X     = zeros((N, N, M), dtype=complex)
    Gamma = zeros((N, N, M))
    A     = zeros((N, N))
    
    nT = 100
    dT = T/nT
    tlist = arange(dT, T+dT/2, dT)

    if f_modes_table_t == None:
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, linspace(0, T, nT+1), H, T, H_args) 

    for t in tlist:
        # TODO: repeated invocations of floquet_modes_t is inefficient...
        # make a and b outer loops and use the mesolve instead of the propagator.
        #f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, H_args)
        f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)   
        for a in range(N):
            for b in range(N):
                k_idx = 0
                for k in range(-kmax,kmax+1, 1):
                    X[a,b,k_idx] += (dT/T) * exp(-1j * k * omega * t) * (f_modes_t[a].dag() * c_op * f_modes_t[b])[0,0]
                    k_idx += 1

    Heaviside = lambda x: ((sign(x)+1)/2.0)
    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax,kmax+1, 1):
                Delta[a,b,k_idx] = f_energies[a] - f_energies[b] + k * omega
                Gamma[a,b,k_idx] = 2 * pi * Heaviside(Delta[a,b,k_idx]) * J_cb(Delta[a,b,k_idx]) * abs(X[a,b,k_idx])**2
                k_idx += 1
                

    for a in range(N):
        for b in range(N):
            for k in range(-kmax,kmax+1, 1):
                k1_idx =   k + kmax;
                k2_idx = - k + kmax;                
                A[a,b] += Gamma[a,b,k1_idx] + n_thermal(abs(Delta[a,b,k1_idx]), w_th) * (Gamma[a,b,k1_idx]+Gamma[b,a,k2_idx])
                
    return Delta, X, Gamma, A
        
    
def floquet_collapse_operators(A):
    """
    Construct
    """
    c_ops = []
    
    N, M = shape(A)
    
    #
    # Here we really need a master equation on Bloch-Redfield form, or perhaps
    # we can use the Lindblad form master equation with some rotating frame
    # approximations? ...
    # 
    for a in range(N):
        for b in range(N):
            if a != b and abs(A[a,b]) > 0.0:
                # only relaxation terms included...
                c_ops.append(sqrt(A[a,b]) * projection(N, a, b))
    
    return c_ops
    
    
def floquet_master_equation_tensor(Alist, f_energies):
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators).
    
    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]
    """

    if isinstance(Alist, list):
        # Alist can be a list of rate matrices corresponding
        # to different operators that couple to the environment
        N, M = shape(Alist[0])
    else:
        # or a simple rate matrix, in which case we put it in a list
        Alist = [Alist]
        N, M = shape(Alist[0])
      
    R = Qobj(scipy.sparse.csr_matrix((N*N,N*N)), [[N,N], [N,N]], [N*N,N*N])

    R.data=R.data.tolil()      
    for I in range(N*N):
        a,b = vec2mat_index(N, I)
        for J in range(N*N):
            c,d = vec2mat_index(N, J)

            R.data[I,J] = - 1.0j * (f_energies[a]-f_energies[b]) * (a == c) * (b == d)
                    
            for A in Alist:               
                s1 = s2 = 0
                for n in range(N):
                    s1 += A[a,n] * (n == c) * (n == d) - A[n,a] * (a == c) * (a == d)
                    s2 += (A[n,a] + A[n,b]) * (a == c) * (b == d)
                       
                dR = (a == b) * s1 - 0.5 * (1 - (a == b)) * s2
                            
                if dR != 0.0:
                    R.data[I,J] += dR

    R.data=R.data.tocsr()
    return R                   

    
def floquet_master_equation_steadystate(H, A):
    """
    Returns the steadystate density matrix (in the floquet basis!) for the
    Floquet-Markov master equation.
    """
    c_ops = floquet_collapse_operators(A)
    rho_ss = steadystate(H, c_ops)   
    return rho_ss
    
    
def floquet_basis_transform(f_modes, f_energies, rho0):
    """
    Make a basis transform that takes rho0 from the floquet basis to the 
    computational basis.
    """
    return rho0.transform(f_modes, True)

#-------------------------------------------------------------------------------
# Floquet-Markov master equation
# 
# 
def floquet_markov_mesolve(R, ekets, rho0, tlist, expt_ops, opt=None):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.   
    """

    if opt == None:
        opt = Odeoptions()

    if opt.tidy:
        R.tidyup()

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = ket2dm(rho0)
       
    #
    # prepare output array
    # 
    n_tsteps  = len(tlist)
    dt        = tlist[1]-tlist[0]
       
    output = Odedata()
    output.times = tlist
        
    if isinstance(expt_ops, FunctionType):
        n_expt_op = 0
        expt_callback = True
        
    elif isinstance(expt_ops, list):
  
        n_expt_op = len(expt_ops)
        expt_callback = False

        if n_expt_op == 0:
            output.states = []
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in expt_ops:
                if op.isherm:
                    output.expect.append(zeros(n_tsteps))
                else:
                    output.expect.append(zeros(n_tsteps,dtype=complex))

    else:
        raise TypeError("Expectation parameter must be a list or a function")


    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis: from computational basis to the floquet basis
    #
    if ekets != None:
        rho0 = rho0.transform(ekets, True)
        if isinstance(expt_ops, list):
            for n in arange(len(e_ops)):             # not working
                e_ops[n] = e_ops[n].transform(ekets) #

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cyq_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                              atol=opt.atol, rtol=opt.rtol,
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

        if expt_callback:
            # use callback method
            expt_ops(t, Qobj(rho))
        else:
            # calculate all the expectation values, or output rho if no operators
            if n_expt_op == 0:
                output.states.append(Qobj(rho)) # copy psi/rho
            else:
                for m in range(0, n_expt_op):
                    output.expect[m][t_idx] = expect(expt_ops[m], rho) # basis OK?

        r.integrate(r.t + dt)
        t_idx += 1
          
    return output

#-------------------------------------------------------------------------------
# Solve the Floquet-Markov master equation
# 
# 
def fmmesolve(H, rho0, tlist, c_ops, e_ops=[], spectra_cb=[], T=None, args={}, options=Odeoptions()):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.  

    .. note:: 
    
        This solver currently does not support multiple collapse operators.
   
    Parameters
    ----------
    
    H : :class:`qutip.Qobj`
        system Hamiltonian.
        
    rho0 / psi0 : :class:`qutip.Qobj`
        initial density matrix or state vector (ket).
     
    tlist : *list* / *array*    
        list of times for :math:`t`.
        
    c_ops : list of :class:`qutip.Qobj`
        list of collapse operators.
    
    expt_ops : list of :class:`qutip.Qobj` / callback function
        list of operators for which to evaluate expectation values.

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.
     
    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and collapse operators.

        This dictionary should also contain an entry 'w_th', which is the temperature
        of the environment (if finite) in the energy/frequency units of the Hamiltonian.
        For example, if the Hamiltonian written in units of 2pi GHz, and the temperature
        is given in K, use the following conversion

        >>> temperature = 25e-3 # unit K
        >>> h = 6.626e-34
        >>> kB = 1.38e-23
        >>> args['w_th'] = temperature * (kB / h) * 2 * pi * 1e-9
     
    options : :class:`qutip.Odeoptions`
        options for the ODE solver.

    Returns
    -------

    output : :class:`qutip.Odedata`

        An instance of the class :class:`qutip.Odedata`, which contains either
        an *array* of expectation values for the times specified by `tlist`.
    """

    if T == None:
        T = max(tlist)

    if len(spectra_cb) == 0:
        for n in range(len(c_ops)):
            spectra_cb.append(lambda w: 1.0) # add white noise callbacks if absent

    f_modes_0, f_energies = floquet_modes(H, T, args)    
    
    kmax = 1

    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies, linspace(0, T, 500+1), H, T, args) 

    # XXX: get w_th from args
    temp = 25e-3
    w_th = temp * (1.38e-23 / 6.626e-34) * 2 * pi * 1e-9   
       
    # calculate the rate-matrices for the floquet-markov master equation
    Delta, X, Gamma, Amat = floquet_master_equation_rates(f_modes_0, f_energies, c_ops, H, T, args, spectra_cb, w_th, kmax, f_modes_table_t)
   
    # the floquet-markov master equation tensor
    R = floquet_master_equation_tensor(Amat, f_energies)
    
    output = fmmesolve(R, f_modes_0, psi0, tlist, [], opt=None) 

    return output
    
