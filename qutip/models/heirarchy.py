"""
Generic Heirarchy equations model version for complex exponents
"""
import numpy as np
import scipy.sparse as sp
import scipy.integrate
from copy import copy
from numpy import matrix
from numpy import linalg
from scipy.misc import factorial
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip import Qobj
from qutip import spre, spost, sprepost, thermal_dm, mesolve, Options, dims
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis, qeye
from qutip import liouvillian as liouv
from qutip import mat2vec, state_number_enumerate, basis
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.solver import Result


def _heom_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.
    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.
    excitations : integer
        The maximum numbers of dimension
    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state


def _heom_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerates all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.
    Example:
        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.
    state : list
        Current state in the iteration. Used internally.
    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.
    idx : integer
        Current index in the iteration. Used internally.
    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.
    """

    if state is None:
        state = np.zeros(len(dims))

    if excitations and sum(state[0:idx]) > excitations:
        pass

    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)

    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


def cot(x):
    """
    Calculate cotangent.
    Parameters
    ----------
    x: Float
        Angle.
    """
    return np.cos(x) / np.sin(x)


class Heirarchy(object):
    """
    The hierarchy model class.

    Parameters
    ==========
    hamiltonian: Qobj
        The system hamiltonian

    coupling: float
        The coupling strength

    temperature: float
        The temperature of the bath in units corresponding to the planck const
        default: 1.

    Nc: int
        Cutoff temperature for the bath
        default=1

    real_coeff: list
        The list of coefficients for the real terms in the exponential
        series expansion of the spectral density

    real_freq: list
        The list of frequencies for the real terms of the exponential series
        expansion of the spectral density

    imaginary_coeff: list
        The list of coefficients for the imaginary terms of the exponential
        series expansion of the spectral density

    imaginary_freq: list
        The list of frequencies for the imaginary terms of the exponential
        series expansion of the spectral density

    planck: float
        default: 1.
        The reduced Planck's constant.

    boltzmann: float
        default: 1.
        The reduced Boltzmann's constant.
    """
    def __init__(self, hamiltonian, coupling, temperature = 1, Nc = 1,
                 real_coeff=[], real_freq=[],
                 imaginary_coeff=[], imaginary_freq=[],
                 planck=1., boltzmann=1.):
        self.hamiltonian = hamiltonian
        self.coupling = coupling
        self.temperature = temperature
        self.Nc = Nc

        # Make this automated from just a normal list of real+complex list
        # Have a check on the lengths of the two lists if this is separate

        self.real_prefactors = real_coeff
        self.real_exponents = real_freq
        self.complex_prefactors = imaginary_coeff
        self.complex_exponents = imaginary_freq
        
        self.planck = planck
        self.boltzmann = boltzmann
        self.NR = len(real_coeff)
        self.NI = len(imaginary_coeff)
        self._N_he = 0

        self.progress_bar = None

    def _rhs(self, progress_bar=False):
        """
        Construct the RHS for the dynamics of this system
        """
        NI = self.NI
        NR = self.NR
        N = NI + NR
        Nc = self.Nc
        Q = self.coupling
        ckAR = self.real_prefactors
        ckAI = self.complex_prefactors
        vkAR = self.real_exponents
        vkAI = self.complex_exponents

        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None 
        # Set by system
        N_temp = 1
        for i in self.hamiltonian.dims[0]:
            N_temp *= N_temp*i
        sup_dim = N_temp**2
        unit = qeye(N_temp)

        # Ntot is the total number of ancillary elements in the hierarchy
        Ntot = int(round(factorial(Nc + N) / (factorial(Nc) * factorial(N))))
        # LD1 = -2.* spre(Q) * spost(Q.dag()) + spre(Q.dag() * Q) + spost(Q.dag() * Q)

        # L12=0.*LD1;
        # Setup liouvillian
        L = liouv(self.hamiltonian, [])
        Ltot = L.data
        unitthing = sp.identity(Ntot, dtype='complex', format='csr')
        Lbig = sp.kron(unitthing, Ltot.tocsr())
        
        nstates, state2idx, idx2state =_heom_state_dictionaries([Nc+1]*(N),Nc)
        self._N_he = nstates
        for nlabelt in _heom_number_enumerate([Nc+1]*(N),Nc):
            nlabel = list(nlabelt)                    
            ntotalcheck = 0
            for ncheck in range(N):
                ntotalcheck = ntotalcheck + nlabel[ncheck]                            
            current_pos = int(round(state2idx[tuple(nlabel)]))
            Ltemp = sp.lil_matrix((Ntot, Ntot))
            Ltemp[current_pos,current_pos] = 1
            Ltemp.tocsr()
            for idx,vki in enumerate(vkAR):
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[idx] * vki * spre(unit).data))

            #treat real and imaginary parts seperately

            for idx,vki in enumerate(vkAI):
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[(NR)+idx] * vki * spre(unit).data))
                #Lbig = Lbig + sp.kron(Ltemp,(-nlabel[1] * vkA[1] * spre(unit).data))
            #for kcount in range(N):
            #    Lbig = Lbig + sp.kron(Ltemp,(-nlabel[kcount] * (vkA[kcount])
            #                    * spre(unit).data))

            for kcount in range(N):
                if nlabel[kcount]>=1:
                #find the position of the neighbour
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] -1
                    current_pos2 = int(round(state2idx[tuple(nlabel)]))
                    Ltemp = sp.lil_matrix((Ntot, Ntot))
                    Ltemp[current_pos, current_pos2] = 1
                    Ltemp.tocsr()
                # renormalized version:    
                    #ci =  (4 * lam0 * gam * kb * Temperature * kcount
                    #      * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                    if kcount<=(NR-1):
                        #DO REAL PARTS Vx
                        c0=ckAR[kcount]
                        c0n=ckAR[kcount]
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(c0n)))
                                         * c0*(spre(Q).data - spost(Q).data)))
                    if kcount>=(NR):     
                        #in=lam
                        #i =  ckA[kcount]
                        c0=ckAI[kcount-NR]
                        c0n=ckAI[kcount-NR]
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(c0n)))
                                         * (1.j*(c0) * (spre(Q).data + spost(Q).data))))
                    nlabel = copy(nlabeltemp)

            for kcount in range(N):
                if ntotalcheck<=(Nc-1):
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] + 1
                    current_pos3 = int(round(state2idx[tuple(nlabel)]))
                if current_pos3<=(Ntot):
                    Ltemp = sp.lil_matrix((Ntot, Ntot))
                    Ltemp[current_pos, current_pos3] = 1
                    Ltemp.tocsr()
                #renormalized   
                    if kcount<=(NR-1):
                        c0n=ckAR[kcount]

                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*((abs(c0n))))
                                      * (spre(Q)- spost(Q)).data)
                    if kcount>=(NR):
                        cin=ckAI[kcount-NR]

                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                      * (spre(Q)- spost(Q)).data)
                nlabel = copy(nlabeltemp)

        rhs = Lbig.tocsr()
        return rhs



    def solve(self, initial_state, tlist, options=None, progress_bar=False):
        """
        """
        if options is None:
            options = Options()

        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None

        N = self.NI + self.NR
        Nc = self.Nc
        rho0 = initial_state
        # Set by system
        N_temp = 1
        for i in self.hamiltonian.dims[0]:
            N_temp = N_temp*i

        sup_dim = N_temp**2
        unit = qeye(N_temp)

        L_helems = self._rhs(progress_bar)
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat

        # Ntot is the total number of ancillary elements in the hierarchy
        r = scipy.integrate.ode(cy_ode_rhs)
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
        r.set_initial_value(rho0_he, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)
        return output

