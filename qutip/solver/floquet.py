# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:58:13 2023

@author: Fenton
"""

__all__ = [
    "FloquetBasis",
    "floquet_tensor",
    "fsesolve",
    "fmmesolve",
    "flimesolve",
    "FLiMESolver",
    "FMESolver",
]

import scipy as scipy
import numpy as np
from qutip.core import data as _data
from qutip import Qobj, QobjEvo, operator_to_vector, expect
from .propagator import Propagator
from .mesolve import MESolver
from .solver_base import Solver
from .result import Result
from time import time
from ..ui.progressbar import progess_bars


class FloquetBasis:
    """
    Utility to compute floquet modes and states.

    Attributes
    ----------
    U : :class:`Propagator`
        The propagator of the Hamiltonian over one period.

    evecs : :class:`qutip.data.Data`
        Matrix where each column is an initial Floquet mode.

    e_quasi : np.ndarray[float]
        The quasi energies of the Hamiltonian.
    """

    def __init__(
        self,
        H,
        T,
        args=None,
        options=None,
        sparse=False,
        sort=True,
        precompute=None,
    ):
        """
        Parameters
        ----------
        H : :class:`Qobj`, :class:`QobjEvo`, QobjEvo compatible format.
            System Hamiltonian, with period `T`.

        T : float
            Period of the Hamiltonian.

        args : None / *dictionary*
            dictionary of parameters for time-dependent Hamiltonians and
            collapse operators.

        options : dict [None]
            Options used by sesolve to compute the floquet modes.

        sparse : bool [False]
            Whether to use the sparse eigen solver when computing the
            quasi-energies.

        sort : bool [True]
            Whether to sort the quasi-energies.

        precompute : list [None]
            If provided, a list of time at which to store the propagators
            for later use when computing modes and states. Default is
            ``linspace(0, T, 101)`` corresponding to the default integration
            steps used for the floquet tensor computation.
        """
        if not T > 0:
            raise ValueError("The period need to be a positive number.")
        self.T = T
        if precompute is not None:
            tlist = np.unique(np.atleast_1d(precompute) % self.T)
            memoize = len(tlist)
            if tlist[0] != 0:
                memoize += 1
            if tlist[-1] != T:
                memoize += 1
        else:
            # Default computation
            tlist = np.linspace(0, T, 101)
            memoize = 101
        self.U = Propagator(H, args=args, options=options, memoize=memoize)
        for t in tlist:
            # Do the evolution by steps to save the intermediate results.
            self.U(t)
        U_T = self.U(self.T)
        if not sparse and isinstance(U_T.data, _data.CSR):
            U_T = U_T.to("Dense")
        evals, evecs = _data.eigs(U_T.data)
        e_quasi = -np.angle(evals) / T
        if sort:
            perm = np.argsort(e_quasi)
            self.evecs = _data.permute.indices(evecs, col_perm=np.argsort(perm))
            self.e_quasi = e_quasi[perm]
        else:
            self.evecs = evecs
            self.e_quasi = e_quasi

    def _as_ketlist(self, kets_mat):
        """
        Split the Data array in a list of kets.
        """
        dims = [self.U(0).dims[0], [1]]
        return [
            Qobj(ket, dims=dims, type="ket")
            for ket in _data.split_columns(kets_mat)
        ]

    def mode(self, t, data=False):
        """
        Calculate the Floquet modes at time ``t``.

        Parameters
        ----------
        t : float
            The time for which to evaluate the Floquet mode.

        data : bool [False]
            Whether to return the states as a single data matrix or a list of
            ket states.

        Returns
        -------
        output : list[:class:`Qobj`], :class:`qutip.data.Data`
            A list of Floquet states for the time ``t`` or the states as column
            in a single matrix.
        """
        t = np.round(t % self.T,8)
        # t = t % self.T
        if t == 0.0:
            kets_mat = self.evecs
        else:
            U = self.U(t).data
            phases = _data.diag(np.exp(1j * t * self.e_quasi))
            kets_mat = U @ self.evecs @ phases
        if data:
            return kets_mat
        else:
            return self._as_ketlist(kets_mat)

    def state(self, t, data=False):
        """
        Evaluate the floquet states at time t.

        Parameters
        ----------
        t : float
            The time for which to evaluate the Floquet states.

        data : bool [False]
            Whether to return the states as a single data matrix or a list of
            ket states.

        Returns
        -------
        output : list[:class:`Qobj`], :class:`qutip.data.Data`
            A list of Floquet states for the time ``t`` or the states as column
            in a single matrix.
        """
        if t:
            phases = _data.diag(np.exp(-1j * t * self.e_quasi))
            states_mat = self.mode(t, True) @ phases
        else:
            states_mat = self.evecs
        if data:
            return states_mat
        else:
            return self._as_ketlist(states_mat)

    def from_floquet_basis(self, floquet_basis, t=0):
        """
        Transform a ket or density matrix from the Floquet basis at time ``t``
        to the lab basis.

        Parameters
        ----------
        floquet_basis : :class:`Qobj`, :class:`qutip.data.Data`
            Initial state in the Floquet basis at time ``t``. May be either a
            ket or density matrix.

        t : float [0]
            The time at which to evaluate the Floquet states.

        Returns
        -------
        output : :class:`Qobj`, :class:`qutip.data.Data`
            The state in the lab basis. The return type is the same as the type
            of the input state.
        """
        is_Qobj = isinstance(floquet_basis, Qobj)
        if is_Qobj:
            dims = floquet_basis.dims
            floquet_basis = floquet_basis.data
            if dims[0] != self.U(0).dims[1]:
                raise ValueError(
                    "Dimensions of the state do not match the Hamiltonian"
                )

        state_mat = self.state(t, True)
        lab_basis = state_mat @ floquet_basis
        if floquet_basis.shape[1] != 1:
            lab_basis = lab_basis @ state_mat.adjoint()

        if is_Qobj:
            return Qobj(lab_basis, dims=dims)
        return lab_basis

    def to_floquet_basis(self, lab_basis, t=0):
        """
        Transform a ket or density matrix in the lab basis
        to the Floquet basis at time ``t``.

        Parameters
        ----------
        lab_basis : :class:`Qobj`, :class:`qutip.data.Data`
            Initial state in the lab basis.

        t : float [0]
            The time at which to evaluate the Floquet states.

        Returns
        -------
        output : :class:`Qobj`, :class:`qutip.data.Data`
            The state in the Floquet basis. The return type is the same as the
            type of the input state.
        """
        is_Qobj = isinstance(lab_basis, Qobj)
        if is_Qobj:
            dims = lab_basis.dims
            lab_basis = lab_basis.data
            if dims[0] != self.U(0).dims[1]:
                raise ValueError(
                    "Dimensions of the state do not match the Hamiltonian"
                )

        state_mat = self.state(t, True)
        floquet_basis = state_mat.adjoint() @ lab_basis
        if lab_basis.shape[1] != 1:
            floquet_basis = floquet_basis @ state_mat

        if is_Qobj:
            return Qobj(floquet_basis, dims=dims)
        return floquet_basis


def _floquet_delta_tensor(f_energies, kmax, T):
    """
    Floquet-Markov master equation X matrices.

    Parameters
    ----------
    f_energies : np.ndarray
        The Floquet energies.

    kmax : int
        The truncation of the number of sidebands (default 5).

    T : float
        The period of the time-dependence of the Hamiltonian.

    Returns
    -------
    delta : np.ndarray
        Floquet delta tensor.
    """
    delta = np.subtract.outer(f_energies, f_energies)
    return np.add.outer(delta, np.arange(-kmax, kmax + 1) * (2 * np.pi / T))


def _floquet_X_matrices(floquet_basis, c_ops, kmax, ntimes=100):
    """
    Floquet-Markov master equation X matrices.

    Parameters
    ----------
    floquet_basis : :class:`FloquetBasis`
        The system Hamiltonian wrapped in a FloquetBasis object.

    c_ops : list of :class:`Qobj`
        The collapse operators describing the dissipation.

    kmax : int
        The truncation of the number of sidebands (default 5).

    ntimes : int [100]
        The number of integration steps (for calculating X) within one period.

    Returns
    -------
    X : list of dict of :class:`qutip.data.Data`
        A dict of the sidebands ``k`` for the X matrices of each c_ops
    """
    T = floquet_basis.T
    N = floquet_basis.U(0).shape[0]
    omega = (2 * np.pi) / T
    tlist = np.linspace(T / ntimes, T, ntimes)
    ks = np.arange(-kmax, kmax + 1)
    out = {k: [_data.csr.zeros(N, N)] * len(c_ops) for k in ks}

    for t in tlist:
        mode = floquet_basis.mode(t, data=True)
        FFs = [mode.adjoint() @ c_op.data @ mode for c_op in c_ops]
        for k, phi in zip(ks, np.exp(-1j * ks * omega * t) / ntimes):
            out[k] = [
                _data.add(prev, new, phi) for prev, new in zip(out[k], FFs)
            ]

    return [{k: out[k][i] for k in ks} for i in range(len(c_ops))]


def _floquet_gamma_matrices(X, delta, J_cb):
    """
    Floquet-Markov master equation gamma matrices.

    Parameters
    ----------
    X : list of dict of :class:`qutip.data.Data`
        Floquet X matrices created by :func:`_floquet_X_matrices`.

    delta : np.ndarray
        Floquet delta tensor created by :func:`_floquet_delta_tensor`.

    J_cb : list of callables
        A list callback functions that compute the noise power spectrum as
        a function of frequency. The list should contain one callable for each
        collapse operator `c_op`, in the same order as the elements of `X`.
        Each callable should accept a numpy array of frequencies and return a
        numpy array of corresponding noise power.

    Returns
    -------
    gammas : dict of :class:`qutip.data.Data`
        A dict mapping the sidebands ``k`` to their gamma matrices.
    """
    N = delta.shape[0]
    kmax = (delta.shape[2] - 1) // 2
    gamma = {k: _data.csr.zeros(N, N) for k in range(-kmax, kmax + 1, 1)}

    for X_c_op, sp in zip(X, J_cb):
        response = sp(delta) * ((2 + 0j) * np.pi)
        response = [
            _data.Dense(response[:, :, k], copy=False)
            for k in range(2 * kmax + 1)
        ]
        for k in range(-kmax, kmax + 1, 1):
            gamma[k] = _data.add(
                gamma[k],
                _data.multiply(
                    _data.multiply(X_c_op[k].conj(), X_c_op[k]),
                    response[k + kmax],
                ),
            )
    return gamma


def _floquet_A_matrix(delta, gamma, w_th):
    """
    Floquet-Markov master equation rate matrix.

    Parameters
    ----------
    delta : np.ndarray
        Floquet delta tensor created by :func:`_floquet_delta_tensor`.

    gamma : dict of :class:`qutip.data.Data`
        Floquet gamma matrices created by :func:`_floquet_gamma_matrices`.

    w_th : float
        The temperature in units of frequency.
    """
    kmax = (delta.shape[2] - 1) // 2

    if w_th > 0.0:
        deltap = np.copy(delta)
        deltap[deltap == 0.0] = np.inf
        thermal = 1.0 / (np.exp(np.abs(deltap) / w_th) - 1.0)
        thermal = [_data.Dense(thermal[:, :, k]) for k in range(2 * kmax + 1)]

        gamma_kk = _data.add(gamma[0], gamma[0].transpose())
        A = _data.add(gamma[0], _data.multiply(thermal[kmax], gamma_kk))

        for k in range(1, kmax + 1):
            g_kk = _data.add(gamma[k], gamma[-k].transpose())
            thermal_kk = _data.multiply(thermal[kmax + k], g_kk)
            A = _data.add(A, _data.add(gamma[k], thermal_kk))
            thermal_kk = _data.multiply(thermal[kmax - k], g_kk.transpose())
            A = _data.add(A, _data.add(gamma[-k], thermal_kk))
    else:
        # w_th is 0, thermal = 0s
        A = gamma[0]
        for k in range(1, kmax + 1):
            A = _data.add(gamma[k], A)
            A = _data.add(gamma[-k], A)

    return A


def _floquet_master_equation_tensor(A):
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators?).

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------
    A : :class:`qutip.data.Data`
        Floquet-Markov master equation rate matrix.

    Returns
    -------
    output : array
        The Floquet-Markov master equation tensor `R`.
    """
    N = A.shape[0]

    # R[i+N*i, j+N*j] = A[j, i]
    cols = np.arange(N, dtype=np.int32)
    rows = np.linspace(1 - 1 / (N + 1), N, N**2 + 1, dtype=np.int32)
    data = np.ones(N, dtype=complex)
    expand = _data.csr.CSR((data, cols, rows), shape=(N**2, N))

    R = expand @ A.transpose() @ expand.transpose()

    # S[i+N*j, j+N*i] = -1/2 * sum_k(A[i, k] + A[j, k])
    ket_1 = _data.Dense(np.ones(N, dtype=complex))
    Asum = A @ ket_1
    to_super = _data.add(_data.kron(Asum, ket_1), _data.kron(ket_1, Asum))
    S = _data.diag(to_super.to_array().flatten() * -0.5, 0)

    return _data.add(R, S)


def floquet_tensor(H, c_ops, spectra_cb, T=0, w_th=0.0, kmax=5, nT=100):
    """
    Construct a tensor that represents the master equation in the floquet
    basis.

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------
    H : :class:`QobjEvo`
        Periodic Hamiltonian

    T : float
        The period of the time-dependence of the hamiltonian.

    c_ops : list of :class:`qutip.qobj`
        list of collapse operators.

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    w_th : float
        The temperature in units of frequency.

    kmax : int
        The truncation of the number of sidebands (default 5).

    Returns
    -------
    output : array
        The Floquet-Markov master equation tensor `R`.
    """
    if isinstance(H, FloquetBasis):
        floquet_basis = H
        T = H.T
    else:
        floquet_basis = FloquetBasis(H, T)
    energy = floquet_basis.e_quasi
    delta = _floquet_delta_tensor(energy, kmax, T)
    x = _floquet_X_matrices(floquet_basis, c_ops, kmax, nT)
    gamma = _floquet_gamma_matrices(x, delta, spectra_cb)
    a = _floquet_A_matrix(delta, gamma, w_th)
    r = _floquet_master_equation_tensor(a)
    dims = floquet_basis.U(0).dims
    return Qobj(
        r, dims=[dims, dims], type="super", superrep="super", copy=False
    )


def fsesolve(H, psi0, tlist, e_ops=None, T=0.0, args=None, options=None):
    """
    Solve the Schrodinger equation using the Floquet formalism.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Periodic system Hamiltonian as :class:`QobjEvo`. List of
        [:class:`Qobj`, :class:`Coefficient`] or callable that
        can be made into :class:`QobjEvo` are also accepted.

    psi0 : :class:`qutip.qobj`
        Initial state vector (ket). If an operator is provided,

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function, optional
        List of operators for which to evaluate expectation values. If this
        list is empty, the state vectors for each time in `tlist` will be
        returned instead of expectation values.

    T : float, default=tlist[-1]
        The period of the time-dependence of the hamiltonian.

    args : dictionary, optional
        Dictionary with variables required to evaluate H.

    options : dict, optional
        Options for the results.

        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.

    Returns
    -------
    output : :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`, which
        contains either an *array* of expectation values or an array of
        state vectors, for the times specified by `tlist`.
    """
    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T or tlist[-1]
        # `fsesolve` is a fallback from `fmmesolve`, for the later, options
        # are for the open system evolution.
        floquet_basis = FloquetBasis(H, T, args, precompute=tlist)

    f_coeff = floquet_basis.to_floquet_basis(psi0)
    result_options = {
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
    }
    result_options.update(options or {})
    result = Result(e_ops, result_options, solver="fsesolve")
    for t in tlist:
        state_t = floquet_basis.from_floquet_basis(f_coeff, t)
        result.add(t, state_t)

    return result

def _floquet_rate_matrix(floquet_basis,tlist,c_ops,c_op_rates,omega,time_sense=0):
    '''
    Parameters
    ----------
    qe : list
        List of Floquet quasienergies of the system.
    tlist : numpy.ndarray
        List of 2**n times distributes evenly over one period of system driving
    c_ops : list
        list of collapse operator matrices, used to calculate the Fourier 
            Amplitudes for each rate matrix
    c_op_rates : list
        List of collapse operator rates/magnitudes.
    omega : float
        the sinusoidal time-dependance of the system
    time_sense : float 0-1, optional
        the time sensitivity/secular approximation restriction for this rate
            matrix as a multiple of the system frequency beatfreq. As far as 
            Ned and I have guessed (cause we aren't positive),0 is completely 
            time independant and in line with, e.g. Blumel's R tensor 
            description (sure of this) and 1 is completely time dependant 
            (unsure of this). The default is 0.

    Returns
    -------
    total_R_tensor : 2D Numpy matrix
        This is the 2D rate matrix for the system, created by summing the 
            Rate matrix of each individual collapse operator. Something 
            something Rate Matrix something something linear Operator.
    '''
    
    def kron(a,b):  
        if a == b:
            value = 1
        else:
            value = 0
        return value
    
    '''
    First, divide all quasienergies by omega to get everything in terms of omega. 
        This way, once I've figured out the result of the exponential term addition, 
        I can just multiply it all by w
        
    Defining the exponential sums in terms of w now to save space below
    '''
    def delta(a,ap,b,bp,l,lp):
        return ((floquet_basis.e_quasi[a]-floquet_basis.e_quasi[ap]) 
               -(floquet_basis.e_quasi[b]-floquet_basis.e_quasi[bp]))/(list(omega.values())[0]/2)\
               +(l-lp)
    
   
    
    Nt = len(tlist)
    Hdim = len(floquet_basis.e_quasi)
    
    total_R_tensor = {}
    
    for cdx, c_op in enumerate(c_ops):

        '''
        These two lines transform the lowering operator into the Floquet mode 
            basis
        '''

        fmodes = np.stack([np.stack([i.full() for i in floquet_basis.mode(t)]) for t in tlist])[...,0]
        fmodes_ct = np.transpose(fmodes,(0,2,1)).conj()
        
        c_op_Floquet_basis = (fmodes_ct @ c_op.full() @ fmodes)
        
        # c_op_Floquet_basis = np.stack([floquet_basis.to_floquet_basis(c_op,t).full() for t in tlist])
        '''
        Performing the 1-D FFT to find the Fourier amplitudes of this specific
            lowering operator in the Floquet basis
            
        Divided by the length of tlist for normalization
        '''
        c_op_Fourier_amplitudes_list = np.array(scipy.fft.fft(c_op_Floquet_basis,axis=0)/len(tlist))
        
        '''
        Next, I want to find all rate-product terms that are nonzero
        '''
        rate_products = np.einsum('lab,kcd->abcdlk',c_op_Fourier_amplitudes_list,np.conj(c_op_Fourier_amplitudes_list))
        rate_products_idx = np.argwhere(rate_products != 0)
        

        
        valid_indices= [tuple(indices) for indices in rate_products_idx if delta(*tuple(indices)) == 0
                        or abs((rate_products[tuple(indices)]/delta(*tuple(indices)))**(-1)) <= time_sense]
        
          
        delta_dict = {}
        for indices in valid_indices:
            try:
                delta_dict[delta(*indices)].append(indices)
            except KeyError:
                delta_dict[delta(*indices)] = [indices]
                
        for key in delta_dict.keys():
            mask_array = np.zeros((Hdim,Hdim,Hdim,Hdim,Nt,Nt))
            for indices in delta_dict[key]:
                mask_array[indices]=True
                
            valid_c_op_products =  rate_products*mask_array
            
            #using c = ap, d = bp, k=lp
            flime_FirstTerm = c_op_rates[cdx]*np.einsum('abcdlk,ma,nc,pb,qd->mnpq',
                                            valid_c_op_products, 
                                            np.eye(Hdim,Hdim), 
                                            np.eye(Hdim,Hdim), 
                                            np.eye(Hdim,Hdim), 
                                            np.eye(Hdim,Hdim))
            
            flime_SecondTerm = c_op_rates[cdx]*np.einsum('abcdlk,ac,md,pb,qn->mnpq', 
                                          valid_c_op_products,
                                          np.eye(Hdim,Hdim), 
                                          np.eye(Hdim,Hdim), 
                                          np.eye(Hdim,Hdim), 
                                          np.eye(Hdim,Hdim))

            flime_ThirdTerm = c_op_rates[cdx]*np.einsum('abcdlk,ac,nb,pm,qd->mnpq',
                                        valid_c_op_products,
                                        np.eye(Hdim,Hdim), 
                                        np.eye(Hdim,Hdim), 
                                        np.eye(Hdim,Hdim), 
                                        np.eye(Hdim,Hdim))
            
            total_R_tensor[key] = np.reshape(flime_FirstTerm-(1/2)*(flime_SecondTerm+flime_ThirdTerm),(Hdim**2,Hdim**2))

    return total_R_tensor
def flimesolve(
        H,
        rho0,
        taulist,
        c_ops_and_rates = [],
        e_ops = [],
        T = 0,
        args = None,
        time_sense = 0,
        quicksolve = False,
        options = None):
    """
    Parameters
    ----------
    
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Periodic system Hamiltonian as :class:`QobjEvo`. List of
        [:class:`Qobj`, :class:`Coefficient`] or callable that
        can be made into :class:`QobjEvo` are also accepted.
    
    rho0 / psi0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).
    tlist : *list* / *array*
        List/Array of times over one period of Hamiltonian evolution. This is used
        to solve for the Rate matrix:math:`t`.
    Taulist:*list* / *array*
        List/Array of times over which the solution(s) is (are) desired.
    
    c_ops : list of :class:`qutip.Qobj`.
        List of collapse operator matrices
    
    c_op_rates : list of floats
        List of collapse operator rates, ordered as above
    
    e_ops : list of :class:`qutip.Qobj` / callback function
        List of operators for which to evaluate expectation values.
        The states are reverted to the lab basis before applying the
    
    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.
    
    args : *dictionary*
        Dictionary of parameters for time-dependent Hamiltonian
    
    T : time-dependance sensitivity
        The value of time dependance to be taken in the secular approximation.
            Ranges from 0 to...a number? Experimental for now until I figure it
            out better.
    
    options : None / dict
        Dictionary of options for the solver.
    
        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - store_floquet_states : bool
          Whether or not to store the density matrices in the floquet basis in
          ``result.floquet_states``.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          Which differential equation integration method to use.
        - atol, rtol : float
          Absolute and relative tolerance of the ODE integrator.
        - nsteps :
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - max_step : float, 0
          Maximum lenght of one internal step. When using pulses, it should be
          less than half the width of the thinnest pulse.
    
        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.
    
    Returns
    -------
    result: :class:`qutip.Result`
    
        An instance of the class :class:`qutip.Result`, which contains
        the expectation values for the times specified by `tlist`, and/or the
        state density matrices corresponding to the times.
    
    """
    if c_ops_and_rates is None:
        return fsesolve(
            H,
            rho0,
            taulist,
            e_ops=e_ops,
            T=T,
            w_th=0,
            args=args,
            options=options,
        )
    
    # if isinstance(c_ops, Qobj):
    #     c_ops = [c_ops]
    # if isinstance(c_op_rates,float):
    #     c_op_rates = [c_op_rates]
    
    '''
    Adding the requirement that tlist is a power of two. If it's not, the 
        code will just bring it to the nearest power of two
    '''
   
    
    #With the tlists sorted, the floquet_basis object can be prepared
    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T
        # t_precompute = taulist
        # are for the open system evolution.
        floquet_basis = FloquetBasis(H, T, args, precompute=None)
    
    # if rho0.type == 'ket':
    #     rho00 = operator_to_vector(ket2dm(rho0))
    # elif rho0.type == 'oper':
    #     rho00 = operator_to_vector(rho0)
    solver = FLiMESolver(floquet_basis,
                         c_ops_and_rates,
                         taulist,
                         args,
                         time_sense = time_sense,
                         quicksolve = quicksolve,
                         options=options)

    return solver.run(rho0, taulist,quicksolve=quicksolve, e_ops=e_ops,)
    
class FloquetResult(Result):
    def _post_init(self, floquet_basis):
        self.floquet_basis = floquet_basis
        if self.options["store_floquet_states"]:
            self.floquet_states = []
        else:
            self.floquet_states = None
        super()._post_init()

    def add(self, t, state):
        if self.options["store_floquet_states"]:
            self.floquet_states.append(state)
        state = self.floquet_basis.from_floquet_basis(state, t)
        super().add(t, state)
    def compadd(self,t,state):
        if self.options["store_floquet_states"]:
            self.floquet_states.append(state)
        super().add(t, state)        
  
class FLiMESolver(MESolver):
    """
    Solver for the Floquet-Markov master equation.

    .. note ::
        Operators (``c_ops`` and ``e_ops``) are in the laboratory basis.

    Parameters
    ----------
    floquet_basis : :class:`qutip.FloquetBasis`
        The system Hamiltonian wrapped in a FloquetBasis object. Choosing a
        different integrator for the ``floquet_basis`` than for the evolution
        of the floquet state can improve the performance.

    tlist : np.ndarray
        List of 2**n times distributed evenly over one period of the 
            Hamiltonian
    taulist: np.ndarray
        List of number_of_periods*2**n times distributed evenly over the
            entire duration of Hamiltonian driving

    Hargs : list
        The time dependence of the Hamiltonian

   c_ops : list of :class:`qutip.Qobj`.
       List of collapse operator matrices
   
   c_op_rates : list of floats
       List of collapse operator rates, ordered as above
   
    options : dict, optional
        Options for the solver, see :obj:`FMESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.
    """

    name = "flimesolve"
    _avail_integrators = {}
    resultclass = FloquetResult
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "method": "adams",
        "store_floquet_states": False,
        
    }
    def __init__(
        self, 
        floquet_basis, 
        c_ops_and_rates, 
        taulist,
        Hargs, 
        *, 
        Nt = None,
        time_sense = 0,
        quicksolve = False,
        options=None
    ):
        if isinstance(floquet_basis, FloquetBasis):
            self.floquet_basis = floquet_basis
        else:
            raise TypeError("The ``floquet_basis`` must be a FloquetBasis")
        self.options = options
        self.Hdim = np.shape(self.floquet_basis.e_quasi)[0]
        if isinstance(floquet_basis, FloquetBasis):
            self.floquet_basis = floquet_basis
        else:
            raise TypeError("The ``floquet_basis`` must be a FloquetBasis")
        
        c_ops = []
        c_op_rates = []
        for entry in c_ops_and_rates:
            c_ops.append(entry[0])
            c_op_rates.append(entry[1])

        self._num_collapse = len(c_ops)
        if not all(
            isinstance(c_op, Qobj)
            for c_op in c_ops
        ):
            raise TypeError("c_ops must be type Qobj")
        
        self.T = 2*np.pi/list(Hargs.values())[0]                                #Period of the Hamiltonian, for calculating the rate matrix
        dtau = taulist[1]-taulist[0]                                            #Finding dtau in taulist
        taulist_zeroed = taulist-taulist[0]                                     #shifting taulist to zero
        Nt_finder = abs(taulist_zeroed+dtau-self.T)                             #subtracting the period from taulist, such that Nt = where this is closest to zero
        
        #Defining tlist, to be used to construct the R tensor
        Nt = Nt if Nt != None \
                else list(np.where( 
                    Nt_finder == np.amin(Nt_finder))
                    )[0][0]+1                                                   #Finding the number of points in taulist that comprises a period                          
        time = self.T                                                           #Length of time of tlist defined to be one period of the system
        dt = time/Nt                                                            #Time point spacing in tlist
        self.tlist_rate_dic = np.linspace(0,time-dt,Nt)
        
        RateDic =  _floquet_rate_matrix(
             floquet_basis,
             self.tlist_rate_dic,
             c_ops,
             c_op_rates,
             Hargs,
             time_sense=time_sense)
            
        
        Rate_Qobj_list = [Qobj(
            RateMat, dims=[[self.Hdim,self.Hdim], [self.Hdim,self.Hdim]], type="super", superrep="super", copy=False) for RateMat in RateDic.values()]
        
        self.R0 = Rate_Qobj_list[0]
        self.Rt_timedep_pairs = [list([Rate_Qobj_list[idx],'exp(1j*'+str(list(RateDic.keys())[idx]*list(Hargs.values())[0])+'*t)']) for idx in range(1,len(Rate_Qobj_list))]
         
        if quicksolve == True:
            self.R0_evals,self.R0_evecs = scipy.linalg.eig(self.R0.full())

        else:
            if self.Rt_timedep_pairs != []: 
                self.rhs = QobjEvo([self.R0,*self.Rt_timedep_pairs])
                
            else: 
                self.rhs = QobjEvo(self.R0)
                
            self._integrator = self._get_integrator()
            self._state_metadata = {}
            self.stats = self._initialize_stats()

        
    def _initialize_stats(self):
        stats = Solver._initialize_stats(self)
        stats.update(
            {
                "solver": "Floquet-Lindblad master equation",
                "num_collapse": self._num_collapse,
            }
        )
        return stats
    
    def start(self, state0, t0, *, floquet=False):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        floquet : bool, optional {False}
            Whether the initial state is in the floquet basis or laboratory
            basis.
        """
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state0, t0)
        super().start(state0, t0)
    def step(self, t, *, args=None, copy=True, floquet=False):
        """
        Evolve the state to ``t`` and return the state as a :class:`Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        floquet : bool, optional {False}
            Whether to return the state in the floquet basis or laboratory
            basis.

        args : dict, optional {None}
            Not supported

        .. note::
            The state must be initialized first by calling ``start`` or
            ``run``. If ``run`` is called, ``step`` will continue from the last
            time and state obtained.
        """
        if args:
            raise ValueError("FMESolver cannot update arguments")
        state = super().step(t)
        if not floquet:
            state = self.floquet_basis.from_floquet_basis(state, t)
        elif copy:
            state = state.copy()
        return state

    def run(self, state00,taulist, *,quicksolve=False, floquet=False, args=None,e_ops=None,):
        """
        Calculate the evolution of the quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        floquet : bool, optional {False}
            Whether the initial state in the floquet basis or laboratory basis.

        args : dict, optional {None}
            Not supported

        e_ops : list {None}
            List of Qobj, QobjEvo or callable to compute the expectation
            values. Function[s] must have the signature
            f(t : float, state : Qobj) -> expect.

        Returns
        -------
        results : :class:`qutip.solver.FloquetResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        
        if args:
            raise ValueError("FMESolver cannot update arguments")
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state00, taulist[0])
        
        dims = np.shape(state00.full())
        fmodes_table = np.stack([np.stack([self.floquet_basis.mode(t)[i].full() for i in range(dims[0])]) for t in (taulist[0]+self.tlist_rate_dic)])[...,0]
        
        tiled_modes = []
        if len(taulist) >= len(self.tlist_rate_dic): 
            for i in range(int(len(taulist)/len(self.tlist_rate_dic))):
                tiled_modes.append(fmodes_table)
            tiled_modes = np.concatenate(tiled_modes)
        else:
            tiled_modes = np.array([fmodes_table[idx] for idx,t in enumerate(taulist)])
        
        quasi_e_table = np.exp(np.einsum('i,k -> ki',-1j*self.floquet_basis.e_quasi,taulist) )
        fstates_table = np.einsum('ijk,ij->ikj',tiled_modes, quasi_e_table  )    
        
        if quicksolve == False:
            
            _time_start = time()
            # if state0.type == 'ket':
            #     state0 = state0*state0.dag()
            # elif state0.type == 'oper':
            #     state0 = state0
            # else:
            #     print('You need to supply a valid ket or operator')
            _data0 = self._prepare_state(state0)
            
            self._integrator.set_state(taulist[0], _data0)
            stats = self._initialize_stats()
            results = self.resultclass(
                e_ops,
                self.options,
                solver=self.name,
                stats=stats,
                floquet_basis=self.floquet_basis,
            )
            stats["preparation time"] += time() - _time_start
            _time_start_solve = time()
           
            sols = [self._restore_state(_data0, copy=False).full()]
            progress_bar = progess_bars[self.options["progress_bar"]]()
            progress_bar.start(len(taulist) - 1, **self.options["progress_kwargs"])
            for t, state in self._integrator.run(taulist):
                progress_bar.update()
                sols.append(self._restore_state(state, copy=False).full())
            progress_bar.finished()
    
            sols = np.array(sols)
            sols_comp = fstates_table @ sols @ np.transpose(fstates_table.conj(),axes=(0,2,1))
            sols_comp = [Qobj(_data.Dense(state),dims=[[2], [2]], type="oper", copy=False) for state in sols_comp] 
        
            # for idx, t in enumerate(taulist):
            #     results.compadd(t,sols_comp[idx])
            
            # stats["run time"] = progress_bar.total_time()
            # return results
        else:
            
                
            _time_start = time()
            stats = {
                "method": 'FLiME_quicksolve',
                "preparation time": 0.0,
                "run time": 0.0,
            }
            results = self.resultclass(
                e_ops,
                self.options,
                solver=self.name,
                stats=stats,
                floquet_basis=self.floquet_basis,
            )
            
            eval_exp_diagonal = np.einsum('ij,ki->kij',np.eye(self.Hdim**2),np.exp(np.einsum('i,k->ki', self.R0_evals,taulist-taulist[0])))
            sol_coeffs = self.R0_evecs @  eval_exp_diagonal @ scipy.linalg.inv(self.R0_evecs)
            
            
            stats["preparation time"] += time() - _time_start
            if state0.type == 'ket':
                state0 = operator_to_vector(state0*state0.dag())
            elif state0.type == 'oper':
                state0 = operator_to_vector(state0)
            else:
                print('You need to supply a valid ket or operator')

            progress_bar = progess_bars[self.options["progress_bar"]]()
            progress_bar.start(len(taulist) - 1, **self.options["progress_kwargs"])
            
            sols = np.reshape(np.einsum('...jk,kz->...jz',sol_coeffs[:len(taulist)], state0.full()),[len(taulist),dims[0],dims[0]],order='F')
            sols_comp = fstates_table @ sols @ np.transpose(fstates_table.conj(),axes=(0,2,1))

            sols_comp = [Qobj(_data.Dense(state),dims=[[2], [2]], type="oper", copy=False) for state in sols_comp] 
            
            '''
            WRITE IN THE EXPECTATION VALUE FUNCTION HERE. THIS IS THE PROBLEM IN CORRELATION FUNCTIONS 
            '''

            
            # results.times = taulist
            # results.states = sols_comp
            # stats["run time"] = time()-_time_start_solve+self.added_time
            # return results
        for idx, t in enumerate(taulist):
            results.compadd(t,sols_comp[idx])
        
        stats["run time"] = progress_bar.total_time()
        return results

    def _argument(self, args):
        if args:
            raise ValueError("FMESolver cannot update arguments")


def fmmesolve(
    H,
    rho0,
    tlist,
    c_ops=None,
    e_ops=None,
    spectra_cb=None,
    T=0,
    w_th=0.0,
    args=None,
    options=None,
):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.

    Parameters
    ----------
    H : :class:`Qobj`, :class:`QobjEvo`, :class:`QobjEvo` compatible format.
        Periodic system Hamiltonian as :class:`QobjEvo`. List of
        [:class:`Qobj`, :class:`Coefficient`] or callable that
        can be made into :class:`QobjEvo` are also accepted.

    rho0 / psi0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    c_ops : list of :class:`qutip.Qobj`
        List of collapse operators. Time dependent collapse operators are not
        supported.

    e_ops : list of :class:`qutip.Qobj` / callback function
        List of operators for which to evaluate expectation values.
        The states are reverted to the lab basis before applying the

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    w_th : float
        The temperature of the environment in units of frequency.
        For example, if the Hamiltonian written in units of 2pi GHz, and the
        temperature is given in K, use the following conversion:

            temperature = 25e-3 # unit K
            h = 6.626e-34
            kB = 1.38e-23
            args['w_th'] = temperature * (kB / h) * 2 * pi * 1e-9

    args : *dictionary*
        Dictionary of parameters for time-dependent Hamiltonian

    options : None / dict
        Dictionary of options for the solver.

        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool, None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - store_floquet_states : bool
          Whether or not to store the density matrices in the floquet basis in
          ``result.floquet_states``.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text', 'enhanced', 'tqdm', ''}
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str ["adams", "bdf", "lsoda", "dop853", "vern9", etc.]
          Which differential equation integration method to use.
        - atol, rtol : float
          Absolute and relative tolerance of the ODE integrator.
        - nsteps :
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - max_step : float, 0
          Maximum lenght of one internal step. When using pulses, it should be
          less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`, which contains
        the expectation values for the times specified by `tlist`, and/or the
        state density matrices corresponding to the times.
    """
    if c_ops is None:
        return fsesolve(
            H,
            rho0,
            tlist,
            e_ops=e_ops,
            T=T,
            
            args=args,
            options=options,
        )

    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T or tlist[-1]
        t_precompute = np.concatenate([tlist, np.linspace(0, T, 101)])
        # `fsesolve` is a fallback from `fmmesolve`, for the later, options
        # are for the open system evolution.
        floquet_basis = FloquetBasis(H, T, args, precompute=t_precompute)

    if not w_th and args:
        w_th = args.get("w_th", 0.0)

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if spectra_cb is None:
        spectra_cb = [lambda w: (w > 0)]
    elif callable(spectra_cb):
        spectra_cb = [spectra_cb]
    if len(spectra_cb) == 1:
        spectra_cb = spectra_cb * len(c_ops)

    a_ops = list(zip(c_ops, spectra_cb))

    solver = FMESolver(floquet_basis, a_ops, w_th=w_th, options=options)
    return solver.run(rho0, tlist, e_ops=e_ops)




class FMESolver(MESolver):
    """
    Solver for the Floquet-Markov master equation.

    .. note ::
        Operators (``c_ops`` and ``e_ops``) are in the laboratory basis.

    Parameters
    ----------
    floquet_basis : :class:`qutip.FloquetBasis`
        The system Hamiltonian wrapped in a FloquetBasis object. Choosing a
        different integrator for the ``floquet_basis`` than for the evolution
        of the floquet state can improve the performance.

    a_ops : list of tuple(:class:`qutip.Qobj`, callable)
        List of collapse operators and the corresponding function for the noise
        power spectrum. The collapse operator must be a :class:`Qobj` and
        cannot be time dependent. The spectrum function must take and return
        an numpy array.

    w_th : float
        The temperature of the environment in units of Hamiltonian frequency.

    kmax : int [5]
        The truncation of the number of sidebands..

    nT : int [20*kmax]
        The number of integration steps (for calculating X) within one period.

    options : dict, optional
        Options for the solver, see :obj:`FMESolver.options` and
        `Integrator <./classes.html#classes-ode>`_ for a list of all options.
    """

    name = "fmmesolve"
    _avail_integrators = {}
    resultclass = FloquetResult
    solver_options = {
        "progress_bar": "text",
        "progress_kwargs": {"chunk_size": 10},
        "store_final_state": False,
        "store_states": None,
        "normalize_output": True,
        "method": "adams",
        "store_floquet_states": False,
    }

    def __init__(
        self, floquet_basis, a_ops, w_th=0.0, *, kmax=5, nT=None, options=None
    ):
        self.options = options
        if isinstance(floquet_basis, FloquetBasis):
            self.floquet_basis = floquet_basis
        else:
            raise TypeError("The ``floquet_basis`` must be a FloquetBasis")

        nT = nT or max(100, 20 * kmax)
        self._num_collapse = len(a_ops)
        c_ops, spectra_cb = zip(*a_ops)
        if not all(
            isinstance(c_op, Qobj) and callable(spectrum)
            for c_op, spectrum in a_ops
        ):
            raise TypeError("a_ops must be tuple of (Qobj, callable)")
        self.rhs = QobjEvo(
            floquet_tensor(
                self.floquet_basis,
                c_ops,
                spectra_cb,
                w_th=w_th,
                kmax=kmax,
                nT=nT,
            )
        )

        self._integrator = self._get_integrator()
        self._state_metadata = {}
        self.stats = self._initialize_stats()

    def _initialize_stats(self):
        stats = Solver._initialize_stats(self)
        stats.update(
            {
                "solver": "Floquet-Markov master equation",
                "num_collapse": self._num_collapse,
            }
        )
        return stats

    def _argument(self, args):
        if args:
            raise ValueError("FMESolver cannot update arguments")

    def start(self, state0, t0, *, floquet=False):
        """
        Set the initial state and time for a step evolution.
        ``options`` for the evolutions are read at this step.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        t0 : double
            Initial time of the evolution.

        floquet : bool, optional {False}
            Whether the initial state is in the floquet basis or laboratory
            basis.
        """
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state0, t0)
        super().start(state0, t0)

    def step(self, t, *, args=None, copy=True, floquet=False):
        """
        Evolve the state to ``t`` and return the state as a :class:`Qobj`.

        Parameters
        ----------
        t : double
            Time to evolve to, must be higher than the last call.

        copy : bool, optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        floquet : bool, optional {False}
            Whether to return the state in the floquet basis or laboratory
            basis.

        args : dict, optional {None}
            Not supported

        .. note::
            The state must be initialized first by calling ``start`` or
            ``run``. If ``run`` is called, ``step`` will continue from the last
            time and state obtained.
        """
        if args:
            raise ValueError("FMESolver cannot update arguments")
        state = super().step(t)
        if not floquet:
            state = self.floquet_basis.from_floquet_basis(state, t)
        elif copy:
            state = state.copy()
        return state

    def run(self, state0, tlist, *, floquet=False, args=None, e_ops=None):
        """
        Calculate the evolution of the quantum system.

        For a ``state0`` at time ``tlist[0]`` do the evolution as directed by
        ``rhs`` and for each time in ``tlist`` store the state and/or
        expectation values in a :class:`Result`. The evolution method and
        stored results are determined by ``options``.

        Parameters
        ----------
        state0 : :class:`Qobj`
            Initial state of the evolution.

        tlist : list of double
            Time for which to save the results (state and/or expect) of the
            evolution. The first element of the list is the initial time of the
            evolution. Each times of the list must be increasing, but does not
            need to be uniformy distributed.

        floquet : bool, optional {False}
            Whether the initial state in the floquet basis or laboratory basis.

        args : dict, optional {None}
            Not supported

        e_ops : list {None}
            List of Qobj, QobjEvo or callable to compute the expectation
            values. Function[s] must have the signature
            f(t : float, state : Qobj) -> expect.

        Returns
        -------
        results : :class:`qutip.solver.FloquetResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """
        if args:
            raise ValueError("FMESolver cannot update arguments")
        if not floquet:
            state0 = self.floquet_basis.to_floquet_basis(state0, tlist[0])
        _time_start = time()
        _data0 = self._prepare_state(state0)
        self._integrator.set_state(tlist[0], _data0)
        stats = self._initialize_stats()
        results = self.resultclass(
            e_ops,
            self.options,
            solver=self.name,
            stats=stats,
            floquet_basis=self.floquet_basis,
        )
        results.add(tlist[0], self._restore_state(_data0, copy=False))
        stats["preparation time"] += time() - _time_start

        progress_bar = progess_bars[self.options["progress_bar"]]()
        progress_bar.start(len(tlist) - 1, **self.options["progress_kwargs"])
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            results.add(t, self._restore_state(state, copy=False))
        progress_bar.finished()

        stats["run time"] = progress_bar.total_time()
        # TODO: It would be nice if integrator could give evolution statistics
        # stats.update(_integrator.stats)
        return results
