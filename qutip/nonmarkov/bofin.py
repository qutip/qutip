"""
This module provides exact solvers for a system-bath setup using the
hierarchy equations of motion (HEOM).
"""

# Authors: Neill Lambert, Tarun Raheja, Shahnawaz Ahmed
# Contact: nwlambert@gmail.com

import warnings
from copy import deepcopy
from math import factorial

import numpy as np
import scipy.sparse as sp
import scipy.integrate
from scipy.sparse.linalg import splu

from qutip import settings
from qutip import state_number_enumerate
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
from qutip.superoperator import liouvillian, spre, spost, vec2mat
from qutip.cy.heom import cy_pad_csr
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.solver import Options, Result
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.ui.progressbar import BaseProgressBar


def add_at_idx(seq, k, val):
    """
    Add (subtract) a value in the tuple at position k
    """
    list_seq = list(seq)
    list_seq[k] += val
    return tuple(list_seq)


def prevhe(current_he, k, ncut):
    """
    Calculate the previous heirarchy index
    for the current index `n`.
    """
    nprev = add_at_idx(current_he, k, -1)
    if nprev[k] < 0:
        return False
    return nprev


def nexthe(current_he, k, ncut):
    """
    Calculate the next heirarchy index
    for the current index `n`.
    """
    nnext = add_at_idx(current_he, k, 1)
    if sum(nnext) > ncut:
        return False
    return nnext


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


def _convert_h_sys(H_sys):
    """ Process input system Hamiltonian, converting and raising as needed.
    """
    if isinstance(H_sys, (Qobj, QobjEvo)):
        pass
    elif isinstance(H_sys, list):
        try:
            H_sys = QobjEvo(H_sys)
        except Exception as err:
            raise ValueError(
                "Hamiltonian (H_sys) of type list cannot be converted to"
                " QObjEvo"
            ) from err
    else:
        raise TypeError(
            f"Hamiltonian (H_sys) has unsupported type: {type(H_sys)!r}")
    return H_sys


def _convert_coup_op(coup_op, coup_op_len):
    """ Convert coup_op to a list of the appropriate length. """
    if isinstance(coup_op, Qobj):
        coup_op = [coup_op] * coup_op_len
    elif (isinstance(coup_op, list)
            and all(isinstance(x, Qobj) for x in coup_op)):
        if len(coup_op) != coup_op_len:
            raise ValueError(
                f"Expected {coup_op_len} coupling operators.")
    else:
        raise TypeError(
            "Coupling operator (coup_op) must be a Qobj or a list of Qobjs"
        )
    return coup_op


def _convert_bath_exponents_bosonic(ckAI, ckAR, vkAI, vkAR):
    all_k = (ckAI, ckAR, vkAI, vkAR)
    if any(not isinstance(k, list) for k in all_k):
        raise TypeError(
            "The bath exponents ckAI, ckAR, vkAI and vkAR must all be lists")
    if len(ckAI) != len(vkAI) or len(ckAR) != len(vkAR):
        raise ValueError(
            "The bath exponent lists ckAI and vkAI, and ckAR and vkAR must"
            " be the same length"
        )
    if any(isinstance(x, list) for k in all_k for x in k):
        raise ValueError(
            "The bath exponent lists ckAI, ckAR, vkAI and vkAR should not"
            " themselves contain lists"
        )
    # warn if any of the vkAR's are close
    for i in range(len(vkAR)):
        for j in range(i + 1, len(vkAR)):
            if np.isclose(vkAR[i], vkAR[j], rtol=1e-5, atol=1e-7):
                warnings.warn(
                    "Expected simplified input. "
                    "Consider collating equal frequency parameters."
                )
    # warn if any of the vkAR's are close
    for i in range(len(vkAI)):
        for j in range(i + 1, len(vkAI)):
            if np.isclose(vkAI[i], vkAI[j], rtol=1e-5, atol=1e-7):
                warnings.warn(
                    "Expected simplified input.  "
                    "Consider collating equal frequency parameters."
                )
    return ckAI, ckAR, vkAI, vkAR


def _mangle_bath_exponents_bosonic(coup_op, ckAR, ckAI, vkAR, vkAI):
    """ Mangle bath exponents by combining similar vkAR and vkAI. """

    common_ck = []
    real_indices = []
    common_vk = []
    img_indices = []
    common_coup_op = []
    coup_op = deepcopy(coup_op)
    nr = len(ckAR)

    for i in range(len(vkAR)):
        for j in range(len(vkAI)):
            if np.isclose(
                vkAR[i], vkAI[j], rtol=1e-5, atol=1e-7) and np.allclose(
                coup_op[i], coup_op[nr + j], rtol=1e-5, atol=1e-7
            ):
                warnings.warn(
                    "Two similar real and imag exponents have been "
                    "collated automatically."
                )
                common_ck.append(ckAR[i])
                common_ck.append(ckAI[j])
                common_vk.append(vkAR[i])
                common_vk.append(vkAI[j])
                real_indices.append(i)
                img_indices.append(j)
                common_coup_op.append(coup_op[i])

    for i in sorted(real_indices, reverse=True):
        ckAR.pop(i)
        vkAR.pop(i)

    for i in sorted(img_indices, reverse=True):
        ckAI.pop(i)
        vkAI.pop(i)

    img_coup_ops = [x + nr for x in img_indices]
    coup_op_indices = real_indices + sorted(img_coup_ops)
    for i in sorted(coup_op_indices, reverse=True):
        coup_op.pop(i)

    coup_op += common_coup_op

    ck = np.array(ckAR + ckAI + common_ck).astype(complex)
    vk = np.array(vkAR + vkAI + common_vk).astype(complex)
    NR = len(ckAR)
    NI = len(ckAI)

    return coup_op, ck, vk, NR, NI


def _convert_bath_exponents_fermionic(ck, vk):
    """ Check the bath exponents for the fermionic solver. """
    if (type(ck) != list or not all(isinstance(x, list) for x in ck)):
        raise TypeError("The bath exponents ck must be a list or lists.")
    if (type(vk) != list or not all(isinstance(x, list) for x in vk)):
        raise TypeError("The bath exponents vk must be a list or lists.")
    if (len(ck) != len(vk)
            or any(len(ck[i]) != len(vk[i]) for i in range(len(ck)))):
        raise ValueError("Exponents ck and vk must be the same length.")
    return ck, vk


def _h_sys_is_hamiltonian(H_sys):
    """ Return True if H_sys is a Hamiltonian and False if it is a Liouvillian.
    """
    if type(H_sys) is QobjEvo:
        H_sys_list = H_sys.to_list()
        H_sys = H_sys_list[0]
    return H_sys.type == "oper"


def _h_sup_dim(H_sys, isHamiltonian):
    """ Return the super operator dimension for the given hamiltonian.
    """
    if type(H_sys) is QobjEvo:
        H_sys_list = H_sys.to_list()
        H_shape = H_sys_list[0].shape
    else:
        H_shape = H_sys.shape
    H_sup_dim = H_shape[0] ** 2 if isHamiltonian else H_shape[0]
    return H_sup_dim


class BosonicHEOMSolver(object):
    """
    This is a class for solvers that use the HEOM method for
    calculating the dynamics evolution.
    The method can compute open system dynamics without using any Markovian
    or rotating wave approximations (RWA) for systems where the bath
    correlations can be approximated to a sum of complex exponentials.
    The method builds a matrix of linked differential equations, which are
    then solved used the same ODE solvers as other qutip solvers
    (e.g. mesolve)

    Attributes
    ----------
    H_sys : Qobj or list
        System Hamiltonian
        Or
        Liouvillian
        Or
        QobjEvo
        Or
        list of Hamiltonians with time dependence

        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1],
        [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the same length
        as ck's and vk's.

    ckAR, ckAI, vkAR, vkAI : lists
        Lists containing coefficients for fitting spectral density correlation

        # Expand bath correlation function as prefactors times exponentials.
        # ck - prefactors
        # vk - frequency of the exponential
        #
        # ck * exp (-vk * t)
        #
        # ckAR * exp (-vkR * t) + i * ckAI * exp (-vkI * t)

    N_cut : int
        # The number of auxiliary density operators (ADO) to retain within the
        # hierarchy.
        #
        # For each exponential in the speci ADO
        #
        # Cutoff parameter for the bath
        #
        # Number of operators kept in the hierarhcy.
        #
        # For each exponential we have a certain number of terms in the
        # hierarchy.
        #
        # Total number of exponentials kept in the hierarchy.
        #
        # A bit like having total number of shared excitations.

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(
        self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None
    ):
        self.H_sys = _convert_h_sys(H_sys)
        self.options = Options() if options is None else options
        self.isTimeDep = isinstance(self.H_sys, QobjEvo)
        self.isHamiltonian = _h_sys_is_hamiltonian(self.H_sys)
        self._sup_dim = _h_sup_dim(self.H_sys, self.isHamiltonian)
        self.N_cut = N_cut
        coup_op = _convert_coup_op(coup_op, len(ckAR) + len(ckAI))
        ckAR, ckAI, vkAR, vkAI = _convert_bath_exponents_bosonic(
            ckAR, ckAI, vkAR, vkAI)
        self.coup_op, self.ck, self.vk, self.NR, self.NI = (
            _mangle_bath_exponents_bosonic(coup_op, ckAR, ckAI, vkAR, vkAI)
        )
        self.progress_bar = BaseProgressBar()

        self._configure_solver()

    def boson_grad_n(self, he_n):
        """
        Get the gradient term for the hierarchy ADM at
        level n
        """

        # skip variable adjusts for common gammas that
        # are passed at the end by process_input
        # by only processing alternate values
        skip = 0

        gradient_sum = 0
        L = self.L.copy()

        for i in range(len(self.vk)):
            # the initial values have different gammas
            # so are processed normally
            if i < self.NR + self.NI:
                gradient_sum += he_n[i] * self.vk[i]

            # the last few values are duplicate so only half need be processed
            else:
                if skip:
                    skip = 0
                    continue
                else:
                    tot_fixed = self.NR + self.NI
                    extra = i + 1 - tot_fixed
                    idx = int(tot_fixed + (extra / 2) + (extra % 2)) - 1
                    gradient_sum += he_n[idx] * self.vk[i]
                    skip = 1

        gradient_sum = -1 * gradient_sum
        sum_op = gradient_sum * sp.eye(
            self.L.shape[0], dtype=complex, format="csr")
        L += sum_op

        # Fill into larger L
        nidx = self.he2idx[he_n]
        L_he_temp = cy_pad_csr(L, self.nhe, self.nhe, nidx, nidx)
        self.L_helems += L_he_temp

    def boson_grad_prev(self, he_n, k, prev_he):
        """
        Get the previous gradient
        """
        nk = he_n[k]
        ck = self.ck

        # processes according to whether index into gammas
        # is in the part of the list with duplicate gammas
        # as processed by process_input
        if k < self.NR:
            norm_prev = nk
            op1 = -1j * norm_prev * ck[k] * (
                self.spreQ[k] - self.spostQ[k]
                )
        elif k >= self.NR and k < self.NR + self.NI:
            norm_prev = nk
            op1 = -1j * norm_prev * 1j * self.ck[k] * (
                self.spreQ[k] + self.spostQ[k]
                )
        else:
            norm_prev = nk
            k1 = self.NR + self.NI + 2 * (k - self.NR - self.NI)
            term1 = -1j * self.ck[k1] * (self.spreQ[k] - self.spostQ[k])
            term2 = self.ck[k1 + 1] * (self.spreQ[k] + self.spostQ[k])
            op1 = norm_prev * (term1 + term2)

        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[prev_he]
        L_he_temp = cy_pad_csr(op1, self.nhe, self.nhe, rowidx, colidx)
        self.L_helems += L_he_temp

    def boson_grad_next(self, he_n, k, next_he):
        """
        Get the next gradient
        """
        norm_next = 1
        op2 = -1j * norm_next * (self.spreQ[k] - self.spostQ[k])

        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[next_he]
        L_he_temp = cy_pad_csr(op2, self.nhe, self.nhe, rowidx, colidx)
        self.L_helems += L_he_temp

    def boson_rhs(self):
        """
        Make the RHS for bosonic case
        """
        for n in self.idx2he:
            he_n = self.idx2he[n]
            self.boson_grad_n(he_n)
            for k in range(self.kcut):
                next_he = nexthe(he_n, k, self.N_cut)
                prev_he = prevhe(he_n, k, self.N_cut)
                if next_he and (next_he in self.he2idx):
                    self.boson_grad_next(he_n, k, next_he)
                if prev_he and (prev_he in self.he2idx):
                    self.boson_grad_prev(he_n, k, prev_he)

    def _boson_solver(self):
        """
        Utility function for bosonic solver.
        """
        # Initialize liouvillians and others using inputs
        self.kcut = int(
            self.NR + self.NI + (len(self.ck) - self.NR - self.NI) / 2
        )
        nhe, he2idx, idx2he = _heom_state_dictionaries(
            [self.N_cut + 1] * self.kcut, self.N_cut
        )
        self.nhe = nhe
        self.he2idx = he2idx
        self.idx2he = idx2he
        total_nhe = int(
            factorial(self.N_cut + self.kcut)
            / (factorial(self.N_cut) * factorial(self.kcut))
        )
        # TODO: Move this assert to the docstring too
        assert total_nhe == nhe

        # Separate cases for Hamiltonian and Liouvillian
        if self.isHamiltonian:

            if self.isTimeDep:
                H_sys_list = self.H_sys.to_list()
                self.N = H_sys_list[0].shape[0]
                self.L = liouvillian(H_sys_list[0], []).data

            else:
                self.N = self.H_sys.shape[0]
                self.L = liouvillian(self.H_sys, []).data

        else:
            if self.isTimeDep:
                H_sys_list = self.H_sys.to_list()
                self.N = int(np.sqrt(H_sys_list[0].shape[0]))
                self.L = H_sys_list[0].data

            else:
                self.N = int(np.sqrt(self.H_sys.shape[0]))
                self.L = self.H_sys.data

        self.L_helems = sp.csr_matrix(
            (self.nhe * self.N ** 2, self.nhe * self.N ** 2),
            dtype=np.complex128,
        )

        # Set coupling operators
        spreQ = []
        spostQ = []
        for coupOp in self.coup_op:
            spreQ.append(spre(coupOp).data)
            spostQ.append(spost(coupOp).data)
        self.spreQ = spreQ
        self.spostQ = spostQ

        # make right hand side
        self.boson_rhs()

        # return output
        return self.L_helems, self.nhe

    def _configure_solver(self):
        """ Set up the solver. """
        RHSmat, nstates = self._boson_solver()
        RHSmat = RHSmat.tocsr()

        if self.isTimeDep:
            solver_params = []
            # TODO: Does the solver require a constant_func still or can
            #       we just add [RHSmat]?
            constant_func = lambda x: 1.0
            h_identity_mat = sp.identity(nstates, format="csr")
            solver_params.append([RHSmat, constant_func])
            H_list = self.H_sys.to_list()

            # Store each time dependent component
            for idx in range(1, len(H_list)):
                temp_mat = sp.kron(
                    h_identity_mat, liouvillian(H_list[idx][0])
                    )
                solver_params.append([temp_mat, H_list[idx][1]])

            solver = scipy.integrate.ode(_dsuper_list_td)
            solver.set_f_params(solver_params)
        else:
            solver = scipy.integrate.ode(cy_ode_rhs)
            solver.set_f_params(RHSmat.data, RHSmat.indices, RHSmat.indptr)

        solver.set_integrator(
            "zvode",
            method=self.options.method,
            order=self.options.order,
            atol=self.options.atol,
            rtol=self.options.rtol,
            nsteps=self.options.nsteps,
            first_step=self.options.first_step,
            min_step=self.options.min_step,
            max_step=self.options.max_step,
        )

        self._ode = solver
        self.RHSmat = RHSmat

    def steady_state(
        self, max_iter_refine=100, use_mkl=False, weighted_matching=False
    ):
        """
        Computes steady state dynamics
        parameters:
        max_iter_refine : Int
            Parameter for the mkl LU solver. If pardiso errors are returned
            this should be increased.
        use_mkl : Boolean
            Optional override default use of mkl if mkl is installed.
        weighted_matching : Boolean
            Setting this true may increase run time, but reduce stability
            (pardisio may not converge).

        Returns
        -------
        steady state :  Qobj
            The steady state density matrix of the system
        solution    :   Numpy array
            Array of the the steady-state and all ADOs.
            Further processing of this can be done with functions provided in
            example notebooks.
        """
        nstates = self.nhe
        sup_dim = self._sup_dim
        n = int(np.sqrt(sup_dim))
        L = deepcopy(self.RHSmat)

        b_mat = np.zeros(sup_dim * nstates, dtype=complex)
        b_mat[0] = 1.0

        L = L.tolil()
        L[0, 0: n ** 2 * nstates] = 0.0
        L = L.tocsr()

        # TODO: Replace with the standard QuTiP configuration options for MKL.
        if settings.has_mkl & use_mkl:
            # TODO: Replace with logging?
            print("Using Intel mkl solver")
            from qutip._mkl.spsolve import mkl_spsolve

            L = L.tocsr() + sp.csr_matrix((
                np.ones(n),
                (np.zeros(n), [num * (n + 1) for num in range(n)])
            ), shape=(n ** 2 * nstates, n ** 2 * nstates))

            L.sort_indices()

            solution = mkl_spsolve(
                L,
                b_mat,
                perm=None,
                verbose=True,
                max_iter_refine=max_iter_refine,
                scaling_vectors=True,
                weighted_matching=weighted_matching,
            )

        else:

            L = L.tocsc() + sp.csc_matrix((
                np.ones(n),
                (np.zeros(n), [num * (n + 1) for num in range(n)])
            ), shape=(n ** 2 * nstates, n ** 2 * nstates))

            # Use superLU solver

            LU = splu(L)
            solution = LU.solve(b_mat)

        dims = self.H_sys.dims
        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:sup_dim]), n, n)
        data = 0.5 * (data + data.H)

        solution = solution.reshape((nstates, sup_dim))

        return Qobj(data, dims=dims), solution

    def run(self, rho0, tlist, full_init=False, return_full=False):
        """
        Function to solve the time dependent evolution of the ODE given
        an initial condition and set of time steps.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system
            (if full_init==False).
            If full_init = True, then rho0 should be a numpy array of
            initial state and all ADOs.

        tlist : list
            Time over which system evolves.

        full_init: Boolean
            Indicates if initial condition is just the system Qobj, or a
            numpy array including all ADOs.

        return_full: Boolean

            Whether to also return as output the full state of all ADOs.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
            If return_full == True, also returns ADOs as an additional
            numpy array.
        """

        sup_dim = self._sup_dim

        solver = self._ode
        dims = self.coup_op[0].dims
        shape = self.coup_op[0].shape

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        if not full_init:
            output.states.append(Qobj(rho0))
            rho0_flat = rho0.full().ravel('F')
            rho0_he = np.zeros([sup_dim*self.nhe], dtype=complex)
            rho0_he[:sup_dim] = rho0_flat
            solver.set_initial_value(rho0_he, tlist[0])
        else:
            output.states.append(Qobj(
                rho0[:sup_dim].reshape(shape, order='F'), dims=dims,
            ))
            rho0_he = rho0
            solver.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)

        if not return_full:
            self.progress_bar.start(n_tsteps)
            for t_idx, t in enumerate(tlist):
                self.progress_bar.update(t_idx)
                if t_idx < n_tsteps - 1:
                    solver.integrate(solver.t + dt[t_idx])
                    rho = Qobj(
                        solver.y[:sup_dim].reshape(shape, order="F"),
                        dims=dims
                    )
                    output.states.append(rho)

            self.progress_bar.finished()
            return output

        else:
            self.progress_bar.start(n_tsteps)
            N_he = self.nhe
            N = shape[0]
            hshape = (N_he, N**2)
            full_hierarchy = [rho0.reshape(hshape)]
            for t_idx, t in enumerate(tlist):
                if t_idx < n_tsteps - 1:
                    solver.integrate(solver.t + dt[t_idx])

                    rho = Qobj(
                        solver.y[:sup_dim].reshape(shape, order='F'),
                        dims=dims
                    )
                    full_hierarchy.append(solver.y.reshape(hshape))
                    output.states.append(rho)
            self.progress_bar.finished()
            return output, full_hierarchy


class HSolverDL(BosonicHEOMSolver):
    """
    HEOM solver based on the Drude-Lorentz model for spectral density.
    Drude-Lorentz bath the correlation functions can be exactly analytically
    expressed as a sum of exponentials.

    This sub-class is included to give backwards compatability with the older
    implentation in qutip.nonmarkov.heom.

    Attributes
    ----------
    coup_strength : float
        Coupling strength.
    temperature : float
        Bath temperature.
    N_cut : int
        Cutoff parameter for the bath
    N_exp : int
        Number of exponential terms used to approximate the bath correlation
        functions
    cut_freq : float
        Bath spectral density cutoff frequency.
    bnd_cut_approx : bool
        Use boundary cut off approximation
    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp, cut_freq, bnd_cut_approx=False, options=None,
    ):
        ckAR, ckAI, vkAR, vkAI = self._calc_matsubara_params(
            lam=coup_strength,
            gamma=cut_freq,
            Nk=N_exp,
            T=temperature
        )

        if bnd_cut_approx:
            L_bnd = self._calc_bound_cut_liouvillian(
                Q=coup_op,
                lam=coup_strength,
                gamma=cut_freq,
                Nk=N_exp,
                T=temperature
            )
            H_sys = _convert_h_sys(H_sys)
            H_sys = liouvillian(H_sys) + L_bnd

        super().__init__(
            H_sys,
            coup_op=coup_op,
            ckAR=ckAR, ckAI=ckAI, vkAR=vkAR, vkAI=vkAI,
            N_cut=N_cut,
        )

        # store input parameters as attributes for politeness
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp
        self.bnd_cut_approx = bnd_cut_approx

    def _calc_bound_cut_liouvillian(self, Q, lam, gamma, Nk, T):
        """ Calculate the hierarchy terminator term for the Liouvillian. """
        beta = 1 / T

        op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)
        approx_factr = ((2 * lam / (beta * gamma)) - 1j * lam)
        approx_factr -= (
            lam * gamma * (-1.0j + 1 / np.tan(gamma / (2 * T))) / gamma
        )

        for k in range(1, Nk + 1):
            vk = 2 * np.pi * k * T
            approx_factr -= (
                (4 * lam * gamma * T * vk / (vk**2 - gamma**2)) / vk
            )

        L_bnd = -approx_factr * op
        return L_bnd

    def _calc_matsubara_params(self, lam, gamma, Nk, T):
        """ Calculate the Matsubara coefficents and frequencies. """
        ckAR = [lam * gamma * (1/np.tan(gamma / (2 * T)))]
        ckAR.extend([
            (4 * lam * gamma * T * 2 * np.pi * k * T /
                ((2 * np.pi * k * T)**2 - gamma**2))
            for k in range(1, Nk + 1)
        ])
        vkAR = [gamma]
        vkAR.extend([2 * np.pi * k * T for k in range(1, Nk + 1)])

        ckAI = [lam * gamma * (-1.0)]
        vkAI = [gamma]

        return ckAR, ckAI, vkAR, vkAI


class FermionicHEOMSolver(object):
    """
    Same as BosonicHEOMSolver, but with Fermionic baths.

    Attributes
    ----------
    H_sys : Qobj or list
        System Hamiltonian
        Or
        Liouvillian
        Or
        QobjEvo
        Or
        list of Hamiltonians with time dependence

        Format for input (if list):
        [time_independent_part, [H1, time_dep_function1],
        [H2, time_dep_function2]]

    coup_op : Qobj or list
        Operator describing the coupling between system and bath.
        Could also be a list of operators, which needs to be the
        same length as ck's and vk's.

    ck, vk : lists
        Lists containing spectral density correlation

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(self, H_sys, coup_op, ck, vk, N_cut, options=None):
        self.H_sys = _convert_h_sys(H_sys)
        self.options = Options() if options is None else options
        self.isTimeDep = isinstance(self.H_sys, QobjEvo)
        self.isHamiltonian = _h_sys_is_hamiltonian(self.H_sys)
        self._sup_dim = _h_sup_dim(self.H_sys, self.isHamiltonian)
        self.N_cut = N_cut
        self.ck, self.vk = _convert_bath_exponents_fermionic(ck, vk)
        self.coup_op = _convert_coup_op(coup_op, len(ck))
        self.progress_bar = BaseProgressBar()

        self._configure_solver()

    def fermion_grad_n(self, he_n):
        """
        Get the gradient term for the hierarchy ADM at
        level n
        """
        gradient_sum = 0
        L = self.L.copy()

        for i in range(len(self.flat_vk)):
            gradient_sum += he_n[i] * self.flat_vk[i]

        gradient_sum = -1 * gradient_sum
        sum_op = gradient_sum * sp.eye(
            self.L.shape[0], dtype=complex, format="csr")
        L += sum_op

        # Fill into larger L
        nidx = self.he2idx[he_n]
        L_he_temp = cy_pad_csr(L, self.nhe, self.nhe, nidx, nidx)
        self.L_helems += L_he_temp

    def fermion_grad_prev(self, he_n, k, prev_he, idx):
        """
        Get next gradient
        """
        ck = self.flat_ck

        # sign1 is based on number of excitations
        # the finicky notation is explicit and correct

        norm_prev = 1
        sign1 = 0
        n_excite = 2
        for i in range(len(he_n)):
            if he_n[i] == 1:
                n_excite += 1
        sign1 = (-1) ** (n_excite - 1)
        upto = self.offsets[k] + idx

        # sign2 is another prefix which looks ugly
        # but is written out explicitly to
        # ensure correctness

        sign2 = 1
        for i in range(upto):
            if prev_he[i]:
                sign2 *= -1
        pref = sign2 * -1j * norm_prev

        op1 = 0
        if k % 2 == 1:
            op1 = pref * (
                (ck[self.offsets[k] + idx] * self.spreQ[k])
                - (sign1 * np.conj(ck[self.offsets[k - 1] + idx]
                   * self.spostQ[k]))
            )
        else:
            op1 = pref * (
                (ck[self.offsets[k] + idx] * self.spreQ[k])
                - (sign1 * np.conj(ck[self.offsets[k + 1] + idx]
                   * self.spostQ[k]))
            )
        # Fill in larger L
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[prev_he]
        L_he_temp = cy_pad_csr(op1, self.nhe, self.nhe, rowidx, colidx)
        self.L_helems += L_he_temp

    def fermion_grad_next(self, he_n, k, next_he, idx):
        """
        Get next gradient
        """
        # sign1 is based on number of excitations
        # the finicky notation is explicit and correct
        norm_next = 1
        sign1 = 0
        n_excite = 2
        for i in range(len(he_n)):
            if he_n[i] == 1:
                n_excite += 1

        sign1 = (-1) ** (n_excite - 1)
        upto = self.offsets[k] + idx

        # sign2 is another prefix which looks ugly
        # but is written out explicitly to
        # ensure correctness

        sign2 = 1
        for i in range(upto):
            if next_he[i]:
                sign2 *= -1
        pref = sign2 * -1j * norm_next

        op2 = pref * ((self.spreQdag[k]) + (sign1 * self.spostQdag[k]))
        rowidx = self.he2idx[he_n]
        colidx = self.he2idx[next_he]
        L_he_temp = cy_pad_csr(op2, self.nhe, self.nhe, rowidx, colidx)
        self.L_helems += L_he_temp

    def fermion_rhs(self):
        """
        Make the RHS for fermionic case
        """
        for n in self.idx2he:
            he_n = self.idx2he[n]
            self.fermion_grad_n(he_n)
            for k in range(self.kcut):
                start = self.offsets[k]
                end = self.offsets[k + 1]
                num_elems = end - start
                for m in range(num_elems):
                    next_he = nexthe(he_n, self.offsets[k] + m, self.N_cut)
                    prev_he = prevhe(he_n, self.offsets[k] + m, self.N_cut)
                    if next_he and (next_he in self.he2idx):
                        self.fermion_grad_next(he_n, k, next_he, m)
                    if prev_he and (prev_he in self.he2idx):
                        self.fermion_grad_prev(he_n, k, prev_he, m)

    def _fermion_solver(self):
        """
        Utility function for fermionic solver.
        """
        self.kcut = len(self.offsets) - 1

        nhe, he2idx, idx2he = _heom_state_dictionaries(
            [2] * len(self.flat_ck), self.N_cut
        )
        self.nhe = nhe
        self.he2idx = he2idx
        self.idx2he = idx2he

        # Separate cases for Hamiltonian and Liouvillian
        if self.isHamiltonian:
            if self.isTimeDep:
                H_sys_list = self.H_sys.to_list()
                self.N = H_sys_list[0].shape[0]
                self.L = liouvillian(H_sys_list[0], []).data

            else:
                self.N = self.H_sys.shape[0]
                self.L = liouvillian(self.H_sys, []).data

        else:

            if self.isTimeDep:
                H_sys_list = self.H_sys.to_list()
                self.N = int(np.sqrt(H_sys_list[0].shape[0]))
                self.L = H_sys_list[0].data

            else:
                self.N = int(np.sqrt(self.H_sys.shape[0]))
                self.L = self.H_sys.data

        self.L_helems = sp.csr_matrix(
            (self.nhe * self.N ** 2, self.nhe * self.N ** 2),
            dtype=np.complex128,
        )
        # Set coupling operators
        spreQ = []
        spostQ = []
        spreQdag = []
        spostQdag = []
        for coupOp in self.coup_op:
            spreQ.append(spre(coupOp).data)
            spostQ.append(spost(coupOp).data)
            spreQdag.append(spre(coupOp.dag()).data)
            spostQdag.append(spost(coupOp.dag()).data)

        self.spreQ = spreQ
        self.spostQ = spostQ
        self.spreQdag = spreQdag
        self.spostQdag = spostQdag
        # make right hand side
        self.fermion_rhs()

        # return output
        return self.L_helems, self.nhe

    def _configure_solver(self):
        """ Configure the solver """
        self.len_list = [len(elem) for elem in self.ck]
        self.flat_ck = [elem for row in self.ck for elem in row]
        self.flat_vk = [elem for row in self.vk for elem in row]
        self.offsets = [0]
        curr_sum = 0
        for i in range(len(self.len_list)):
            self.offsets.append(curr_sum + self.len_list[i])
            curr_sum += self.len_list[i]

        RHSmat, nstates = self._fermion_solver()
        RHSmat = RHSmat.tocsr()

        if self.isTimeDep:
            solver_params = []
            h_identity_mat = sp.identity(nstates, format="csr")
            H_list = self.H_sys.to_list()

            # Store each time dependent component
            for idx in range(1, len(H_list)):
                temp_mat = sp.kron(
                    h_identity_mat,
                    liouvillian(H_list[idx][0])
                )

                solver_params.append([temp_mat, H_list[idx][1]])

            solver = scipy.integrate.ode(_dsuper_list_td)
            solver.set_f_params(solver_params)
        else:
            solver = scipy.integrate.ode(cy_ode_rhs)
            solver.set_f_params(RHSmat.data, RHSmat.indices, RHSmat.indptr)

        solver.set_integrator(
            "zvode",
            method=self.options.method,
            order=self.options.order,
            atol=self.options.atol,
            rtol=self.options.rtol,
            nsteps=self.options.nsteps,
            first_step=self.options.first_step,
            min_step=self.options.min_step,
            max_step=self.options.max_step,
        )

        self._ode = solver
        self.RHSmat = RHSmat

    def steady_state(
        self, max_iter_refine=100, use_mkl=False, weighted_matching=False
    ):
        """
        Computes steady state dynamics

        max_iter_refine : Int
            Parameter for the mkl LU solver. If pardiso errors are returned
            this should be increased.
        use_mkl : Boolean
            Optional override default use of mkl if mkl is installed.
        weighted_matching : Boolean
            Setting this true may increase run time, but reduce stability
            (pardisio may not converge).

        Returns
        -------
        steady state :  Qobj
            The steady state density matrix of the system
        solution    :   Numpy array
            Array of the the steady-state and all ADOs.
            Further processing of this can be done with functions provided in
            example notebooks.
        """
        nstates = self.nhe
        sup_dim = self._sup_dim
        n = int(np.sqrt(sup_dim))
        L = deepcopy(self.RHSmat)

        b_mat = np.zeros(sup_dim * nstates, dtype=complex)
        b_mat[0] = 1.0

        L = L.tolil()
        L[0, 0: n ** 2 * nstates] = 0.0
        L = L.tocsr()

        # TODO: Use the usual QuTiP MKL configuration here
        if settings.has_mkl & use_mkl:
            print("Using Intel mkl solver")
            from qutip._mkl.spsolve import mkl_spsolve

            L = L.tocsr() + sp.csr_matrix((
                np.ones(n),
                (np.zeros(n), [num * (n + 1) for num in range(n)])
            ), shape=(n ** 2 * nstates, n ** 2 * nstates))

            L.sort_indices()

            solution = mkl_spsolve(
                L,
                b_mat,
                perm=None,
                verbose=True,
                max_iter_refine=max_iter_refine,
                scaling_vectors=True,
                weighted_matching=weighted_matching,
            )

        else:

            L = L.tocsc() + sp.csc_matrix((
                np.ones(n),
                (np.zeros(n), [num * (n + 1) for num in range(n)])
            ), shape=(n ** 2 * nstates, n ** 2 * nstates))

            # Use superLU solver

            LU = splu(L)
            solution = LU.solve(b_mat)

        if self.isTimeDep:
            H_sys_list = self.H_sys.to_list()
            dims = H_sys_list[0].dims
        else:
            dims = self.H_sys.dims
        data = dense2D_to_fastcsr_fmode(vec2mat(solution[:sup_dim]), n, n)
        data = 0.5 * (data + data.H)

        solution = solution.reshape((nstates, sup_dim))

        return Qobj(data, dims=dims), solution

    def run(self, rho0, tlist, full_init=False, return_full=False):
        """
        Function to solve for an open quantum system using the
        HEOM model.

        Parameters
        ----------
        rho0 : Qobj
            Initial state (density matrix) of the system
            (if full_init==False).
            If full_init = True, then rho0 should be a numpy array of
            initial state and all ADOs.

        tlist : list
            Time over which system evolves.

        full_init: Boolean
            Indicates if initial condition is just the system Qobj, or a
            numpy array including all ADOs.

        return_full: Boolean

            Whether to also return as output the full state of all ADOs.

        Returns
        -------
        results : :class:`qutip.solver.Result`
            Object storing all results from the simulation.
            If return_full == True, also returns ADOs as an additional
            numpy array.
        """
        sup_dim = self._sup_dim

        solver = self._ode
        dims = self.coup_op[0].dims
        shape = self.coup_op[0].shape

        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []

        if not full_init:
            output.states.append(Qobj(rho0))
            rho0_flat = rho0.full().ravel('F')
            rho0_he = np.zeros([sup_dim*self.nhe], dtype=complex)
            rho0_he[:sup_dim] = rho0_flat
            solver.set_initial_value(rho0_he, tlist[0])
        else:
            output.states.append(Qobj(
                rho0[:sup_dim].reshape(shape, order='F'), dims=dims
            ))
            rho0_he = rho0
            solver.set_initial_value(rho0_he, tlist[0])

        dt = np.diff(tlist)
        n_tsteps = len(tlist)

        if not return_full:
            self.progress_bar.start(n_tsteps)
            for t_idx, t in enumerate(tlist):
                self.progress_bar.update(t_idx)
                if t_idx < n_tsteps - 1:
                    solver.integrate(solver.t + dt[t_idx])
                    rho = Qobj(
                        solver.y[:sup_dim].reshape(shape, order="F"),
                        dims=dims
                    )
                    output.states.append(rho)

            self.progress_bar.finished()
            return output

        else:
            self.progress_bar.start(n_tsteps)
            N_he = self.nhe
            N = shape[0]
            hshape = (N_he, N**2)
            full_hierarchy = [rho0.reshape(hshape)]
            for t_idx, t in enumerate(tlist):
                if t_idx < n_tsteps - 1:
                    solver.integrate(solver.t + dt[t_idx])
                    rho = Qobj(
                               solver.y[:sup_dim].reshape(shape, order='F'),
                               dims=dims
                    )
                    full_hierarchy.append(solver.y.reshape(hshape))
                    output.states.append(rho)
            self.progress_bar.finished()
            return output, full_hierarchy


def _dsuper_list_td(t, y, L_list):
    """
    Auxiliary function for the integration.
    Is called at every time step.
    """
    L = L_list[0][0]
    for n in range(1, len(L_list)):
        L = L + L_list[n][0] * L_list[n][1](t)
    return L * y
