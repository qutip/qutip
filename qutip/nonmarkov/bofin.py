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


def _check_Hsys(H_sys):
    # Check if Hamiltonians are one of the allowed types
    if not isinstance(H_sys, (Qobj, QobjEvo, list)):
        msg = "Hamiltonian format is incorrect."
        raise RuntimeError(msg)

    # Check if Hamiltonians supplied in the list are correct
    if isinstance(H_sys, list):
        for H in H_sys:
            # If not a list of time dependent Hamiltonians
            if not isinstance(H, list):
                # Just check if it is a Qobj
                _check_Hsys(H)

            # Check if time dependent Hamiltonian terms are correct
            # in the list fomat if it is a tuple of [H, callable]
            elif isinstance(H[0], Qobj):
                if not callable(H[1]):
                    msg = "Incorrect time dependent function for Hamiltonian."
                    raise RuntimeError(msg)

            else:
                _check_Hsys(H[0])

        return True

    else:
        return True


def _check_coup_ops(coup_op, length):

    if (type(coup_op) != Qobj) and (
        type(coup_op) == list and type(coup_op[0]) != Qobj
    ):
        raise RuntimeError(
            "Coupling operator must be a QObj or list " + " of QObjs."
        )

    if type(coup_op) == list:
        if len(coup_op) != (length):
            raise RuntimeError(
                "Expected " + str(length)
                + " coupling operators."
            )


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

    N_cut : int
        Cutoff parameter for the bath

    options : :class:`qutip.solver.Options`
        Generic solver options.
        If set to None the default options will be used
    """

    def __init__(
        self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None
    ):

        self.reset()
        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = BaseProgressBar()
        # set other attributes
        self.configure(H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options)

    def reset(self):
        """
        Reset any attributes to default values
        """
        self.H_sys = None
        self.coup_op = None
        self.ckAR = []
        self.ckAI = []
        self.vkAR = []
        self.vkAI = []
        self.N_cut = 5
        self.options = None
        self.ode = None

    def process_input(
        self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None
    ):
        """
        Type-checks provided input
        Merges certain bath properties if conditions are met.
        """

        # Checks for Hamiltonian

        _check_Hsys(H_sys)

        # Checks for coupling operator
        _check_coup_ops(coup_op, len(ckAR) + len(ckAI))

        # Checks for ckAR, ckAI, vkAR, vkAI

        if (
            type(ckAR) != list
            or type(vkAR) != list
            or type(ckAR) != list
            or type(ckAI) != list
        ):
            raise RuntimeError("Expected list for coefficients.")

        if (
            type(ckAR[0]) == list
            or type(vkAR[0]) == list
            or type(ckAR[0]) == list
            or type(ckAI[0]) == list
        ):
            raise RuntimeError(
                "Lists of coefficients should be one dimensional."
            )

        if len(ckAR) != len(vkAR) or len(ckAI) != len(vkAI):
            raise RuntimeError(
                "Spectral density correlation coefficients not "
                + "specified correctly."
            )

        # Check for close vk's.
        # Just gives warning, and continues.
        for i in range(len(vkAR)):
            for j in range(i + 1, len(vkAR)):
                if np.isclose(vkAR[i], vkAR[j], rtol=1e-5, atol=1e-7):
                    warnings.warn(
                        "Expected simplified input. "
                        "Consider collating equal frequency parameters."
                    )

        for i in range(len(vkAI)):
            for j in range(i + 1, len(vkAI)):
                if np.isclose(vkAI[i], vkAI[j], rtol=1e-5, atol=1e-7):
                    warnings.warn(
                        "Expected simplified input.  "
                        "Consider collating equal frequency parameters."
                    )

        if type(H_sys) == list:
            self.H_sys = QobjEvo(H_sys)
        else:
            self.H_sys = H_sys

        nr = len(ckAR)
        ni = len(ckAI)
        ckAR = list(ckAR)
        ckAI = list(ckAI)
        vkAR = list(vkAR)
        vkAI = list(vkAI)
        coup_op = deepcopy(coup_op)

        # Check to make list of coupling operators

        if type(coup_op) != list:
            coup_op = [coup_op for i in range(nr + ni)]

        # Check for handling the case where real and imaginary exponents
        # are close.
        # This happens in the normal overdamped drude-lorentz case.
        # We give a warning to tell the user this collation is being done
        # automatically.
        common_ck = []
        real_indices = []
        common_vk = []
        img_indices = []
        common_coup_op = []
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

        # Check to similarly truncate coupling operators

        img_coup_ops = [x + nr for x in img_indices]
        coup_op_indices = real_indices + sorted(img_coup_ops)
        for i in sorted(coup_op_indices, reverse=True):
            coup_op.pop(i)

        coup_op += common_coup_op

        # Assigns to attributes

        self.coup_op = coup_op
        self.ckAR = ckAR
        self.ckAI = ckAI
        self.vkAR = vkAR
        self.vkAI = vkAI
        self.common_ck = common_ck
        self.common_vk = common_vk
        self.N_cut = int(N_cut)
        self.ck = np.array(ckAR + ckAI + common_ck).astype(complex)
        self.vk = np.array(vkAR + vkAI + common_vk).astype(complex)
        self.NR = len(ckAR)
        self.NI = len(ckAI)

        # Checks and sets flags for Hamiltonian type

        self.isHamiltonian = True
        self.isTimeDep = False

        if type(self.H_sys) is QobjEvo:
            self.H_sys_list = self.H_sys.to_list()

            if self.H_sys_list[0].type == "oper":
                self.isHamiltonian = True
            else:
                self.isHamiltonian = False

            self.isTimeDep = True

        else:
            if self.H_sys.type == "oper":
                self.isHamiltonian = True
            else:
                self.isHamiltonian = False

        if isinstance(options, Options):
            self.options = options

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
        block = self.N ** 2
        pos = int(nidx * block)
        # indlist = np.array(list(range(pos, pos+block)))
        # self.L_helems[indlist[:, None], indlist] += L
        pos = int(nidx * (block))
        L_he_temp = _pad_csr(L, self.nhe, self.nhe, nidx, nidx)
        # self.L_helems[pos : pos + block, pos : pos + block] += L
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
        block = self.N ** 2

        rowpos = int(rowidx * (block))
        colpos = int(colidx * (block))
        L_he_temp = _pad_csr(op1, self.nhe, self.nhe, rowidx, colidx)
        # self.L_helems[rowpos : rowpos + block, colpos : colpos + block] \
        #     += op1
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
        block = self.N ** 2
        rowpos = int(rowidx * (block))
        colpos = int(colidx * (block))

        rowpos = int(rowidx * (block))
        colpos = int(colidx * (block))
        L_he_temp = _pad_csr(op2, self.nhe, self.nhe, rowidx, colidx)
        # self.L_helems[rowpos : rowpos + block, colpos : colpos + block] \
        #     += op2
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

        # Separate cases for Hamiltonian and Liouvillian
        if self.isHamiltonian:

            if self.isTimeDep:
                self.N = self.H_sys_list[0].shape[0]
                self.L = liouvillian(self.H_sys_list[0], []).data

            else:
                self.N = self.H_sys.shape[0]
                self.L = liouvillian(self.H_sys, []).data

        else:
            if self.isTimeDep:
                self.N = int(np.sqrt(self.H_sys_list[0].shape[0]))
                self.L = self.H_sys_list[0].data

            else:
                self.N = int(np.sqrt(self.H_sys.shape[0]))
                self.L = self.H_sys.data

        self.L_helems = sp.csr_matrix(
            (self.nhe * self.N ** 2, self.nhe * self.N ** 2),
            dtype=np.complex,
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

    def configure(
        self, H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None
    ):
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used
        """

        # Type checks the input and truncates exponents if necessary

        self.process_input(
            H_sys, coup_op, ckAR, ckAI, vkAR, vkAI, N_cut, options=None)

        # Sets variables locally for configuring solver

        options = self.options
        H = self.H_sys
        Q = self.coup_op
        Nc = self.N_cut
        ckAR = self.ckAR
        ckAI = self.ckAI
        vkAR = self.vkAR
        vkAI = self.vkAI
        common_ck = self.common_ck
        common_vk = self.common_vk
        NR = self.NR
        NI = self.NI
        ck = self.ck
        vk = self.vk

        # Passing data to bosonic solver

        RHSmat, nstates = self._boson_solver()
        RHSmat = RHSmat.tocsr()

        # Setting up solver

        solver = None

        if self.isTimeDep:

            solver_params = []
            constant_func = lambda x: 1.0
            h_identity_mat = sp.identity(nstates, format="csr")
            solver_params.append([RHSmat, constant_func])
            H_list = self.H_sys_list

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

        # Sets options for solver

        solver.set_integrator(
            "zvode",
            method=options.method,
            order=options.order,
            atol=options.atol,
            rtol=options.rtol,
            nsteps=options.nsteps,
            first_step=options.first_step,
            min_step=options.min_step,
            max_step=options.max_step,
        )

        # Sets attributes related to solver

        self._ode = solver
        self.RHSmat = RHSmat
        self._configured = True

        if self.isHamiltonian:
            if self.isTimeDep:
                self._sup_dim = (
                    self.H_sys_list[0].shape[0] * self.H_sys_list[0].shape[0]
                )
            else:
                self._sup_dim = H.shape[0] * H.shape[0]
        else:
            if self.isTimeDep:
                self._sup_dim = (
                    self.H_sys_list[0].shape[0]
                )
            else:
                self._sup_dim = H.shape[0]

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
        unit_h_elems = sp.identity(nstates, format="csr")
        L = deepcopy(self.RHSmat)  # + sp.kron(unit_h_elems,
        # liouvillian(H).data)

        b_mat = np.zeros(sup_dim * nstates, dtype=complex)
        b_mat[0] = 1.0

        L = L.tolil()
        L[0, 0: n ** 2 * nstates] = 0.0
        L = L.tocsr()

        if settings.has_mkl & use_mkl:
            print("Using Intel mkl solver")
            from qutip._mkl.spsolve import mkl_splu, mkl_spsolve

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

        solution = solution.reshape((nstates, self.H_sys.shape[0] ** 2))

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

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

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
    implentation in qutip.


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
        self.reset()

        if options is None:
            self.options = Options()
        else:
            self.options = options

        self.progress_bar = BaseProgressBar()

        # the other attributes will be set in the configure method
        self.configure(
            H_sys, coup_op, coup_strength, temperature,
            N_cut, N_exp, cut_freq,
            bnd_cut_approx=bnd_cut_approx,
        )

    def reset(self):
        """
        Reset any attributes to default values
        """
        BosonicHEOMSolver.reset(self)

        self.coup_strength = 0.0
        self.cut_freq = 0.0
        self.temperature = 1.0
        self.N_exp = 2

    def configure(
        self, H_sys, coup_op, coup_strength, temperature,
        N_cut, N_exp,  cut_freq,
        bnd_cut_approx=None, options=None
    ):
        """
        Configure the correlation function parameters using the required
        decompostion, and then use the parent class BosonicHEOMSolver to check
        input and construct RHS.

        The parameters are described in the class attributes, unless there
        is some specific behaviour
        """
        self.coup_strength = coup_strength
        self.cut_freq = cut_freq
        self.temperature = temperature
        self.N_exp = N_exp

        if bnd_cut_approx is not None:
            self.bnd_cut_approx = bnd_cut_approx

        options = self.options
        progress_bar = self.progress_bar

        ckAR, ckAI, vkAR, vkAI = self._calc_matsubara_params()

        Q = coup_op

        if bnd_cut_approx:
            # do version with tanimura terminator
            lam = self.coup_strength
            gamma = self.cut_freq
            T = self.temperature
            beta = 1/T
            Nk = self.N_exp

            op = -2*spre(Q)*spost(Q.dag()) + spre(Q.dag()*Q) + spost(Q.dag()*Q)
            approx_factr = ((2 * lam / (beta * gamma)) - 1j*lam)
            approx_factr -= (
                lam * gamma * (-1.0j + 1 / np.tan(gamma / (2 * T))) / gamma
            )

            for k in range(1, Nk + 1):
                vk = 2 * np.pi * k * T
                approx_factr -= (
                    (4 * lam * gamma * T * vk / (vk**2 - gamma**2)) / vk
                )

            L_bnd = -approx_factr*op
            H_sys = liouvillian(H_sys) + L_bnd

        NR = len(ckAR)
        NI = len(ckAI)
        Q2 = [Q for kk in range(NR+NI)]

        BosonicHEOMSolver.configure(
            self, H_sys, Q2, ckAR, ckAI, vkAR, vkAI, N_cut, options
        )

    def _calc_matsubara_params(self):
        """
        Calculate the Matsubara coefficents and frequencies
        Returns
        -------
        ckAR, ckAI, vkAR, vkAI: list(complex)
        """
        lam = self.coup_strength
        gamma = self.cut_freq
        Nk = self.N_exp
        T = self.temperature

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
        self.reset()
        if options is None:
            self.options = Options()
        else:
            self.options = options
        # set other attributes
        self.configure(H_sys, coup_op, ck, vk, N_cut, options)
        self.progress_bar = BaseProgressBar()

    def reset(self):
        """
        Reset any attributes to default values
        """
        self.H_sys = None
        self.coup_op = None
        self.ck = []
        self.vk = []
        self.N_cut = 5
        self.options = None
        self.ode = None

    def process_input(self, H_sys, coup_op, ck, vk, N_cut, options=None):
        """
        Type-checks provided input

        """
        # Checks for Hamiltonian
        _check_Hsys(H_sys)

        # Checks for coupling operator
        _check_coup_ops(coup_op, len(ck))

        # Checks for cks and vks
        if (
            type(ck) != list
            or type(vk) != list
            or type(ck[0]) != list
            or type(vk[0]) != list
        ):
            raise RuntimeError("Expected list of lists.")

        if len(ck) != len(vk):
            raise RuntimeError("Exponents supplied incorrectly.")

        for idx in range(len(ck)):
            if len(ck[idx]) != len(vk[idx]):
                raise RuntimeError("Exponents supplied incorrectly.")

        # Make list of coupling operators
        if type(coup_op) != list:
            coup_op = [coup_op for elem in range(len(ck))]

        if type(H_sys) == list:
            self.H_sys = QobjEvo(H_sys)
        else:
            self.H_sys = H_sys

        self.coup_op = coup_op
        self.ck = ck
        self.vk = vk
        self.N_cut = int(N_cut)

        # Checks and sets flags for Hamiltonian type

        self.isHamiltonian = True
        self.isTimeDep = False

        if type(self.H_sys) is QobjEvo:
            self.H_sys_list = self.H_sys.to_list()

            if self.H_sys_list[0].type == "oper":
                self.isHamiltonian = True
            else:
                self.isHamiltonian = False

            self.isTimeDep = True

        else:
            if self.H_sys.type == "oper":
                self.isHamiltonian = True
            else:
                self.isHamiltonian = False

        if isinstance(options, Options):
            self.options = options

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
        block = self.N ** 2
        pos = int(nidx * block)

        L_he_temp = _pad_csr(L, self.nhe, self.nhe, nidx, nidx)
        # self.L_helems[pos : pos + block, pos : pos + block] += L
        self.L_helems += L_he_temp

    def fermion_grad_prev(self, he_n, k, prev_he, idx):
        """
        Get next gradient
        """
        nk = he_n[k]
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
        block = self.N ** 2
        rowpos = int(rowidx * block)
        colpos = int(colidx * block)
        L_he_temp = _pad_csr(op1, self.nhe, self.nhe, rowidx, colidx)

        self.L_helems += L_he_temp

        # self.L_helems[rowpos : rowpos + block, colpos : colpos + block] \
        #     += op1

    def fermion_grad_next(self, he_n, k, next_he, idx):
        """
        Get next gradient
        """
        # nk = he_n[k]
        ck = self.flat_ck

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
        block = self.N ** 2
        rowpos = int(rowidx * block)
        colpos = int(colidx * block)
        L_he_temp = _pad_csr(op2, self.nhe, self.nhe, rowidx, colidx)

        self.L_helems += L_he_temp
        # self.L_helems[rowpos : rowpos + block, colpos : colpos + block] \
        #     += op2

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
                self.N = self.H_sys_list.shape[0]
                self.L = liouvillian(self.H_sys_list[0], []).data

            else:
                self.N = self.H_sys.shape[0]
                self.L = liouvillian(self.H_sys, []).data

        else:

            if self.isTimeDep:
                self.N = int(np.sqrt(self.H_sys_list[0].shape[0]))
                self.L = self.H_sys_list[0].data

            else:
                self.N = int(np.sqrt(self.H_sys.shape[0]))
                self.L = self.H_sys.data

        self.L_helems = sp.csr_matrix(
            (self.nhe * self.N ** 2, self.nhe * self.N ** 2), dtype=np.complex
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

    def configure(self, H_sys, coup_op, ck, vk, N_cut, options=None):
        """
        Configure the solver using the passed parameters
        The parameters are described in the class attributes, unless there
        is some specific behaviour

        Parameters
        ----------
        options : :class:`qutip.solver.Options`
            Generic solver options.
            If set to None the default options will be used
        """
        # Type check input
        self.process_input(H_sys, coup_op, ck, vk, N_cut, options)

        # Setting variables locally

        options = self.options
        H = self.H_sys
        Q = self.coup_op
        ck = self.ck
        vk = self.vk
        Nc = self.N_cut

        self.len_list = [len(elem) for elem in ck]
        self.flat_ck = [elem for row in self.ck for elem in row]
        self.flat_vk = [elem for row in self.vk for elem in row]
        self.offsets = [0]
        curr_sum = 0
        for i in range(len(self.len_list)):
            self.offsets.append(curr_sum + self.len_list[i])
            curr_sum += self.len_list[i]

        # Passing Hamiltonian
        # Passing data to fermionic solver

        RHSmat, nstates = self._fermion_solver()
        RHSmat = RHSmat.tocsr()

        # Setting up solver

        solver = None

        if self.isTimeDep:

            solver_params = []
            h_identity_mat = sp.identity(nstates, format="csr")
            H_list = self.H_sys_list

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

        # Sets options for solver

        solver.set_integrator(
            "zvode",
            method=options.method,
            order=options.order,
            atol=options.atol,
            rtol=options.rtol,
            nsteps=options.nsteps,
            first_step=options.first_step,
            min_step=options.min_step,
            max_step=options.max_step,
        )

        # Sets attributes related to solver

        self._ode = solver
        self.RHSmat = RHSmat
        self._configured = True
        if self.isHamiltonian:
            if self.isTimeDep:
                self._sup_dim = (
                    self.H_sys_list[0].shape[0] * self.H_sys_list[0].shape[0]
                )
            else:
                self._sup_dim = H.shape[0] * H.shape[0]
        else:
            if self.isTimeDep:
                self._sup_dim = (
                    self.H_sys_list[0].shape[0]
                )
            else:
                self._sup_dim = H.shape[0]

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
        unit_h_elems = sp.identity(nstates, format="csr")
        L = deepcopy(self.RHSmat)  # + sp.kron(unit_h_elems,
        # liouvillian(H).data)

        b_mat = np.zeros(sup_dim * nstates, dtype=complex)
        b_mat[0] = 1.0

        L = L.tolil()
        L[0, 0: n ** 2 * nstates] = 0.0
        L = L.tocsr()

        if settings.has_mkl & use_mkl:
            print("Using Intel mkl solver")
            from qutip._mkl.spsolve import mkl_splu, mkl_spsolve

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

        solution = solution.reshape((nstates, self.H_sys.shape[0] ** 2))

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

        if not self._configured:
            raise RuntimeError("Solver must be configured before it is run")

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


def _pad_csr(A, row_scale, col_scale, insertrow=0, insertcol=0):
    """
    Expand the input csr_matrix to a greater space as given by the scale.
    Effectively inserting A into a larger matrix
         zeros([A.shape[0]*row_scale, A.shape[1]*col_scale]
    at the position [A.shape[0]*insertrow, A.shape[1]*insertcol]
    The same could be achieved through using a kron with a matrix with
    one element set to 1. However, this is more efficient
    """

    # ajgpitch 2016-03-08:
    # Clearly this is a very simple operation in dense matrices
    # It seems strange that there is nothing equivalent in sparse however,
    # after much searching most threads suggest directly addressing
    # the underlying arrays, as done here.
    # This certainly proved more efficient than other methods such as stacking
    # TODO: Perhaps cythonize and move to spmatfuncs

    if not isinstance(A, sp.csr_matrix):
        raise TypeError("First parameter must be a csr matrix")
    nrowin = A.shape[0]
    ncolin = A.shape[1]
    nrowout = nrowin*row_scale
    ncolout = ncolin*col_scale

    A._shape = (nrowout, ncolout)
    if insertcol == 0:
        pass
    elif insertcol > 0 and insertcol < col_scale:
        A.indices = A.indices + insertcol*ncolin
    else:
        raise ValueError("insertcol must be >= 0 and < col_scale")

    if insertrow == 0:
        A.indptr = np.concatenate((
            A.indptr, np.array([A.indptr[-1]]*(row_scale-1)*nrowin)
        ))
    elif insertrow == row_scale-1:
        A.indptr = np.concatenate((
            np.array([0] * (row_scale - 1) * nrowin), A.indptr
        ))
    elif insertrow > 0 and insertrow < row_scale - 1:
        A.indptr = np.concatenate((
            np.array([0] * insertrow * nrowin),
            A.indptr,
            np.array([A.indptr[-1]] * (row_scale - insertrow - 1) * nrowin)
        ))
    else:
        raise ValueError("insertrow must be >= 0 and < row_scale")

    return A
