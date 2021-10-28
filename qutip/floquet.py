__all__ = ['floquet_modes', 'floquet_modes_t', 'floquet_modes_table',
           'floquet_modes_t_lookup', 'floquet_states', 'floquet_states_t',
           'floquet_wavefunction', 'floquet_wavefunction_t',
           'floquet_state_decomposition', 'fsesolve',
           'floquet_master_equation_rates', 'floquet_collapse_operators',
           'floquet_master_equation_tensor',
           'floquet_master_equation_steadystate', 'floquet_basis_transform',
           'floquet_markov_mesolve', 'fmmesolve']

import numpy as np
import scipy.linalg as la
import scipy
import warnings
from copy import copy
from numpy import angle, pi, exp, sqrt
from types import FunctionType
from qutip.qobj import Qobj, isket
from qutip.superoperator import vec2mat_index, mat2vec, vec2mat
#from qutip.mesolve import mesolve
from qutip.sesolve import sesolve
from qutip.rhs_generate import rhs_clear
from qutip.steadystate import steadystate
from qutip.states import ket2dm
from qutip.states import projection
from qutip.solver import Options
from qutip.propagator import propagator
from qutip.solver import Result, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.expect import expect
from qutip.utilities import n_thermal


def floquet_modes(H, T, args=None, sort=False, U=None, options=None):
    """
    Calculate the initial Floquet modes Phi_alpha(0) for a driven system with
    period T.

    Returns a list of :class:`qutip.qobj` instances representing the Floquet
    modes and a list of corresponding quasienergies, sorted by increasing
    quasienergy in the interval [-pi/T, pi/T]. The optional parameter `sort`
    decides if the output is to be sorted in increasing quasienergies or not.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    args : dictionary
        dictionary with variables required to evaluate H

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    U : :class:`qutip.qobj`
        The propagator for the time-dependent Hamiltonian with period `T`.
        If U is `None` (default), it will be calculated from the Hamiltonian
        `H` using :func:`qutip.propagator.propagator`.

    options : :class:`qutip.solver`
        options for the ODE solver. For the propagator U.

    Returns
    -------

    output : list of kets, list of quasi energies

        Two lists: the Floquet modes as kets and the quasi energies.

    """

    if U is None:
        # get the unitary propagator
        U = propagator(H, T, [], args=args, options=copy(options))

    # find the eigenstates for the propagator
    evals, evecs = la.eig(U.full())

    eargs = angle(evals)

    # make sure that the phase is in the interval [-pi, pi], so that
    # the quasi energy is in the interval [-pi/T, pi/T] where T is the
    # period of the driving.  eargs += (eargs <= -2*pi) * (2*pi) +
    # (eargs > 0) * (-2*pi)
    eargs += (eargs <= -pi) * (2 * pi) + (eargs > pi) * (-2 * pi)
    e_quasi = -eargs / T

    # sort by the quasi energy
    if sort:
        order = np.argsort(-e_quasi)
    else:
        order = list(range(len(evals)))

    # prepare a list of kets for the floquet states
    new_dims = [U.dims[0], [1] * len(U.dims[0])]
    new_shape = [U.shape[0], 1]
    kets_order = [Qobj(np.array(evecs[:, o]).T,
                       dims=new_dims, shape=new_shape) for o in order]

    return kets_order, e_quasi[order]


def floquet_modes_t(f_modes_0, f_energies, t, H, T, args=None,
                    options=None):
    """
    Calculate the Floquet modes at times tlist Phi_alpha(tlist) propagting the
    initial Floquet modes Phi_alpha(0)

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at :math:`t`

    f_energies : list
        Floquet energies.

    t : float
        The time at which to evaluate the floquet modes.

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    args : dictionary
        dictionary with variables required to evaluate H

    T : float
        The period of the time-dependence of the hamiltonian.

    options : :class:`qutip.solver`
        options for the ODE solver. For the propagator.

    Returns
    -------

    output : list of kets

        The Floquet modes as kets at time :math:`t`

    """
    # find t in [0,T] such that t_orig = t + n * T for integer n
    t = t - int(t / T) * T
    f_modes_t = []

    # get the unitary propagator from 0 to t
    if t > 0.0:
        U = propagator(H, t, [], args, options=copy(options))

        for n in np.arange(len(f_modes_0)):
            f_modes_t.append(U * f_modes_0[n] * exp(1j * f_energies[n] * t))
    else:
        f_modes_t = f_modes_0

    return f_modes_t


def floquet_modes_table(f_modes_0, f_energies, tlist, H, T, args=None,
                        options=None):
    """
    Pre-calculate the Floquet modes for a range of times spanning the floquet
    period. Can later be used as a table to look up the floquet modes for
    any time.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        Floquet modes at :math:`t`

    f_energies : list
        Floquet energies.

    tlist : array
        The list of times at which to evaluate the floquet modes.

    H : :class:`qutip.qobj`
        system Hamiltonian, time-dependent with period `T`

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        dictionary with variables required to evaluate H

    options : :class:`qutip.solver`
        options for the ODE solver.

    Returns
    -------

    output : nested list

        A nested list of Floquet modes as kets for each time in `tlist`

    """
    options = copy(options) or Options()
    # truncate tlist to the driving period
    tlist_period = tlist[np.where(tlist <= T)]

    f_modes_table_t = [[] for t in tlist_period]

    opt = options
    opt.rhs_reuse = True
    rhs_clear()

    for n, f_mode in enumerate(f_modes_0):
        output = sesolve(H, f_mode, tlist_period, [], args, opt)
        for t_idx, f_state_t in enumerate(output.states):
            f_modes_table_t[t_idx].append(
                f_state_t * exp(1j * f_energies[n] * tlist_period[t_idx]))

    return f_modes_table_t


def floquet_modes_t_lookup(f_modes_table_t, t, T):
    """
    Lookup the floquet mode at time t in the pre-calculated table of floquet
    modes in the first period of the time-dependence.

    Parameters
    ----------

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table`.

    t : float
        The time for which to evaluate the Floquet modes.

    T : float
        The period of the time-dependence of the hamiltonian.

    Returns
    -------

    output : nested list

        A list of Floquet modes as kets for the time that most closely matching
        the time `t` in the supplied table of Floquet modes.
    """

    # find t_wrap in [0,T] such that t = t_wrap + n * T for integer n
    t_wrap = t - int(t / T) * T

    # find the index in the table that corresponds to t_wrap (= tlist[t_idx])
    t_idx = int(t_wrap / T * len(f_modes_table_t))

    # XXX: might want to give a warning if the cast of t_idx to int discard
    # a significant fraction in t_idx, which would happen if the list of time
    # values isn't perfect matching the driving period
    # if debug: print "t = %f -> t_wrap = %f @ %d of %d" % (t, t_wrap, t_idx,
    # N)

    return f_modes_table_t[t_idx]


def floquet_states(f_modes_t, f_energies, t):
    """
    Evaluate the floquet states at time t given the Floquet modes at that time.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of Floquet modes for time :math:`t`.

    f_energies : array
        The Floquet energies.

    t : float
        The time for which to evaluate the Floquet states.

    Returns
    -------

    output : list

        A list of Floquet states for the time :math:`t`.

    """

    return [(f_modes_t[i] * exp(-1j * f_energies[i] * t))
            for i in np.arange(len(f_energies))]


def floquet_states_t(f_modes_0, f_energies, t, H, T, args=None,
                     options=None):
    """
    Evaluate the floquet states at time t given the initial Floquet modes.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes (for time :math:`t=0`).

    f_energies : array
        The Floquet energies.

    t : float
        The time for which to evaluate the Floquet states.

    H : :class:`qutip.qobj`
        System Hamiltonian, time-dependent with period `T`.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    options : :class:`qutip.solver`
        options for the ODE solver.

    Returns
    -------

    output : list

        A list of Floquet states for the time :math:`t`.

    """

    f_modes_t = floquet_modes_t(f_modes_0, f_energies, t, H, T, args,
                                options=options)
    return [(f_modes_t[i] * exp(-1j * f_energies[i] * t))
            for i in np.arange(len(f_energies))]


def floquet_wavefunction(f_modes_t, f_energies, f_coeff, t):
    """
    Evaluate the wavefunction for a time t using the Floquet state
    decompositon, given the Floquet modes at time `t`.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes (for time :math:`t=0`).

    f_energies : array
        The Floquet energies.

    f_coeff : array
        The coefficients for Floquet decomposition of the initial wavefunction.

    t : float
        The time for which to evaluate the Floquet states.

    Returns
    -------

    output : :class:`qutip.qobj`

        The wavefunction for the time :math:`t`.

    """
    return sum([f_modes_t[i] * exp(-1j * f_energies[i] * t) * f_coeff[i]
                for i in np.arange(len(f_energies))])


def floquet_wavefunction_t(f_modes_0, f_energies, f_coeff, t, H, T, args=None,
                           options=None):
    """
    Evaluate the wavefunction for a time t using the Floquet state
    decompositon, given the initial Floquet modes.

    Parameters
    ----------

    f_modes_t : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes (for time :math:`t=0`).

    f_energies : array
        The Floquet energies.

    f_coeff : array
        The coefficients for Floquet decomposition of the initial wavefunction.

    t : float
        The time for which to evaluate the Floquet states.

    H : :class:`qutip.qobj`
        System Hamiltonian, time-dependent with period `T`.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    Returns
    -------

    output : :class:`qutip.qobj`

        The wavefunction for the time :math:`t`.

    """

    f_states_t = floquet_states_t(f_modes_0, f_energies, t, H, T, args,
                                  options=options)
    return sum([f_states_t[i] * f_coeff[i]
                for i in np.arange(len(f_energies))])


def floquet_state_decomposition(f_states, f_energies, psi):
    r"""
    Decompose the wavefunction `psi` (typically an initial state) in terms of
    the Floquet states, :math:`\psi = \sum_\alpha c_\alpha \psi_\alpha(0)`.

    Parameters
    ----------

    f_states : list of :class:`qutip.qobj` (kets)
        A list of Floquet modes.

    f_energies : array
        The Floquet energies.

    psi : :class:`qutip.qobj`
        The wavefunction to decompose in the Floquet state basis.

    Returns
    -------

    output : array

        The coefficients :math:`c_\alpha` in the Floquet state decomposition.

    """
    # [:1,:1][0, 0] patch around scipy 1.3.0 bug
    return [(f_states[i].dag() * psi).data[:1, :1][0, 0]
            for i in np.arange(len(f_energies))]


def fsesolve(H, psi0, tlist, e_ops=[], T=None, args={}, Tsteps=100,
             options_modes=None):
    """
    Solve the Schrodinger equation using the Floquet formalism.

    Parameters
    ----------

    H : :class:`qutip.qobj.Qobj`
        System Hamiltonian, time-dependent with period `T`.

    psi0 : :class:`qutip.qobj`
        Initial state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values. If this
        list is empty, the state vectors for each time in `tlist` will be
        returned instead of expectation values.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    Tsteps : integer
        The number of time steps in one driving period for which to
        precalculate the Floquet modes. `Tsteps` should be an even number.

    options_modes : :class:`qutip.solver`
        options for the ODE solver.

    Returns
    -------

    output : :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which
        contains either an *array* of expectation values or an array of
        state vectors, for the times specified by `tlist`.
    """

    if not T:
        # assume that tlist span exactly one period of the driving
        T = tlist[-1]

    if options_modes is None:
        options_modes_table = Options()
    else:
        options_modes_table = options_modes

    # find the floquet modes for the time-dependent hamiltonian
    f_modes_0, f_energies = floquet_modes(H, T, args,
                                          options=options_modes)

    # calculate the wavefunctions using the from the floquet modes
    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                          np.linspace(0, T, Tsteps + 1),
                                          H, T, args,
                                          options=options_modes_table)

    # setup Result for storing the results
    output = Result()
    output.times = tlist
    output.solver = "fsesolve"

    if isinstance(e_ops, FunctionType):
        output.num_expect = 0
        expt_callback = True

    elif isinstance(e_ops, list):

        output.num_expect = len(e_ops)
        expt_callback = False

        if output.num_expect == 0:
            output.states = []
        else:
            output.expect = []
            for op in e_ops:
                if op.isherm:
                    output.expect.append(np.zeros(len(tlist)))
                else:
                    output.expect.append(np.zeros(len(tlist), dtype=complex))

    else:
        raise TypeError("e_ops must be a list Qobj or a callback function")

    psi0_fb = psi0.transform(f_modes_0)
    for t_idx, t in enumerate(tlist):
        f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
        f_states_t = floquet_states(f_modes_t, f_energies, t)
        psi_t = psi0_fb.transform(f_states_t, True)

        if expt_callback:
            # use callback method
            e_ops(t, psi_t)
        else:
            # calculate all the expectation values, or output psi if
            # no expectation value operators where defined
            if output.num_expect == 0:
                output.states.append(Qobj(psi_t))
            else:
                for e_idx, e in enumerate(e_ops):
                    output.expect[e_idx][t_idx] = expect(e, psi_t)

    return output


def floquet_master_equation_rates(f_modes_0, f_energies, c_op, H, T,
                                  args, J_cb, w_th, kmax=5,
                                  f_modes_table_t=None):
    """
    Calculate the rates and matrix elements for the Floquet-Markov master
    equation.

    Parameters
    ----------

    f_modes_0 : list of :class:`qutip.qobj` (kets)
        A list of initial Floquet modes.

    f_energies : array
        The Floquet energies.

    c_op : :class:`qutip.qobj`
        The collapse operators describing the dissipation.

    H : :class:`qutip.qobj`
        System Hamiltonian, time-dependent with period `T`.

    T : float
        The period of the time-dependence of the hamiltonian.

    args : dictionary
        Dictionary with variables required to evaluate H.

    J_cb : callback functions
        A callback function that computes the noise power spectrum, as
        a function of frequency, associated with the collapse operator `c_op`.

    w_th : float
        The temperature in units of frequency.

    k_max : int
        The truncation of the number of sidebands (default 5).

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets)
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table` (optional).

    options : :class:`qutip.solver`
        options for the ODE solver.

    Returns
    -------

    output : list

        A list (Delta, X, Gamma, A) containing the matrices Delta, X, Gamma
        and A used in the construction of the Floquet-Markov master equation.

    """

    N = len(f_energies)
    M = 2 * kmax + 1

    omega = (2 * pi) / T

    Delta = np.zeros((N, N, M))
    X = np.zeros((N, N, M), dtype=complex)
    Gamma = np.zeros((N, N, M))
    A = np.zeros((N, N))

    nT = 100
    dT = T / nT
    tlist = np.arange(dT, T + dT / 2, dT)

    if f_modes_table_t is None:
        f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                              np.linspace(0, T, nT + 1), H, T,
                                              args)

    for t in tlist:
        # TODO: repeated invocations of floquet_modes_t is
        # inefficient...  make a and b outer loops and use the mesolve
        # instead of the propagator.

        f_modes_t = np.hstack([f.full() for f in floquet_modes_t_lookup(
            f_modes_table_t, t, T)])
        FF = f_modes_t.T.conj() @ c_op.full() @ f_modes_t
        phi = exp(-1j * np.arange(-kmax, kmax+1) * omega * t)
        X += (dT / T) * np.einsum("ij,k->ijk", FF, phi)

    Heaviside = lambda x: ((np.sign(x) + 1) / 2.0)
    for a in range(N):
        for b in range(N):
            k_idx = 0
            for k in range(-kmax, kmax + 1, 1):
                Delta[a, b, k_idx] = f_energies[a] - f_energies[b] + k * omega
                Gamma[a, b, k_idx] = 2 * pi * Heaviside(Delta[a, b, k_idx]) * \
                    J_cb(Delta[a, b, k_idx]) * abs(X[a, b, k_idx]) ** 2
                k_idx += 1

    for a in range(N):
        for b in range(N):
            for k in range(-kmax, kmax + 1, 1):
                k1_idx = k + kmax
                k2_idx = -k + kmax
                A[a, b] += Gamma[a, b, k1_idx] + \
                    n_thermal(abs(Delta[a, b, k1_idx]), w_th) * \
                    (Gamma[a, b, k1_idx] + Gamma[b, a, k2_idx])

    return Delta, X, Gamma, A


def floquet_collapse_operators(A):
    """
    Construct collapse operators corresponding to the Floquet-Markov
    master-equation rate matrix `A`.

    .. note::

        Experimental.

    """
    c_ops = []

    N, M = np.shape(A)

    #
    # Here we really need a master equation on Bloch-Redfield form, or perhaps
    # we can use the Lindblad form master equation with some rotating frame
    # approximations? ...
    #
    for a in range(N):
        for b in range(N):
            if a != b and abs(A[a, b]) > 0.0:
                # only relaxation terms included...
                c_ops.append(sqrt(A[a, b]) * projection(N, a, b))

    return c_ops


def floquet_master_equation_tensor(Alist, f_energies):
    """
    Construct a tensor that represents the master equation in the floquet
    basis (with constant Hamiltonian and collapse operators).

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------

    Alist : list
        A list of Floquet-Markov master equation rate matrices.

    f_energies : array
        The Floquet energies.

    Returns
    -------

    output : array

        The Floquet-Markov master equation tensor `R`.

    """

    if isinstance(Alist, list):
        # Alist can be a list of rate matrices corresponding
        # to different operators that couple to the environment
        N, M = np.shape(Alist[0])
    else:
        # or a simple rate matrix, in which case we put it in a list
        Alist = [Alist]
        N, M = np.shape(Alist[0])

    Rdata_lil = scipy.sparse.lil_matrix((N * N, N * N), dtype=complex)

    AsumList = [np.sum(A, axis=1) for A in Alist]

    for k in range(len(Alist)):
        for i in range(N):
            Rdata_lil[i+N*i, i+N*i] -= -Alist[k][i, i] + AsumList[k][i]
            for j in range(i+1, N):
                Rdata_lil[i+N*i, j+N*j] += Alist[k][j, i]
                Rdata_lil[j+N*j, i+N*i] += Alist[k][i, j]
                a_term = -(1/2)*(AsumList[k][i] + AsumList[k][j])
                Rdata_lil[i+N*j, i+N*j] += a_term
                Rdata_lil[j+N*i, j+N*i] += a_term

    return Qobj(Rdata_lil, dims=[[N, N], [N, N]], shape=(N*N, N*N))


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

# -----------------------------------------------------------------------------
# Floquet-Markov master equation
#
#


def floquet_markov_mesolve(
    R, rho0, tlist, e_ops, options=None, floquet_basis=True,
    f_modes_0=None, f_modes_table_t=None, f_energies=None, T=None,
):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.

    .. note::

        It is important to understand in which frame and basis the results
        are returned here.

    Parameters
    ----------

    R : array
        The Floquet-Markov master equation tensor `R`.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.  If ``f_modes_0`` is not passed, this density
        matrix is assumed to be in the Floquet picture.

    tlist : *list* / *array*
        list of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values.

    options : :class:`qutip.solver`
        options for the ODE solver.

    floquet_basis: bool, True
        If ``True``, states and expectation values will be returned in the
        Floquet basis.  If ``False``, a transformation will be made to the
        computational basis; this will be in the lab frame if
        ``f_modes_table``, ``T` and ``f_energies`` are all supplied, or the
        interaction picture (defined purely be f_modes_0) if they are not.

    f_modes_0 : list of :class:`qutip.qobj` (kets), optional
        A list of initial Floquet modes, used to transform the given starting
        density matrix into the Floquet basis.  If this is not passed, it is
        assumed that ``rho`` is already in the Floquet basis.

    f_modes_table_t : nested list of :class:`qutip.qobj` (kets), optional
        A lookup-table of Floquet modes at times precalculated by
        :func:`qutip.floquet.floquet_modes_table`.  Necessary if
        ``floquet_basis`` is ``False`` and the transformation should be made
        back to the lab frame.

    f_energies : array_like of float, optional
        The precalculated Floquet quasienergies.  Necessary if
        ``floquet_basis`` is ``False`` and the transformation should be made
        back to the lab frame.

    T : float, optional
        The time period of driving.  Necessary if ``floquet_basis`` is
        ``False`` and the transformation should be made back to the lab frame.

    Returns
    -------

    output : :class:`qutip.solver.Result`
        An instance of the class :class:`qutip.solver.Result`, which
        contains either an *array* of expectation values or an array of
        state vectors, for the times specified by `tlist`.
    """
    opt = options or Options()
    if opt.tidy:
        R.tidyup()
    rho0 = rho0.proj() if rho0.isket else rho0

    # Prepare output object.
    dt = tlist[1] - tlist[0]
    output = Result()
    output.solver = "fmmesolve"
    output.times = tlist
    if isinstance(e_ops, FunctionType):
        expt_callback = True
        store_states = opt.store_states or False
    else:
        expt_callback = False
        try:
            e_ops = list(e_ops)
        except TypeError:
            raise TypeError("`e_ops` must be iterable or a function") from None
        n_expt_op = len(e_ops)
        if n_expt_op == 0:
            store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                dtype = np.float64 if op.isherm else np.complex128
                output.expect.append(np.zeros(len(tlist), dtype=dtype))
        store_states = opt.store_states or (n_expt_op == 0)
    if store_states:
        output.states = []

    # Choose which frame transformations should be done on the initial and
    # evolved states.
    lab_lookup = [f_modes_table_t, f_energies, T]
    if (
        any(x is None for x in lab_lookup)
        and not all(x is None for x in lab_lookup)
    ):
        warnings.warn(
            "if transformation back to the computational basis in the lab"
            "frame is desired, all of `f_modes_t`, `f_energies` and `T` must"
            "be supplied."
        )
        f_modes_table_t = f_energies = T = None

    # Initial state.
    if f_modes_0 is not None:
        rho0 = rho0.transform(f_modes_0)

    # Evolved states.
    if floquet_basis:
        def transform(rho, t):
            return rho
    elif f_modes_table_t is not None:
        # Lab frame, computational basis.
        def transform(rho, t):
            f_modes_t = floquet_modes_t_lookup(f_modes_table_t, t, T)
            f_states_t = floquet_states(f_modes_t, f_energies, t)
            return rho.transform(f_states_t, True)
    elif f_modes_0 is not None:
        # Interaction picture, computational basis.
        def transform(rho, t):
            return rho.transform(f_modes_0, False)
    else:
        raise ValueError(
            "cannot transform out of the Floquet basis without some knowledge "
            "of the Floquet modes.  Pass `f_modes_0`, or all of `f_modes_t`, "
            "`f_energies` and `T`."
        )

    # Setup integrator.
    initial_vector = mat2vec(rho0.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    # Main evolution loop.
    for t_idx, t in enumerate(tlist):
        if not r.successful():
            break
        rho = transform(Qobj(vec2mat(r.y), rho0.dims, rho0.shape), t)
        if expt_callback:
            e_ops(t, rho)
        else:
            for m, e_op in enumerate(e_ops):
                output.expect[m][t_idx] = expect(e_op, rho)
        if store_states:
            output.states.append(rho)
        r.integrate(r.t + dt)
    return output

# -----------------------------------------------------------------------------
# Solve the Floquet-Markov master equation
#
#


def fmmesolve(H, rho0, tlist, c_ops=[], e_ops=[], spectra_cb=[], T=None,
              args={}, options=Options(), floquet_basis=True, kmax=5,
              _safe_mode=True, options_modes=None):
    """
    Solve the dynamics for the system using the Floquet-Markov master equation.

    .. note::

        This solver currently does not support multiple collapse operators.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        system Hamiltonian.

    rho0 / psi0 : :class:`qutip.qobj`
        initial density matrix or state vector (ket).

    tlist : *list* / *array*
        list of times for :math:`t`.

    c_ops : list of :class:`qutip.qobj`
        list of collapse operators.

    e_ops : list of :class:`qutip.qobj` / callback function
        list of operators for which to evaluate expectation values.

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    T : float
        The period of the time-dependence of the hamiltonian. The default value
        'None' indicates that the 'tlist' spans a single period of the driving.

    args : *dictionary*
        dictionary of parameters for time-dependent Hamiltonians and
        collapse operators.

        This dictionary should also contain an entry 'w_th', which is
        the temperature of the environment (if finite) in the
        energy/frequency units of the Hamiltonian.  For example, if
        the Hamiltonian written in units of 2pi GHz, and the
        temperature is given in K, use the following conversion

        >>> temperature = 25e-3 # unit K # doctest: +SKIP
        >>> h = 6.626e-34 # doctest: +SKIP
        >>> kB = 1.38e-23 # doctest: +SKIP
        >>> args['w_th'] = temperature * (kB / h) * 2 * pi * 1e-9 \
            #doctest: +SKIP

    options : :class:`qutip.solver`
        options for the ODE solver. For solving the master equation.

    floquet_basis : bool
        Will return results in Floquet basis or computational basis
        (optional).

    k_max : int
        The truncation of the number of sidebands (default 5).

    options_modes : :class:`qutip.solver`
        options for the ODE solver. For computing Floquet modes.

    Returns
    -------

    output : :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.
    """

    if _safe_mode:
        _solver_safety_check(H, rho0, c_ops, e_ops, args)

    if options_modes is None:
        options_modes_table = Options()
    else:
        options_modes_table = options_modes

    if T is None:
        T = max(tlist)

    if len(spectra_cb) == 0:
        # add white noise callbacks if absent
        spectra_cb = [lambda w: 1.0] * len(c_ops)

    f_modes_0, f_energies = floquet_modes(H, T, args,
                                          options=options_modes)

    f_modes_table_t = floquet_modes_table(f_modes_0, f_energies,
                                          np.linspace(0, T, 500 + 1),
                                          H, T, args,
                                          options=options_modes_table)

    # get w_th from args if it exists
    if 'w_th' in args:
        w_th = args['w_th']
    else:
        w_th = 0

    # TODO: loop over input c_ops and spectra_cb, calculate one R for each set

    # calculate the rate-matrices for the floquet-markov master equation
    Delta, X, Gamma, Amat = floquet_master_equation_rates(
        f_modes_0, f_energies, c_ops[0], H, T, args, spectra_cb[0],
        w_th, kmax, f_modes_table_t)

    # the floquet-markov master equation tensor
    R = floquet_master_equation_tensor(Amat, f_energies)

    return floquet_markov_mesolve(R, rho0, tlist, e_ops,
                                  options=options,
                                  floquet_basis=floquet_basis,
                                  f_modes_0=f_modes_0,
                                  f_modes_table_t=f_modes_table_t,
                                  T=T,
                                  f_energies=f_energies)
