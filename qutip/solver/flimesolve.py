# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:44:30 2023

@author: Fenton
"""

import numpy.ma as ma
import scipy as scp
import itertools as itertools

from ..core import stack_columns
import numpy as np
from qutip.core import data as _data
from qutip import Qobj, QobjEvo, operator_to_vector, vector_to_operator, ket2dm
from .propagator import Propagator
from .mesolve import MESolver
from .solver_base import Solver
from .integrator import Integrator
from .result import Result
from time import time
from ..ui.progressbar import progess_bars


def _floquet_rate_matrix(floquet_basis,
                         tlist,
                         c_ops,
                         c_op_rates,
                         omega,
                         time_sense=0):
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
    time_sense : float 0-inf, optional
        the time sensitivity/secular approximation restriction for this rate
            matrix as a multiple of the system frequency beatfreq. The current
            implimentation is alright, but this could use some fixing up

    Returns
    -------
    total_R_tensor : HdimxHdimxHdimxHdim Numpy matrix
        This is the 4D rate matrix for the system, created by summing the
            Rate matrix of each individual collapse operator. Something
            something Rate Matrix something something linear Operator.
    '''

    def kron(a, b):
        if a == b:
            value = 1
        else:
            value = 0
        return value

    '''
    First, divide all quasienergies by omega to get everything in terms of
        omega. This way, once I've figured out the result of the exponential
        term addition, I can just multiply it all by w

    Defining the exponential sums in terms of w now to save space below
    '''
    def delta(a, ap, b, bp, l, lp):
        return ((floquet_basis.e_quasi[a]-floquet_basis.e_quasi[ap])
                - (floquet_basis.e_quasi[b]-floquet_basis.e_quasi[bp])
                + (l-lp))/(list(omega.values())[0]/2)

    Nt = len(tlist)
    Hdim = len(floquet_basis.e_quasi)

    Rate_matrix_list = []
    for cdx, c_op in enumerate(c_ops):

        '''
        These two lines transform the lowering operator into the Floquet mode
            basis
        '''
        fmodes = np.stack([np.stack([i.full() for i in floquet_basis.mode(t)])
                           for t in tlist])[..., 0]
        fmodes_ct = np.transpose(fmodes, (0, 2, 1))

        c_op_Floquet_basis = (fmodes_ct @ c_op.full() @ fmodes)

        '''
        Performing the 1-D FFT to find the Fourier amplitudes of this specific
            lowering operator in the Floquet basis

        Divided by the length of tlist for normalization
        '''
        c_op_Fourier_amplitudes_list = np.array(
            scp.fft.fft(c_op_Floquet_basis, axis=0)/len(tlist)
            )

        '''
        The loops over n,m,p,q are to set the n,m,p,q elements of the R matrix
            for a given time dependence. The other loops are to do sums within
            those elements. The Full form of the FLime as written here can be
            found on the "Matrix Form FLiME" OneNote page on my report tab.
            Too much to explain it here!
        '''
        # Iterating over all possible l, lp
        iterations_test_list_back = [Hdx for Hdx
                                     in itertools.product(range(0, len(tlist)),
                                                          repeat=2)]

        # Iterating over all possible combinations of A, AP, B, BP
        iterations_test_list = [Hdx for Hdx
                                in itertools.product(range(0, Hdim),
                                                     repeat=4)]

        full_iterations_test_list = [(*x, *y) for x in iterations_test_list
                                     for y in iterations_test_list_back]

        # Finding the actual time dependance (as a function)
        time_dependence_list = [delta(*test_itx)
                                for test_itx in full_iterations_test_list]

        '''
        Currently, I have this set such that any argument in the time dependent
            exponential that is less than the desired time sensitivy is kept,
            while others are ignore. This was my naive way of understanding
            the time-dependancy scaling when I first made this section. Because
            I've been working mostly in the zero time-dependence case, I
            haven't really bothered to fix it yet.'

        Instead, this needs to be changed to keep terms based on FFT amplitude,
            rather than time-dependence. That is, I need to keep terms based on
            the normalized value of each point in the FFT. This is less
            directly related to the time dependence values, but it's easier to
            say "hey this Fourier component is more important because its
            amplitude is much higher than all the others"
        '''

        # Recovering the indices of valid time arguments
        valid_TDXs = (~ma.getmaskarray(ma.masked_where(
                        np.absolute(time_dependence_list)
                        > time_sense, time_dependence_list))).nonzero()[0]

        # Creates a list tuples that are the valid (a,b,ap,bp,l,lp)
        #     indices to construct R(t) with the given secular constraint
        valid_time_dep_sum_index_vals = \
            [tuple(full_iterations_test_list[valid_index])
             for valid_index in valid_TDXs]

        # Creating empty dictionary to hold {t_value : R_tensor(t_value)} pairs
        c_op_R_tensor = {}

        # For every entry in the list of tuples, create R(t)
        for vdx, vals in enumerate(valid_time_dep_sum_index_vals):
            a  = vals[0]
            ap = vals[1]
            b  = vals[2]
            bp = vals[3]
            l  = vals[4]
            lp = vals[5]

            R_slice = np.zeros((Hdim**2, Hdim**2), dtype=complex)
            # Iterating over the indices of R_slice to set each value.
            # I Should figure out something faster later
            for idx in np.ndindex(Hdim, Hdim, Hdim, Hdim):
                m = idx[0]
                n = idx[1]
                p = idx[2]
                q = idx[3]

                R_slice[m+Hdim*n, p+Hdim*q] = \
                    c_op_rates[cdx] \
                    * c_op_Fourier_amplitudes_list[l, a, b] \
                    * np.conj(c_op_Fourier_amplitudes_list[lp, ap, bp])\
                    * (        kron(m,  a) * kron(n, ap) * kron(p, b) * kron(q, bp)
                       - (1/2)*kron(a, ap) * kron(m, bp) * kron(p, b) * kron(q,  n)
                       - (1/2)*kron(a, ap) * kron(n,  b) * kron(p, m) * kron(q, bp)
                       )

            try:
                # If this time-dependence entry already exists,
                #     add this "slice" to it
                c_op_R_tensor[time_dependence_list[valid_TDXs[vdx]]] += R_slice
            except KeyError:
                # If this time-dependence entry doesn't already exist, make it
                c_op_R_tensor[time_dependence_list[valid_TDXs[vdx]]] = R_slice

        Rate_matrix_list.append(c_op_R_tensor)

    total_R_tensor = {}
    for Rdic_idx in Rate_matrix_list:
        for key in Rdic_idx:
            try:
                # If this time-dependence entry already exists,
                #      add this "slice" to it
                total_R_tensor[key] += Rdic_idx[key]
            except KeyError:
                # If this time-dependence entry doesn't already exist, make it
                total_R_tensor[key] = Rdic_idx[key]

    return total_R_tensor


def flimesolve(
        H,
        rho0,
        tlist,
        c_ops=[],
        c_op_rates=[],
        e_ops=[],
        T=0,
        args=None,
        time_sense=0,
        options=None):
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
        List of times for :math:`t`.

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
    if c_ops is None:
        return fsesolve(
            H,
            rho0,
            tlist,
            e_ops=e_ops,
            T=T,
            w_th=0,
            args=args,
            options=options,
        )

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]
    if isinstance(c_op_rates, float):
        c_op_rates = [c_op_rates]

    '''
    This section imposes that there are 2**n time points per period. At one
        point there was something in the working theory that required this,
        but I'm unsure without checking back that it's still a requirement.
        This is an area that definitely needs testing, for that reason!
    '''
    '''
    Adding the requirement that tlist is a power of two. If it's not, the
        code will just bring it to the nearest power of two
    '''
    def next_power_of_2(x):
        x = int(x)
        return 1 if x == 0 else 2**(x - 1).bit_length()

    if T == 0:
        len_one_period = len(tlist)
        number_of_periods = 1
        T = tlist[-1]
    else:
        len_one_period = np.where(abs((tlist/(T))-(1+(tlist[0]/T)))
                                  == np.amin(abs((tlist/(T))-(1+(tlist[0]/T))))
                                  )[0][0]
        number_of_periods = int(np.round((tlist[-1]-tlist[0])/T))

    # Imposing that the input has 2**n points per period
    Nt_one_period = int(next_power_of_2(len_one_period))
    Nt = int(Nt_one_period * number_of_periods)
    timet = T*number_of_periods
    dt = timet/Nt
    tlist = np.linspace(0, timet-dt, Nt)
    tlist_one_period = np.linspace(0, T-dt, Nt_one_period)

    # With the tlists sorted, the floquet_basis object can be prepared
    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        T = T or tlist[-1]
        t_precompute = tlist
        # `fsesolve` is a fallback from `fmmesolve`, for the later, options
        # are for the open system evolution.
        floquet_basis = FloquetBasis(H, T, args, precompute=t_precompute)

    if rho0.type == 'ket':
        rho00 = operator_to_vector(ket2dm(rho0))
    elif rho0.type == 'oper':
        rho00 = operator_to_vector(rho0)

    solver = FLiMESolver(floquet_basis,
                         tlist_one_period,
                         tlist,
                         args,
                         c_ops,
                         c_op_rates,
                         time_sense=time_sense,
                         options=options)
    return solver.run(rho0, tlist, e_ops=e_ops)


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
        tlist,
        taulist,
        Hargs,
        c_ops,
        c_op_rates,
        *,
        time_sense=0,
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

        self._num_collapse = len(c_ops)
        if not all(
            isinstance(c_op, Qobj)
            for c_op in c_ops
        ):
            raise TypeError("c_ops must be type Qobj")

        RateDic = _floquet_rate_matrix(
             floquet_basis,
             tlist,
             c_ops,
             c_op_rates,
             Hargs,
             time_sense=time_sense)

        Rate_Qobj_list = [Qobj(
                               RateMat,
                               dims=[[self.Hdim, self.Hdim],
                                     [self.Hdim, self.Hdim]
                                     ],
                               type="super",
                               superrep="super",
                               copy=False)
                          for RateMat in RateDic.values()]

        R0 = Rate_Qobj_list[0]
        Rt_timedep_pairs = [list([Rate_Qobj_list[idx],
                                  'exp(1j*' + str(list(RateDic.keys())[idx]
                                                  * list(Hargs.values())[0])
                                  + '*t)'])
                            for idx in range(1, len(Rate_Qobj_list))]

        if Rt_timedep_pairs != []:
            self.rhs = QobjEvo([R0, *Rt_timedep_pairs])
        else:
            self.rhs = QobjEvo(R0)

        '''
        Qobjevo automatically reduces tensor rank, like we do by unfolding the
        R tensor. I think I need to refold it before putting it into this
        thing?
        '''

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

    def _initialize_stats(self):
        stats = Solver._initialize_stats(self)
        stats.update(
            {
                "solver": "Floquet-Lindblad master equation",
                "num_collapse": self._num_collapse,
            }
        )
        return stats

    def _argument(self, args):
        if args:
            raise ValueError("FMESolver cannot update arguments")
