__all__ = [
    "flimesolve",
    "FLiMESolver",
]

import numpy as np
from qutip.core import data as _data
from qutip import Qobj, QobjEvo, operator_to_vector
from qutip.solver.mesolve import MESolver
from qutip.solver.solver_base import Solver
from qutip.solver.result import Result
from time import time
from qutip.ui.progressbar import progress_bars
from qutip.solver.floquet import fsesolve, FloquetBasis

from qutip.core.environment import (
    BosonicEnvironment,
)


def _c_op_Fourier_amplitudes(floquet_basis, tlist, c_op):
    """
    Parameters
    ----------
    floquet_basis : FloquetBasis Object
        The FloquetBasis object formed from qutip.floquet.FloquetBasis
    c_ops : list of :obj:`.Qobj`, :obj:`.QobjEvo`
        Single collapse operator, or list of collapse operators
    tlist: list, array
        A list of times over a single period for which to calculate the
        Fourier transform of the collapse operators in the Floquet basis

    Returns
    -------
    c_op_Fourier_ampltiides : A list of Fourier amplitudes for the system
        collapse operators in the Floquet basis, used to calculate the rate
        matrices

    """
    # Transforming the lowering operator into the Floquet Basis
    #     and taking the FFT
    modes_table = _floquet_mode_table(floquet_basis, tlist)

    c_op_Floquet_basis = np.einsum(
        "xij,jk,xkl->xil",
        np.transpose(modes_table.conj(), (0, 2, 1)),
        c_op.full(),
        modes_table,
    )
    c_op_Fourier_amplitudes_list = (
        (np.fft.fft(c_op_Floquet_basis, axis=0))
    ) / len(tlist)
    return c_op_Fourier_amplitudes_list


def _rate_matrix_indices(
    relative_secular_cutoff, c_op_Fourier_amplitudes_list, Nt, floquet_basis
):
    omega = 2 * np.pi / floquet_basis.T
    quasis = floquet_basis.e_quasi
    c_op_conj = np.conj(c_op_Fourier_amplitudes_list)

    xx, yy, zz = np.array(c_op_Fourier_amplitudes_list).nonzero()
    xx = np.array(xx) - int(Nt / 2)

    idx_length = len(xx)
    delta_dict = {}
    for idx1 in range(idx_length):
        x00 = xx[idx1]
        x01 = yy[idx1]
        x02 = zz[idx1]
        q01 = quasis[x01]
        q02 = quasis[x02]
        for idx2 in range(idx_length):
            x10 = xx[idx2]
            x11 = yy[idx2]
            x12 = zz[idx2]
            q11 = quasis[x11]
            q12 = quasis[x12]
            cop1 = c_op_Fourier_amplitudes_list[x00, x01, x02]
            cop2 = c_op_conj[x10, x11, x12]
            ordered_idx = (
                yy[idx1],
                zz[idx1],
                yy[idx2],
                zz[idx2],
                xx[idx1],
                xx[idx2],
            )
            delts = ((q01 - q02) - (q11 - q12)) / omega + (x00 - x10)
            if delts == 0 or (
                abs(delts / (cop1 * cop2)) <= relative_secular_cutoff
            ):
                try:
                    delta_dict[delts].append(ordered_idx)
                except KeyError:
                    delta_dict[delts] = [ordered_idx]
    return delta_dict


def _Rate_Matrix_Builder(
    delta_dict, c_op_Fourier_amplitudes_list, power_spectrum, floquet_basis
):
    quasis = floquet_basis.e_quasi
    omega = 2 * np.pi / floquet_basis.T
    Hdim = len(quasis)

    c_op_conj = np.conj(c_op_Fourier_amplitudes_list)
    matrix_it = np.nditer(
        np.zeros((Hdim, Hdim, Hdim, Hdim), dtype="complex"),
        flags=["multi_index"],
    )
    matrix_idx = [matrix_it.multi_index for idx in matrix_it]

    R_tensor = {}
    for key in delta_dict.keys():
        valid_c_op_prods_list = delta_dict[key]

        flime_FirstTerm = np.zeros(
            len(valid_c_op_prods_list * Hdim**4), dtype=complex
        )
        flime_SecondTerm = np.zeros(
            len(valid_c_op_prods_list * Hdim**4), dtype=complex
        )
        for idx, indices in enumerate(valid_c_op_prods_list):
            a, b, ap, bp, k, kp = indices

            gam_plus = (
                c_op_Fourier_amplitudes_list[k, a, b]
                * c_op_conj[kp, ap, bp]
                * power_spectrum(quasis[a] - quasis[b] - k * omega)
            )
            gam_minus_prime = (
                c_op_Fourier_amplitudes_list[k, a, b]
                * c_op_conj[kp, ap, bp]
                * power_spectrum(quasis[ap] - quasis[bp] - kp * omega)
            )

            for itx, indixes in enumerate(matrix_idx):
                m, n, p, q = indixes

                t1 = (b == p) & (q == bp) & (m == a) & (ap == n)
                t2 = (m == bp) & (q == n) & (ap == a) & (b == p)
                # t3 = t1 Keeping this for reference
                t4 = (q == bp) & (ap == a) & (m == p) & (b == n)

                flime_FirstTerm[idx * Hdim**4 + itx] = (
                    gam_plus * (t1 ^ t2) * (1 - 2 * t2)
                )

                flime_SecondTerm[idx * Hdim**4 + itx] = (
                    gam_minus_prime * (t1 ^ t4) * (1 - 2 * t4)
                )

        try:
            R_tensor[key] += (1 / 2) * np.sum(
                np.reshape(
                    np.add(flime_FirstTerm, flime_SecondTerm),
                    (len(valid_c_op_prods_list), Hdim**2, Hdim**2),
                ),
                axis=0,
            )
        except KeyError:
            R_tensor[key] = (1 / 2) * np.sum(
                np.reshape(
                    np.add(flime_FirstTerm, flime_SecondTerm),
                    (len(valid_c_op_prods_list), Hdim**2, Hdim**2),
                ),
                axis=0,
            )

    return R_tensor


def _floquet_rate_matrix(
    floquet_basis,
    Nt,
    c_ops,
    power_spectra=None,
    relative_secular_cutoff=0,
):
    """
    Parameters
    ----------
    floquet_basis : FloquetBasis Object
        The FloquetBasis object formed from qutip.floquet.FloquetBasis
    Nt : Int
        Number of points in one period of the Hamiltonian
    c_ops : list of :obj:`.Qobj`, :obj:`.QobjEvo`
        Single collapse operator, or list of collapse operators
    relative_secular_cutoff : float 0-1,optional
        the relative secular approximation restriction of
        FLiMESolve. Decides "acceptable" values of (frequency/rate) for rate
        matrix entries. Lower values imply rate occurs much faster than
        rotation frtequency, i.e. more important matrix entries. Higher
        values meaning rates cause changes slower than the Hamiltonian rotates,
        i.e. the changes average out on "longer" time scales.
    power_spectrum : list of functions
        The power spectra of the autocorrelation function(s) as a function of
        w, given by Gamma(w) = int_0^inf(e^i Delta t)Tr_B{B(t)B\rho_B},
        supplied as either a list of callable power spectrum functions, as
        a list of BosonicEnvironment objects, or some combination thereof.

    Returns
    -------
    total_R_tensor : Dictionary of {frequency: 2D rate matrix} pairs
        This function returns a list of dictionaries whose keys are the
        frequency of a specific rate matrix term, and whose values are the
        rate matrices for the associated frequency.
    """
    Hdim = len(floquet_basis.e_quasi)  # Dimensionality of the Hamiltonian

    # Forming tlist to take FFT of collapse operators
    timet = floquet_basis.T
    dt = timet / Nt
    tlist = np.linspace(0, timet - dt, Nt)

    Nt = len(tlist)
    Hdim = len(floquet_basis.e_quasi)

    total_R_tensor = {0: np.zeros((Hdim**2, Hdim**2), dtype="complex")}

    for cdx, c_op in enumerate(c_ops):
        power_spectrum = power_spectra[cdx]

        c_op_Fourier_amplitudes_list = _c_op_Fourier_amplitudes(
            floquet_basis, tlist, c_op
        )
        """
        Finding all terms that are either DC or that are "important" enough
            to include as decided by the Relative Secular Approximation
        """
        delta_dict = _rate_matrix_indices(
            relative_secular_cutoff,
            c_op_Fourier_amplitudes_list,
            Nt,
            floquet_basis,
        )

        """
        Below takes all the indices that correspond to a single frequency,
            and builds that frequency value of the Rate Matrix
        """
        R_tensor = _Rate_Matrix_Builder(
            delta_dict,
            c_op_Fourier_amplitudes_list,
            power_spectrum,
            floquet_basis,
        )
        for key in R_tensor:
            try:
                total_R_tensor[key] += R_tensor[key]
            except KeyError:
                total_R_tensor[key] = R_tensor[key]

    return total_R_tensor


def flimesolve(
    H,
    rho0,
    tlist,
    T,
    c_ops=None,
    *,
    e_ops=None,
    args=None,
    relative_secular_cutoff=0,
    options={"Nt": 2**4},
):
    """
    Solve system dynamics using the Floquet-Lindblad Master Equation

    Parameters
    ----------

    H : :class:`Qobj`,:class:`QobjEvo`,:class:`QobjEvo` compatible format.
        Periodic system Hamiltonian as :class:`QobjEvo`. List of
        :class:`Qobj`, :class:`Coefficient` or callable that
        can be made into :class:`QobjEvo` are also accepted.

    rho0 / psi0 : :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    T : float
        The period of the time-dependence of the hamiltonian.

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        Single collapse operator, or list of collapse operators, or
        list of [collapse operator, power spectrum function] pairs, or
        list of [collapse operator, BosonicEnvironment] pairs, or
        some combination of the three options.

    e_ops : list of :class:`qutip.Qobj` / callback function
        List of operators for which to evaluate expectation values.
        The states are reverted to the lab basis before applying the

    args : *dictionary*
        Dictionary of parameters for time-dependent Hamiltonian

    relative_secular_cutoff : float
        Value of the relative secular approximation to use when constructing the rate
        matrix R(t). Default value of zero uses the fully time-independent/most
        strict secular approximation, and values greater than zero have time
        dependence. The default integration method change depending
        on this value, "diag" for `0`, "adams" otherwise.

    options : None / dict
        Dictionary of options for the solver.
        - Nt : Int
          The number of points within one period of the Hamiltonian, used for
          forming the rate matrix. If none is supplied, flimesolve will default
          to using 16 points per period.
        - store_final_state : bool
          Whether or not to store the final state of the evolution in the
          result class.
        - store_states : bool,None
          Whether or not to store the state vectors or density matrices.
          On `None` the states will be saved if no expectation operators are
          given.
        - store_floquet_states : bool
          Whether or not to store the density matrices in the floquet basis in
          ``result.floquet_states``.
        - normalize_output : bool
          Normalize output state to hide ODE numerical errors.
        - progress_bar : str {'text','enhanced','tqdm',''}
          How to present the solver progress.
          'tqdm' uses the python module of the same name and raise an error
          if not installed. Empty string or False will disable the bar.
        - progress_kwargs : dict
          kwargs to pass to the progress_bar. Qutip's bars use `chunk_size`.
        - method : str ["adams","bdf","lsoda","dop853","vern9",etc.]
          Which differential equation integration method to use.
        - atol,rtol : float
          Absolute and relative tolerance of the ODE integrator.
        - nsteps :
          Maximum number of (internally defined) steps allowed in one ``tlist``
          step.
        - max_step : float,0
          Maximum lenght of one internal step. When using pulses,it should be
          less than half the width of the thinnest pulse.

        Other options could be supported depending on the integration method,
        see `Integrator <./classes.html#classes-ode>`_.

    Returns
    -------
    result: :class:`qutip.Result`

        An instance of the class :class:`qutip.Result`,which contains
        the expectation values for the times specified by `tlist`,and/or the
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

    if isinstance(H, FloquetBasis):
        floquet_basis = H
    else:
        floquet_basis = FloquetBasis(H, T, args, precompute=None)

    solver = FLiMESolver(
        floquet_basis,
        c_ops,
        relative_secular_cutoff=relative_secular_cutoff,
        options=options,
    )
    return solver.run(rho0, tlist, e_ops=e_ops)


def _floquet_mode_table(floquet_basis, tlist):
    """
    Parameters
    ----------

    floquet_basis : Floquet Basis object for the solver/Hamiltonian of interest.

    tlist : tlist : *list* / *array*
        List of times for :math:`t`.

    Returns
    -------
    result: :class:`np.array`

        An array, indexed by (i,j,k), for the jth row, kth column, and
            ith time point of the Floquet modes used to transform into the
            FLoquet Basis
    """
    return np.stack(
        [floquet_basis.mode(t, data=True).to_array() for t in tlist]
    )


def _floquet_state_table(floquet_basis, tlist):
    """
    Parameters
    ----------

    floquet_basis : Floquet Basis object for the solver/Hamiltonian of interest.

    tlist : tlist : *list* / *array*
        List of times for :math:`t`.

    Returns
    -------
    result: :class:`np.array`

        An array, indexed by (i,j,k), for the jth row, kth column, and
            ith time point of the Floquet States, specifically found
            to hit every t%tau in the system evolution, even if a
            certain time point doesn't happen every period'

    """

    dims = len(floquet_basis.e_quasi)

    taulist_test = np.array(tlist) % floquet_basis.T
    taulist_correction = np.argwhere(
        abs(taulist_test - floquet_basis.T) < 1e-10
    )
    taulist_test_new = np.round(taulist_test, 10)
    taulist_test_new[taulist_correction] = 0

    tally = {}
    for i, item in enumerate(taulist_test_new):
        try:
            tally[item].append(i)
        except KeyError:
            tally[item] = [i]

    sorted_time_args = {key: tally[key] for key in np.sort(list(tally.keys()))}

    fmodes_core_dict = {
        t: floquet_basis.mode(t, data=True).to_array()
        for t in (sorted_time_args.keys())
    }

    tiled_modes = np.zeros((len(tlist), dims, dims), dtype=complex)
    for key in fmodes_core_dict:
        tiled_modes[sorted_time_args[key]] = fmodes_core_dict[key]
    quasi_e_table = np.exp(
        np.einsum("i,k -> ik", -1j * np.array(tlist), floquet_basis.e_quasi)
    )
    fstates_table = np.einsum("ijk,ik->ijk", tiled_modes, quasi_e_table)
    return fstates_table


class FloquetResult(Result):
    def _post_init(self, floquet_basis):
        self.floquet_basis = floquet_basis
        if self.options["store_floquet_states"]:
            self.floquet_states = []
        else:
            self.floquet_states = None
        super()._post_init()

    def add(self, t, state, floquet_state):
        if self.options["store_floquet_states"]:
            self.floquet_states.append(floquet_state)
        super().add(t, state)


class FLiMESolver(MESolver):
    """
    Solver for the Floquet-Lindblad master equation.

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

    c_ops : list of (:obj:`.QobjEvo`, :obj:`.QobjEvo` compatible format)
        Single collapse operator, or list of collapse operators, or
        list of [collapse operator, power spectrum function] pairs, or
        list of [collapse operator, BosonicEnvironment] pairs, or
        some combination of the three options.

    relative_secular_cutoff : float
        Value of the relative secular approximation to use when constructing the rate
        matrix R(t).Default value of zero uses the fully time-independent/most
        strict secular approximation, and will utilize the "diag" solver method.
        Values greater than zero have time dependence, and will subsequently
        use the "Adams" method for the ODE solver.

    options : dict,optional
         Options for the solver,see :obj:`FLiMESolver.options` and
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
        "method": None,
        "store_floquet_states": False,
        "atol": 1e-8,
        "rtol": 1e-6,
        "Nt": 2**4,
    }

    def __init__(
        self,
        floquet_basis,
        c_ops,
        relative_secular_cutoff=0,
        *,
        options=None,
    ):
        if relative_secular_cutoff == 0:
            self.solver_options["method"] = "diag"
        else:
            self.solver_options["method"] = "adams"

        if isinstance(floquet_basis, FloquetBasis):
            self.floquet_basis = floquet_basis
        else:
            raise TypeError("The ``floquet_basis`` must be a FloquetBasis")

        self.options = options
        self._Nt = self.options["Nt"]

        def dummy_power_spectrum(omega):
            return 1

        spectra_new = []
        c_ops_new = []
        for c_op in c_ops:
            if isinstance(c_op, Qobj):  # C_op is just a rate
                c_ops_new.append(c_op)
                spectra_new.append(dummy_power_spectrum)
            elif isinstance(
                c_op[0], Qobj
            ):  # C_op is an operator/spectrum or operator/environment pair
                if callable(c_op[1]):
                    c_ops_new.append(c_op[0])
                    spectra_new.append(c_op[1])
                elif isinstance(c_op[1], BosonicEnvironment):
                    c_ops_new.append(c_op[0])
                    spectra_new.append(c_op[1].power_spectrum)
                else:
                    print(
                        "c_ops must be either a [rate*operator] quantum object, "
                        "a [Qobj,power spectrum] pair, or a bosonic bath "
                        "Environment"
                    )

            else:
                print(
                    "c_ops must be either a [rate*operator] quantum object, "
                    "a [Qobj,power spectrum] pair, or a bosonic bath "
                    "Environment"
                )

        self.c_ops = c_ops_new
        self.power_spectra = spectra_new

        self._relative_secular_cutoff = relative_secular_cutoff
        self._num_collapse = len(c_ops)
        self.dims = len(self.floquet_basis.e_quasi)
        self._build_rhs()
        self._state_metadata = {}

        self.stats = self._initialize_stats()

    def _initialize_stats(self):
        stats = Solver._initialize_stats(self)
        stats.update(
            {
                "solver": "Floquet-Lindblad master equation",
                "num_collapse": self._num_collapse,
                "init rhs time": self._init_rhs_time,
            }
        )
        return stats

    @property
    def Nt(self):
        return self._Nt

    @Nt.setter
    def Nt(self, new):
        self._Nt = new
        self._build_rhs()

    @property
    def relative_secular_cutoff(self):
        return self._relative_secular_cutoff

    @relative_secular_cutoff.setter
    def relative_secular_cutoff(self, new):
        self._relative_secular_cutoff = new

        self._build_rhs()

    def _build_rhs(self):
        self.rate_matrix = _floquet_rate_matrix(
            self.floquet_basis,
            self._Nt,
            self.c_ops,
            relative_secular_cutoff=self._relative_secular_cutoff,
            power_spectra=self.power_spectra,
        )
        self.rhs = self._create_rhs(self.rate_matrix)
        self._integrator = self._get_integrator()

    def _create_rhs(self, rate_matrix_dictionary):
        _time_start = time()

        Rate_Qobj_list = {
            key: Qobj(
                rate_matrix_dictionary[key],
                dims=[
                    self.floquet_basis.U(0).dims,
                    self.floquet_basis.U(0).dims,
                ],
                copy=False,
            ).to("csr")
            for key in rate_matrix_dictionary
        }

        Rt_timedep_pairs = [
            Rate_Qobj_list[0],
            *[
                [
                    Rate_Qobj_list[key],
                    lambda t, key=key: np.exp(
                        (1j * key * 2 * np.pi * t) / self.floquet_basis.T
                    ),
                ]
                for key in set(Rate_Qobj_list) - {0}
            ],
        ]

        rhs = QobjEvo(Rt_timedep_pairs)

        self._init_rhs_time = time() - _time_start
        return rhs

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

        floquet : bool,optional {False}
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
            Time to evolve to,must be higher than the last call.

        copy : bool,optional {True}
            Whether to return a copy of the data or the data in the ODE solver.

        floquet : bool,optional {False}
            Whether to return the state in the floquet basis or laboratory
            basis.

        args : dict,optional {None}
            Not supported

        .. note::
            The state must be initialized first by calling ``start`` or
            ``run``. If ``run`` is called,``step`` will continue from the last
            time and state obtained.
        """
        if args:
            raise ValueError("FLiMESolver cannot update arguments")
        state = super().step(t)
        if not floquet:
            state = self.floquet_basis.from_floquet_basis(state, t)
        elif copy:
            state = state.copy()
        return state

    def run(
        self,
        state0,
        tlist,
        *,
        floquet=False,
        args=None,
        e_ops=None,
    ):
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
            evolution. Each times of the list must be increasing,but does not
            need to be uniformy distributed.

        floquet : bool,optional {False}
            Whether the initial state in the floquet basis or laboratory basis.

        args : dict,optional {None}
            Not supported

        e_ops : list {None}
            List of Qobj,QobjEvo or callable to compute the expectation
            values. Function[s] must have the signature
            f(t : float,state : Qobj) -> expect.

        Returns
        -------
        results : :class:`qutip.solver.FloquetResult`
            Results of the evolution. States and/or expect will be saved. You
            can control the saved data in the options.
        """

        if args:
            raise ValueError("FLiMESolver cannot update arguments")
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

        if state0.type == "ket":
            state0 = operator_to_vector(state0 * state0.dag())
        elif state0.type == "oper":
            state0 = operator_to_vector(state0)
        else:
            raise ValueError("You need to supply a valid ket or operator")

        fstates_table = _floquet_state_table(self.floquet_basis, tlist)

        stats["preparation time"] += time() - _time_start

        sols = [self._restore_state(_data0, copy=False).full()]
        progress_bar = progress_bars[self.options["progress_bar"]](
            len(tlist) - 1, **self.options["progress_kwargs"]
        )
        for t, state in self._integrator.run(tlist):
            progress_bar.update()
            sols.append(self._restore_state(state, copy=False).full())
        progress_bar.finished()
        sols = np.array(sols)

        sols_comp_arr = np.einsum(
            "xij,xjk,xkl->xil",
            fstates_table,
            sols,
            np.transpose(fstates_table.conj(), axes=(0, 2, 1)),
            order="F",
        )
        dims = self.floquet_basis.U(0)._dims
        sols_comp = [
            Qobj(
                _data.Dense(state, copy=False),
                dims=dims,
                copy=False,
            )
            for state in sols_comp_arr
        ]
        for idx, state in enumerate(sols_comp):
            results.add(
                tlist[idx], state, Qobj(fstates_table[idx], copy=False)
            )
        stats["run time"] = time() - _time_start
        return results

    def _argument(self, args):
        if args:
            raise ValueError("FLiMESolver cannot update arguments")
