from __future__ import print_function

__all__ = ['SolverOptions', 'ExpectOps']

import os
import sys
import warnings
import datetime
import numpy as np
from collections import OrderedDict
from types import FunctionType, BuiltinFunctionType

from .. import __version__, Qobj, QobjEvo
from .optionsclass import optionsclass
from ..core import data as _data

solver_safe = {}


class SolverSystem():
    pass


class ExpectOps:
    """
        Contain and compute expectation values
    """
    def __init__(self, e_ops=[], super_=False):
        # take care of expectation values, if any
        self.isfunc = False
        self.e_ops_dict = False
        self.raw_e_ops = e_ops
        self.e_ops_qoevo = []
        self.e_num = 0
        self.e_ops_isherm = []

        if isinstance(e_ops, (Qobj, QobjEvo)):
            e_ops = [e_ops]
        elif isinstance(e_ops, dict):
            self.e_ops_dict = e_ops
            e_ops = [e for e in e_ops.values()]

        self.e_ops = e_ops
        if isinstance(e_ops, list):
            self.e_num = len(e_ops)
            self.e_ops_isherm = [e.isherm for e in e_ops]
            if not super_:
                self.e_ops_qoevo = np.array([QobjEvo(e) for e in e_ops],
                                            dtype=object)
            else:
                self.e_ops_qoevo = np.array([QobjEvo(spre(e)) for e in e_ops],
                                            dtype=object)
        elif callable(e_ops):
            self.isfunc = True
            self.e_num = 1

    def init(self, tlist):
        self.tlist = tlist
        if self.isfunc:
            self.raw_out = []
        else:
            self.raw_out = np.zeros((self.e_num, len(tlist)), dtype=complex)

    def check_dims(self, dims):
        if not self.isfunc:
            for op in self.e_ops_qoevo:
                if isinstance(op, QobjEvo) and op.dims[1] != dims[0]:
                    raise TypeError(f"e_ops dims ({op.dims}) are not "
                                    f"compatible with the system's ({dims})")

    def copy(self):
        out = ExpectOps.__new__(ExpectOps)
        out.isfunc = self.isfunc
        out.e_ops_dict = self.e_ops_dict
        out.raw_e_ops = self.raw_e_ops
        out.e_ops = self.e_ops
        out.e_num = self.e_num
        out.e_ops_isherm = self.e_ops_isherm
        out.e_ops_qoevo = self.e_ops_qoevo
        return out

    def step(self, iter_, state):
        if self.isfunc:
            self.raw_out.append(self.e_ops(t, state))
        else:
            state = _data.dense.fast_from_numpy(state)
            t = self.tlist[iter_]
            for ii in range(self.e_num):
                if isinstance(self.e_ops_qoevo[ii], QobjEvo):
                    self.raw_out[ii, iter_] = \
                        self.e_ops_qoevo[ii].expect_data(t, state)
                elif callable(self.e_ops_qoevo[ii]):
                    self.raw_out[ii, iter_] = \
                        self.e_ops_qoevo[ii](t, state)

    def finish(self):
        if self.isfunc:
            result = self.raw_out
        else:
            result = []
            for ii in range(self.e_num):
                if self.e_ops_isherm[ii]:
                    result.append(np.real(self.raw_out[ii, :]))
                else:
                    result.append(self.raw_out[ii, :])
            if self.e_ops_dict:
                result = {e: result[n]
                          for n, e in enumerate(self.e_ops_dict.keys())}
        return result

    def __eq__(self, other):
        if isinstance(other, ExpectOps):
            other = other.raw_e_ops
        return self.raw_e_ops == other

    def __ne__(self, other):
        return not (self == other)

    def __bool__(self):
        return bool(self.e_num)


@optionsclass("solver")
class SolverOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(order=10, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.order = 10

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.solver['order'] = 10

    Supported options:

        atol : float {1e-8}
            Absolute tolerance.
        rtol : float {1e-6}
            Relative tolerance.
        method : str {'adams','bdf'}
            Integration method.
        order : int {12}
            Order of integrator (<=12 'adams', <=5 'bdf')
        nsteps : int {2500}
            Max. number of internal steps/call.
        first_step : float {0}
            Size of initial step (0 = automatic).
        min_step : float {0}
            Minimum step size (0 = automatic).
        max_step : float {0}
            Maximum step size (0 = automatic)
        tidy : bool {True,False}
            Tidyup Hamiltonian and initial state by removing small terms.
        average_states : bool {False}
            Average states values over trajectories in stochastic solvers.
        average_expect : bool {True}
            Average expectation values over trajectories for stochastic solvers.
        ntraj : int {500}
            Number of trajectories in stochastic solvers.
        store_final_state : bool {False, True}
            Whether or not to store the final state of the evolution in the
            result class.
        store_states : bool {False, True}
            Whether or not to store the state vectors or density matrices in the
            result class, even if expectation values operators are given. If no
            expectation are provided, then states are stored by default and this
            option has no effect.
    """
    options = {
        # Absolute tolerance (default = 1e-8)
        "atol": 1e-8,
        # Relative tolerance (default = 1e-6)
        "rtol": 1e-6,
        # Integration method (default = 'adams', for stiff 'bdf')
        "method": 'adams',
        # Maximum order used by integrator (<=12 for 'adams', <=5 for 'bdf')
        "order": 12,
        # Max. number of internal steps/call
        "nsteps": 1000,
        # Size of initial step (0 = determined by solver)
        "first_step": 0,
        # Max step size (0 = determined by solver)
        "max_step": 0,
        # Minimal step size (0 = determined by solver)
        "min_step": 0,
        # Average expectation values over trajectories (default = True)
        "average_expect": True,
        # average expectation values
        "average_states": False,
        # tidyup Hamiltonian before calculation (default = True)
        "tidy": True,
        # Number of trajectories (default = 500)
        "ntraj": 500,
        "gui": False,
        # store final state?
        "store_final_state": False,
        # store states even if expectation operators are given?
        "store_states": False,
        # average mcsolver density matricies assuming steady state evolution
        "steady_state_average": False,
        # Normalize output of solvers
        # (turned off for batch unitary propagator mode)
        "normalize_output": True
    }


@optionsclass("mcsolve", SolverOptions)
class McOptions:
    """
    Class of options for evolution solvers such as :func:`qutip.mesolve` and
    :func:`qutip.mcsolve`. Options can be specified either as arguments to the
    constructor::

        opts = SolverOptions(norm_tol=1e-3, ...)

    or by changing the class attributes after creation::

        opts = SolverOptions()
        opts.['norm_tol'] = 1e-3

    Returns options class to be used as options in evolution solvers.

    The default can be changed by::

        qutip.settings.options.montecarlo['norm_tol'] = 1e-3

    Options
    -------

    norm_tol : float {1e-4}
        Tolerance used when finding wavefunction norm in mcsolve.
    norm_t_tol : float {1e-6}
        Tolerance used when finding wavefunction time in mcsolve.
    norm_steps : int {5}
        Max. number of steps used to find wavefunction norm to within norm_tol
        in mcsolve.
    mc_corr_eps : float {1e-10}
        Arbitrarily small value for eliminating any divide-by-zero errors in
        correlation calculations when using mcsolve.
    """
    options = {
        # Tolerance for wavefunction norm (mcsolve only)
        "norm_tol": 1e-4,
        # Tolerance for collapse time precision (mcsolve only)
        "norm_t_tol": 1e-6,
        # Max. number of steps taken to find wavefunction norm to within
        # norm_tol (mcsolve only)
        "norm_steps": 5,
        # small value in mc solver for computing correlations
        "mc_corr_eps": 1e-10,
    }


class Result():
    """Class for storing simulation results from any of the dynamics solvers.

    Attributes
    ----------

    solver : str
        Which solver was used [e.g., 'mesolve', 'mcsolve', 'brmesolve', ...]
    times : list/array
        Times at which simulation data was collected.
    expect : list/array
        Expectation values (if requested) for simulation.
    states : array
        State of the simulation (density matrix or ket) evaluated at ``times``.
    num_expect : int
        Number of expectation value operators in simulation.
    num_collapse : int
        Number of collapse operators in simualation.
    ntraj : int/list
        Number of trajectories (for stochastic solvers). A list indicates
        that averaging of expectation values was done over a subset of total
        number of trajectories.
    col_times : list
        Times at which state collpase occurred. Only for Monte Carlo solver.
    col_which : list
        Which collapse operator was responsible for each collapse in
        ``col_times``. Only for Monte Carlo solver.

    """
    def __init__(self):
        self.solver = None
        self.times = None
        self.states = []
        self.expect = []
        self.num_expect = 0
        self.num_collapse = 0
        self.ntraj = None
        self.seeds = None
        self.col_times = None
        self.col_which = None

    def __str__(self):
        s = "Result object "
        if self.solver:
            s += "with " + self.solver + " data.\n"
        else:
            s += "missing solver information.\n"
        s += "-" * (len(s) - 1) + "\n"
        if self.states is not None and len(self.states) > 0:
            s += "states = True\n"
        elif self.expect is not None and len(self.expect) > 0:
            s += "expect = True\nnum_expect = " + str(self.num_expect) + ", "
        else:
            s += "states = True, expect = True\n" + \
                "num_expect = " + str(self.num_expect) + ", "
        s += "num_collapse = " + str(self.num_collapse)
        if self.solver == 'mcsolve':
            s += ", ntraj = " + str(self.ntraj)
        return s

    def __repr__(self):
        return self.__str__()

    def __getstate__(self):
        # defines what happens when Qobj object gets pickled
        self.__dict__.update({'qutip_version': __version__[:5]})
        return self.__dict__

    def __setstate__(self, state):
        # defines what happens when loading a pickled Qobj
        if 'qutip_version' in state.keys():
            del state['qutip_version']
        (self.__dict__).update(state)


# %%%%%%%%%%% remove ?
class SolverConfiguration():
    def __init__(self):

        self.cgen_num = 0

        self.reset()

    def reset(self):
        # General stuff
        self.tlist = None       # evaluations times
        self.ntraj = None       # number / list of trajectories
        self.options = None     # options for solvers
        self.norm_tol = None    # tolerance for wavefunction norm
        self.norm_steps = None  # max. number of steps to take in finding
        # Initial state stuff
        self.psi0 = None        # initial state
        self.psi0_dims = None   # initial state dims
        self.psi0_shape = None  # initial state shape

        # flags for setting time-dependence, collapse ops, and number of times
        # codegen has been run
        self.cflag = 0     # Flag signaling collapse operators
        self.tflag = 0     # Flag signaling time-dependent problem

        self.soft_reset()

    def soft_reset(self):
        # Hamiltonian stuff
        self.h_td_inds = []  # indicies of time-dependent Hamiltonian operators
        self.h_tdterms = []  # List of td strs and funcs
        self.h_data = None   # List of sparse matrix data
        self.h_ind = None    # List of sparse matrix indices
        self.h_ptr = None    # List of sparse matrix ptrs

        # Expectation operator stuff
        self.e_num = 0        # number of expect ops
        self.e_ops_data = []  # expect op data
        self.e_ops_ind = []   # expect op indices
        self.e_ops_ptr = []   # expect op indptrs
        self.e_ops_isherm = []  # expect op isherm

        # Collapse operator stuff
        self.c_num = 0          # number of collapse ops
        self.c_const_inds = []  # indicies of constant collapse operators
        self.c_td_inds = []     # indicies of time-dependent collapse operators
        self.c_ops_data = []    # collapse op data
        self.c_ops_ind = []     # collapse op indices
        self.c_ops_ptr = []     # collapse op indptrs
        self.c_args = []        # store args for time-dependent collapse func.

        # Norm collapse operator stuff
        self.n_ops_data = []  # norm collapse op data
        self.n_ops_ind = []   # norm collapse op indices
        self.n_ops_ptr = []   # norm collapse op indptrs

        # holds executable strings for time-dependent collapse evaluation
        self.col_expect_code = None
        self.col_spmv_code = None

        # hold stuff for function list based time dependence
        self.h_td_inds = []
        self.h_td_data = []
        self.h_td_ind = []
        self.h_td_ptr = []
        self.h_funcs = None
        self.h_func_args = None
        self.c_funcs = None
        self.c_func_args = None

        # time-dependent (TD) function stuff
        self.tdfunc = None     # Placeholder for TD RHS function.
        self.tdname = None     # Name of td .pyx file
        self.colspmv = None    # Placeholder for TD col-spmv function.
        self.colexpect = None  # Placeholder for TD col_expect function.
        self.string = None     # Holds string of variables passed to td solver


def _format_time(t, tt=None, ttt=None):
    time_str = str(datetime.timedelta(seconds=t))
    if tt is not None and ttt is not None:
        sect_percent = 100*t/tt
        solve_percent = 100*t/ttt
        time_str += " ({:03.2f}% section, {:03.2f}% total)".format(
                                            sect_percent, solve_percent)
    elif tt is not None:
        sect_percent = 100*t/tt
        time_str += " ({:03.2f}% section)".format(sect_percent)

    elif ttt is not None:
        solve_percent = 100*t/ttt
        time_str += " ({:03.2f}% total)".format(solve_percent)

    return time_str


class Stats:
    """
    Statistical information on the solver performance
    Statistics can be grouped into sections.
    If no section names are given in the the contructor, then all statistics
    will be added to one section 'main'

    Parameters
    ----------
    section_names : list
        list of keys that will be used as keys for the sections
        These keys will also be used as names for the sections
        The text in the output can be overidden by setting the header property
        of the section
        If no names are given then one section called 'main' is created

    Attributes
    ----------
    sections : OrderedDict of _StatsSection
        These are the sections that are created automatically on instantiation
        or added using add_section

    header : string
        Some text that will be used as the heading in the report
        By default there is None

    total_time : float
        Time in seconds for the solver to complete processing
        Can be None, meaning that total timing percentages will be reported
    """

    def __init__(self, section_names=None):
        self._def_section_name = 'main'
        self.sections = OrderedDict()
        self.total_time = None
        self.header = None
        if isinstance(section_names, list):
            c = 0
            for name in section_names:
                self.sections[name] = _StatsSection(name, self)
                if c == 0:
                    self._def_section_name = name
                c += 1

        else:
            self.sections[self._def_section_name] = \
                        _StatsSection(self._def_section_name)

    def _get_section(self, section):
        if section is None:
            return self.sections[self._def_section_name]
        elif isinstance(section, _StatsSection):
            return section
        else:
            sect = self.sections.get(section, None)
            if sect is None:
                raise ValueError("Unknown section {}".format(section))
            else:
                return sect

    def add_section(self, name):
        """
        Add another section with the given name

        Parameters
        ----------
        name : string
            will be used as key for sections dict
            will also be the header for the section

        Returns
        -------
        section : :class:`_StatsSection`
            The new section
        """
        sect = _StatsSection(name, self)
        self.sections[name] = sect
        return sect

    def add_count(self, key, value, section=None):
        """
        Add value to count. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be an integer

        Parameters
        ----------
        key : string
            key for the section.counts dictionary
            reusing a key will result in numerical addition of value

        value : int
            Initial value of the count, or added to an existing count

        section : string or :class:`_StatsSection`
            Section which to add the count to.
            If None given, the default (first) section will be used
        """

        self._get_section(section).add_count(key, value)

    def add_timing(self, key, value, section=None):
        """
        Add value to timing. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be a float, and given in seconds.

        Parameters
        ----------
        key : string
            key for the section.timings dictionary
            reusing a key will result in numerical addition of value

        value : int
            Initial value of the timing, or added to an existing timing

        section: string or `class` : _StatsSection
            Section which to add the timing to.
            If None given, the default (first) section will be used
        """
        self._get_section(section).add_timing(key, value)

    def add_message(self, key, value, section=None, sep=";"):
        """
        Add value to message. If key does not already exist in section then
        it is created with this value.
        If key already exists the value is added to the message
        The value will be converted to a string

        Parameters
        ----------
        key : string
            key for the section.messages dictionary
            reusing a key will result in concatenation of value

        value : int
            Initial value of the message, or added to an existing message

        sep : string
            Message will be prefixed with this string when concatenating

        section: string or `class` : _StatsSection
            Section which to add the message to.
            If None given, the default (first) section will be used
        """
        self._get_section(section).add_message(key, value, sep=sep)

    def set_total_time(self, value, section=None):
        """
        Sets the total time for the complete solve or for a specific section
        value is expected to be a float, and given in seconds

        Parameters
        ----------
        value : float
            Time in seconds to complete the solver section

        section : string or `class` : _StatsSection
            Section which to set the total_time for
            If None given, the total_time for complete solve is set
        """
        if not isinstance(value, float):
            try:
                value = float(value)
            except:
                raise TypeError("value is expected to be a float")

        if section is None:
            self.total_time = value
        else:
            sect = self._get_section(section)
            sect.total_time = value

    def report(self, output=sys.stdout):
        """
        Report the counts, timings and messages from the sections.
        Sections are reported in the order that the names were supplied
        in the constructor.
        The counts, timings and messages are reported in the order that they
        are added to the sections
        The output can be written to anything that supports a write method,
        e.g. a file or the console (default)
        The output is intended to in markdown format

        Parameters
        ----------
        output : stream
            file or console stream - anything that support write - where
            the output will be written
        """

        if not hasattr(output, 'write'):
            raise TypeError("output must have a write method")

        if self.header:
            output.write("{}\n{}\n".format(self.header,
                                     ("="*len(self.header))))
        for name, sect in self.sections.items():
            sect.report(output)

        if self.total_time is not None:
            output.write("\nSummary\n-------\n")
            output.write("{}\t solver total time\n".format(
                                            _format_time(self.total_time)))

    def clear(self):
        """
        Clear counts, timings and messages from all sections
        """
        for sect in self.sections.values():
            sect.clear()
        self.total_time = None


class _StatsSection(object):
    """
    Not intended to be directly instantiated
    This is the type for the SolverStats.sections values

    The method parameter descriptions are the same as for those the parent
    with the same method name

    Parameters
    ----------
    name : string
        key for the parent sections dictionary
        will also be used as the header

    parent : `class` :  SolverStats
        The container for all the sections

    Attributes
    ----------
    name : string
        key for the parent sections dictionary
        will also be used as the header

    parent : `class` :  SolverStats
        The container for all the sections

    header : string
        Used as heading for section in report

    counts : OrderedDict
        The integer type statistics for the stats section

    timings : OrderedDict
        The timing type statistics for the stats section
        Expected to contain float values representing values in seconds

    messages : OrderedDict
        Text type output to be reported

    total_time : float
        Total time for processing in the section
        Can be None, meaning that section timing percentages will be reported
    """
    def __init__(self, name, parent):
        self.parent = parent
        self.header = str(name)
        self.name = name
        self.counts = OrderedDict()
        self.timings = OrderedDict()
        self.messages = OrderedDict()
        self.total_time = None

    def add_count(self, key, value):
        """
        Add value to count. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the given value
        value is expected to be an integer
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except:
                raise TypeError("value is expected to be an integer")

        if key in self.counts:
            self.counts[key] += value
        else:
            self.counts[key] = value

    def add_timing(self, key, value):
        """
        Add value to timing. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be a float, and given in seconds.
        """
        if not isinstance(value, float):
            try:
                value = float(value)
            except:
                raise TypeError("value is expected to be a float")

        if key in self.timings:
            self.timings[key] += value
        else:
            self.timings[key] = value

    def add_message(self, key, value, sep=";"):
        """
        Add value to message. If key does not already exist in section then
        it is created with this value.
        If key already exists the value is added to the message
        The value will be converted to a string
        """
        value = str(value)

        if key in self.messages:
            if sep is not None:
                try:
                    value = sep + value
                except:
                    TypeError("It is not possible to concatenate the value "
                              "with the given seperator")
            self.messages[key] += value
        else:
            self.messages[key] = value

    def report(self, output=sys.stdout):
        """
        Report the counts, timings and messages for this section.
        Note the percentage of the section and solver total times will be
        given if the parent and or section total_time is set
        """
        if self.header:
            output.write("\n{}\n{}\n".format(self.header,
                                     ("-"*len(self.header))))

        # TODO: Make the timings and counts ouput in a table format
        #       Generally make more pretty

        # Report timings
        try:
            ttt = self.parent.total_time
        except:
            ttt = None

        tt = self.total_time

        output.write("### Timings:\n")
        for key, value in self.timings.items():
            l = " - {}\t{}\n".format(_format_time(value, tt, ttt), key)
            output.write(l)
        if tt is not None:
            output.write(" - {}\t{} total time\n".format(_format_time(tt),
                                                     self.name))

        # Report counts
        output.write("### Counts:\n")
        for key, value in self.counts.items():
            l = " - {}\t{}\n".format(value, key)
            output.write(l)

        # Report messages
        output.write("### Messages:\n")
        for key, value in self.messages.items():
            l = " - {}:\t{}\n".format(key, value)
            output.write(l)


    def clear(self):
        """
        Clear counts, timings and messages from this section
        """
        self.counts.clear()
        self.timings.clear()
        self.messages.clear()
        self.total_time = None


def _solver_safety_check(H, state=None, c_ops=[], e_ops=[], args={}):
    # Input is std Qobj (Hamiltonian or Liouvillian)
    if isinstance(H, Qobj):
        Hdims = H.dims
        Htype = H.type
        _structure_check(Hdims, Htype, state)
    # Input H is function
    elif isinstance(H, (FunctionType, BuiltinFunctionType)):
        Hdims = H(0,args).dims
        Htype = H(0,args).type
        _structure_check(Hdims, Htype, state)
    # Input is td-list
    elif isinstance(H, list):
        if isinstance(H[0], Qobj):
            Hdims = H[0].dims
            Htype = H[0].type
        elif isinstance(H[0], list):
            Hdims = H[0][0].dims
            Htype = H[0][0].type
        elif isinstance(H[0], (FunctionType, BuiltinFunctionType)):
            Hdims = H[0](0,args).dims
            Htype = H[0](0,args).type
        else:
            raise Exception('Invalid td-list element.')
        # Check all operators in list
        for ii in range(len(H)):
            if isinstance(H[ii], Qobj):
                _temp_dims = H[ii].dims
                _temp_type = H[ii].type
            elif isinstance(H[ii], list):
                _temp_dims = H[ii][0].dims
                _temp_type = H[ii][0].type
            elif isinstance(H[ii], (FunctionType, BuiltinFunctionType)):
                _temp_dims = H[ii](0,args).dims
                _temp_type = H[ii](0,args).type
            else:
                raise Exception('Invalid td-list element.')
            _structure_check(_temp_dims,_temp_type,state)

    else:
        raise Exception('Invalid time-dependent format.')


    for ii in range(len(c_ops)):
        do_tests = True
        if isinstance(c_ops[ii], Qobj):
            _temp_state = c_ops[ii]
        elif isinstance(c_ops[ii], list):
            if isinstance(c_ops[ii][0], Qobj):
                _temp_state = c_ops[ii][0]
            elif isinstance(c_ops[ii][0], tuple):
                do_tests = False
                for kk in range(len(c_ops[ii][0])):
                    _temp_state = c_ops[ii][0][kk]
                    _structure_check(Hdims, Htype, _temp_state)
        else:
            raise Exception('Invalid td-list element.')
        if do_tests:
            _structure_check(Hdims, Htype, _temp_state)

    if isinstance(e_ops, list):
        for ii in range(len(e_ops)):
            if isinstance(e_ops[ii], Qobj):
                _temp_state = e_ops[ii]
            elif isinstance(e_ops[ii], list):
                _temp_state = e_ops[ii][0]
            else:
                raise Exception('Invalid td-list element.')
            _structure_check(Hdims,Htype,_temp_state)
    elif isinstance(e_ops, FunctionType):
        pass
    else:
        raise Exception('Invalid e_ops specification.')


def _structure_check(Hdims, Htype, state):
    if state is not None:
        # Input state is a ket vector
        if state.type == 'ket':
            # Input is Hamiltonian
            if Htype == 'oper':
                if Hdims[1] != state.dims[0]:
                    raise Exception('Input operator and ket do not '
                                    'share same structure.')
            # Input is super and state is ket
            elif Htype == 'super':
                if Hdims[1][1] != state.dims[0]:
                    raise Exception('Input operator and ket do not '
                                    'share same structure.')
            else:
                raise Exception('Invalid input operator.')
        # Input state is a density matrix
        elif state.type == 'oper':
            # Input is Hamiltonian and state is density matrix
            if Htype == 'oper':
                if Hdims[1] != state.dims[0]:
                    raise Exception('Input operators do not '
                                    'share same structure.')
            # Input is super op. and state is density matrix
            elif Htype == 'super':
                if Hdims[1] != state.dims:
                    raise Exception('Input operators do not '
                                    'share same structure.')


#
# create a global instance of the SolverConfiguration class
#
config = SolverConfiguration()

# for backwards compatibility
# Odeoptions = Options
# Odedata = Result
