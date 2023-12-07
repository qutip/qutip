"""Time-dependent Quantum Object (Qobj) class.
"""
__all__ = ['QobjEvo']

from qutip.qobj import Qobj
import qutip.settings as qset
from qutip.interpolate import Cubic_Spline
from scipy.interpolate import CubicSpline, interp1d
from functools import partial
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.qobjevo_codegen import (_compile_str_single, _compiled_coeffs,
                                   _compiled_coeffs_python)
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi,
                                 spmv)
from qutip.cy.cqobjevo import (CQobjCte, CQobjCteDense, CQobjEvoTd,
                               CQobjEvoTdMatched, CQobjEvoTdDense)
from qutip.cy.cqobjevo_factor import (InterCoeffT, InterCoeffCte,
                                      InterpolateCoeff, StrCoeff,
                                      StepCoeffCte, StepCoeffT)
import sys
import scipy
import os
from re import sub

if qset.has_openmp:
    from qutip.cy.openmp.cqobjevo_omp import (CQobjCteOmp, CQobjEvoTdOmp,
                                              CQobjEvoTdMatchedOmp)

safePickle = [False]
if sys.platform == 'win32':
    safePickle[0] = True


if qset.has_cython:
    import cython
    use_cython = [True]
else:
    use_cython = [False]


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


str_env = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "pi": np.pi,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "erf": scipy.special.erf,
    "zerf": scipy.special.erf,
    "sqrt": np.sqrt,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
    "abs": np.abs,
    "norm": lambda x: np.abs(x)**2,
    "arg": np.angle,
    "proj": proj,
    "np": np,
    "spe": scipy.special}


class _file_list:
    """
    Contain temp a list .pyx to clean
    """

    def __init__(self):
        self.files = []

    def add(self, file_):
        self.files += [file_]

    def clean(self):
        to_del = []
        for i, file_ in enumerate(self.files):
            try:
                os.remove(file_)
                to_del.append(i)
            except Exception:
                if not os.path.isfile(file_):
                    to_del.append(i)

        for i in to_del[::-1]:
            del self.files[i]

    def __del__(self):
        self.clean()


coeff_files = _file_list()


class _StrWrapper:
    def __init__(self, code):
        self.code = "_out = " + code

    def __call__(self, t, args={}):
        env = {"t": t}
        env.update(args)
        exec(self.code, str_env, env)
        return env["_out"]


class _CubicSplineWrapper:
    # Using scipy's CubicSpline since Qutip's one
    # only accept linearly distributed tlist
    def __init__(self, tlist, coeff, args=None):
        self.coeff = coeff
        self.tlist = tlist
        try:
            use_step_func = args["_step_func_coeff"]
        except KeyError:
            use_step_func = 0
        if use_step_func:
            self.func = interp1d(
                self.tlist, self.coeff, kind="previous",
                bounds_error=False, fill_value=0.)
        else:
            self.func = CubicSpline(self.tlist, self.coeff)

    def __call__(self, t, args={}):
        return self.func([t])[0]


class _StateAsArgs:
    # old with state (f(t, psi, args)) to new (args["state"] = psi)
    def __init__(self, coeff_func):
        self.coeff_func = coeff_func

    def __call__(self, t, args={}):
        return self.coeff_func(t, args["_state_vec"], args)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class StateArgs:
    """Object to indicate to use the state in args outside solver.
    args[key] = StateArgs(type, op)
    """

    def __init__(self, type="Qobj", op=None):
        self.dyn_args = (type, op)

    def __call__(self):
        return self.dyn_args


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# object for each time dependent element of the QobjEvo
# qobj : the Qobj of element ([*Qobj*, f])
# get_coeff : a callable that take (t, args) and return the coeff at that t
# coeff : The coeff as a string, array or function as provided by the user.
# type : flag for the type of coeff
class EvoElement:
    """
    Internal type used to represent the time-dependent parts of a
    :class:`~qutip.QobjEvo`.

    Availables "types" are

    1. function
    2. string
    3. ``np.ndarray``
    4. :class:`.Cubic_Spline`
    """

    def __init__(self, qobj, get_coeff, coeff, type):
        self.qobj = qobj
        self.get_coeff = get_coeff
        self.coeff = coeff
        self.type = type

    @classmethod
    def make(cls, list_):
        return cls(*list_)

    def __getitem__(self, i):
        if i == 0:
            return self.qobj
        if i == 1:
            return self.get_coeff
        if i == 2:
            return self.coeff
        if i == 3:
            return self.type


class QobjEvo:
    """
    A class for representing time-dependent quantum objects, such as quantum
    operators and states.

    Basic math operations are defined:

    - ``+``, ``-`` : :class:`~qutip.QobjEvo`, :class:`~qutip.Qobj`, scalars.
    - ``*``: :class:`~qutip.Qobj`, C number
    - ``/`` : C number

    This object is constructed by passing a list of :obj:`~qutip.Qobj`
    instances, each of which *may* have an associated scalar time dependence.
    The list is summed to produce the final result.  In other words, if an
    instance of this class is :math:`Q(t)`, then it is constructed from a set
    of constant:obj:`~qutip.Qobj` :math:`\\{Q_k\\}` and time-dependent scalars
    :math:`f_k(t)` by

    .. math::

        Q(t) = \\sum_k f_k(t) Q_k

    If a scalar :math:`f_k(t)` is not passed with a given :obj:`~qutip.Qobj`,
    then that term is assumed to be constant.  The next section contains more
    detail on the allowed forms of the constants, and gives several examples
    for how to build instances of this class.

    **Time-dependence formats**

    There are three major formats for specifying a time-dependent scalar:

    - Python function
    - string
    - array

    For function format, the function signature must be
    ``f(t: float, args: dict) -> complex``, for example

    .. code-block:: python

        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        def f2_t(t, args):
            return np.cos(t * args["w2"])

        H = QobjEvo([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1., "w2":2.})

    For string-based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:

    .. code-block::

        pi   exp   log   log10
        erf  zerf  norm  proj
        real imag conj abs arg
        sin  sinh  asin  asinh
        cos  cosh  acos  acosh
        tan  tanh  atan  atanh
        numpy as np
        scipy.special as spe

    A couple more simple examples:

    .. code-block:: python

        H = QobjEvo([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})

    For numpy array format, the array must be an 1d of dtype ``np.float64`` or
    ``np.complex128``.  A list of times (``np.float64``) at which the
    coeffients must be given as ``tlist``.  The coeffients array must have the
    same length as the tlist.  The times of the tlist do not need to be
    equidistant, but must be sorted.  By default, a cubic spline interpolation
    will be used for the coefficient at time t.  If the coefficients are to be
    treated as step functions, use the arguments
    ``args = {"_step_func_coeff": True}``.  Examples of array-format usage are:

    .. code-block:: python

        tlist = np.logspace(-5,0,100)
        H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    Mixing time formats is allowed.  It is not possible to create a single
    ``QobjEvo`` that contains different ``tlist`` values, however.

    **Passing arguments**

    ``args`` is a dict of (name: object). The name must be a valid Python
    identifier string, and in general the object can be any type that is
    supported by the code to be compiled in the string.

    There are some "magic" names that can be specified, whose objects will be
    overwritten when used within :func:`.sesolve`, :func:`.mesolve` and
    :func:`.mcsolve`.  This allows access to the solvers' internal states, and
    they are updated at every call.  The initial values of these dictionary
    elements is unimportant.  The magic names available are:

    - ``"state"``: the current state as a :class:`~qutip.Qobj`
    - ``"state_vec"``: the current state as a column-stacked 1D ``np.ndarray``
    - ``"state_mat"``: the current state as a 2D ``np.ndarray``
    - ``"expect_op_<n>"``: the current expectation value of the element
      ``e_ops[n]``, which is an argument to the solvers.  Replace ``<n>`` with
      an integer literal, e.g. ``"expect_op_0"``.  This will be either real- or
      complex-valued, depending on whether the state and operator are both
      Hermitian or not.
    - ``"collapse"``: (:func:`.mcsolve` only) a list of the collapses that have
      occurred during the evolution.  Each element of the list is a 2-tuple
      ``(time: float, which: int)``, where ``time`` is the time this collapse
      happened, and ``which`` is an integer indexing the ``c_ops`` argument to
      :func:`.mcsolve`.

    Parameters
    ----------
    Q_object : list, :class:`~qutip.Qobj` or :class:`~qutip.QobjEvo`
        The time-dependent description of the quantum object.  This is of the
        same format as the first parameter to the general ODE solvers; in
        general, it is a list of ``[Qobj, time_dependence]`` pairs that are
        summed to make the whole object.  The ``time_dependence`` can be any of
        the formats discussed in the previous section.  If a particular term
        has no time-dependence, then you should just give the ``Qobj`` instead
        of the 2-element list.

    args : dict, optional
        Mapping of ``{str: object}``, discussed in greater detail above.  The
        strings can be any valid Python identifier, and the objects are of the
        consumable types.  See the previous section for details on the "magic"
        names used to access solver internals.

    tlist : array_like, optional
        List of the times any numpy-array coefficients describe.  This is used
        only in at least one of the time dependences in ``Q_object`` is given
        in Numpy-array format.  The times must be sorted, but need not be
        equidistant.  Values inbetween will be interpolated.


    Attributes
    ----------
    cte : :class:`~qutip.Qobj`
        Constant part of the QobjEvo.

    ops : list of :class:`.EvoElement`
        Internal representation of the time-dependence structure of the
        elements.

    args : dict
        The current value of the ``args`` dictionary passed into the
        constructor.

    dynamics_args : list
        Names of the dynamic arguments that the solvers will generate.  These
        are the magic names that were found in the ``args`` parameter.

    tlist : array_like
        List of times at which the numpy-array coefficients are applied.

    compiled : str
        A string representing the properties of the low-level Cython class
        backing this object (may be empty).

    compiled_qobjevo : ``CQobjCte`` or ``CQobjEvoTd``
        Cython version of the QobjEvo.

    coeff_get : callable
        Object called to obtain a list of all the coefficients at a particular
        time.

    coeff_files : list
        Runtime created files to delete with the instance.

    dummy_cte : bool
        Is self.cte an empty Qobj

    const : bool
        Indicates if quantum object is constant

    type : {"cte", "string", "func", "array", "spline", "mixed_callable", \
"mixed_compilable"}
        Information about the type of coefficients used in the entire object.

    num_obj : int
        Number of :obj:`~qutip.Qobj` in the QobjEvo.

    use_cython : bool
        Flag to compile string to Cython or Python

    safePickle : bool
        Flag to not share pointers between thread.
    """

    def __init__(self, Q_object=[], args={}, copy=True,
                 tlist=None, state0=None, e_ops=[]):
        if isinstance(Q_object, QobjEvo):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args)
                for i, dargs in enumerate(self.dynamics_args):
                    e_int = dargs[1] == "expect" and isinstance(dargs[2], int)
                    if e_ops and e_int:
                        self.dynamics_args[i] = (dargs[0], "expect",
                                                 e_ops[dargs[2]])
                if state0 is not None:
                    self._dynamics_args_update(0., state0)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args.copy()
        self.dynamics_args = []
        self.cte = None
        self.tlist = np.asarray(tlist) if tlist is not None else None
        self.compiled = ""
        self.compiled_qobjevo = None
        self.coeff_get = None
        self.type = "none"
        self.omp = 0
        self.coeff_files = []
        self.use_cython = use_cython[0]
        self.safePickle = safePickle[0]

        # Attempt to determine if a 2-element list is a single, time-dependent
        # operator, or a list with 2 possibly time-dependent elements.
        if isinstance(Q_object, list) and len(Q_object) == 2:
            try:
                # Test if parsing succeeds on this as a single element.
                self._td_op_type(Q_object)
                Q_object = [Q_object]
            except (TypeError, ValueError):
                pass

        op_type = self._td_format_check(Q_object)
        self.ops = []

        if isinstance(op_type, int):
            if op_type == 0:
                self.cte = Q_object
                self.const = True
                self.type = "cte"
            elif op_type == 1:
                raise TypeError("The Qobj must not already be a function")
            elif op_type == -1:
                pass
        else:
            op_type_count = [0, 0, 0, 0]
            for type_, op in zip(op_type, Q_object):
                if type_ == 0:
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                elif type_ == 1:
                    op_type_count[0] += 1
                    self.ops.append(EvoElement(op[0], op[1], op[1], "func"))
                elif type_ == 2:
                    op_type_count[1] += 1
                    self.ops.append(EvoElement(op[0], _StrWrapper(op[1]),
                                    op[1], "string"))
                elif type_ == 3:
                    op_type_count[2] += 1
                    self.ops.append(EvoElement(
                        op[0],
                        _CubicSplineWrapper(tlist, op[1], args=self.args),
                        op[1].copy(), "array"))
                elif type_ == 4:
                    op_type_count[3] += 1
                    self.ops.append(EvoElement(op[0], op[1], op[1], "spline"))

            nops = sum(op_type_count)
            if all([op_t == 0 for op_t in op_type]):
                self.type = "cte"
            elif op_type_count[0] == nops:
                self.type = "func"
            elif op_type_count[1] == nops:
                self.type = "string"
            elif op_type_count[2] == nops:
                self.type = "array"
            elif op_type_count[3] == nops:
                self.type = "spline"
            elif op_type_count[0]:
                self.type = "mixed_callable"
            else:
                self.type = "mixed_compilable"

            try:
                if not self.cte:
                    self.cte = self.ops[0].qobj
                    # test is all qobj are compatible (shape, dims)
                    for op in self.ops[1:]:
                        self.cte += op.qobj
                    self.cte *= 0.
                    self.dummy_cte = True
                else:
                    cte_copy = self.cte.copy()
                    # test is all qobj are compatible (shape, dims)
                    for op in self.ops:
                        cte_copy += op.qobj
            except Exception as e:
                raise TypeError("Qobj not compatible.") from e

            if not self.ops:
                self.const = True
        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)
        self._args_checks()
        if e_ops:
            for i, dargs in enumerate(self.dynamics_args):
                if dargs[1] == "expect" and isinstance(dargs[2], int):
                    self.dynamics_args[i] = (dargs[0], "expect",
                                             QobjEvo(e_ops[dargs[2]]))
        if state0 is not None:
            self._dynamics_args_update(0., state0)

    def _td_format_check(self, Q_object):
        if isinstance(Q_object, Qobj):
            return 0
        if isinstance(Q_object, (FunctionType, BuiltinFunctionType, partial)):
            return 1
        if isinstance(Q_object, list):
            return [self._td_op_type(element) for element in Q_object] or -1
        raise TypeError("Incorrect Q_object specification")

    def _td_op_type(self, element):
        if isinstance(element, Qobj):
            return 0
        try:
            op, td = element
        except (TypeError, ValueError) as exc:
            raise TypeError("Incorrect Q_object specification") from exc
        if (not isinstance(op, Qobj)) or isinstance(td, Qobj):
            # Qobj is itself callable, so we need an extra check to make sure
            # that we don't have a two-element list where both are Qobj.
            raise TypeError("Incorrect Q_object specification")
        if isinstance(td, Cubic_Spline):
            out = 4
        elif callable(td):
            out = 1
        elif isinstance(td, str):
            out = 2
        elif isinstance(td, np.ndarray):
            if self.tlist is None or td.shape != self.tlist.shape:
                raise ValueError("Time lists are not compatible")
            out = 3
        else:
            raise TypeError("Incorrect Q_object specification")
        return out

    def _args_checks(self):
        statedims = [self.cte.dims[1], [1]]
        for key in self.args:
            if key == "state" or key == "state_qobj":
                self.dynamics_args += [(key, "Qobj", None)]
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims)

            if key == "state_mat":
                self.dynamics_args += [("state_mat", "mat", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full()
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full()

            if key == "state_vec":
                self.dynamics_args += [("state_vec", "vec", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full().ravel("F")
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full().ravel("F")

            if key.startswith("expect_op_"):
                e_op_num = int(key[10:])
                self.dynamics_args += [(key, "expect", e_op_num)]

            if isinstance(self.args[key], StateArgs):
                self.dynamics_args += [(key, *self.args[key]())]
                self.args[key] = 0.

    def _check_old_with_state(self):
        add_vec = False
        for op in self.ops:
            if op.type == "func":
                try:
                    op.get_coeff(0., self.args)
                except TypeError as e:
                    nfunc = _StateAsArgs(self.coeff)
                    op = EvoElement((op.qobj, nfunc, nfunc, "func"))
                    add_vec = True
        if add_vec:
            self.dynamics_args += [("_state_vec", "vec", None)]

    def __del__(self):
        for file_ in self.coeff_files:
            try:
                os.remove(file_)
            except:
                pass

    def __call__(self, t, data=False, state=None, args={}):
        """
        Return a single :obj:`~qutip.Qobj` at the given time ``t``.
        """
        try:
            t = float(t)
        except Exception as e:
            raise TypeError("Time must be a real scalar.") from e

        if state is not None:
            self._dynamics_args_update(t, state)

        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            old_args = self.args.copy()
            old_compiled = self.compiled
            self.compiled = False
            self.args.update(args)
            op_t = self.__call__(t, data=data)
            self.args = old_args
            self.compiled = old_compiled
        elif self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled and self.compiled.split()[0] != "dense":
            op_t = self.compiled_qobjevo.call(t, data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part.qobj.data * part.get_coeff(t, self.args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part.qobj * part.get_coeff(t, self.args)

        return op_t

    def _dynamics_args_update(self, t, state):
        if isinstance(state, Qobj):
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.full().ravel("F")
                elif what == "mat":
                    self.args[name] = state.full()
                elif what == "Qobj":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)

        elif isinstance(state, np.ndarray) and state.ndim == 1:
            s1 = self.cte.shape[1]
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[0] == s1 and self.cte.issuper:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=self.cte.dims[1])
                elif state.shape[0] == s1:
                    mat = state.reshape((-1, 1))
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(
                            mat, dims=[self.cte.dims[1], [1]])
                elif state.shape[0] == s1*s1:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1],
                                                          self.cte.dims[1]])

        elif isinstance(state, np.ndarray) and state.ndim == 2:
            s1 = self.cte.shape[1]
            new_l = int(np.sqrt(s1))
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.ravel("F")
                elif what == "mat":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[1] == 1:
                    self.args[name] = Qobj(state, dims=[self.cte.dims[1], [1]])
                elif state.shape[1] == s1:
                    self.args[name] = Qobj(state, dims=self.cte.dims)
                else:
                    self.args[name] = Qobj(state)

        else:
            raise TypeError("state must be a Qobj or np.ndarray")

    def copy(self):
        """Return a copy of this object."""
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.num_obj = self.num_obj
        new.type = self.type
        new.compiled = False
        new.compiled_qobjevo = None
        new.coeff_get = None
        new.coeff_files = []
        new.use_cython = self.use_cython
        new.safePickle = self.safePickle

        for op in self.ops:
            if op.type == "array":
                new_coeff = op.coeff.copy()
            else:
                new_coeff = op.coeff
            new.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                      new_coeff, op.type))

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.num_obj = other.num_obj
        self.type = other.type
        self.compiled = ""
        self.compiled_qobjevo = None
        self.coeff_get = None
        self.ops = []
        self.coeff_files = []
        self.use_cython = other.use_cython
        self.safePickle = other.safePickle

        for op in other.ops:
            if op.type == "array":
                new_coeff = op.coeff.copy()
            else:
                new_coeff = op.coeff
            self.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                       new_coeff, op.type))

    def arguments(self, new_args):
        """
        Update the scoped variables that were passed as ``args`` to new values.
        """
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        # remove dynamics_args that are to be refreshed
        self.dynamics_args = [dargs for dargs in self.dynamics_args
                              if dargs[0] not in new_args]
        self.args.update(new_args)
        self._args_checks()
        if self.compiled and self.compiled.split()[2] != "cte":
            if isinstance(self.coeff_get, StrCoeff):
                self.coeff_get.set_args(self.args)
                self.coeff_get._set_dyn_args(self.dynamics_args)
            elif isinstance(self.coeff_get, _UnitedFuncCaller):
                self.coeff_get.set_args(self.args, self.dynamics_args)

    def solver_set_args(self, new_args, state, e_ops):
        self.dynamics_args = []
        self.args.update(new_args)
        self._args_checks()
        for i, dargs in enumerate(self.dynamics_args):
            if dargs[1] == "expect" and isinstance(dargs[2], int):
                self.dynamics_args[i] = (dargs[0], "expect",
                                         QobjEvo(e_ops[dargs[2]]))
                if self.compiled:
                    self.dynamics_args[i][2].compile()
        self._dynamics_args_update(0., state)
        if self.compiled and self.compiled.split()[2] != "cte":
            if isinstance(self.coeff_get, StrCoeff):
                self.coeff_get.set_args(self.args)
                self.coeff_get._set_dyn_args(self.dynamics_args)
            elif isinstance(self.coeff_get, _UnitedFuncCaller):
                self.coeff_get.set_args(self.args, self.dynamics_args)

    def to_list(self):
        """
        Return this operator in the list-like form used to initialised it, like
        can be passed to :func:`~mesolve`.
        """
        list_qobj = []
        if not self.dummy_cte:
            list_qobj.append(self.cte)
        for op in self.ops:
            list_qobj.append([op.qobj, op.coeff])
        return list_qobj

    # Math function
    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, QobjEvo):
            self.cte += other.cte
            l = len(self.ops)
            for op in other.ops:
                if op.type == "array":
                    new_coeff = op.coeff.copy()
                else:
                    new_coeff = op.coeff
                self.ops.append(EvoElement(op.qobj.copy(), op.get_coeff,
                                           new_coeff, op.type))
                l += 1
            self.args.update(**other.args)
            self.dynamics_args += other.dynamics_args
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            if self.type != other.type:
                if self.type in ["func", "mixed_callable"] or \
                        other.type in ["func", "mixed_callable"]:
                    self.type = "mixed_callable"
                else:
                    self.type = "mixed_compilable"
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None

            if self.tlist is None:
                self.tlist = other.tlist
            else:
                if other.tlist is None:
                    pass
                elif len(other.tlist) != len(self.tlist) or \
                        other.tlist[-1] != self.tlist[-1]:
                    raise ValueError("Time lists are not compatible")
        else:
            self.cte += other
            self.dummy_cte = False

        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)
        self._reset_type()
        return self

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = -self.copy()
        res += other
        return res

    def __isub__(self, other):
        self += (-other)
        return self

    def __mul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):
        res = self.copy()
        if isinstance(other, Qobj):
            res.cte = other * res.cte
            for op in res.ops:
                op.qobj = other * op.qobj
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
            self.cte *= other
            for op in self.ops:
                op.qobj *= other
            return self
        if isinstance(other, QobjEvo):
            if other.const:
                self.cte *= other.cte
                for op in self.ops:
                    op.qobj *= other.cte
            elif self.const:
                cte = self.cte.copy()
                self = other.copy()
                self.cte = cte * self.cte
                for op in self.ops:
                    op.qobj = cte*op.qobj
            else:
                cte = self.cte.copy()
                self.cte *= other.cte
                new_terms = []
                old_ops = self.ops
                if not other.dummy_cte:
                    for op in old_ops:
                        new_terms.append(self._ops_mul_cte(op, other.cte, "R"))
                if not self.dummy_cte:
                    for op in other.ops:
                        new_terms.append(self._ops_mul_cte(op, cte, "L"))

                for op_left in old_ops:
                    for op_right in other.ops:
                        new_terms.append(self._ops_mul_(op_left,
                                                        op_right))
                self.ops = new_terms
                self.args.update(other.args)
                self.dynamics_args += other.dynamics_args
                self.dummy_cte = self.dummy_cte and other.dummy_cte
                self.num_obj = (len(self.ops) if
                                self.dummy_cte else len(self.ops) + 1)
            self._reset_type()
            return self
        return NotImplemented

    def __div__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            res = self.copy()
            res *= other**(-1)
            return res
        return NotImplemented

    def __idiv__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            self *= other**(-1)
            return self
        return NotImplemented

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.cte = -res.cte
        for op in res.ops:
            op.qobj = -op.qobj
        return res

    def _ops_mul_(self, opL, opR):
        new_f = _Prod(opL.get_coeff, opR.get_coeff)
        new_op = [opL.qobj*opR.qobj, new_f, None, 0]
        if opL.type == opR.type and opL.type == "string":
            new_op[2] = "(" + opL.coeff + ") * (" + opR.coeff + ")"
            new_op[3] = "string"
        elif opL[3] == opR[3] and opL[3] == "array":
            new_op[2] = opL[2]*opR[2]
            new_op[3] = "array"
        else:
            new_op[2] = new_f
            new_op[3] = "func"
            if self.type not in ["func", "mixed_callable"]:
                self.type = "mixed_callable"
        return EvoElement.make(new_op)

    def _ops_mul_cte(self, op, cte, side):
        new_op = [None, op.get_coeff, op.coeff, op.type]
        if side == "R":
            new_op[0] = op.qobj * cte
        if side == "L":
            new_op[0] = cte * op.qobj
        return EvoElement.make(new_op)

    # Transformations
    def trans(self):
        """Return the matrix transpose."""
        res = self.copy()
        res.cte = res.cte.trans()
        for op in res.ops:
            op.qobj = op.qobj.trans()
        return res

    def conj(self):
        """Return the matrix elementwise conjugation."""
        res = self.copy()
        res.cte = res.cte.conj()
        for op in res.ops:
            op.qobj = op.qobj.conj()
        res._f_conj()
        return res

    def dag(self):
        """Return the matrix conjugate-transpose (dagger)."""
        res = self.copy()
        res.cte = res.cte.dag()
        for op in res.ops:
            op.qobj = op.qobj.dag()
        res._f_conj()
        return res

    def _cdc(self):
        """Return ``a.dag * a``."""
        if not self.num_obj == 1:
            res = self.dag()
            res *= self
        else:
            res = self.copy()
            res.cte = res.cte.dag() * res.cte
            for op in res.ops:
                op.qobj = op.qobj.dag() * op.qobj
            res._f_norm2()
        return res

    # Unitary function of Qobj
    def tidyup(self, atol=None):
        """Removes small elements from this quantum object inplace."""
        self.cte = self.cte.tidyup(atol)
        for op in self.ops:
            op.qobj = op.qobj.tidyup(atol)
        return self

    def _compress_make_set(self):
        sets = []
        callable_flags = ["func", "spline"]
        for i, op1 in enumerate(self.ops):
            already_matched = False
            for _set in sets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.qobj == op2.qobj:
                        same_flag = op1.type == op2.type
                        callable_1 = op1.type in callable_flags
                        callable_2 = op2.type in callable_flags
                        if (same_flag or (callable_1 and callable_2)):
                            this_set.append(j+i+1)
                sets.append(this_set)

        fsets = []
        for i, op1 in enumerate(self.ops):
            already_matched = False
            for _set in fsets:
                already_matched = already_matched or i in _set
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.type != op2.type:
                        pass
                    elif op1.type == "array":
                        if np.allclose(op1.coeff, op2.coeff):
                            this_set.append(j+i+1)
                    else:
                        if op1.coeff is op2.coeff:
                            this_set.append(j+i+1)
                fsets.append(this_set)
        return sets, fsets

    def _compress_merge_qobj(self, sets):
        callable_flags = ["func", "spline"]
        new_ops = []
        for _set in sets:
            if len(_set) == 1:
                new_ops.append(self.ops[_set[0]])

            elif self.ops[_set[0]].type in callable_flags:
                new_op = [self.ops[_set[0]].qobj, None, None, "func"]
                fs = []
                for i in _set:
                    fs += [self.ops[i].get_coeff]
                new_op[1] = _Add(fs)
                new_op[2] = new_op[1]
                new_ops.append(EvoElement.make(new_op))

            elif self.ops[_set[0]].type == "string":
                new_op = [self.ops[_set[0]].qobj, None, None, "string"]
                new_str = "(" + self.ops[_set[0]].coeff + ")"
                for i in _set[1:]:
                    new_str += " + (" + self.ops[i].coeff + ")"
                new_op[1] = _StrWrapper(new_str)
                new_op[2] = new_str
                new_ops.append(EvoElement.make(new_op))

            elif self.ops[_set[0]].type == "array":
                new_op = [self.ops[_set[0]].qobj, None, None, "array"]
                new_array = (self.ops[_set[0]].coeff).copy()
                for i in _set[1:]:
                    new_array += self.ops[i].coeff
                new_op[2] = new_array
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_array, args=self.args)
                new_ops.append(EvoElement.make(new_op))

        self.ops = new_ops

    def _compress_merge_func(self, fsets):
        new_ops = []
        for _set in fsets:
            base = self.ops[_set[0]]
            new_op = [None, base.get_coeff, base.coeff, base.type]
            if len(_set) == 1:
                new_op[0] = base.qobj
            else:
                new_op[0] = base.qobj.copy()
                for i in _set[1:]:
                    new_op[0] += self.ops[i].qobj
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops

    def compress(self):
        """
        Merge together elements that share the same time-dependence, to reduce
        the number of matrix multiplications and additions that need to be done
        to evaluate this object.

        Modifies the object inplace.
        """
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        num_ops = len(self.ops)

        if N_sets < num_ops and N_fsets < num_ops:
            # Both could be better
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            if N_sets < N_fsets:
                self._compress_merge_qobj(sets)
            else:
                self._compress_merge_func(fsets)
            sets, fsets = self._compress_make_set()
            N_sets = len(sets)
            N_fsets = len(fsets)
            num_ops = len(self.ops)

        if N_sets < num_ops:
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            self._compress_merge_qobj(sets)
        elif N_fsets < num_ops:
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
            self._compress_merge_func(fsets)
        self._reset_type()

    def _reset_type(self):
        op_type_count = [0, 0, 0, 0]
        for op in self.ops:
            if op.type == "func":
                op_type_count[0] += 1
            elif op.type == "string":
                op_type_count[1] += 1
            elif op.type == "array":
                op_type_count[2] += 1
            elif op.type == "spline":
                op_type_count[3] += 1

        nops = sum(op_type_count)
        if not self.ops and self.dummy_cte is False:
            self.type = "cte"
        elif op_type_count[0] == nops:
            self.type = "func"
        elif op_type_count[1] == nops:
            self.type = "string"
        elif op_type_count[2] == nops:
            self.type = "array"
        elif op_type_count[3] == nops:
            self.type = "spline"
        elif op_type_count[0]:
            self.type = "mixed_callable"
        else:
            self.type = "mixed_compilable"

        self.num_obj = (len(self.ops) if self.dummy_cte else len(self.ops) + 1)

    def permute(self, order):
        """
        Permute the tensor structure of the underlying matrices into a new
        format.

        See Also
        --------
        Qobj.permute : the same operation on constant quantum objects.
        """
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op.qobj = op.qobj.permute(order)
        return res

    def apply(self, function, *args, **kw_args):
        """
        Apply the linear function ``function`` to every ``Qobj`` included in
        this time-dependent object, and return a new ``QobjEvo`` with the
        result.

        Any additional arguments or keyword arguments will be appended to every
        function call.
        """
        self.compiled = ""
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op.qobj = function(op.qobj, *args, **kw_args)
        return res

    def apply_decorator(self, function, *args,
                        str_mod=None, inplace_np=False, **kw_args):
        """
        Apply the given function to every time-dependent coefficient in the
        quantum object, and return a new object with the result.

        Any additional arguments and keyword arguments will be appended to the
        function calls.

        Parameters
        ----------
        function : callable
            ``(time_dependence, *args, **kwargs) -> time_dependence``.  Called
            on each time-dependent coefficient to produce a new coefficient.
            The additional arguments and keyword arguments are the ones given
            to this function.

        str_mod : list
            A 2-element list of strings, that will additionally wrap any string
            time-dependences.  An existing time-dependence string ``x`` will
            become ``str_mod[0] + x + str_mod[1]``.

        inplace_np : bool, default False
            Whether this function should modify Numpy arrays inplace, or be
            used like a regular decorator.  Some decorators create incorrect
            arrays as some transformations ``f'(t) = f(g(t))`` create a
            mismatch between the array and the associated time list.
        """
        res = self.copy()
        for op in res.ops:
            op.get_coeff = function(op.get_coeff, *args, **kw_args)
            if op.type == ["func", "spline"]:
                op.coeff = op.get_coeff
                op.type = "func"
            elif op.type == "string":
                if str_mod is None:
                    op.coeff = op.get_coeff
                    op.type = "func"
                else:
                    op.coeff = str_mod[0] + op.coeff + str_mod[1]
            elif op.type == "array":
                if inplace_np:
                    # keep the original function, change the array
                    def f(a):
                        return a
                    ff = function(f, *args, **kw_args)
                    for i, v in enumerate(op.coeff):
                        op.coeff[i] = ff(v)
                    op.get_coeff = _CubicSplineWrapper(
                        self.tlist, op.coeff, args=self.args)
                else:
                    op.coeff = op.get_coeff
                    op.type = "func"
        if self.type == "string" and str_mod is None:
            res.type = "mixed_callable"
        elif self.type == "array" and not inplace_np:
            res.type = "mixed_callable"
        elif self.type == "spline":
            res.type = "mixed_callable"
        elif self.type == "mixed_compilable":
            for op in res.ops:
                if op.type == "func":
                    res.type = "mixed_callable"
        return res

    def _f_norm2(self):
        self.compiled = ""
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Norm2(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = "norm(" + op.coeff + ")"
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = np.abs(op.coeff)**2
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_op[2], args=self.args)
            elif op.type == "spline":
                new_op[1] = _Norm2(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops
        return self

    def _f_conj(self):
        self.compiled = ""
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Conj(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = "conj(" + op.coeff + ")"
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = np.conj(op.coeff)
                new_op[1] = _CubicSplineWrapper(
                    self.tlist, new_op[2], args=self.args)
            elif op.type == "spline":
                new_op[1] = _Conj(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops
        return self

    def _shift(self):
        self.compiled = ""
        self.args.update({"_t0": 0})
        new_ops = []
        for op in self.ops:
            new_op = [op.qobj, None, None, op.type]
            if op.type == "func":
                new_op[1] = _Shift(op.get_coeff)
                new_op[2] = new_op[1]
            elif op.type == "string":
                new_op[2] = sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                                "(t+_t0)", " " + op.coeff + " ")
                new_op[1] = _StrWrapper(new_op[2])
            elif op.type == "array":
                new_op[2] = _Shift(op.get_coeff)
                new_op[1] = new_op[2]
                new_op[3] = "func"
                self.type = "mixed_callable"
            elif op.type == "spline":
                new_op[1] = _Shift(op.get_coeff)
                new_op[2] = new_op[1]
                new_op[3] = "func"
                self.type = "mixed_callable"
            new_ops.append(EvoElement.make(new_op))
        self.ops = new_ops
        return self

    def expect(self, t, state, herm=False):
        """
        Calculate the expectation value of this operator on the given
        (time-independent) state at a particular time.

        This is more efficient than ``expect(QobjEvo(t), state)``.

        Parameters
        ----------
        t : float
            The time to evaluate this operator at.

        state : Qobj or np.ndarray
            The state to take the expectation value around.

        herm : bool, default False
            Whether this operator and the state are both Hermitian.  If True,
            only the real part of the result will be returned.

        See Also
        --------
        expect : General-purpose expectation values.
        """
        if not isinstance(t, (int, float)):
            raise TypeError("Time must be a real scalar")
        if isinstance(state, Qobj):
            if self.cte.dims[1] == state.dims[0]:
                vec = state.full().ravel("F")
            elif self.cte.dims[1] == state.dims:
                vec = state.full().ravel("F")
            else:
                raise ValueError("Dimensions do not fit")
        elif isinstance(state, np.ndarray):
            vec = state.ravel("F")
        else:
            raise TypeError("The vector must be an array or Qobj")

        if vec.shape[0] == self.cte.shape[1]:
            if self.compiled:
                exp = self.compiled_qobjevo.expect(t, vec)
            elif self.cte.issuper:
                self._dynamics_args_update(t, state)
                exp = cy_expect_rho_vec(self.__call__(t, data=True), vec, 0)
            else:
                self._dynamics_args_update(t, state)
                exp = cy_expect_psi(self.__call__(t, data=True), vec, 0)
        elif vec.shape[0] == self.cte.shape[1]**2:
            if self.compiled:
                exp = self.compiled_qobjevo.overlap(t, vec)
            else:
                self._dynamics_args_update(t, state)
                exp = (self.__call__(t, data=True) *
                       vec.reshape((self.cte.shape[1],
                                    self.cte.shape[1])).T).trace()
        else:
            raise ValueError("The shapes do not match")

        if herm:
            return exp.real
        else:
            return exp

    def mul_vec(self, t, vec):
        """
        Multiply this object evaluated at time `t` by a vector.

        Parameters
        ----------
        t : float
            The time to evaluate this object at.

        vec : Qobj or np.ndarray
            The state-vector to multiply this object by.

        Returns
        -------
        vec: Qobj or np.ndarray
            The vector result in the same type as the input.
        """
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("Time must be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise ValueError("Dimensions do not fit")
            was_Qobj = True
            dims = vec.dims
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise ValueError(f"The vector must be 1d, but is {vec.ndim}d")
        if vec.shape[0] != self.cte.shape[1]:
            raise ValueError("The lengths do not match")

        if self.compiled:
            out = self.compiled_qobjevo.mul_vec(t, vec)
        else:
            self._dynamics_args_update(t, vec)
            out = spmv(self.__call__(t, data=True), vec)

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def mul_mat(self, t, mat):
        """
        Multiply this object evaluated at time `t` by a matrix (from the
        right).

        Parameters
        ----------
        t : float
            The time to evaluate this object at.

        mat : Qobj or np.ndarray
            The matrix that is multiplied by this object.

        Returns
        -------
        mat: Qobj or np.ndarray
            The matrix result in the same type as the input.
        """
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("Time must be a real scalar")
        if isinstance(mat, Qobj):
            if self.cte.dims[1] != mat.dims[0]:
                raise ValueError("Dimensions do not fit")
            was_Qobj = True
            dims = mat.dims
            mat = mat.full()
        if not isinstance(mat, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if mat.ndim != 2:
            raise ValueError(f"The matrix must be 2d, but is {mat.ndim}d")
        if mat.shape[0] != self.cte.shape[1]:
            raise ValueError("The lengths do not match")

        if self.compiled:
            out = self.compiled_qobjevo.mul_mat(t, mat)
        else:
            self._dynamics_args_update(t, mat)
            out = self.__call__(t, data=True) * mat

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        """
        Create an associated Cython object for faster usage.  This function is
        called automatically by the solvers.

        Parameters
        ----------
        code : bool, default False
            Return the code string generated by compilation of any strings.

        matched : bool, default False
            If True, the underlying sparse matrices used to represent each
            element of the type will have their structures unified.  This may
            include adding explicit zeros to sparse matrices, but can be faster
            in some cases due to not having to deal with repeated structural
            mismatches.

        dense : bool, default False
            Whether to swap to using dense matrices to back the data.

        omp : int, optional
            The number of OpenMP threads to use when doing matrix
            multiplications, if QuTiP was compiled with OpenMP.

        Returns
        -------
        compiled_str : str
            (Only if `code` was set to True).  The code-generated string of
            compiled calling code.
        """
        self.tidyup()
        Code = None
        if self.compiled:
            return
        for _, _, op in self.dynamics_args:
            if isinstance(op, QobjEvo):
                op.compile(code, matched, dense, omp)
        if not qset.has_openmp:
            omp = 0
        if omp:
            nnz = [self.cte.data.nnz]
            for part in self.ops:
                nnz += [part.qobj.data.nnz]
            if all(qset.openmp_thresh < nz for nz in nnz):
                omp = 0

        if self.const:
            if dense:
                self.compiled_qobjevo = CQobjCteDense()
                self.compiled = "dense single cte"
            elif omp:
                self.compiled_qobjevo = CQobjCteOmp()
                self.compiled = "csr omp cte"
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjCte()
                self.compiled = "csr single cte"
            self.compiled_qobjevo.set_data(self.cte)
        else:
            if matched:
                if omp:
                    self.compiled_qobjevo = CQobjEvoTdMatchedOmp()
                    self.compiled = "matched omp "
                    self.compiled_qobjevo.set_threads(omp)
                    self.omp = omp
                else:
                    self.compiled_qobjevo = CQobjEvoTdMatched()
                    self.compiled = "matched single "
            elif dense:
                self.compiled_qobjevo = CQobjEvoTdDense()
                self.compiled = "dense single "
            elif omp:
                self.compiled_qobjevo = CQobjEvoTdOmp()
                self.compiled = "csr omp "
                self.compiled_qobjevo.set_threads(omp)
                self.omp = omp
            else:
                self.compiled_qobjevo = CQobjEvoTd()
                self.compiled = "csr single "
            self.compiled_qobjevo.set_data(self.cte, self.ops)
            self.compiled_qobjevo.has_dyn_args(bool(self.dynamics_args))

            if self.type in ["func"]:
                # funclist = []
                # for part in self.ops:
                #    funclist.append(part.get_coeff)
                funclist = [part.get_coeff for part in self.ops]
                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args, self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)

            elif self.type in ["mixed_callable"] and self.use_cython:
                funclist = []
                for part in self.ops:
                    if isinstance(part.get_coeff, _StrWrapper):
                        get_coeff, file_ = _compile_str_single(
                            part.coeff,
                            self.args)
                        coeff_files.add(file_)
                        self.coeff_files.append(file_)
                        funclist.append(get_coeff)
                    else:
                        funclist.append(part.get_coeff)

                self.coeff_get = _UnitedFuncCaller(funclist, self.args,
                                                   self.dynamics_args,
                                                   self.cte)
                self.compiled += "pyfunc"
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
            elif self.type in ["mixed_callable"]:
                funclist = [part.get_coeff for part in self.ops]
                _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                    self.ops,
                    self.args,
                    self.dynamics_args,
                    self.tlist)
                coeff_files.add(file_)
                self.coeff_files.append(file_)
                self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                                  self.dynamics_args,
                                                  self.cte)
                self.compiled_qobjevo.set_factor(func=self.coeff_get)
                self.compiled += "pyfunc"
            elif self.type in ["string", "mixed_compilable"]:
                if self.use_cython:
                    # All factor can be compiled
                    self.coeff_get, Code, file_ = _compiled_coeffs(
                        self.ops,
                        self.args,
                        self.dynamics_args,
                        self.tlist)
                    coeff_files.add(file_)
                    self.coeff_files.append(file_)
                    self.compiled_qobjevo.set_factor(obj=self.coeff_get)
                    self.compiled += "cyfactor"
                else:
                    # All factor can be compiled
                    _UnitedStrCaller, Code, file_ = _compiled_coeffs_python(
                        self.ops,
                        self.args,
                        self.dynamics_args,
                        self.tlist)
                    coeff_files.add(file_)
                    self.coeff_files.append(file_)
                    funclist = [part.get_coeff for part in self.ops]
                    self.coeff_get = _UnitedStrCaller(funclist, self.args,
                                                      self.dynamics_args,
                                                      self.cte)
                    self.compiled_qobjevo.set_factor(func=self.coeff_get)
                    self.compiled += "pyfunc"

            elif self.type == "array":
                try:
                    use_step_func = self.args["_step_func_coeff"]
                except KeyError:
                    use_step_func = 0
                if np.allclose(np.diff(self.tlist),
                               self.tlist[1] - self.tlist[0]):
                    if use_step_func:
                        self.coeff_get = StepCoeffCte(
                            self.ops, None, self.tlist)
                    else:
                        self.coeff_get = InterCoeffCte(
                            self.ops, None, self.tlist)
                else:
                    if use_step_func:
                        self.coeff_get = StepCoeffT(
                            self.ops, None, self.tlist)
                    else:
                        self.coeff_get = InterCoeffT(
                            self.ops, None, self.tlist)
                self.compiled += "cyfactor"
                self.compiled_qobjevo.set_factor(obj=self.coeff_get)

            elif self.type == "spline":
                self.coeff_get = InterpolateCoeff(self.ops, None, None)
                self.compiled += "cyfactor"
                self.compiled_qobjevo.set_factor(obj=self.coeff_get)

            else:
                pass

            coeff_files.clean()
            if code:
                return Code

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            out.append(part.get_coeff(t, self.args))
        return out

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key != "compiled_qobjevo"}
        if self.compiled:
            return (_dict_, self.compiled_qobjevo.__getstate__())
        else:
            return (_dict_,)

    def __setstate__(self, state):
        self.__dict__ = state[0]
        self.compiled_qobjevo = None
        if self.compiled:
            mat_type, threading, td = self.compiled.split()
            if mat_type == "csr":
                if self.safePickle:
                    # __getstate__ and __setstate__ of compiled_qobjevo pass pointers
                    # In 'safe' mod, these pointers are not used.
                    if td == "cte":
                        if threading == "single":
                            self.compiled_qobjevo = CQobjCte()
                            self.compiled_qobjevo.set_data(self.cte)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjCteOmp()
                            self.compiled_qobjevo.set_data(self.cte)
                            self.compiled_qobjevo.set_threads(self.omp)
                    else:
                        # time dependence is pyfunc or cyfactor
                        if threading == "single":
                            self.compiled_qobjevo = CQobjEvoTd()
                            self.compiled_qobjevo.set_data(self.cte, self.ops)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjEvoTdOmp()
                            self.compiled_qobjevo.set_data(self.cte, self.ops)
                            self.compiled_qobjevo.set_threads(self.omp)

                        if td == "pyfunc":
                            self.compiled_qobjevo.set_factor(
                                func=self.coeff_get)
                        elif td == "cyfactor":
                            self.compiled_qobjevo.set_factor(
                                obj=self.coeff_get)
                else:
                    if td == "cte":
                        if threading == "single":
                            self.compiled_qobjevo = CQobjCte.__new__(CQobjCte)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjCteOmp.__new__(
                                CQobjCteOmp)
                            self.compiled_qobjevo.set_threads(self.omp)
                    else:
                        # time dependence is pyfunc or cyfactor
                        if threading == "single":
                            self.compiled_qobjevo = CQobjEvoTd.__new__(
                                CQobjEvoTd)
                        elif threading == "omp":
                            self.compiled_qobjevo = CQobjEvoTdOmp.__new__(
                                CQobjEvoTdOmp)
                            self.compiled_qobjevo.set_threads(self.omp)
                    self.compiled_qobjevo.__setstate__(state[1])

            elif mat_type == "dense":
                if td == "cte":
                    self.compiled_qobjevo = \
                        CQobjCteDense.__new__(CQobjCteDense)
                else:
                    CQobjEvoTdDense.__new__(CQobjEvoTdDense)
                self.compiled_qobjevo.__setstate__(state[1])

            elif mat_type == "matched":
                if threading == "single":
                    self.compiled_qobjevo = \
                        CQobjEvoTdMatched.__new__(CQobjEvoTdMatched)
                elif threading == "omp":
                    self.compiled_qobjevo = \
                        CQobjEvoTdMatchedOmp.__new__(CQobjEvoTdMatchedOmp)
                    self.compiled_qobjevo.set_threads(self.omp)
                self.compiled_qobjevo.__setstate__(state[1])


# Function defined inside another function cannot be pickled,
# Using class instead
class _UnitedFuncCaller:
    def __init__(self, funclist, args, dynamics_args, cte):
        self.funclist = funclist
        self.args = args
        self.dynamics_args = dynamics_args
        self.dims = cte.dims
        self.shape = cte.shape

    def set_args(self, args, dynamics_args):
        self.args = args
        self.dynamics_args = dynamics_args

    def dyn_args(self, t, state, shape):
        # 1d array are to F ordered
        mat = state.reshape(shape, order="F")
        for name, what, op in self.dynamics_args:
            if what == "vec":
                self.args[name] = state
            elif what == "mat":
                self.args[name] = mat
            elif what == "Qobj":
                if self.shape[1] == shape[1]:  # oper
                    self.args[name] = Qobj(mat, dims=self.dims)
                elif shape[1] == 1:  # ket
                    self.args[name] = Qobj(mat, dims=[self.dims[1], [1]])
                else:  # rho
                    self.args[name] = Qobj(mat, dims=self.dims[1])
            elif what == "expect":
                if shape[1] == op.cte.shape[1]:  # same shape as object
                    self.args[name] = op.mul_mat(t, mat).trace()
                else:
                    self.args[name] = op.expect(t, state)

    def __call__(self, t, args={}):
        if args:
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        out = []
        for func in self.funclist:
            out.append(func(t, now_args))
        return out

    def get_args(self):
        return self.args


class _Norm2():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return self.func(t, args)*np.conj(self.func(t, args))


class _Shift():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return np.conj(self.func(t + args["_t0"], args))


class _Conj():
    def __init__(self, f):
        self.func = f

    def __call__(self, t, args):
        return np.conj(self.func(t, args))


class _Prod():
    def __init__(self, f, g):
        self.func_1 = f
        self.func_2 = g

    def __call__(self, t, args):
        return self.func_1(t, args)*self.func_2(t, args)


class _Add():
    def __init__(self, fs):
        self.funcs = fs

    def __call__(self, t, args):
        return np.sum([f(t, args) for f in self.funcs])
