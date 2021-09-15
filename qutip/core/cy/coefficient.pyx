#cython: language_level=3
from .inter import _prep_cubic_spline
from .inter cimport (_spline_complex_cte_second,
                     _spline_complex_t_second,
                     _step_complex_t, _step_complex_cte)
from .interpolate cimport interp, zinterp
from ..interpolate import Cubic_Spline
import inspect
import pickle
import scipy
import numpy as np
cimport numpy as cnp
cimport cython
import qutip

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         norm(double complex x)


def coefficient_function_parameters(func, style=None):
    """
    Return the function style (either "pythonic" or not) and a list of
    additional parameters accepted. Used by :obj:`FunctionCoefficient`
    and :obj:`_FuncElement` to determine the call signature of the
    supplied function based on the
    :obj:`qutip.core.settings.function_coefficient_signature` setting and
    the supplied function signature.

    Parameters
    ----------
    func : function
        The :obj:`FunctionCoefficient` to inspect. The first argument
        of the function is assumed to be ``t`` (the time at which to
        evaluate the coefficient). The remaining arguments depend on
        the signature style (see below).

    style : {None, "pythonic", "dict", "auto"}
        The style of the signature used. If style is ``None``,
        the value of :obj:`qutip.core.settings.function_coefficient_signature`
        is used. Otherwise the supplied value overrides the global setting.

    Returns
    -------
    (f_is_pythonic, f_parameters)

    f_is_pythonic : bool
        True if the function should be called as ``f(t, **kw)`` and False
        if the function should be called as ``f(t, kw_dict)``.

    f_parameters : set or None
        The set of parameters (other than ``t``) of the function or
        ``None`` if the function accepts arbitrary parameters.
    """
    sig = inspect.signature(func)
    f_has_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if style is None:
        style = qutip.settings.core["function_coefficient_signature"]
    if style == "auto":
        if tuple(sig.parameters.keys()) == ("t", "args") and not f_has_kw:
            # if the signature is exactly f(t, args), then assume parameters
            # are supplied in an argument dictionary
            style = "dict"
        else:
            style = "pythonic"
    if style == "dict" or f_has_kw:
        # f might accept any parameter
        f_parameters = None
    else:
        # f accepts only t and the named parameters
        f_parameters = set(list(sig.parameters.keys())[1:])
    return (style == "pythonic", f_parameters)


cdef class Coefficient:
    """
    `Coefficient` are the time-dependant scalar of a `[Qobj, coeff]` pair
    composing time-dependant operator in list format for :obj:`QobjEvo`.

    :obj:`Coefficient` are immutable.
    """
    def __init__(self):
        raise NotImplementedError("Only sub-classes should be initiated.")

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return self

    def __call__(self, double t, dict _args=None, **kwargs):
        """
        Return the coefficient value at time `t`.
        Stored arguments can overwriten with `_args` or as keywords parameters.

        Parameters
        ----------
        t : float
            Time at which to evaluate the :obj:`Coefficient`.

        _args : dict
            Dictionary of arguments to use instead of the stored ones.

        **kwargs
            Arguments to overwrite for this call.
        """
        if _args is not None or kwargs:
            return (<Coefficient> self.replace_arguments(_args,
                                                         **kwargs))._call(t)
        return self._call(t)

    cdef double complex _call(self, double t) except *:
        """Core computation of the :obj:`Coefficient`."""
        raise NotImplementedError("All Coefficient sub-classes "
                                  "should overwrite this.")

    def __add__(left, right):
        if (
            isinstance(left, InterCoefficient)
            and isinstance(right, InterCoefficient)
        ):
            return add_inter(left, right)
        if isinstance(left, Coefficient) and isinstance(right, Coefficient):
            return SumCoefficient(left.copy(), right.copy())
        return NotImplemented

    def __mul__(left, right):
        if isinstance(left, Coefficient) and isinstance(right, Coefficient):
            return MulCoefficient(left.copy(), right.copy())
        if isinstance(left, qutip.Qobj):
            return qutip.QobjEvo([left.copy(), right.copy()])
        if isinstance(right, qutip.Qobj):
            return qutip.QobjEvo([right.copy(), left.copy()])
        return NotImplemented

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return pickle.loads(pickle.dumps(self))

    def conj(self):
        """ Return a conjugate :obj:`Coefficient` of this"""
        return ConjCoefficient(self)

    def _cdc(self):
        """ Return a :obj:`Coefficient` being the norm of this"""
        return NormCoefficient(self)

    def _shift(self):
        """ Return a :obj:`Coefficient` with a time shift"""
        return ShiftCoefficient(self, 0)

@cython.auto_pickle(True)
cdef class FunctionCoefficient(Coefficient):
    """
    :obj:`Coefficient` wrapping a Python function.

    Parameters
    ----------
    func : callable(t : float, ...) -> complex
        Function returning the coefficient value at time ``t``.

    args : dict
        Values of the arguments to pass to ``func``.

    f_is_pythonic : bint
        Set to true if ``func`` should be called in the old QuTiP 4 style
        as ``func(t, args)`` where ``args`` is a dictionary that contains
        all the arguments. Otherwise set to false and ``func`` will be
        called as ``f(t, **args)``.

    f_parameters : set or None
        The set of argument names ``func`` accepts or ``None`` is ``func``
        accepts all possible arguments (e.g. via a ``**kw`` argument).
    """
    cdef object func
    cdef bint _f_is_pythonic
    cdef object _f_parameters

    def __init__(self, func, dict args, bint f_is_pythonic, f_parameters):
        self.func = func
        self.args = args
        self._f_is_pythonic = f_is_pythonic
        self._f_parameters = f_parameters

    @classmethod
    def by_inspection(cls, func, args):
        f_is_pythonic, f_parameters = coefficient_function_parameters(func)
        if f_parameters is not None:
            args = {k: args[k] for k in f_parameters & args.keys()}
        return cls(
            func, args, f_is_pythonic=f_is_pythonic, f_parameters=f_parameters)

    cdef complex _call(self, double t) except *:
        if self._f_is_pythonic:
            return self.func(t, **self.args)
        return self.func(t, self.args)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return FunctionCoefficient(
            self.func,
            self.args.copy(),
            f_is_pythonic=self._f_is_pythonic,
            f_parameters=self._f_parameters,
        )

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        if _args:
            kwargs.update(_args)
        if self._f_parameters is not None:
            kwargs = {k: kwargs[k] for k in self._f_parameters & kwargs.keys()}
        if not kwargs:
            return self
        return FunctionCoefficient(
            self.func,
            {**self.args, **kwargs},
            f_is_pythonic=self._f_is_pythonic,
            f_parameters=self._f_parameters,
        )


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


cdef class StrFunctionCoefficient(Coefficient):
    """
    A :obj:`Coefficient` defined by a string containing a simple Python expression.

    The string should contain a compilable Python expression that results in a complex number.
    The time ``t`` is available as a local variable, as are the individual arguments (i.e. the
    keys of ``args``). The ``args`` dictionary itself is not accessible.

    The following symbols are defined:
        ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``pi``,
        ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``,
        ``exp``, ``log``, ``log10``, ``erf``, ``zerf``, ``sqrt``,
        ``real``, ``imag``, ``conj``, ``abs``, ``norm``, ``arg``, ``proj``,
        ``numpy`` as ``np`` and ``scipy.special`` as ``spe``.

    Examples
    --------
    >>> StrFunctionCoefficient("sin(w * pi * t)", {'w': 1j})

    Parameters
    ----------
    base : str
        A string representing a compilable Python expression that results in a complex number.

    args : dict
        A dictionary of variable used in the code string. It may include unused
        variables.
    """
    cdef object func
    cdef str base

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

    def __init__(self, base, dict args):
        args2var = "\n".join(["    {} = args['{}']".format(key, key)
                              for key in args])
        code = f"""
def coeff(t, args):
{args2var}
    return {base}"""
        lc = {}
        exec(code, self.str_env, lc)
        self.base = base
        self.func = lc["coeff"]
        self.args = args

    cdef complex _call(self, double t) except *:
        return self.func(t, self.args)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return StrFunctionCoefficient(self.base, self.args.copy())

    def __reduce__(self):
        return (StrFunctionCoefficient, (self.base, self.args))

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        if _args:
            kwargs.update(_args)
        if kwargs:
            return StrFunctionCoefficient(self.base, {**self.args, **kwargs})
        return self


cdef class InterpolateCoefficient(Coefficient):
    """
    A :obj:`Coefficient` built from a :class:`qutip.Cubic_Spline` object.

    Parameters
    ----------
    splineObj : :class:`qutip.Cubic_Spline`
        Spline interpolation object representing the coefficient as a function
        of the time.
    """

    cdef double lower_bound, higher_bound
    cdef complex[::1] spline_data
    cdef object spline

    def __init__(self, splineObj):
        self.lower_bound = splineObj.a
        self.higher_bound = splineObj.b
        self.spline_data = splineObj.coeffs.astype(np.complex128)
        self.spline = splineObj

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return zinterp(t,
                       self.lower_bound,
                       self.higher_bound,
                       self.spline_data)

    def __reduce__(self):
        return InterpolateCoefficient, (self.spline,)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return InterpolateCoefficient(self.spline)


cdef class InterCoefficient(Coefficient):
    """
    A :obj:`Coefficient` built from a cubic spline interpolation of a numpy array.

    Parameters
    ----------
    coeff_arr : np.ndarray
        The array of coefficient values to interpolate.

    tlist : np.ndarray
        An array of times corresponding to each coefficient value. The times must be
        increasing, but do not need to be uniformly spaced.
    """
    cdef int n_t, constant
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr, second_derr
    cdef object tlist_np, coeff_np, second_np

    def __init__(self, coeff_arr, tlist, _second=None, _constant=None):
        self.tlist_np = tlist
        self.tlist = tlist
        self.coeff_np = coeff_arr
        self.coeff_arr = coeff_arr
        if _second is None:
            self.second_np, self.constant = _prep_cubic_spline(coeff_arr,
                                                                tlist)
        else:
            self.second_np = _second
            self.constant = _constant
        self.second_derr = self.second_np
        self.dt = tlist[1] - tlist[0]

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.constant:
            coeff = _spline_complex_cte_second(t,
                                               self.tlist,
                                               self.coeff_arr,
                                               self.second_derr)
        else:
            coeff = _spline_complex_t_second(t,
                                             self.tlist,
                                             self.coeff_arr,
                                             self.second_derr)
        return coeff

    def __reduce__(self):
        return (InterCoefficient,
                (self.coeff_np, self.tlist_np, self.second_np, self.constant))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return InterCoefficient(self.coeff_np, self.tlist_np,
                                self.second_np, self.constant)


cdef Coefficient add_inter(InterCoefficient left, InterCoefficient right):
    if np.array_equal(left.tlist_np, right.tlist_np):
        return InterCoefficient(left.coeff_np + right.coeff_np,
                                left.tlist_np,
                                left.second_np + right.second_np,
                                left.constant
                               )
    else:
        return SumCoefficient(left.copy(), right.copy())


cdef class StepCoefficient(Coefficient):
    """
    A step function :obj:`Coefficient` whose values are specified in a numpy array.

    At each point in time, the value of the coefficient is the most recent previous value given in the numpy array.

    Parameters
    ----------
    coeff_arr : np.ndarray
        The array of coefficient values to interpolate.

    tlist : np.ndarray
        An array of times corresponding to each coefficient value. The times must be
        increasing, but do not need to be uniformly spaced.
    """
    cdef int n_t, constant
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr
    cdef object tlist_np, coeff_np

    def __init__(self, coeff_arr, tlist, _constant=None):
        self.tlist_np = tlist
        self.tlist = self.tlist_np
        self.coeff_np = coeff_arr
        self.coeff_arr = self.coeff_np
        if _constant is None:
            self.constant = np.allclose(np.diff(tlist), tlist[1]-tlist[0])
        else:
            self.constant = _constant
        self.dt = tlist[1] - tlist[0]
        self.args = {}

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.constant:
            coeff = _step_complex_cte(t, self.tlist, self.coeff_arr)
        else:
            coeff = _step_complex_t(t, self.tlist, self.coeff_arr)
        return coeff

    def __reduce__(self):
        return (StepCoefficient, (self.coeff_np, self.tlist_np, self.constant))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return StepCoefficient(self.coeff_np, self.tlist_np, self.constant)


@cython.auto_pickle(True)
cdef class SumCoefficient(Coefficient):
    """
    A :obj:`Coefficient` built from the sum of two other coefficients.

    A :obj:`SumCoefficient` is returned as the result of the addition of two coefficients, e.g. ::

        coefficient("t * t") + coefficient("t")  # SumCoefficient
    """
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cdef complex _call(self, double t) except *:
        return self.first._call(t) + self.second._call(t)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return SumCoefficient(self.first.copy(), self.second.copy())

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return SumCoefficient(
            self.first.replace_arguments(_args, **kwargs),
            self.second.replace_arguments(_args, **kwargs)
        )


@cython.auto_pickle(True)
cdef class MulCoefficient(Coefficient):
    """
    A :obj:`Coefficient` built from the product of two other coefficients.

    A :obj:`MulCoefficient` is returned as the result of the multiplication of two coefficients, e.g. ::

        coefficient("w * t", args={'w': 1}) * coefficient("t")  # MulCoefficient
    """
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cdef complex _call(self, double t) except *:
        return self.first._call(t) * self.second._call(t)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return MulCoefficient(self.first.copy(), self.second.copy())

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return MulCoefficient(
            self.first.replace_arguments(_args, **kwargs),
            self.second.replace_arguments(_args, **kwargs)
        )


@cython.auto_pickle(True)
cdef class ConjCoefficient(Coefficient):
    """
    The conjugate of a :obj:`Coefficient`.

    A :obj:`ConjCoefficient` is returned by ``Coefficient.conj()`` and ``qutip.coefficent.conj(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t) except *:
        return conj(self.base._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return ConjCoefficient(self.base.copy())

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return ConjCoefficient(
            self.base.replace_arguments(_args, **kwargs)
        )


@cython.auto_pickle(True)
cdef class NormCoefficient(Coefficient):
    """
    The L2 :func:`norm` of a :obj:`Coefficient`. A shortcut for ``conj(coeff) * coeff``.

    :obj:`NormCoefficient` is returned by ``qutip.coefficent.norm(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return NormCoefficient(
            self.base.replace_arguments(_args, **kwargs)
        )

    cdef complex _call(self, double t) except *:
        return norm(self.base._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return NormCoefficient(self.base.copy())


@cython.auto_pickle(True)
cdef class ShiftCoefficient(Coefficient):
    """
    Introduce a time shift into the :obj:`Coefficient`.

    Used internally within qutip when calculating correlations.

    :obj:ShiftCoefficient is returned by ``qutip.coefficent.shift(Coefficient)``.
    """
    cdef Coefficient base
    cdef double _t0

    def __init__(self, Coefficient base, double _t0):
        self.base = base
        self._t0 = _t0

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`Coefficient` if the coefficient has arguments, or the original coefficient if it does not.
        Arguments to replace may be supplied either in a dictionary as the first position argument, or passed as
        keywords, or as a combination of the two. Arguments not replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        if _args:
            kwargs.update(_args)
        try:
            _t0 = kwargs["_t0"]
            del kwargs["_t0"]
        except KeyError:
            _t0 = self._t0
        return ShiftCoefficient(self.base.replace_arguments(**kwargs), _t0)

    cdef complex _call(self, double t) except *:
        return self.base._call(t + self._t0)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return ShiftCoefficient(self.base.copy(), self._t0)
