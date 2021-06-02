#cython: language_level=3
from .inter import _prep_cubic_spline
from .inter cimport (_spline_complex_cte_second,
                     _spline_complex_t_second,
                     _step_complex_t, _step_complex_cte)
from .interpolate cimport interp, zinterp
from ..interpolate import Cubic_Spline
import pickle
import scipy
import numpy as np
cimport numpy as cnp
cimport cython
import qutip

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         norm(double complex x)


cdef class Coefficient:
    """
    `Coefficient` are the time-dependant scalar of a `[Qobj, coeff]` pair
    composing time-dependant operator in list format for :obj:`QobjEvo`.

    :obj:`Coefficient` are immutable.
    """
    def __init__(self):
        raise NotImplementedError("Only sub-classes should be initiated.")

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.

        """
        return self

    def __call__(self, double t, dict args=None):
        """Return the coefficient value at `t` with given `args`."""
        if args is not None:
            return (<Coefficient> self.replace(arguments=args))._call(t)
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
    func : callable(t : float, args : dict) -> complex
        Function computing the coefficient for a :obj:`QobjEvo`.

    args : dict
        Dictionary of variable to pass to `func`.
    """
    cdef object func

    def __init__(self, func, dict args):
        self.func = func
        self.args = args

    cdef complex _call(self, double t) except *:
        return self.func(t, self.args)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return FunctionCoefficient(self.func, self.args.copy())

    def replace(self, *, arguments=None, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        arguments : dict
            New arguments for function and str based :obj:`Coefficient`.
            The dictionary do not need to include all keys, but only those
            which need to be updated.
        """
        if arguments:
            return FunctionCoefficient(
                self.func,
                {**self.args, **arguments}
            )
        return self


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


cdef class StrFunctionCoefficient(Coefficient):
    """
    :obj:`Coefficient` wrapping a string into a python function.
    The string must represent compilable python code resulting in a complex.
    The time is available as the local variable `t` and the keys of `args`
    are also available as local variables. The `args` dictionary itself is not
    available.
    The following symbols are defined:
        ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``pi``,
        ``sinh``, ``cosh``, ``tanh``, ``asinh``, ``acosh``, ``atanh``,
        ``exp``, ``log``, ``log10``, ``erf``, ``zerf``, ``sqrt``,
        ``real``, ``imag``, ``conj``, ``abs``, ``norm``, ``arg``, ``proj``,
        ``numpy`` as ``np`` and ``scipy.special`` as ``spe``.

    Examples
    --------
    >>> StrFunctionCoefficient("sin(w*pi*t)", {'w': 1j})

    Parameters
    ----------
    base : str
        A string representing a compilable python code resulting in a complex.

    args : dict
        Dictionary of variable used in the code string. May include unused
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

    def replace(self, *, arguments=None, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        arguments : dict
            New arguments for function and str based :obj:`Coefficient`.
            The dictionary do not need to include all keys, but only those
            which need to be updated.
        """
        if arguments:
            return StrFunctionCoefficient(
                self.base,
                {**self.args, **arguments}
            )
        return self


cdef class InterpolateCoefficient(Coefficient):
    """
    :obj:`Coefficient` built from a :class:`qutip.Cubic_Spline` object.

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

    def replace(self, *, tlist=None, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        tlist : np.array
            New array of times for the array coefficients.
        """
        if tlist is not None:
            return InterpolateCoefficient(
                Cubic_Spline(tlist[0], tlist[-1],
                             self.spline.array,
                             *self.spline.bounds)
                )
        else:
            return self


cdef class InterCoefficient(Coefficient):
    """
    :obj:`Coefficient` built from a cubic spline interpolation of a numpy array.

    Parameters
    ----------
    coeff_arr : np.ndarray
        Array of coefficients to interpolate.

    tlist : np.ndarray
        Array of times corresponding to each coefficient. The time must be
        inscreasing, but do not need to be uniformly spaced.
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

    def replace(self, *, tlist=None, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        tlist : np.array
            New array of times for the array coefficients.
        """
        if tlist:
            return InterCoefficient(self.coeff_np, tlist)
        else:
            return self

    @property
    def array(self):
        # Fro QIP tests
        return self.coeff_np


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
    :obj:`Coefficient` built from a numpy array interpolated using previous
    value.

    Parameters
    ----------
    coeff_arr : np.ndarray
        Array of coefficients to interpolate.

    tlist : np.ndarray
        Array of times corresponding to each coefficient. The time must be
        inscreasing, but do not need to be uniformly spaced.
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

    def replace(self, *, tlist=None, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        tlist : np.array
            New array of times for the array coefficients.
        """
        if tlist:
            return StepCoefficient(self.coeff_np, tlist)
        else:
            return self

    @property
    def array(self):
        # Fro QIP tests
        return np.array(self.coeff_arr)


@cython.auto_pickle(True)
cdef class SumCoefficient(Coefficient):
    """
    :obj:`Coefficient` built from the sum of 2 other Coefficients.
    Result of :obj:`Coefficient` + :obj:`Coefficient`.
    """
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cpdef void arguments(self, dict args) except *:
        self.first.arguments(args)
        self.second.arguments(args)

    cdef complex _call(self, double t) except *:
        return self.first._call(t) + self.second._call(t)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return SumCoefficient(self.first.copy(), self.second.copy())

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.
        """
        return SumCoefficient(
            self.first.replace(**kwargs),
            self.second.replace(**kwargs)
        )


@cython.auto_pickle(True)
cdef class MulCoefficient(Coefficient):
    """
    :obj:`Coefficient` built from the product of 2 other Coefficients.
    Result of :obj:`Coefficient` * :obj:`Coefficient`.

    Methods
    -------
    conj():
        Conjugate of the :obj:`Coefficient`.
    copy():
        Create a copy of the :obj:`Coefficient`.
    replace(arguments, tlist):
        Create a new :obj:`Coefficient` with updated arguments and/or tlist.
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

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.
        """
        return MulCoefficient(
            self.first.replace(**kwargs),
            self.second.replace(**kwargs)
        )


@cython.auto_pickle(True)
cdef class ConjCoefficient(Coefficient):
    """
    Conjugate of a :obj:`Coefficient`.

    Result of ``Coefficient.conj()`` or ``qutip.coefficent.conj(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t) except *:
        return conj(self.base._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return ConjCoefficient(self.base.copy())

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.
        """
        return ConjCoefficient(
            self.base.replace(**kwargs)
        )


@cython.auto_pickle(True)
cdef class NormCoefficient(Coefficient):
    """
    Norm of a :obj:`Coefficient`.
    Used as a shortcut of conj(coeff) * coeff
    Result of ``Coefficient._cdc()`` or ``qutip.coefficent.norm(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.
        """
        return NormCoefficient(
            self.base.replace(**kwargs)
        )

    cdef complex _call(self, double t) except *:
        return norm(self.base._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return NormCoefficient(self.base.copy())


@cython.auto_pickle(True)
cdef class ShiftCoefficient(Coefficient):
    """
    Introduce a time shift in the :obj:`Coefficient`.
    Used intenally in correlation.
    Result of ``Coefficient._shift()`` or
    ``qutip.coefficent.shift(Coefficient)``.
    """
    cdef Coefficient base
    cdef double _t0

    def __init__(self, Coefficient base, double _t0):
        self.base = base
        self._t0 = _t0

    def replace(self, **kwargs):
        """
        Return a :obj:`Coefficient` with args or tlist changed.

        Parameters
        ----------
        **kwargs : dict
            Attributes to change.

            Keys
            ----
            arguments : dict
                Can be replaced in str and function based coefficients.

            tlist : np.array
                Can be replaced array based coefficients.
        """
        try:
            _t0 = kwargs["arguments"]["_t0"]
        except KeyError:
            _t0 = self._t0
        return ShiftCoefficient(self.base.replace(**kwargs), _t0)

    cdef complex _call(self, double t) except *:
        return self.base._call(t + self._t0)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`Coefficient`."""
        return ShiftCoefficient(self.base.copy(), self._t0)
