#cython: language_level=3
#cython: c_api_binop_methods=True

import inspect
import pickle
import typing
import scipy
from scipy.interpolate import make_interp_spline
import numpy as np
cimport numpy as cnp
cimport cython
import qutip

cdef extern from "<complex>" namespace "std" nogil:
    double complex conj(double complex x)
    double         norm(double complex x)


__all__ = [
    "Coefficient",  "InterCoefficient", "FunctionCoefficient",
    "StrFunctionCoefficient", "ConjCoefficient", "NormCoefficient"
]


def coefficient_function_parameters(func, style=None):
    """
    Return the function style (either "pythonic" or not) and a list of
    additional parameters accepted.

    Used by :obj:`FunctionCoefficient` and :obj:`_FuncElement` to determine the
    call signature of the supplied function based on the given style (or
    ``qutip.settings.core["function_coefficient_style"]`` if no style is given)
    and the signature of the given function.

    Parameters
    ----------
    func : function
        The :obj:`FunctionCoefficient` to inspect. The first argument
        of the function is assumed to be ``t`` (the time at which to
        evaluate the coefficient). The remaining arguments depend on
        the signature style (see below).

    style : {None, "pythonic", "dict", "auto"}
        The style of the signature used. If style is ``None``,
        the value of ``qutip.settings.core["function_coefficient_style"]``
        is used. Otherwise the supplied value overrides the global setting.

    Returns
    -------
    (f_pythonic, f_parameters)

    f_pythonic : bool
        True if the function should be called as ``f(t, **kw)`` and False
        if the function should be called as ``f(t, kw_dict)``.

    f_parameters : set or None
        The set of parameters (other than ``t``) of the function or
        ``None`` if the function accepts arbitrary parameters.
    """
    sig = inspect.signature(func)
    f_has_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if style is None:
        style = qutip.settings.core["function_coefficient_style"]
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
    composing time-dependant operator in list format for :obj:`.QobjEvo`.

    :obj:`.Coefficient` are immutable.
    """
    def __init__(self, args, **_):
        self.args = args

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return self

    def __call__(self, t, dict _args=None, **kwargs):
        """
        Return the coefficient value at time `t`.
        Stored arguments can overwriten with `_args` or as keywords parameters.

        Parameters
        ----------
        t : float
            Time at which to evaluate the :obj:`.Coefficient`.

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
        """Core computation of the :obj:`.Coefficient`."""
        # All Coefficient sub-classes should overwrite this or __call__
        return complex(self(t))

    def __add__(left, right):
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
        """Return a copy of the :obj:`.Coefficient`."""
        return pickle.loads(pickle.dumps(self))

    def conj(self):
        """ Return a conjugate :obj:`.Coefficient` of this"""
        return ConjCoefficient(self)

    def __eq__(self, other):
        return NotImplemented


@cython.auto_pickle(True)
cdef class FunctionCoefficient(Coefficient):
    """
    :obj:`.Coefficient` wrapping a Python function.

    Parameters
    ----------
    func : callable(t : float, ...) -> complex
        Function returning the coefficient value at time ``t``.

    args : dict
        Values of the arguments to pass to ``func``.

    style : {None, "pythonic", "dict", "auto"}
        The style of function signature used. If style is ``None``,
        the value of ``qutip.settings.core["function_coefficient_style"]``
        is used. Otherwise the supplied value overrides the global setting.

    The parameters ``_f_pythonic`` and ``_f_parameters`` override function
    style and parameter detection and are not intended to be part of
    the public interface.
    """
    cdef object func
    cdef bint _f_pythonic
    cdef set _f_parameters

    _UNSET = object()

    def __init__(self, func, dict args, style=None, _f_pythonic=_UNSET,
                 _f_parameters=_UNSET, **_):
        if _f_pythonic is self._UNSET or _f_parameters is self._UNSET:
            if not (_f_pythonic is self._UNSET
                    and _f_parameters is self._UNSET):
                raise TypeError(
                    "_f_pythonic and _f_parameters should "
                    "always be given together."
                )
            _f_pythonic, _f_parameters = coefficient_function_parameters(
                func, style=style)
            if _f_parameters is not None:
                args = {k: args[k] for k in _f_parameters & args.keys()}
            else:
                args = args.copy()
        self.func = func
        self.args = args
        self._f_pythonic = _f_pythonic
        self._f_parameters = _f_parameters

    def __call__(self, t, dict _args=None, **kwargs):
        if _args is not None or kwargs:
            return self.replace_arguments(_args, **kwargs)(t)
        if self._f_pythonic:
            return self.func(t, **self.args)
        return self.func(t, self.args)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return FunctionCoefficient(
            self.func,
            self.args.copy(),
            _f_pythonic=self._f_pythonic,
            _f_parameters=self._f_parameters,
        )

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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
            _f_pythonic=self._f_pythonic,
            _f_parameters=self._f_parameters,
        )

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef FunctionCoefficient self = left
        cdef FunctionCoefficient other = right
        return self.func is other.func and self.args == other.args

    def conj(self):
        if typing.get_type_hints(self.func).get("return", complex) is float:
            return self
        return ConjCoefficient(self)


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


cdef class StrFunctionCoefficient(Coefficient):
    """
    A :obj:`.Coefficient` defined by a string containing a simple Python
    expression.

    The string should contain a compilable Python expression that results in a
    complex number. The time ``t`` is available as a local variable, as are the
    individual arguments (i.e. the keys of ``args``). The ``args`` dictionary
    itself is not accessible.

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
        A string representing a compilable Python expression that results in a
        complex number.

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

    def __init__(self, base, dict args, **_):
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
        """Return a copy of the :obj:`.Coefficient`."""
        return StrFunctionCoefficient(self.base, self.args.copy())

    def __reduce__(self):
        return (StrFunctionCoefficient, (self.base, self.args))

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef StrFunctionCoefficient self = left
        cdef StrFunctionCoefficient other = right
        return self.base is other.base and self.args == other.args


cdef class InterCoefficient(Coefficient):
    """
    A :obj:`.Coefficient` built from an interpolation of a numpy array.

    Parameters
    ----------
    coeff_arr : np.ndarray
        The array of coefficient values to interpolate.

    tlist : np.ndarray
        An array of times corresponding to each coefficient value. The times
        must be increasing, but do not need to be uniformly spaced.

    order : int
        Order of the interpolation. Order ``0`` uses the previous (i.e. left)
        value. The order will be reduced to ``len(tlist) - 1`` if it is larger.

    boundary_conditions : 2-Tuple, str or None, optional
        Boundary conditions for spline evaluation. Default value is `None`.
        Correspond to `bc_type` of scipy.interpolate.make_interp_spline.
        Refer to Scipy's documentation for further details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_interp_spline.html
    """
    cdef int order
    cdef double dt
    cdef double[::1] tlist
    cdef complex[:, :] poly
    cdef object np_arrays
    cdef object boundary_conditions

    def __init__(self, coeff_arr, tlist, int order, boundary_conditions, **_):
        tlist = np.array(tlist, dtype=np.float64)
        coeff_arr = np.array(coeff_arr, dtype=np.complex128)

        if coeff_arr.ndim != 1:
            raise ValueError("The array to interpolate must be a 1D array")
        if coeff_arr.shape != tlist.shape:
            raise ValueError("tlist must be the same len "
                             "as the array to interpolate")
        if order < 0:
            raise ValueError("order must be a positive integer")

        order = min(order, len(tlist) - 1)

        if order == 0:
            coeff_arr = coeff_arr.reshape((1, -1))
        elif order == 1:
            coeff_arr = np.vstack([
                    np.diff(coeff_arr, append=-1) / np.diff(tlist, append=-1),
                    coeff_arr
                ])
        elif order >= 2:
            # Use scipy to compute the spline and transform it to polynomes
            # as used in scipy's PPoly which is easier for us to use.
            spline = make_interp_spline(tlist, coeff_arr, k=order,
                                        bc_type=boundary_conditions)
            # Scipy can move knots, we add them to tlist
            tlist = np.sort(np.unique(np.concatenate([spline.t, tlist])))
            a = np.arange(spline.k+1)
            a[0] = 1
            fact = np.cumprod(a)
            coeff_arr = np.concatenate([
                spline(tlist, i) / fact[i]
                for i in range(spline.k, -1, -1)
            ]).reshape((spline.k+1, -1))

        self._prepare(tlist, coeff_arr)
        self.boundary_conditions = boundary_conditions

    def _prepare(self, np_tlist, np_poly, dt=None):
        self.np_arrays = (np_tlist, np_poly)
        self.tlist = np_tlist
        self.poly = np_poly
        self.order = self.poly.shape[0] - 1
        diff = np.diff(self.np_arrays[0])
        if dt is not None:
            self.dt = dt
        elif len(diff) >= 1 and np.allclose(diff[0], diff):
            self.dt = diff[0]
        else:
            self.dt = 0

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef size_t _binary_search(self, double x):
        # Binary search for the interval
        # return the indice of the of the biggest element where t <= x
        cdef size_t low = 0
        cdef size_t high = self.tlist.shape[0]
        cdef size_t middle
        cdef size_t count = 0
        while low+1 != high and count < 64:
            middle = (low + high)//2
            if x < self.tlist[middle]:
                high = middle
            else:
                low = middle
            # We keep a count to be sure that it never get into an infinit loop
            # even if tlist has an unexpected format.
            count += 1
        return low

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double complex _call(self, double t) except *:
        cdef size_t idx, i
        cdef double factor
        cdef double complex out
        cdef double complex[:] slice
        if t <= self.tlist[0]:
            return self.poly[-1, 0]
        elif t >= self.tlist[-1]:
            return self.poly[-1, -1]
        if self.dt:
            idx = <size_t>((t - self.tlist[0]) / self.dt)
        else:
            idx = self._binary_search(t)
        if self.order == 0:
            out = self.poly[0, idx]
        else:
            factor = t - self.tlist[idx]
            slice = self.poly[:, idx]
            out = 0.
            for i in range(self.order+1):
                out *= factor
                out += slice[i]
        return out

    def __reduce__(self):
        return (InterCoefficient.restore, (*self.np_arrays, self.dt))

    @classmethod
    def restore(cls, np_tlist, np_poly, dt=None):
        cdef InterCoefficient out = cls.__new__(cls)
        out._prepare(np_tlist, np_poly, dt)
        return out

    @classmethod
    def from_PPoly(cls, ppoly, **_):
        return cls.restore(ppoly.x, np.asarray(ppoly.c, complex))

    @classmethod
    def from_Bspline(cls, spline, **_):
        tlist = np.unique(spline.t)
        a = np.arange(spline.k+1)
        a[0] = 1
        fact = np.cumprod(a) + 0j
        poly = np.concatenate([
            spline(tlist, i) / fact[i]
            for i in range(spline.k, -1, -1)
        ]).reshape((spline.k+1, -1)).astype(complex, copy=False)
        return cls.restore(tlist, poly)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return InterCoefficient.restore(*self.np_arrays, self.dt)

    def __add__(left, right):
        cdef InterCoefficient self
        cdef InterCoefficient other

        # Pre-cython 3 support
        if type(left) is InterCoefficient:
            self = left
        else:
            self = right
            right = left

        if isinstance(right, InterCoefficient):
            other = <InterCoefficient> right
            if (
                self.np_arrays[0].shape == other.np_arrays[0].shape
                and (self.order == other.order)
                and np.allclose(
                    self.np_arrays[0], other.np_arrays[0],
                    rtol=1e-15, atol=1e-15
                )
            ):
                return InterCoefficient.restore(
                    self.np_arrays[0],
                    self.np_arrays[1] + other.np_arrays[1],
                    self.dt
                )

        if isinstance(right, ConstantCoefficient):
            value = (<ConstantCoefficient> right).value
            poly = self.np_arrays[1].copy()
            poly[-1, :] += value
            return InterCoefficient.restore(
                self.np_arrays[0], poly, self.dt
            )

        return SumCoefficient(left, right)

    def __mul__(left, right):
        cdef InterCoefficient self
        cdef InterCoefficient other
        cdef int i, j, inv1, inv2, N, idx

        # Pre-cython 3 support
        if type(left) is InterCoefficient:
            self = left
        else:
            self = right
            right = left

        if not isinstance(right, Coefficient):
            return Coefficient.__mul__(self, right)

        if isinstance(right, InterCoefficient):
            """
            We create a spline of the same order as the input.
            The pure mathematical product should add orders, creating a 6th
            order for the product of 2 cubic splines, etc. We don't want to
            increase the polynome size without limit and recreating the spline
            is still a good approximation.

            Note: Should we add an option for this? Or limit to CubicSpline?
            """
            other = <InterCoefficient> right
            if (
                self.np_arrays[0].shape == other.np_arrays[0].shape
                and (self.order == other.order)
                and (self.boundary_conditions == other.boundary_conditions)
                and np.allclose(
                    self.np_arrays[0], other.np_arrays[0],
                    rtol=1e-15, atol=1e-15
                )
            ):
                coeff1 = self.np_arrays[1][-1, :]
                coeff2 = other.np_arrays[1][-1, :]
                return InterCoefficient(
                    coeff1 * coeff2,
                    self.tlist,
                    self.order,
                    self.boundary_conditions
                )

        if isinstance(right, ConstantCoefficient):
            value = (<ConstantCoefficient> right).value
            return InterCoefficient.restore(
                self.np_arrays[0], self.np_arrays[1] * value, self.dt
            )

        return MulCoefficient(left, right)

    def conj(InterCoefficient self):
        if np.isreal(self.np_arrays[1]).all():
            return self
        return InterCoefficient.restore(
            self.np_arrays[0], np.conj(self.np_arrays[1]), self.dt
        )

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef InterCoefficient self = left
        cdef InterCoefficient other = right
        return (
            self.np_arrays[0].shape == other.np_arrays[0].shape and
            self.np_arrays[1].shape == other.np_arrays[1].shape and
            np.allclose(self.np_arrays[0], other.np_arrays[0]) and
            np.allclose(self.np_arrays[1], other.np_arrays[1]) and
            np.allclose(self.dt, other.dt) and
            self.boundary_conditions == other.boundary_conditions
        )


@cython.auto_pickle(True)
cdef class SumCoefficient(Coefficient):
    """
    A :obj:`.Coefficient` built from the sum of two other coefficients.

    A :obj:`SumCoefficient` is returned as the result of the addition of two
    coefficients, e.g. ::

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
        """Return a copy of the :obj:`.Coefficient`."""
        return SumCoefficient(self.first.copy(), self.second.copy())

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef SumCoefficient self = left
        cdef SumCoefficient other = right
        return (
            (self.first == other.first and self.second == other.second) or
            (self.second == other.first and self.first == other.second)
        )


@cython.auto_pickle(True)
cdef class MulCoefficient(Coefficient):
    """
    A :obj:`.Coefficient` built from the product of two other coefficients.

    A :obj:`MulCoefficient` is returned as the result of the multiplication of
    two coefficients, e.g. ::

        coefficient("w * t", args={'w': 1}) * coefficient("t")
    """
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cdef complex _call(self, double t) except *:
        return self.first._call(t) * self.second._call(t)

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return MulCoefficient(self.first.copy(), self.second.copy())

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef MulCoefficient self = left
        cdef MulCoefficient other = right
        return (
            (self.first == other.first and self.second == other.second) or
            (self.second == other.first and self.first == other.second)
        )


@cython.auto_pickle(True)
cdef class ConjCoefficient(Coefficient):
    """
    The conjugate of a :obj:`.Coefficient`.

    A :obj:`ConjCoefficient` is returned by ``Coefficient.conj()`` and
    ``qutip.coefficent.conj(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cdef complex _call(self, double t) except *:
        return conj(self.base._call(t))

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return ConjCoefficient(self.base)

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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

    def conj(self, other):
        return self.base

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef ConjCoefficient self = left
        cdef ConjCoefficient other = right
        return self.base == other.base


@cython.auto_pickle(True)
cdef class NormCoefficient(Coefficient):
    """
    The L2 :func:`norm` of a :obj:`.Coefficient`. A shortcut for
    ``conj(coeff) * coeff``.

    :obj:`NormCoefficient` is returned by
    ``qutip.coefficent.norm(Coefficient)``.
    """
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

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
        """Return a copy of the :obj:`.Coefficient`."""
        return NormCoefficient(self.base.copy())

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef NormCoefficient self = left
        cdef NormCoefficient other = right
        return self.base == other.base


@cython.auto_pickle(True)
cdef class ConstantCoefficient(Coefficient):
    """
    A time-independent coefficient.

    :obj:`ConstantCoefficient` is returned by ``qutip.coefficent.const(value)``.
    """
    cdef complex value

    def __init__(self, complex value, **_):
        self.value = value

    def replace_arguments(self, _args=None, **kwargs):
        """
        Replace the arguments (``args``) of a coefficient.

        Returns a new :obj:`.Coefficient` if the coefficient has arguments, or
        the original coefficient if it does not. Arguments to replace may be
        supplied either in a dictionary as the first position argument, or
        passed as keywords, or as a combination of the two. Arguments not
        replaced retain their previous values.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        """
        return self

    cdef complex _call(self, double t) except *:
        return self.value

    cpdef Coefficient copy(self):
        """Return a copy of the :obj:`.Coefficient`."""
        return self

    def __add__(self, other):
        if (
            isinstance(self, ConstantCoefficient) and
            isinstance(other, ConstantCoefficient)
        ):
            return ConstantCoefficient(
                (<ConstantCoefficient> self).value +
                (<ConstantCoefficient> other).value
            )
        return NotImplemented

    def __mul__(self, other):
        if (
            isinstance(self, ConstantCoefficient) and
            isinstance(other, ConstantCoefficient)
        ):
            return ConstantCoefficient(
                (<ConstantCoefficient> self).value *
                (<ConstantCoefficient> other).value
            )
        return Coefficient.__mul__(self, other)

    def conj(ConstantCoefficient self):
        if self.value.imag == 0:
            return self
        return ConstantCoefficient(conj(self.value))

    def __eq__(left, right):
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        cdef ConstantCoefficient self = left
        cdef ConstantCoefficient other = right
        return self.value == other.value
