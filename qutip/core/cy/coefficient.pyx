#cython: language_level=3
cdef extern from "<complex>" namespace "std" nogil:
    double complex cos(double complex x)
    double complex conj(double complex x)
    double         norm(double complex x)
    double complex exp(double complex x)

from .inter import _prep_cubic_spline
from .inter cimport (_spline_complex_cte_second,
                             _spline_complex_t_second,
                             _step_complex_t, _step_complex_cte)
from .interpolate cimport (interp, zinterp)
import pickle
import numpy as np
cimport numpy as cnp
cimport cython

cdef class Coefficient:
    cdef complex _call(self, double t) except *:
        return 0j

    cpdef void arguments(self, dict args) except *:
        self.args = args

    def __call__(self, double t, dict args={}):
        """update args and return the
        """
        if args:
            self.arguments(args)
        return self._call(t)

    def optstr(self):
        return ""

    def __add__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        return SumCoefficient(self, other)

    def __mul__(self, other):
        if not isinstance(other, Coefficient):
            return NotImplemented
        return MulCoefficient(self, other)

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def conj(self):
        return ConjCoefficient(self)

    def _cdc(self):
        return NormCoefficient(self)

    def _shift(self):
        return ShiftCoefficient(self, 0)


@cython.auto_pickle(True)
cdef class FunctionCoefficient(Coefficient):
    cdef object func

    def __init__(self, func, args):
        self.func = func
        self.args = args

    cdef complex _call(self, double t) except *:
        return self.func(t, self.args)

    def copy(self):
        return FunctionCoefficient(self.func, self.args.copy())


cdef class InterpolateCoefficient(Coefficient):
    cdef double lower_bound, higher_bound
    cdef complex[::1] spline_data

    def __init__(self, splineObj):
        self.lower_bound = splineObj.a
        self.higher_bound = splineObj.b
        self.spline_data = splineObj.coeffs.astype(np.complex128)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return zinterp(t,
                       self.lower_bound,
                       self.higher_bound,
                       self.spline_data)

    def __reduce__(self):
        return (MakeInterpolateCoefficient,
            (self.lower_bound, self.higher_bound, np.array(self.spline_data)))

    def _setstate_(self, state):
        self.lower_bound = state[0]
        self.higher_bound = state[1]
        self.spline_data = state[2]


def MakeInterpolateCoefficient(*state):
    obj = InterpolateCoefficient.__new__(InterpolateCoefficient)
    obj._setstate_(state)
    return obj


cdef class InterCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr, second_derr

    def __init__(self, cnp.ndarray[complex, ndim=1] coeff_arr,
                 cnp.ndarray[double, ndim=1] tlist):
        self.second_derr, self.cte = _prep_cubic_spline(coeff_arr, tlist)
        self.tlist = tlist
        self.coeff_arr = coeff_arr
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.cte:
            coeff = _spline_complex_cte_second(t,
                                               self.tlist,
                                               self.coeff_arr,
                                               self.second_derr,
                                               self.n_t,
                                               self.dt)
        else:
            coeff = _spline_complex_t_second(t,
                                             self.tlist,
                                             self.coeff_arr,
                                             self.second_derr,
                                             self.n_t)
        return coeff

    def __reduce__(self):
        return MakeInterCoefficient, (self.n_t, self.cte, self.dt,
            np.array(self.tlist), np.array(self.coeff_arr),
            np.array(self.second_derr))

    def _setstate_(self, state):
        self.n_t = state[0]
        self.cte = state[1]
        self.dt = state[2]
        self.tlist = state[3]
        self.coeff_arr = state[4]
        self.second_derr = state[5]


def MakeInterCoefficient(*state):
    obj = InterCoefficient.__new__(InterCoefficient)
    obj._setstate_(state)
    return obj


cdef class StepCoefficient(Coefficient):
    cdef int n_t, cte
    cdef double dt
    cdef double[::1] tlist
    cdef complex[::1] coeff_arr

    def __init__(self, complex[::1] coeff_arr, double[::1] tlist):
        self.cte = np.allclose(np.diff(tlist), tlist[1]-tlist[0])
        self.tlist = tlist
        self.coeff_arr = coeff_arr.copy()
        self.dt = tlist[1] - tlist[0]
        self.n_t = len(tlist)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        cdef complex coeff
        if self.cte:
            coeff = _step_complex_cte(t, self.tlist, self.coeff_arr, self.n_t)
        else:
            coeff = _step_complex_t(t, self.tlist, self.coeff_arr, self.n_t)
        return coeff

    def __reduce__(self):
        return (MakeStepCoefficient,
            (np.array(self.coeff_arr), np.array(self.tlist)))


def MakeStepCoefficient(coeff_arr, tlist):
    return StepCoefficient(coeff_arr, tlist)


@cython.auto_pickle(True)
cdef class SumCoefficient(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cpdef void arguments(self, dict args) except *:
        self.first.arguments(args)
        self.second.arguments(args)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return self.first._call(t) + self.second._call(t)

    """
    def __setstate__(self, state):
        self.first.__setstate__(state[0])
        self.second.__setstate__(state[1])

    def __getstate__(self):
        return (self.first.__getstate__(), self.second.__getstate__())"""

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})+({})".format(str1, str2)
        return ""


@cython.auto_pickle(True)
cdef class MulCoefficient(Coefficient):
    cdef Coefficient first
    cdef Coefficient second

    def __init__(self, Coefficient first, Coefficient second):
        self.first = first
        self.second = second

    cpdef void arguments(self, dict args) except *:
        self.first.arguments(args)
        self.second.arguments(args)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return self.first._call(t) * self.second._call(t)

    """def __setstate__(self, state):
        self.first.__setstate__(state[0])
        self.second.__setstate__(state[1])

    def __getstate__(self):
        return (self.first.__getstate__(), self.second.__getstate__())"""

    def optstr(self):
        str1 = self.first.optstr()
        str2 = self.second.optstr()
        if str1 and str2:
            return "({})*({})".format(str1, str2)
        return ""


@cython.auto_pickle(True)
cdef class ConjCoefficient(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cpdef void arguments(self, dict args) except *:
        self.base.arguments(args)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return conj(self.base._call(t))

    """def __setstate__(self, state):
        self.base.__setstate__(state)

    def __getstate__(self):
        return self.base.__getstate__()"""

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "conj({})".format(str1)
        return ""


@cython.auto_pickle(True)
cdef class NormCoefficient(Coefficient):
    cdef Coefficient base

    def __init__(self, Coefficient base):
        self.base = base

    cpdef void arguments(self, dict args) except *:
        self.base.arguments(args)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return norm(self.base._call(t))

    """def __setstate__(self, state):
        self.base.__setstate__(state)

    def __getstate__(self):
        return self.base.__getstate__()"""

    def optstr(self):
        str1 = self.base.optstr()
        if str1:
            return "norm({})".format(str1)
        return ""


@cython.auto_pickle(True)
cdef class ShiftCoefficient(Coefficient):
    cdef Coefficient base
    cdef double _t0

    def __init__(self, Coefficient base, double _t0):
        self.base = base
        self._t0 = _t0

    cpdef void arguments(self, dict args) except *:
        self._t0 = args["_t0"]
        self.base.arguments(args)

    @cython.initializedcheck(False)
    cdef complex _call(self, double t) except *:
        return self.base._call(t + self._t0)

    """def __setstate__(self, state):
        self.base.__setstate__(state)

    def __getstate__(self):
        return self.base.__getstate__()"""

    def optstr(self):
        from re import sub
        str1 = self.base.optstr()
        if str1:
            return sub("(?<=[^0-9a-zA-Z_])t(?=[^0-9a-zA-Z_])",
                       "(t+_t0)", " " + str1 + " ")
        return ""
